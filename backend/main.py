import os
import json
import logging
import posixpath
import re
from urllib.parse import urlparse, unquote, parse_qs
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth

from forecasting import (
    download_and_parse_csv,
    validate_and_prepare_sales_data,
    train_sarima_model,
    generate_forecast_payload,
    _DEFAULT_SUPABASE_HOST,
    CsvDownloadError,
    merge_sales_sources
)

from database import (
    insert_forecast_record,
    get_user_forecast_history,
    get_forecast_by_id,
    delete_forecast_by_id,
    create_daily_sales,
    update_daily_sales,
    get_user_daily_sales,
    get_user_all_daily_sales,
    check_database_readiness,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseIdempotencyConflictError,
    DailySalesConflictError
)

from dotenv import load_dotenv
_backend_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_backend_env_path)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
ALLOW_UNVERIFIED_DEV_MODE = os.getenv("ALLOW_UNVERIFIED_DEV_MODE", "false").strip().lower() == "true"

# Security Policy:
# Development bypass is ONLY allowed when ENVIRONMENT is explicitly "development" AND ALLOW_UNVERIFIED_DEV_MODE is true.
# In production (or any non-development environment), unverified mode is strictly forbidden.
IS_DEV_BYPASS_ACTIVE = (ENVIRONMENT == "development") and ALLOW_UNVERIFIED_DEV_MODE

if IS_DEV_BYPASS_ACTIVE:
    logger.warning("=" * 70)
    logger.warning("CRITICAL SECURITY NOTICE: Running in unverified development mode.")
    logger.warning("Token signatures are not verified. NEVER use this in production.")
    logger.warning("Set ENVIRONMENT=production in .env to enforce strict cryptographic authentication.")
    logger.warning("=" * 70)

# Initialize Firebase Admin
try:
    if not firebase_admin._apps:
        firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        if firebase_creds_path and os.path.exists(firebase_creds_path):
            cred = credentials.Certificate(firebase_creds_path)
        else:
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            private_key = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')
            client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
            
            if project_id and private_key and client_email:
                cred = credentials.Certificate({
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key": private_key,
                    "client_email": client_email,
                    "token_uri": "https://oauth2.googleapis.com/token",
                })
            else:
                cred = None
                
        if cred:
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK successfully initialized.")
        else:
            logger.warning("No Firebase Admin credentials configured.")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin: {type(e).__name__}")

def get_firestore_client():
    if firebase_admin._apps:
        try:
            return firestore.client()
        except Exception:
            return None
    return None

db = get_firestore_client()

app = FastAPI(title="SMART BPI Sales Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    sales_csv_url: Optional[str] = None
    forecast_days: int = 30
    idempotency_key: Optional[str] = None

class DailySalesCreateRequest(BaseModel):
    sale_date: str
    daily_revenue: float

class DailySalesUpdateRequest(BaseModel):
    daily_revenue: float
    sale_date: Optional[str] = None

async def verify_user(authorization: Optional[str] = Header(None)) -> str:
    # 1. Require Authorization Bearer header
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected 'Bearer <id_token>'."
        )
    
    token = authorization.split("Bearer ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Empty authentication token provided."
        )
    
    is_firebase_configured = bool(firebase_admin._apps)
    
    # 2. In production (or any non-dev environment), fail closed if credentials missing
    if ENVIRONMENT != "development":
        if not is_firebase_configured:
            logger.critical("PRODUCTION SECURITY FAULT: Request rejected because Firebase Admin SDK is not configured.")
            raise HTTPException(
                status_code=503,
                detail="Authentication service unavailable: Firebase Admin SDK is not configured."
            )
        
        try:
            decoded_token = firebase_auth.verify_id_token(token, check_revoked=True)
            uid = decoded_token.get("uid")
            if not uid:
                raise HTTPException(status_code=401, detail="Authentication failed: Token missing UID.")
            return uid
        except firebase_auth.ExpiredIdTokenError:
            raise HTTPException(status_code=401, detail="Authentication failed: Token has expired.")
        except firebase_auth.RevokedIdTokenError:
            raise HTTPException(status_code=401, detail="Authentication failed: Token has been revoked.")
        except firebase_auth.InvalidIdTokenError:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed: Invalid ID token."
            )
        except Exception as e:
            logger.error(f"Token verification error: {type(e).__name__}")
            raise HTTPException(status_code=401, detail="Authentication failed: Invalid ID token.")
            
    # 3. In development environment:
    if is_firebase_configured:
        try:
            decoded_token = firebase_auth.verify_id_token(token)
            return decoded_token["uid"]
        except firebase_auth.ExpiredIdTokenError:
            if not IS_DEV_BYPASS_ACTIVE:
                raise HTTPException(status_code=401, detail="Authentication failed: Token has expired.")
        except firebase_auth.RevokedIdTokenError:
            if not IS_DEV_BYPASS_ACTIVE:
                raise HTTPException(status_code=401, detail="Authentication failed: Token has been revoked.")
        except firebase_auth.InvalidIdTokenError:
            if not IS_DEV_BYPASS_ACTIVE:
                raise HTTPException(status_code=401, detail="Authentication failed: Invalid ID token.")
        except Exception as e:
            logger.error(f"Dev token verification error: {type(e).__name__}")
            if not IS_DEV_BYPASS_ACTIVE:
                raise HTTPException(status_code=401, detail="Authentication failed: Invalid ID token.")
    
    if IS_DEV_BYPASS_ACTIVE and ENVIRONMENT == "development":
        try:
            import jwt
            decoded = jwt.decode(token, options={"verify_signature": False})
            uid = decoded.get("user_id") or decoded.get("uid") or decoded.get("sub")
            if uid:
                return str(uid)
            return "demo-user"
        except Exception:
            return "demo-user"

    raise HTTPException(
        status_code=503,
        detail="Firebase Admin SDK is not configured. For development testing, configure credentials or enable ALLOW_UNVERIFIED_DEV_MODE=True in development only."
    )


def validate_sales_storage_url_ownership(url_str: str, authenticated_uid: str) -> None:
    """
    Validates that url_str is a safe, authentic storage URL belonging to authenticated_uid.
    Strictly prevents URL encoding bypasses, double encoding, directory traversal,
    untrusted hosts/buckets, and cross-tenant access.

    Raises:
        HTTPException(400): If URL format, scheme, host, port, bucket, or structure is invalid.
        HTTPException(403): If the URL points to another user's sales data.
    """
    if not url_str or not isinstance(url_str, str):
        raise HTTPException(status_code=400, detail="Invalid storage URL: URL cannot be empty.")

    # 1. Reject backslashes which can confuse path normalizers
    if "\\" in url_str:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Backslashes are not permitted.")

    # Reject encoded backslashes (%5c, %5C)
    if "%5c" in url_str.lower():
        raise HTTPException(status_code=400, detail="Invalid storage URL: Encoded backslashes are not permitted.")

    try:
        parsed = urlparse(url_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Failed to parse URL.")

    # 2. Scheme must be strictly HTTPS
    if parsed.scheme.lower() != "https":
        raise HTTPException(status_code=400, detail="Invalid storage URL: Only HTTPS URLs are permitted.")

    # 3. Port: None or 443 only (handle malformed ports as HTTP 400)
    try:
        port = parsed.port
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Malformed port.")

    if port and port != 443:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Unexpected port.")

    # 4. No embedded user credentials (userinfo)
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Embedded credentials are not permitted.")

    # 5. No fragments
    if parsed.fragment:
        raise HTTPException(status_code=400, detail="Invalid storage URL: URL fragments are not permitted.")

    # 6. Check for malformed percent encoding or double URL encoding
    raw_path = parsed.path
    if "%" in raw_path:
        # Check for malformed hex like %zz or truncated %
        if re.search(r"%[^0-9a-fA-F]|%[0-9a-fA-F]?$", raw_path):
            raise HTTPException(status_code=400, detail="Invalid storage URL: Malformed percent-encoding.")
        # Check for double encoding (e.g. %252e)
        first_unquote = unquote(raw_path)
        second_unquote = unquote(first_unquote)
        if first_unquote != second_unquote:
            raise HTTPException(status_code=400, detail="Invalid storage URL: Double URL encoding is not permitted.")
    else:
        first_unquote = raw_path

    # Validate decoded raw path segments before normalization so traversal cannot be silently collapsed
    raw_segments = [s for s in first_unquote.split("/") if s]
    if any(s in (".", "..") for s in raw_segments):
        raise HTTPException(status_code=400, detail="Invalid storage URL: Path traversal segments are not permitted.")

    # Normalize path (collapsing /./ and /../)
    normalized_path = posixpath.normpath(first_unquote)

    hostname = (parsed.hostname or "").lower()
    trusted_supabase_host = os.getenv("TRUSTED_SUPABASE_HOST", _DEFAULT_SUPABASE_HOST).strip().lower()
    trusted_firebase_host = os.getenv("TRUSTED_FIREBASE_STORAGE_HOST", "firebasestorage.googleapis.com").strip().lower()

    path_owner = None

    if hostname == trusted_supabase_host:
        # In Supabase Storage, encoded path separators (%2f, %2F) are not permitted
        if "%2f" in raw_path.lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid storage URL: Encoded path separators are not permitted in storage URL."
            )

        # Supabase Storage structure:
        # /storage/v1/object/public/{bucket}/{uid}/{filename...}
        # Validate using exact path segments (not startswith) on raw_segments
        if len(raw_segments) < 7:
            raise HTTPException(
                status_code=400,
                detail="Invalid storage URL: Missing user folder or filename in storage path."
            )

        if (
            raw_segments[0] != "storage"
            or raw_segments[1] != "v1"
            or raw_segments[2] != "object"
            or raw_segments[3] != "public"
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid storage URL: Invalid Supabase storage path structure."
            )

        expected_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "service-data").strip().lower()
        if raw_segments[4].lower() != expected_bucket:
            raise HTTPException(
                status_code=400,
                detail="Invalid storage URL: Path does not match the expected service-data storage bucket."
            )

        path_owner = raw_segments[5]

        # Explicit safe query-parameter policy for Supabase URLs
        if parsed.query:
            sb_params = parse_qs(parsed.query)
            allowed_sb_params = {"token", "download", "t", "v", "version"}
            for key in sb_params.keys():
                if key not in allowed_sb_params:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid storage URL: Unsupported query parameter '{key}'."
                    )

    elif hostname == trusted_firebase_host:
        # Firebase Storage structure:
        # /v0/b/{bucket}/o/{object_path}
        if "/o/" in raw_path:
            prefix_part = raw_path.split("/o/")[0]
            if "%2f" in prefix_part.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Invalid storage URL: Encoded path separators in storage prefix are not permitted."
                )

        if not normalized_path.startswith("/v0/b/"):
            raise HTTPException(status_code=400, detail="Invalid storage URL: Invalid Firebase storage path structure.")
        parts = [p for p in normalized_path.split("/") if p]
        if len(parts) < 5 or parts[0] != "v0" or parts[1] != "b" or parts[3] != "o":
            raise HTTPException(status_code=400, detail="Invalid storage URL: Malformed Firebase storage segments.")
        
        expected_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "final-business-os.firebasestorage.app").strip().lower()
        if parts[2].lower() != expected_bucket:
            raise HTTPException(status_code=400, detail="Invalid storage URL: Storage bucket does not match configured bucket.")
        
        obj_path = "/".join(parts[4:])
        obj_segments = [s for s in obj_path.split("/") if s]
        if not obj_segments:
            raise HTTPException(status_code=400, detail="Invalid storage URL: Missing object path.")

        # Reject path escapes and traversal segments inside the object path
        for seg in obj_segments:
            if seg in (".", "..") or "\\" in seg:
                raise HTTPException(status_code=400, detail="Invalid storage URL: Invalid path segment.")
        
        if obj_segments[0] == "service-data":
            if len(obj_segments) < 3:
                raise HTTPException(status_code=400, detail="Invalid storage URL: Missing user folder or filename.")
            path_owner = obj_segments[1]
        else:
            if len(obj_segments) < 2:
                raise HTTPException(status_code=400, detail="Invalid storage URL: Missing user folder or filename.")
            path_owner = obj_segments[0]

        if parsed.query:
            params = parse_qs(parsed.query)
            allowed_params = {"alt", "token"}
            for key in params.keys():
                if key not in allowed_params:
                    raise HTTPException(status_code=400, detail=f"Invalid storage URL: Unsupported query parameter '{key}'.")
    else:
        raise HTTPException(status_code=400, detail="Invalid storage URL: Storage host is not trusted.")

    # 7. Check path owner against authenticated Firebase UID
    if not path_owner or path_owner in (".", "..") or path_owner != authenticated_uid:
        logger.warning(
            f"Tenant isolation breach attempt: user '{authenticated_uid}' attempted accessing storage path belonging to another tenant."
        )
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to access another user's sales data."
        )


@app.get("/live")
def liveness_check():
    """
    Liveness probe: confirms that the FastAPI application process is up and running.
    """
    return {"status": "alive"}


@app.get("/ready")
def readiness_check():
    """
    Readiness probe: confirms that the database configuration is available
    and a lightweight database query succeeds.
    Fails closed with HTTP 503 if database is unconfigured or unreachable.
    """
    db_readiness = check_database_readiness()
    if not db_readiness.get("ready", False):
        raise HTTPException(
            status_code=503,
            detail="Database service is not ready."
        )
    return {
        "status": "ready",
        "database": "connected"
    }


@app.get("/")
@app.get("/health")
def health_check(response: Response):
    """
    Consolidated health check endpoint for monitoring dashboards.
    Returns 200 when database is configured and ready, or 503 when degraded.
    """
    db_readiness = check_database_readiness()
    is_ready = bool(db_readiness.get("ready", False))
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "healthy" if is_ready else "degraded",
        "environment": ENVIRONMENT,
        "auth_configured": bool(firebase_admin._apps),
        "database_configured": is_ready
    }


@app.post("/forecast")
def generate_forecast(
    req: ForecastRequest,
    uid: str = Depends(verify_user),
    idempotency_header: Optional[str] = Header(None, alias="Idempotency-Key")
):
    if req.forecast_days not in [30, 180, 365]:
        raise HTTPException(status_code=400, detail="forecast_days must be 30, 180, or 365")
    
    url = str(req.sales_csv_url).strip() if req.sales_csv_url else None
    df_csv = None

    if url:
        # 1. Path-level User Ownership Isolation & URL Structure Validation
        validate_sales_storage_url_ownership(url, uid)

        # 2. Firestore Business Profile Ownership Validation
        current_db = get_firestore_client() or db
        if current_db is not None:
            try:
                doc_ref = current_db.collection("businesses").document(uid)
                doc_snap = doc_ref.get()
                if not doc_snap.exists:
                    raise HTTPException(status_code=404, detail="Business profile not found for this user.")
                
                business_data = doc_snap.to_dict() or {}
                services_data = business_data.get("servicesData", {})
                stored_url = services_data.get("salesCsvUrl")
                
                if not stored_url:
                    raise HTTPException(status_code=404, detail="No sales CSV found in user's business profile.")
                
                if stored_url != url:
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: The provided sales_csv_url does not match the sales data registered to your business."
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Firestore verification error: {type(e).__name__}")
                raise HTTPException(status_code=500, detail="Failed to verify business ownership.")
        else:
            if ENVIRONMENT != "development" or not IS_DEV_BYPASS_ACTIVE:
                raise HTTPException(
                    status_code=503,
                    detail="Database service unavailable: Firestore is not configured for ownership verification."
                )
            logger.warning(f"Development mode: Skipping Firestore document check for user '{uid}'. Path isolation verified.")

        # 3. Download and parse CSV (Sanitized error reporting)
        try:
            df_csv = download_and_parse_csv(url)
        except CsvDownloadError as e:
            err_msg = str(e).lower()
            if "too large" in err_msg or "limit" in err_msg:
                category = "FILE_TOO_LARGE"
                status_code = 400
                detail = "Sales CSV exceeds the maximum allowed file size."
            elif "404" in err_msg or "not found" in err_msg:
                category = "NOT_FOUND"
                status_code = 404
                detail = "Sales CSV file was not found in storage."
            elif "timeout" in err_msg or "timed out" in err_msg:
                category = "TIMEOUT"
                status_code = 504
                detail = "Storage server timed out while retrieving sales CSV."
            elif "connection" in err_msg or "dropped" in err_msg:
                category = "CONNECTION_FAILED"
                status_code = 502
                detail = "Storage service connection failed while retrieving sales CSV."
            elif "invalid csv format" in err_msg or "format" in err_msg:
                category = "MALFORMED_CSV"
                status_code = 400
                detail = "Invalid CSV format: file could not be parsed as tabular CSV."
            else:
                category = "DOWNLOAD_ERROR"
                status_code = 400
                detail = "Unable to retrieve or parse the provided sales CSV."

            logger.error(f"CSV download error for user '{uid}': category={category}")
            raise HTTPException(status_code=status_code, detail=detail)
        except ValueError as e:
            logger.error(f"CSV parse error for user '{uid}': {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Invalid CSV data format.")
        except Exception as e:
            logger.error(f"Unexpected CSV retrieval error for user '{uid}': {type(e).__name__}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the sales CSV.")

    # 4. Fetch authenticated manual daily sales entries
    try:
        manual_entries = get_user_all_daily_sales(uid)
    except Exception as e:
        logger.warning(f"Could not load manual daily sales for user '{uid}': {e}")
        manual_entries = []

    # 5. Merge sales sources according to SMART BPI Data-Source Policy
    try:
        merged_df = merge_sales_sources(df_csv, manual_entries)
    except DailySalesConflictError as e:
        logger.warning(f"Sales data conflict detected for user '{uid}': {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        logger.warning(f"Sales data merge error for user '{uid}': {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # 6. Validate and prepare data
    try:
        series = validate_and_prepare_sales_data(merged_df)
    except ValueError as e:
        logger.warning(f"Sales data validation failure for user '{uid}': {type(e).__name__}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected validation error for user '{uid}': {type(e).__name__}")
        raise HTTPException(status_code=400, detail="Sales data validation failed.")

    # 7. Train SARIMA
    try:
        model_results = train_sarima_model(series)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model training for user '{uid}': {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error during model training.")

    # 8. Generate forecast payload
    try:
        response_payload = generate_forecast_payload(model_results, series, req.forecast_days)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 9. Persist forecast in Supabase database
    custom_key = req.idempotency_key or idempotency_header
    try:
        stored_record = insert_forecast_record(
            user_id=uid,
            forecast_days=req.forecast_days,
            csv_source_url=url or "manual_daily_sales",
            series=series,
            forecast_payload=response_payload,
            custom_idempotency_key=custom_key
        )
        # Augment response payload with persisted record identifier and timestamp
        response_payload["id"] = stored_record.get("id")
        if "created_at" in stored_record:
            response_payload["created_at"] = stored_record["created_at"]
    except DatabaseIdempotencyConflictError as e:
        logger.warning(f"Idempotency conflict for user '{uid}': {type(e).__name__}")
        raise HTTPException(
            status_code=409,
            detail="Idempotency key was previously used with different forecast parameters or data."
        )
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Persistence error during /forecast: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save forecast record to database. Persistence aborted."
        )
    except Exception as e:
        logger.error(f"Unexpected persistence error during /forecast: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while saving forecast results."
        )

    return response_payload


@app.get("/forecast/history")
def get_forecast_history(
    limit: int = 20,
    offset: int = 0,
    uid: str = Depends(verify_user)
):
    """
    Retrieves the authenticated user's forecast history, sorted newest first.
    Strictly scoped to the verified Firebase UID.
    """
    try:
        history_result = get_user_forecast_history(user_id=uid, limit=limit, offset=offset)
        return history_result
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in /forecast/history: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to retrieve forecast history from database.")
    except Exception as e:
        logger.error(f"Unexpected error in /forecast/history: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error retrieving forecast history.")


@app.get("/forecast/{forecast_id}")
def get_single_forecast(
    forecast_id: str,
    uid: str = Depends(verify_user)
):
    """
    Retrieves a single forecast record by ID.
    Strictly scoped to both the forecast ID AND verified Firebase UID.
    """
    try:
        record = get_forecast_by_id(user_id=uid, forecast_id=forecast_id)
        if not record:
            raise HTTPException(status_code=404, detail="Forecast record not found.")
        return record
    except HTTPException:
        raise
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in GET /forecast/{forecast_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to retrieve forecast record.")
    except Exception as e:
        logger.error(f"Unexpected error in GET /forecast/{forecast_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error retrieving forecast record.")


@app.delete("/forecast/{forecast_id}")
def delete_single_forecast(
    forecast_id: str,
    uid: str = Depends(verify_user)
):
    """
    Deletes a single forecast record owned by the authenticated user.
    """
    try:
        deleted = delete_forecast_by_id(user_id=uid, forecast_id=forecast_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Forecast record not found.")
        return {"message": "Forecast record deleted successfully.", "id": forecast_id}
    except HTTPException:
        raise
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in DELETE /forecast/{forecast_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to delete forecast record.")
    except Exception as e:
        logger.error(f"Unexpected error in DELETE /forecast/{forecast_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error deleting forecast record.")


# ---------------------------------------------------------------------------
# Daily Sales Endpoints
# ---------------------------------------------------------------------------

@app.post("/sales/daily", status_code=status.HTTP_201_CREATED)
def create_daily_sales_entry(
    req: DailySalesCreateRequest,
    uid: str = Depends(verify_user)
):
    """
    Creates an authenticated daily sales record for the user's business.
    Fails closed with HTTP 409 if a record for the date already exists.
    """
    try:
        record = create_daily_sales(
            user_id=uid,
            sale_date=req.sale_date,
            daily_revenue=req.daily_revenue
        )
        return record
    except DailySalesConflictError as e:
        logger.warning(f"Daily sales conflict for user '{uid}': {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        logger.warning(f"Daily sales validation error for user '{uid}': {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in POST /sales/daily for user '{uid}': {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to save daily sales entry to database.")
    except Exception as e:
        logger.error(f"Unexpected error in POST /sales/daily for user '{uid}': {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error creating daily sales entry.")


@app.get("/sales/daily")
def list_daily_sales_entries(
    limit: int = 50,
    offset: int = 0,
    uid: str = Depends(verify_user)
):
    """
    Retrieves paginated daily sales entries for the authenticated user, sorted by date DESC.
    """
    try:
        res = get_user_daily_sales(user_id=uid, limit=limit, offset=offset)
        return res
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in GET /sales/daily: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to retrieve daily sales entries.")
    except Exception as e:
        logger.error(f"Unexpected error in GET /sales/daily: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error retrieving daily sales entries.")


@app.put("/sales/daily/{entry_id}")
def update_daily_sales_entry(
    entry_id: str,
    req: DailySalesUpdateRequest,
    uid: str = Depends(verify_user)
):
    """
    Updates an existing daily sales record owned by the authenticated user.
    """
    try:
        record = update_daily_sales(
            user_id=uid,
            entry_id=entry_id,
            daily_revenue=req.daily_revenue,
            sale_date=req.sale_date
        )
        if not record:
            raise HTTPException(status_code=404, detail="Daily sales entry not found.")
        return record
    except HTTPException:
        raise
    except DailySalesConflictError as e:
        logger.warning(f"Daily sales update conflict for user '{uid}': {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        logger.warning(f"Daily sales update validation error for user '{uid}': {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except (DatabaseConfigurationError, DatabaseConnectionError) as e:
        logger.error(f"Database error in PUT /sales/daily/{entry_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to update daily sales entry.")
    except Exception as e:
        logger.error(f"Unexpected error in PUT /sales/daily/{entry_id}: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Unexpected error updating daily sales entry.")
