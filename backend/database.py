"""
SMART BPI — Backend Database Persistence Infrastructure
Connects to Supabase PostgreSQL via the Supabase Python SDK using service-role credentials.

Security Guarantees:
- Runs exclusively on the FastAPI backend.
- Credentials (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) are read from backend environment only.
- Credentials, private URLs, and internal SQL connection strings are NEVER exposed in return values, exceptions, or logs.
- All database operations are strictly scoped to the verified Firebase UID.
- Idempotency key mechanism prevents duplicate records upon network retries.
"""

import os
import math
from datetime import datetime, timezone
import hashlib
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure backend-specific .env is loaded regardless of process launch directory
_backend_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_backend_env_path)
load_dotenv()

# Singleton client instance
_supabase_client = None


class DatabaseConfigurationError(Exception):
    """Raised when required Supabase database environment variables are missing or invalid."""
    pass


class DatabaseConnectionError(Exception):
    """Raised when the database connection or operation fails."""
    pass


class DatabaseIdempotencyConflictError(Exception):
    """Raised when an idempotency key is reused with conflicting forecast parameters or data."""
    pass


class DailySalesConflictError(ValueError):
    """Raised when attempting to insert or update a daily sales record on a date that already exists."""
    pass


def get_supabase_config() -> tuple[str, str]:
    """
    Safely retrieves Supabase credentials from the backend environment.
    Never logs or exposes the secret values.
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return url, key


def get_supabase_client():
    """
    Returns an initialized Supabase Client instance.
    Raises DatabaseConfigurationError if environment variables are missing.
    Raises DatabaseConnectionError if SDK initialization fails.
    
    Guarantees:
    - Never prints or logs the secret key.
    - Error messages are sanitized before propagation.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url, key = get_supabase_config()

    if not url:
        logger.error("Database connection failure: SUPABASE_URL is not configured.")
        raise DatabaseConfigurationError("Supabase database configuration missing: SUPABASE_URL not configured.")

    if not key:
        logger.error("Database connection failure: SUPABASE_SERVICE_ROLE_KEY is not configured.")
        raise DatabaseConfigurationError("Supabase database configuration missing: SUPABASE_SERVICE_ROLE_KEY not configured.")

    try:
        from supabase import create_client, Client
    except ImportError:
        logger.error("Supabase Python SDK is not installed in the backend environment.")
        raise DatabaseConnectionError("Supabase Python SDK ('supabase') is not installed. Please install requirements.") from None

    try:
        _supabase_client = create_client(url, key)
        logger.info("Supabase database client successfully initialized.")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {type(e).__name__}")
        raise DatabaseConnectionError("Database initialization failure. Please verify Supabase service credentials.") from None


def reset_supabase_client() -> None:
    """Resets the singleton client instance (primarily used for unit testing)."""
    global _supabase_client
    _supabase_client = None


def check_database_readiness() -> Dict[str, Any]:
    """
    Performs a safe, lightweight readiness inspection of the database connection
    and verifies that the 'sales_forecasts' table is accessible.
    Does not crash on unconfigured environments or missing dependencies.
    Never exposes credentials, private URLs, or internal keys.
    """
    url, key = get_supabase_config()
    url_configured = bool(url and url.startswith("https://"))
    key_configured = bool(key and len(key) > 20)

    # 1. Check SDK availability
    sdk_installed = False
    try:
        import supabase
        sdk_installed = True
    except ImportError:
        sdk_installed = False

    if not sdk_installed:
        return {
            "ready": False,
            "sdk_installed": False,
            "url_configured": url_configured,
            "service_role_key_configured": key_configured,
            "table_accessible": False,
            "status": "missing_dependency",
            "message": "Supabase Python SDK ('supabase') is not installed in the Python environment."
        }

    # 2. Check configuration presence
    if not url_configured or not key_configured:
        missing = []
        if not url_configured:
            missing.append("SUPABASE_URL")
        if not key_configured:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")

        return {
            "ready": False,
            "sdk_installed": True,
            "url_configured": url_configured,
            "service_role_key_configured": key_configured,
            "table_accessible": False,
            "status": "missing_configuration",
            "message": f"Database unconfigured: missing {', '.join(missing)}."
        }

    # 3. Attempt client creation and verify table reachability
    try:
        client = get_supabase_client()
        # Safe, lightweight query: check if table is reachable without loading heavy rows
        client.table("sales_forecasts").select("id").limit(1).execute()
        return {
            "ready": True,
            "sdk_installed": True,
            "url_configured": True,
            "service_role_key_configured": True,
            "table_accessible": True,
            "status": "ready",
            "message": "Database and sales_forecasts table are accessible."
        }
    except DatabaseConfigurationError:
        return {
            "ready": False,
            "sdk_installed": True,
            "url_configured": url_configured,
            "service_role_key_configured": key_configured,
            "table_accessible": False,
            "status": "missing_configuration",
            "message": "Database configuration missing."
        }
    except Exception as e:
        logger.error(f"Database readiness query failed: {type(e).__name__}")
        return {
            "ready": False,
            "sdk_installed": True,
            "url_configured": True,
            "service_role_key_configured": True,
            "table_accessible": False,
            "status": "unreachable",
            "message": "Database table unreachable or connection failed."
        }


def _canonical_numeric_str(val: Any) -> str:
    """
    Returns a deterministic, lossless canonical string for a numeric value.
    Rejects NaN and infinity according to input validation rules.
    Uses exact IEEE-754 hex representation for floats so that differences smaller
    than 0.0001 are preserved losslessly without arbitrary rounding.
    """
    if val is None or pd.isna(val):
        raise ValueError("Invalid numeric value in dataset: NaN or None is not permitted.")
    try:
        f = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Non-numeric value in sales data: {val}")

    if math.isnan(f) or math.isinf(f):
        raise ValueError("Invalid numeric value in dataset: NaN or infinity is not permitted.")

    # Canonicalize -0.0 to 0.0
    if f == 0.0:
        f = 0.0

    return f.hex()


def compute_series_hash(series: Any) -> str:
    """
    Computes a deterministic, lossless SHA-256 fingerprint of the historical sales series.
    
    Guarantees:
    - Sorts the dataset by chronological date index, guaranteeing deterministic ordering.
    - Uses exact, lossless IEEE-754 representation (float.hex) for all revenue values.
    - Rejects NaN, infinity, and invalid numeric values.
    - Accurately differentiates datasets with differences smaller than 0.0001.
    - Reordered datasets with identical dates and values yield the exact same fingerprint.
    """
    if series is None:
        return ""

    if hasattr(series, "items"):
        # Sort series by its index (dates) for stable deterministic ordering
        try:
            sorted_series = series.sort_index()
        except Exception as e:
            raise ValueError(f"Failed to sort sales series by date index: {type(e).__name__}") from e

        entries = []
        for k, v in sorted_series.items():
            if pd.isna(k):
                raise ValueError("Missing or invalid date index detected in sales data.")
            # Format date index stably
            if hasattr(k, "strftime"):
                date_str = k.strftime("%Y-%m-%d")
            else:
                date_str = str(k).strip()
            val_str = _canonical_numeric_str(v)
            entries.append(f"{date_str}:{val_str}")
        content = "|".join(entries)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    elif hasattr(series, "tolist"):
        entries = [_canonical_numeric_str(x) for x in series.tolist()]
        content = "|".join(entries)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    else:
        raise ValueError(f"Unsupported sales series input type '{type(series).__name__}'. Expected pandas Series or list.")


DEFAULT_MODEL_VERSION = "damped_ets_v3"


def compute_idempotency_key(
    user_id: str,
    forecast_days: int,
    series: Any = None,
    data_hash: Optional[str] = None,
    csv_source_url: str = "",
    historical_end_date: Optional[str] = None,
    forecast_start_date: str = "",
    custom_idempotency_key: Optional[str] = None,
    model_version: Optional[str] = DEFAULT_MODEL_VERSION
) -> str:
    """
    Computes a deterministic SHA-256 idempotency key for a forecast run.
    Incorporates:
    - Authenticated Firebase UID (user_id)
    - Forecast horizon (forecast_days: 30, 180, 365)
    - Validated historical sales data hash (or historical_end_date fallback if series is omitted)
    - Forecast start date
    - CSV source URL
    - Model version identifier (e.g. 'sarima_v2_stationary')

    Guarantees:
    - Always scoped to authenticated user_id.
    - If custom_idempotency_key is provided, scopes it to user_id to prevent cross-tenant collision:
      f"custom:{user_id}:{custom_idempotency_key}".
    - Network retries with identical parameters & sales data produce the exact same key.
    - If the sales CSV is updated with new or changed data at the same URL, the series data hash changes,
      producing a distinct key and allowing a new forecast to be stored.
    - Changing the forecast horizon (e.g. 30 vs 180 vs 365) produces distinct keys.
    - Different model versions produce distinct keys, allowing algorithm upgrades without serving stale forecasts.
    """
    if custom_idempotency_key:
        clean_key = str(custom_idempotency_key).strip()
        return f"custom:{user_id}:{clean_key}"

    if not data_hash:
        if series is not None:
            data_hash = compute_series_hash(series)
        else:
            data_hash = f"hist_end_{historical_end_date or ''}"

    version_suffix = f":{model_version.strip()}" if model_version and str(model_version).strip() else ""
    raw_material = f"{user_id}:{forecast_days}:{data_hash}:{forecast_start_date}:{csv_source_url}{version_suffix}"
    return hashlib.sha256(raw_material.encode("utf-8")).hexdigest()


def insert_forecast_record(
    user_id: str,
    forecast_days: int,
    csv_source_url: str,
    series: Any,
    forecast_payload: Dict[str, Any],
    custom_idempotency_key: Optional[str] = None,
    model_version: Optional[str] = DEFAULT_MODEL_VERSION
) -> Dict[str, Any]:
    """
    Persists a generated forecast run in the Supabase PostgreSQL 'sales_forecasts' table.
    
    Reliability & Security Guarantees:
    - user_id is strictly sourced from the verified Firebase UID.
    - Idempotency key prevents duplicate insertions.
    - Model version prevents stale cache hits from older or incompatible models.
    - If idempotency lookup succeeds and finds an existing record, validates that it matches
      the logical request parameters before returning it (prevents conflicting key reuse).
    - If idempotency lookup fails due to a database error, aborts persistence immediately
      (never silently converts a database outage into an idempotency miss).
    - If concurrent identical inserts race on the unique constraint, handles the conflict,
      confirms user ownership and request matching, and retrieves the existing record.
    - Error details are sanitized before raising.
    """
    client = get_supabase_client()

    summary = dict(forecast_payload.get("summary", {}))
    records = forecast_payload.get("forecast", [])

    # Always compute the dataset fingerprint regardless of whether custom_idempotency_key is provided
    data_hash = compute_series_hash(series)
    summary["dataset_fingerprint"] = data_hash
    if model_version:
        summary["model_version"] = model_version

    # Extract historical date boundaries
    historical_start_date = None
    historical_end_date = None
    try:
        if hasattr(series, "index") and len(series) > 0:
            historical_start_date = series.index[0].strftime("%Y-%m-%d")
            historical_end_date = series.index[-1].strftime("%Y-%m-%d")
    except Exception:
        pass

    # Map forecast_total and forecast_average based on horizon
    if forecast_days == 30:
        forecast_total = summary.get("forecast_30_days_total", 0.0)
        forecast_average = summary.get("forecast_30_days_average", 0.0)
    elif forecast_days == 180:
        forecast_total = summary.get("forecast_6_months_total", 0.0)
        forecast_average = summary.get("forecast_6_months_average", 0.0)
    else:  # 365
        forecast_total = summary.get("forecast_1_year_total", 0.0)
        forecast_average = summary.get("forecast_1_year_average", 0.0)

    forecast_start_date = summary.get("forecast_start_date", "")
    forecast_end_date = summary.get("forecast_end_date", "")

    idempotency_key = compute_idempotency_key(
        user_id=user_id,
        forecast_days=forecast_days,
        series=series,
        data_hash=data_hash,
        csv_source_url=csv_source_url,
        historical_end_date=historical_end_date,
        forecast_start_date=forecast_start_date,
        custom_idempotency_key=custom_idempotency_key,
        model_version=model_version
    )

    # 1. Idempotency Check: check if this forecast already exists
    try:
        existing_query = client.table("sales_forecasts") \
            .select("id, user_id, forecast_days, forecast_start_date, forecast_end_date, forecast_total, forecast_average, expected_growth_percentage, csv_source_url, idempotency_key, summary_metrics, created_at") \
            .eq("user_id", user_id) \
            .eq("idempotency_key", idempotency_key) \
            .limit(1) \
            .execute()

        if existing_query.data and len(existing_query.data) > 0:
            existing = existing_query.data[0]
            existing_summary = existing.get("summary_metrics") or {}
            existing_fingerprint = existing_summary.get("dataset_fingerprint") if isinstance(existing_summary, dict) else None

            # Verify that the existing record matches the same logical request parameters and dataset fingerprint.
            # Fail closed: existing record MUST have a valid dataset_fingerprint matching the incoming data_hash.
            if (
                existing.get("forecast_days") != forecast_days
                or existing.get("csv_source_url") != csv_source_url
                or existing.get("forecast_start_date") != forecast_start_date
                or not existing_fingerprint
                or existing_fingerprint != data_hash
            ):
                logger.warning(f"Conflicting idempotency key reuse for user '{user_id}'.")
                raise DatabaseIdempotencyConflictError(
                    "Idempotency key was previously used with different forecast parameters or data."
                )
            logger.info("Idempotent forecast hit: returning existing record.")
            return existing

        # Fallback check for existing stationary 365-day forecast saved before model versioning was introduced
        if forecast_days == 365 and model_version == DEFAULT_MODEL_VERSION and not custom_idempotency_key:
            legacy_key = compute_idempotency_key(
                user_id=user_id,
                forecast_days=forecast_days,
                series=series,
                data_hash=data_hash,
                csv_source_url=csv_source_url,
                historical_end_date=historical_end_date,
                forecast_start_date=forecast_start_date,
                custom_idempotency_key=None,
                model_version=None
            )
            if legacy_key != idempotency_key:
                legacy_query = client.table("sales_forecasts") \
                    .select("id, user_id, forecast_days, forecast_start_date, forecast_end_date, forecast_total, forecast_average, expected_growth_percentage, csv_source_url, idempotency_key, summary_metrics, created_at") \
                    .eq("user_id", user_id) \
                    .eq("idempotency_key", legacy_key) \
                    .limit(1) \
                    .execute()
                if legacy_query.data and len(legacy_query.data) > 0:
                    candidate = legacy_query.data[0]
                    cand_summary = candidate.get("summary_metrics") or {}
                    cand_fingerprint = cand_summary.get("dataset_fingerprint")
                    one_year_tot = float(cand_summary.get("forecast_1_year_total", 0.0))
                    # Confirm that candidate is the valid stationary 365-day run (< 10^13, matching fingerprint & date)
                    if cand_fingerprint == data_hash and candidate.get("forecast_start_date") == forecast_start_date and one_year_tot < 1e13:
                        logger.info("Idempotent forecast hit: matched existing verified stationary 365-day forecast record.")
                        return candidate
    except DatabaseIdempotencyConflictError:
        raise
    except Exception as e:
        logger.error(f"Database error during idempotency check: {type(e).__name__}")
        raise DatabaseConnectionError("Failed to check existing forecast in database. Persistence aborted.") from None

    # 2. Prepare payload for insertion
    insert_data = {
        "user_id": user_id,
        "forecast_days": int(forecast_days),
        "historical_start_date": historical_start_date,
        "historical_end_date": historical_end_date,
        "historical_total_sales": float(summary.get("historical_total_sales", 0.0)),
        "historical_average_daily_sales": float(summary.get("historical_average_daily_sales", 0.0)),
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": forecast_end_date,
        "forecast_total": round(float(forecast_total), 2),
        "forecast_average": round(float(forecast_average), 2),
        "expected_growth_percentage": float(summary.get("expected_growth_percentage", 0.0)),
        "csv_source_url": csv_source_url,
        "forecast_records": records,
        "summary_metrics": summary,
        "idempotency_key": idempotency_key
    }

    try:
        res = client.table("sales_forecasts").insert(insert_data).execute()
        if not res.data or len(res.data) == 0:
            raise DatabaseConnectionError("Database insert returned no data.")
        logger.info(f"Forecast successfully persisted for user '{user_id}'. Record ID: {res.data[0].get('id')}")
        return res.data[0]
    except Exception as e:
        # Check if a race-condition unique constraint violation occurred on idempotency_key
        err_str = str(e).lower()
        if "idempotency_key" in err_str or "duplicate" in err_str or "unique" in err_str or "23505" in err_str:
            logger.info("Unique constraint conflict on idempotency_key; retrieving existing record.")
            try:
                retry_lookup = client.table("sales_forecasts") \
                    .select("id, user_id, forecast_days, forecast_start_date, forecast_end_date, forecast_total, forecast_average, expected_growth_percentage, csv_source_url, idempotency_key, summary_metrics, created_at") \
                    .eq("idempotency_key", idempotency_key) \
                    .limit(1) \
                    .execute()
                if retry_lookup.data and len(retry_lookup.data) > 0:
                    rec = retry_lookup.data[0]
                    # Verify owner matches authenticated user
                    if rec.get("user_id") != user_id:
                        logger.error(f"Security fault: Idempotency collision across different users ({rec.get('user_id')} != {user_id})")
                        raise DatabaseConnectionError("Idempotency conflict could not be resolved safely.")

                    rec_summary = rec.get("summary_metrics") or {}
                    rec_fingerprint = rec_summary.get("dataset_fingerprint") if isinstance(rec_summary, dict) else None

                    # Verify matching logical parameters and dataset fingerprint.
                    # Fail closed: conflicting record MUST have a valid dataset_fingerprint matching incoming data_hash.
                    if (
                        rec.get("forecast_days") != forecast_days
                        or rec.get("csv_source_url") != csv_source_url
                        or rec.get("forecast_start_date") != forecast_start_date
                        or not rec_fingerprint
                        or rec_fingerprint != data_hash
                    ):
                        logger.warning(f"Conflicting idempotency key reuse on concurrent conflict for user '{user_id}'.")
                        raise DatabaseIdempotencyConflictError(
                            "Idempotency key was previously used with different forecast parameters or data."
                        )
                    return rec
                else:
                    logger.error("Unique constraint error occurred but conflicting record could not be found.")
                    raise DatabaseConnectionError("Unique constraint conflict could not be resolved.")
            except DatabaseIdempotencyConflictError:
                raise
            except DatabaseConnectionError:
                raise
            except Exception as lookup_err:
                logger.error(f"Failed to retrieve conflicting record: {type(lookup_err).__name__}")
                raise DatabaseConnectionError("Concurrent insertion conflict could not be resolved.") from None

        logger.error(f"Failed to persist forecast in Supabase: {type(e).__name__}")
        raise DatabaseConnectionError("Failed to store forecast in database. Persistence aborted.") from None


def get_user_forecast_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Retrieves paginated forecast history exclusively for the authenticated user.
    Results are sorted newest first.
    Ommits heavy timeseries arrays from list view for network efficiency while retaining KPIs.
    """
    client = get_supabase_client()

    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    try:
        # Select summary fields along with forecast_records to derive exact start/end windows
        select_cols = (
            "id, user_id, forecast_days, "
            "historical_start_date, historical_end_date, "
            "historical_total_sales, historical_average_daily_sales, "
            "forecast_start_date, forecast_end_date, "
            "forecast_total, forecast_average, "
            "expected_growth_percentage, csv_source_url, "
            "forecast_records, "
            "created_at"
        )
        
        query = client.table("sales_forecasts") \
            .select(select_cols, count="exact") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .range(safe_offset, safe_offset + safe_limit - 1)
        
        res = query.execute()

        items = res.data or []
        for item in items:
            records = item.get("forecast_records")
            if records and isinstance(records, list) and len(records) > 0:
                first_rec = records[0]
                last_rec = records[-1]
                if isinstance(first_rec, dict) and first_rec.get("date"):
                    item["forecast_start_date"] = str(first_rec["date"]).strip()
                if isinstance(last_rec, dict) and last_rec.get("date"):
                    item["forecast_end_date"] = str(last_rec["date"]).strip()

        total = res.count if res.count is not None else len(items)

        return {
            "history": items,
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset
        }
    except Exception as e:
        logger.error(f"Failed to fetch forecast history: {type(e).__name__}")
        raise DatabaseConnectionError("Failed to retrieve forecast history from database.") from None


def get_forecast_by_id(
    user_id: str,
    forecast_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single complete forecast record by its ID.
    Strictly scoped to both record ID AND verified Firebase UID.
    Returns None if the record does not exist or belongs to another user.
    """
    client = get_supabase_client()

    try:
        res = client.table("sales_forecasts") \
            .select("*") \
            .eq("id", forecast_id) \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        if res.data and len(res.data) > 0:
            rec = res.data[0]
            records = rec.get("forecast_records")
            if records and isinstance(records, list) and len(records) > 0:
                first_rec = records[0]
                last_rec = records[-1]
                if isinstance(first_rec, dict) and first_rec.get("date"):
                    rec["forecast_start_date"] = str(first_rec["date"]).strip()
                if isinstance(last_rec, dict) and last_rec.get("date"):
                    rec["forecast_end_date"] = str(last_rec["date"]).strip()
            return rec
        return None
    except Exception as e:
        logger.error(f"Failed to fetch forecast by ID: {type(e).__name__}")
        raise DatabaseConnectionError("Failed to retrieve forecast record from database.") from None


def delete_forecast_by_id(
    user_id: str,
    forecast_id: str
) -> bool:
    """
    Deletes a single forecast record owned by the authenticated user.
    Strictly scoped to both record ID AND verified Firebase UID.
    Returns True if deleted, False if not found or unauthorized.
    """
    client = get_supabase_client()

    try:
        res = client.table("sales_forecasts") \
            .delete() \
            .eq("id", forecast_id) \
            .eq("user_id", user_id) \
            .execute()

        if res.data and len(res.data) > 0:
            logger.info(f"Forecast record '{forecast_id}' deleted by user '{user_id}'.")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete forecast record: {type(e).__name__}")
        raise DatabaseConnectionError("Failed to delete forecast record from database.") from None


# ---------------------------------------------------------------------------
# Daily Sales Operations
# ---------------------------------------------------------------------------

def validate_daily_sales_inputs(sale_date: str, daily_revenue: Any) -> tuple[str, float]:
    """
    Validates that sale_date is a valid YYYY-MM-DD calendar date and daily_revenue
    is a finite, non-negative numeric value. Normalizes revenue to 2 decimal places.
    """
    if not sale_date or not isinstance(sale_date, str):
        raise ValueError("Sales date must be a non-empty string in YYYY-MM-DD format.")

    clean_date = sale_date.strip()
    try:
        parsed_date = datetime.strptime(clean_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Invalid date format. Expected 'YYYY-MM-DD' representing a valid calendar date.")

    if daily_revenue is None:
        raise ValueError("Daily revenue is required.")

    try:
        rev_float = float(daily_revenue)
    except (ValueError, TypeError):
        raise ValueError("Daily revenue must be a valid numeric amount.")

    if math.isnan(rev_float) or math.isinf(rev_float):
        raise ValueError("Daily revenue must be a finite numeric value.")

    if rev_float < 0:
        raise ValueError("Negative sales values are not permitted.")

    return clean_date, round(rev_float, 2)


def create_daily_sales(
    user_id: str,
    sale_date: str,
    daily_revenue: float
) -> Dict[str, Any]:
    """
    Persists a manual daily sales entry in Supabase 'daily_sales' table.
    Enforces user ownership, input validation, and date uniqueness per user.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Authenticated user_id is required.")

    clean_date, clean_rev = validate_daily_sales_inputs(sale_date, daily_revenue)
    client = get_supabase_client()

    # Pre-check for duplicate date to provide an immediate caller-safe conflict response
    try:
        existing = client.table("daily_sales") \
            .select("id, user_id, sale_date, daily_revenue") \
            .eq("user_id", user_id) \
            .eq("sale_date", clean_date) \
            .limit(1) \
            .execute()
        if existing.data and len(existing.data) > 0:
            raise DailySalesConflictError(
                f"A sales record for date '{clean_date}' already exists. Please edit the existing entry."
            )
    except DailySalesConflictError:
        raise
    except Exception as e:
        err_msg = str(e).lower()
        if "pgrst205" in err_msg or "schema cache" in err_msg or "not find the table" in err_msg:
            logger.critical("Database table 'daily_sales' not found in Supabase schema.")
            raise DatabaseConnectionError("Database table 'daily_sales' does not exist. Please apply migration 002.") from None
        logger.error(f"Error checking existing daily sales for user '{user_id}': {type(e).__name__}")
        raise DatabaseConnectionError("Failed to check existing daily sales record.") from None

    # Insert into daily_sales
    insert_data = {
        "user_id": user_id,
        "sale_date": clean_date,
        "daily_revenue": clean_rev,
    }

    try:
        res = client.table("daily_sales").insert(insert_data).execute()
        if not res.data or len(res.data) == 0:
            raise DatabaseConnectionError("Database insert returned no data.")
        logger.info(f"Daily sales entry saved for user '{user_id}' on '{clean_date}': ₹{clean_rev}")
        return res.data[0]
    except Exception as e:
        err_str = str(e).lower()
        if "unique" in err_str or "duplicate" in err_str or "uq_daily_sales_user_date" in err_str or "23505" in err_str:
            logger.warning(f"Unique constraint race condition caught on daily sales for user '{user_id}' on '{clean_date}'.")
            raise DailySalesConflictError(
                f"A sales record for date '{clean_date}' already exists. Please edit the existing entry."
            )
        logger.error(f"Failed to persist daily sales record for user '{user_id}': {type(e).__name__}")
        raise DatabaseConnectionError("Failed to store daily sales record in database.") from None


def update_daily_sales(
    user_id: str,
    entry_id: str,
    daily_revenue: float,
    sale_date: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Updates an existing daily sales record owned by the authenticated user.
    Strictly scoped to both record ID AND verified Firebase UID.
    Returns the updated record or None if not found / unauthorized.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Authenticated user_id is required.")
    if not entry_id or not isinstance(entry_id, str):
        raise ValueError("Entry ID is required.")

    # Validate inputs
    dummy_date = sale_date or "2024-01-01"
    clean_date, clean_rev = validate_daily_sales_inputs(dummy_date, daily_revenue)

    payload: Dict[str, Any] = {
        "daily_revenue": clean_rev,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if sale_date:
        payload["sale_date"] = clean_date

    client = get_supabase_client()
    try:
        res = client.table("daily_sales") \
            .update(payload) \
            .eq("id", entry_id) \
            .eq("user_id", user_id) \
            .execute()
        if res.data and len(res.data) > 0:
            logger.info(f"Daily sales entry '{entry_id}' updated by user '{user_id}'.")
            return res.data[0]
        return None
    except Exception as e:
        err_str = str(e).lower()
        if "unique" in err_str or "duplicate" in err_str or "uq_daily_sales_user_date" in err_str or "23505" in err_str:
            raise DailySalesConflictError(
                f"Cannot update: a sales record for date '{clean_date}' already exists."
            )
        logger.error(f"Failed to update daily sales record '{entry_id}': {type(e).__name__}")
        raise DatabaseConnectionError("Failed to update daily sales record in database.") from None


def get_user_daily_sales(
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Retrieves paginated daily sales entries exclusively for the authenticated user,
    sorted by sale_date DESC.
    """
    client = get_supabase_client()
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    try:
        res = client.table("daily_sales") \
            .select("id, user_id, sale_date, daily_revenue, created_at, updated_at", count="exact") \
            .eq("user_id", user_id) \
            .order("sale_date", desc=True) \
            .range(safe_offset, safe_offset + safe_limit - 1) \
            .execute()

        items = res.data or []
        total = res.count if res.count is not None else len(items)
        return {
            "entries": items,
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset
        }
    except Exception as e:
        err_msg = str(e).lower()
        if "pgrst205" in err_msg or "schema cache" in err_msg or "not find the table" in err_msg:
            logger.warning("Database table 'daily_sales' not found in Supabase. Returning empty list.")
            return {"entries": [], "total": 0, "limit": safe_limit, "offset": safe_offset}
        logger.error(f"Failed to fetch daily sales for user '{user_id}': {type(e).__name__}")
        raise DatabaseConnectionError("Failed to retrieve daily sales from database.") from None


def get_user_all_daily_sales(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all daily sales records for the authenticated user ordered by sale_date ASC.
    Used for merging with forecasting pipeline.
    """
    client = get_supabase_client()
    try:
        res = client.table("daily_sales") \
            .select("id, sale_date, daily_revenue, created_at, updated_at") \
            .eq("user_id", user_id) \
            .order("sale_date", desc=False) \
            .execute()
        return res.data or []
    except Exception as e:
        err_msg = str(e).lower()
        if "pgrst205" in err_msg or "schema cache" in err_msg or "not find the table" in err_msg:
            logger.warning("Database table 'daily_sales' not found in Supabase. Returning empty list.")
            return []
        logger.error(f"Failed to fetch all daily sales for user '{user_id}': {type(e).__name__}")
        raise DatabaseConnectionError("Failed to retrieve user daily sales.") from None

