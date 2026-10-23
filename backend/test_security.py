"""
SMART BPI - Security & Authentication Test Suite
Tests all required security scenarios:
1. Missing token handling (401 Unauthorized)
2. Expired, invalid, and revoked tokens (401 Unauthorized, generic sanitized messages)
3. Valid authenticated user access (200 OK)
4. Storage URL Ownership & Anti-Exploit Suite:
   - 4A: Direct access to another user's Supabase storage path (403 Forbidden without UID leakage)
   - 4B: Untrusted storage host (400 Bad Request)
   - 4C: Untrusted storage bucket (400 Bad Request)
   - 4D: Path traversal (/..) normalized and blocked (403 Forbidden)
   - 4E: URL-encoded traversal (%2e%2e) normalized and blocked (403 Forbidden)
   - 4F: Unexpected port (:8080) rejected (400 Bad Request)
   - 4G: Embedded credentials (user:pass@) rejected (400 Bad Request)
   - 4H: URL fragment (#frag) rejected (400 Bad Request)
   - 4I: Backslash injection rejected (400 Bad Request)
   - 4J: Malformed percent-encoding (%zz) rejected (400 Bad Request)
   - 4K: Double URL-encoding (%252e) rejected (400 Bad Request)
   - 4L: Valid Firebase Storage URL supported and verified
   - 4M: Unauthorized Firebase Storage URL rejected with 403 (no UID leakage)
   - 4N: Firebase Storage URL with unsupported query params rejected (400 Bad Request)
5. CSV Download & Parsing Error Sanitization:
   - 5A: File not found in storage returns 404 with sanitized message
   - 5B: File too large returns 400 with sanitized message
   - 5C: Storage connection dropped/failed returns 502 Bad Gateway
   - 5D: Storage timeout returns 504 Gateway Timeout
   - 5E: Malformed tabular CSV returns 400 with sanitized message
   - 5F: No internal URLs, file paths, or credentials leaked in CSV error responses
6. Firestore business profile mismatch (403 Forbidden)
7. Production mode with missing Firebase credentials fails closed (503 Service Unavailable)
8. Development bypass behavior (active only in development environment)
9. Liveness & Readiness Probes:
   - 9A: GET /live confirms application process is running (200 OK)
   - 9B: GET /ready succeeds when DB query succeeds (200 OK)
   - 9C: GET /ready fails closed when DB is unreachable (503 Service Unavailable)
   - 9D: GET /health returns 200 when ready, 503 when degraded
10. Custom Idempotency Key Conflict Handling:
   - 10A: Conflicting key reuse with different horizon returns 409 Conflict
11. GET /forecast/history requires auth and scopes strictly to caller
12. GET /forecast/{id} cross-tenant access blocked (404 Not Found)
13. DELETE /forecast/{id} cross-tenant deletion blocked (404 Not Found)
14. Database persistence failure produces safe 500 error and prevents forecast output
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
import firebase_admin
from firebase_admin import auth as firebase_auth
import database
from database import DatabaseIdempotencyConflictError, DatabaseConnectionError
import forecasting
from forecasting import CsvDownloadError
import main
from main import app

client = TestClient(app)

SAMPLE_CSV_URL_USER1 = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_123/sales_sample.csv"
SAMPLE_CSV_URL_USER2 = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_victim/sales_sample.csv"
SAMPLE_FIREBASE_URL_USER1 = "https://firebasestorage.googleapis.com/v0/b/final-business-os.firebasestorage.app/o/service-data%2Fuser_123%2Fsales.csv?alt=media"
SAMPLE_FIREBASE_URL_USER2 = "https://firebasestorage.googleapis.com/v0/b/final-business-os.firebasestorage.app/o/service-data%2Fuser_victim%2Fsales.csv?alt=media"

MOCK_FORECAST_RESP = {
    "summary": {
        "historical_total_sales": 1000.0,
        "historical_average_daily_sales": 50.0,
        "forecast_30_days_total": 1500.0,
        "forecast_30_days_average": 50.0,
        "forecast_6_months_total": 9000.0,
        "forecast_6_months_average": 50.0,
        "forecast_1_year_total": 18000.0,
        "forecast_1_year_average": 50.0,
        "expected_growth_percentage": 5.0,
        "forecast_start_date": "2026-11-01",
        "forecast_end_date": "2026-11-30"
    },
    "forecast": [{"date": "2026-11-01", "predicted_sales": 50.0, "lower_bound": 45.0, "upper_bound": 55.0}]
}


def run_tests():
    print("=" * 70)
    print("RUNNING SMART BPI SECURITY & AUTHENTICATION AUDIT TESTS")
    print("=" * 70)
    
    passed = 0
    total = 14

    # --------------------------------------------------------------------------
    # TEST 1: Missing Token
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Missing Token Check...")
    res = client.post("/forecast", json={
        "sales_csv_url": SAMPLE_CSV_URL_USER1,
        "forecast_days": 30
    })
    assert res.status_code == 401, f"Expected 401, got {res.status_code}: {res.text}"
    assert "Missing or invalid Authorization header" in res.json().get("detail", "")
    print(" -> PASS: Correctly rejected with 401 Unauthorized for missing token.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 2: Expired, Invalid, and Revoked Tokens (Generic 401 Responses)
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Expired, Invalid, and Revoked Token Checks...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}):
        
        # Test 2a: Expired token
        with patch.object(firebase_auth, "verify_id_token", side_effect=firebase_auth.ExpiredIdTokenError("Token expired", None)):
            res = client.post("/forecast", 
                headers={"Authorization": "Bearer expired.token.jwt"},
                json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
            )
            assert res.status_code == 401, f"Expected 401, got {res.status_code}: {res.text}"
            assert res.json().get("detail") == "Authentication failed: Token has expired."

        # Test 2b: Invalid/Forged token
        with patch.object(firebase_auth, "verify_id_token", side_effect=firebase_auth.InvalidIdTokenError("Signature verification failed")):
            res = client.post("/forecast", 
                headers={"Authorization": "Bearer forged.invalid.token"},
                json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
            )
            assert res.status_code == 401, f"Expected 401, got {res.status_code}: {res.text}"
            assert res.json().get("detail") == "Authentication failed: Invalid ID token."

        # Test 2c: Revoked token
        with patch.object(firebase_auth, "verify_id_token", side_effect=firebase_auth.RevokedIdTokenError("Token revoked")):
            res = client.post("/forecast", 
                headers={"Authorization": "Bearer revoked.token.jwt"},
                json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
            )
            assert res.status_code == 401, f"Expected 401, got {res.status_code}: {res.text}"
            assert res.json().get("detail") == "Authentication failed: Token has been revoked."

    print(" -> PASS: Correctly returned generic, safe 401 Unauthorized responses for expired, invalid, and revoked tokens.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 3: Valid Authenticated User Access
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Valid Authenticated User Access...")
    mock_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.exists = True
    mock_doc.to_dict.return_value = {
        "servicesData": {"salesCsvUrl": SAMPLE_CSV_URL_USER1}
    }
    mock_db.collection.return_value.document.return_value.get.return_value = mock_doc

    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch.object(main, "get_firestore_client", return_value=mock_db), \
         patch("main.download_and_parse_csv", return_value=MagicMock()), \
         patch("main.validate_and_prepare_sales_data", return_value=MagicMock()), \
         patch("main.train_sarima_model", return_value=MagicMock()), \
         patch("main.generate_forecast_payload", return_value=MOCK_FORECAST_RESP), \
         patch("main.insert_forecast_record", return_value={"id": "mock_forecast_uuid_123", "created_at": "2026-10-23T00:00:00Z"}):

        res = client.post("/forecast", 
            headers={"Authorization": "Bearer valid.jwt.token"},
            json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
        )
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert "summary" in data
        assert "forecast" in data
        assert data["summary"]["forecast_30_days_total"] == 1500.0
        assert data.get("id") == "mock_forecast_uuid_123"

    print(" -> PASS: Valid authenticated user successfully verified and granted forecast access.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 4: Storage URL Ownership & Anti-Exploit Suite
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Storage URL Ownership & Anti-Exploit Suite...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_attacker"}):
        
        # 4A: Direct access to another tenant's path (403, no UID leakage)
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": SAMPLE_CSV_URL_USER2, "forecast_days": 30}
        )
        assert res.status_code == 403
        detail = res.json().get("detail", "")
        assert "user_victim" not in detail, "Security flaw: Victim UID leaked in 403 detail!"
        assert "You do not have permission to access another user's sales data" in detail
        print("   * 4A Another Tenant Path: Blocked with 403 without UID leakage.")

        # 4B: Untrusted host -> 400
        untrusted_host = "https://attacker.evil.com/storage/v1/object/public/service-data/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": untrusted_host, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Storage host is not trusted" in res.json().get("detail", "")
        print("   * 4B Untrusted Host: Blocked with 400 Bad Request.")

        # 4C: Untrusted bucket -> 400
        untrusted_bucket = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/private-keys/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": untrusted_bucket, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Path does not match the expected service-data storage bucket" in res.json().get("detail", "")
        print("   * 4C Untrusted Bucket: Blocked with 400 Bad Request.")

        # 4D: Path traversal (/..) rejected before normalization (400 Bad Request)
        traversal_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/../user_victim/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": traversal_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "user_victim" not in res.json().get("detail", "")
        assert "Path traversal" in res.json().get("detail", "")
        print("   * 4D Path Traversal (/..): Rejected before normalization with 400 Bad Request.")

        # 4E: URL-encoded traversal (%2e%2e) -> unquoted and rejected before normalization (400 Bad Request)
        encoded_traversal = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/%2e%2e/user_victim/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": encoded_traversal, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "user_victim" not in res.json().get("detail", "")
        assert "Path traversal" in res.json().get("detail", "")
        print("   * 4E Encoded Traversal (%2e%2e): Unquoted and rejected before normalization with 400 Bad Request.")

        # 4F: Unexpected port (:8080) -> 400
        port_url = "https://dzwzsnnrfstsmtpueilo.supabase.co:8080/storage/v1/object/public/service-data/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": port_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Unexpected port" in res.json().get("detail", "")
        print("   * 4F Unexpected Port: Blocked with 400 Bad Request.")

        # 4G: User credentials in URL (user:pass@) -> 400
        userinfo_url = "https://attacker:secret@dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": userinfo_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Embedded credentials" in res.json().get("detail", "")
        print("   * 4G Embedded Credentials: Blocked with 400 Bad Request.")

        # 4H: Fragment in URL (#frag) -> 400
        frag_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/sales.csv#secret"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": frag_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "fragments" in res.json().get("detail", "").lower()
        print("   * 4H URL Fragment: Blocked with 400 Bad Request.")

        # 4I: Backslash injection -> 400
        backslash_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage\\v1\\object/public/service-data/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": backslash_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Backslashes" in res.json().get("detail", "")
        print("   * 4I Backslash Injection: Blocked with 400 Bad Request.")

        # 4J: Malformed percent-encoding (%zz) -> 400
        malformed_pct_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/sales%zz.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": malformed_pct_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Malformed percent-encoding" in res.json().get("detail", "")
        print("   * 4J Malformed Percent Encoding: Blocked with 400 Bad Request.")

        # 4K: Double URL encoding (%252e) -> 400
        double_enc_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/%252e%252e/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": double_enc_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Double URL encoding" in res.json().get("detail", "")
        print("   * 4K Double URL Encoding: Blocked with 400 Bad Request.")

        # 4L: Valid Firebase Storage URL supported
        # When user_123 accesses own file in Firebase Storage
        mock_fb_doc = MagicMock()
        mock_fb_doc.exists = True
        mock_fb_doc.to_dict.return_value = {
            "servicesData": {"salesCsvUrl": SAMPLE_FIREBASE_URL_USER1}
        }
        mock_fb_db = MagicMock()
        mock_fb_db.collection.return_value.document.return_value.get.return_value = mock_fb_doc

        with patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
             patch.object(main, "get_firestore_client", return_value=mock_fb_db), \
             patch("main.download_and_parse_csv", return_value=MagicMock()), \
             patch("main.validate_and_prepare_sales_data", return_value=MagicMock()), \
             patch("main.train_sarima_model", return_value=MagicMock()), \
             patch("main.generate_forecast_payload", return_value=MOCK_FORECAST_RESP), \
             patch("main.insert_forecast_record", return_value={"id": "f_firebase_123", "created_at": "2026-10-23T00:00:00Z"}):

            res = client.post("/forecast", 
                headers={"Authorization": "Bearer valid.jwt.token"},
                json={"sales_csv_url": SAMPLE_FIREBASE_URL_USER1, "forecast_days": 30}
            )
            assert res.status_code == 200, f"Expected 200 for Firebase URL, got {res.status_code}: {res.text}"
            print("   * 4L Valid Firebase Storage URL: Successfully parsed and authorized.")

        # 4M: Unauthorized Firebase Storage URL rejected with 403 (no UID leakage)
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": SAMPLE_FIREBASE_URL_USER2, "forecast_days": 30}
        )
        assert res.status_code == 403
        assert "user_victim" not in res.json().get("detail", "")
        print("   * 4M Unauthorized Firebase Storage URL: Blocked with 403 (no UID leakage).")

        # 4N: Firebase Storage URL with unsupported query params rejected with 400
        suspicious_fb_url = "https://firebasestorage.googleapis.com/v0/b/final-business-os.firebasestorage.app/o/service-data%2Fuser_attacker%2Fsales.csv?alt=media&admin_override=true"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": suspicious_fb_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Unsupported query parameter" in res.json().get("detail", "")
        print("   * 4N Unsupported Query Params on Storage: Blocked with 400 Bad Request.")

        # 4O: Encoded backslash (%5c) rejected
        encoded_backslash_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker%5csales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": encoded_backslash_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Encoded backslashes" in res.json().get("detail", "")
        print("   * 4O Encoded Backslash (%5c): Blocked with 400 Bad Request.")

        # 4P: Encoded path separator (%2f) in Supabase URL rejected
        encoded_separator_sb = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data%2Fuser_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": encoded_separator_sb, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Encoded path separators" in res.json().get("detail", "")
        print("   * 4P Encoded Separator (%2f) in Supabase URL: Blocked with 400 Bad Request.")

        # 4Q: Encoded path separator (%2f) in Firebase storage prefix rejected
        encoded_separator_fb = "https://firebasestorage.googleapis.com/v0/b%2Ffinal-business-os.firebasestorage.app/o/service-data%2Fuser_attacker%2Fsales.csv?alt=media"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": encoded_separator_fb, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Encoded path separators" in res.json().get("detail", "")
        print("   * 4Q Encoded Separator (%2f) in Firebase prefix: Blocked with 400 Bad Request.")

        # 4R: Malformed port rejected with 400
        malformed_port_url = "https://dzwzsnnrfstsmtpueilo.supabase.co:abc/storage/v1/object/public/service-data/user_attacker/sales.csv"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": malformed_port_url, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Malformed port" in res.json().get("detail", "")
        print("   * 4R Malformed Port: Blocked with 400 Bad Request.")

        # 4S: Supabase URL with unsupported query parameter rejected with 400
        unsupported_sb_query = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/sales.csv?admin_override=true"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": unsupported_sb_query, "forecast_days": 30}
        )
        assert res.status_code == 400
        assert "Unsupported query parameter" in res.json().get("detail", "")
        print("   * 4S Unsupported Query Parameter on Supabase: Blocked with 400 Bad Request.")

        # 4T: Supabase URL with safe query parameter (?t=123456) accepted
        safe_sb_query = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_attacker/sales.csv?t=123456"
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer attacker.jwt.token"},
            json={"sales_csv_url": safe_sb_query, "forecast_days": 30}
        )
        # Should proceed past URL validation and reach Firestore profile check (or fail on profile mismatch/download, not query param)
        assert res.status_code != 400 or "query parameter" not in res.json().get("detail", "")
        print("   * 4T Safe Query Parameter on Supabase (?t=...): Allowed.")

    print(" -> PASS: Storage URL ownership and anti-exploit suite strictly verified.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 5: CSV Download & Parsing Error Sanitization
    # --------------------------------------------------------------------------
    print("\n[TEST 5] CSV Download & Parsing Error Sanitization...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch.object(main, "get_firestore_client", return_value=mock_db):

        # 5A: File not found in storage -> 404 with sanitized message
        with patch("main.download_and_parse_csv", side_effect=CsvDownloadError("CSV download failed: server returned HTTP 404")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 404
            assert "Sales CSV file was not found" in res.json().get("detail", "")

        # 5B: File too large -> 400 with sanitized message
        with patch("main.download_and_parse_csv", side_effect=CsvDownloadError("CSV file is too large. Download aborted after 12,000,000 bytes (limit: 10 MB).")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 400
            assert "Sales CSV exceeds the maximum allowed file size" in res.json().get("detail", "")

        # 5C: Storage connection failure -> 502 Bad Gateway
        with patch("main.download_and_parse_csv", side_effect=CsvDownloadError("CSV download connection error: Connection dropped mid-stream")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 502
            assert "Storage service connection failed" in res.json().get("detail", "")

        # 5D: Storage timeout -> 504 Gateway Timeout
        with patch("main.download_and_parse_csv", side_effect=CsvDownloadError("CSV download timed out. The storage server did not respond in time.")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 504
            assert "Storage server timed out" in res.json().get("detail", "")

        # 5E: Malformed tabular CSV -> 400
        with patch("main.download_and_parse_csv", side_effect=CsvDownloadError("Invalid CSV format: Error tokenizing data. C error: Expected 2 fields")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 400
            assert "Invalid CSV format" in res.json().get("detail", "")

        # 5F: No internal URLs, file paths, or credentials leaked
        with patch("main.download_and_parse_csv", side_effect=Exception("Failed connecting to internal s3://secret-path/db_token_12345")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 500
            detail = res.json().get("detail", "")
            assert "s3://" not in detail
            assert "secret-path" not in detail
            assert "db_token" not in detail
            assert "unexpected error occurred while processing the sales CSV" in detail

        # 5G: Logging during CSV errors records safe category and excludes sensitive URLs/tokens
        with patch("main.logger.error") as mock_log_err, \
             patch("main.download_and_parse_csv", side_effect=CsvDownloadError("CSV download failed: 404 at https://secret-storage.co/obj?token=secret_abc")):
            res = client.post("/forecast", headers={"Authorization": "Bearer valid.jwt.token"}, json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30})
            assert res.status_code == 404
            mock_log_err.assert_called()
            log_call_str = str(mock_log_err.call_args)
            assert "category=NOT_FOUND" in log_call_str
            assert "secret_abc" not in log_call_str
            assert "https://" not in log_call_str
            print("   * 5G Safe Logging: CSV download errors log category only, no sensitive tokens or URLs.")

    print(" -> PASS: CSV errors cleanly categorized and sanitized without information leakage.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 6: Firestore Business Profile Mismatch Check
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Firestore Business Profile Ownership Validation...")
    mismatched_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/user_123/sales_different.csv"
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch.object(main, "get_firestore_client", return_value=mock_db):
        
        res = client.post("/forecast", 
            headers={"Authorization": "Bearer valid.jwt.token"},
            json={"sales_csv_url": mismatched_url, "forecast_days": 30}
        )
        assert res.status_code == 403, f"Expected 403 for profile mismatch, got {res.status_code}: {res.text}"
        assert "does not match the sales data registered to your business" in res.json().get("detail", "")
        print(" -> PASS: Firestore business profile ownership strictly enforced.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 7: Production Mode with Missing Firebase Credentials Fails Closed
    # --------------------------------------------------------------------------
    print("\n[TEST 7] Production Mode with Missing Firebase Credentials...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(main, "ALLOW_UNVERIFIED_DEV_MODE", True), \
         patch.object(firebase_admin, "_apps", {}):

        res = client.post("/forecast", 
            headers={"Authorization": "Bearer forged.or.unverified.token"},
            json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
        )
        assert res.status_code == 503, f"Expected 503 fail-closed, got {res.status_code}: {res.text}"
        assert "Firebase Admin SDK is not configured" in res.json().get("detail", "")
        print(" -> PASS: Production strictly fails closed (503 Service Unavailable). Dev bypass ignored.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 8: Development Bypass Behavior
    # --------------------------------------------------------------------------
    print("\n[TEST 8] Development Bypass Behavior Check...")
    import jwt
    fake_jwt = jwt.encode({"uid": "dev_test_user"}, "secret", algorithm="HS256")
    with patch.object(main, "ENVIRONMENT", "development"), \
         patch.object(main, "IS_DEV_BYPASS_ACTIVE", True), \
         patch.object(firebase_admin, "_apps", {}):
        
        mock_dev_db = MagicMock()
        mock_dev_doc = MagicMock()
        mock_dev_doc.exists = True
        mock_dev_doc.to_dict.return_value = {
            "servicesData": {"salesCsvUrl": "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/dev_test_user/sales.csv"}
        }
        mock_dev_db.collection.return_value.document.return_value.get.return_value = mock_dev_doc

        with patch.object(main, "get_firestore_client", return_value=mock_dev_db), \
             patch("main.download_and_parse_csv", return_value=MagicMock()), \
             patch("main.validate_and_prepare_sales_data", return_value=MagicMock()), \
             patch("main.train_sarima_model", return_value=MagicMock()), \
             patch("main.generate_forecast_payload", return_value=MOCK_FORECAST_RESP), \
             patch("main.insert_forecast_record", return_value={"id": "dev_f1", "created_at": "2026-10-23T00:00:00Z"}):

            res = client.post("/forecast",
                headers={"Authorization": f"Bearer {fake_jwt}"},
                json={"sales_csv_url": "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/dev_test_user/sales.csv", "forecast_days": 30}
            )
            assert res.status_code == 200, f"Expected 200 in dev bypass, got {res.status_code}: {res.text}"
            print(" -> PASS: Development bypass active only in development environment.")
            passed += 1

    # --------------------------------------------------------------------------
    # TEST 9: Liveness & Readiness Probes Suite
    # --------------------------------------------------------------------------
    print("\n[TEST 9] Liveness & Readiness Probes Suite...")
    # 9A: Liveness confirms application process is running
    res = client.get("/live")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}
    print("   * 9A Liveness Probe (/live): Process is up and returns 200 OK.")

    # 9B: Readiness succeeds when database is ready
    with patch("main.check_database_readiness", return_value={"ready": True, "status": "ready"}):
        res = client.get("/ready")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"
        
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"
        assert res_health.json()["database_configured"] is True
    print("   * 9B Readiness Probe (/ready & /health): Returns 200 OK when database is accessible.")

    # 9C: Readiness fails closed when database is unreachable (503 Service Unavailable)
    with patch("main.check_database_readiness", return_value={"ready": False, "status": "unreachable"}):
        res = client.get("/ready")
        assert res.status_code == 503
        assert "Database service is not ready" in res.json().get("detail", "")
        
        res_health = client.get("/health")
        assert res_health.status_code == 503
        assert res_health.json()["status"] == "degraded"
        assert res_health.json()["database_configured"] is False
    print("   * 9C Readiness Failure (/ready & /health): Correctly returns 503 Service Unavailable.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 10: Custom Idempotency Key Conflict Handling
    # --------------------------------------------------------------------------
    print("\n[TEST 10] Custom Idempotency Key Conflict Handling...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch.object(main, "get_firestore_client", return_value=mock_db), \
         patch("main.download_and_parse_csv", return_value=MagicMock()), \
         patch("main.validate_and_prepare_sales_data", return_value=MagicMock()), \
         patch("main.train_sarima_model", return_value=MagicMock()), \
         patch("main.generate_forecast_payload", return_value=MOCK_FORECAST_RESP), \
         patch("main.insert_forecast_record", side_effect=DatabaseIdempotencyConflictError("Idempotency key was previously used with different forecast parameters or data.")):

        res = client.post("/forecast", 
            headers={"Authorization": "Bearer valid.jwt.token"},
            json={
                "sales_csv_url": SAMPLE_CSV_URL_USER1,
                "forecast_days": 180,
                "idempotency_key": "conflicting_custom_key_001"
            }
        )
        assert res.status_code == 409, f"Expected 409 Conflict, got {res.status_code}: {res.text}"
        assert "previously used with different forecast parameters" in res.json().get("detail", "")
        print(" -> PASS: Conflicting idempotency key reuse correctly returns 409 Conflict.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 11: GET /forecast/history Authentication & Scoping
    # --------------------------------------------------------------------------
    print("\n[TEST 11] GET /forecast/history Authentication & Scoping...")
    # Missing token
    res = client.get("/forecast/history")
    assert res.status_code == 401
    
    # Authenticated call scopes strictly to user
    mock_history = {"history": [{"id": "h1", "user_id": "user_123"}], "total": 1, "limit": 20, "offset": 0}
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch("main.get_user_forecast_history", return_value=mock_history) as mock_get_hist:
        
        res = client.get("/forecast/history", headers={"Authorization": "Bearer valid.jwt.token"})
        assert res.status_code == 200
        assert res.json()["total"] == 1
        mock_get_hist.assert_called_once_with(user_id="user_123", limit=20, offset=0)
    print(" -> PASS: /forecast/history strictly requires authentication and scopes to caller.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 12: GET /forecast/{id} Cross-Tenant Access Blocked
    # --------------------------------------------------------------------------
    print("\n[TEST 12] GET /forecast/{id} Cross-Tenant Access Blocked...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_attacker"}), \
         patch("main.get_forecast_by_id", return_value=None):

        res = client.get("/forecast/victim_forecast_id", headers={"Authorization": "Bearer attacker.jwt.token"})
        assert res.status_code == 404, f"Expected 404, got {res.status_code}"
        assert "not found" in res.json().get("detail", "").lower()
    print(" -> PASS: User A cannot retrieve User B's forecast record (returns 404 Not Found).")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 13: DELETE /forecast/{id} Cross-Tenant Deletion Blocked
    # --------------------------------------------------------------------------
    print("\n[TEST 13] DELETE /forecast/{id} Cross-Tenant Deletion Blocked...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_attacker"}), \
         patch("main.delete_forecast_by_id", return_value=False):

        res = client.delete("/forecast/victim_forecast_id", headers={"Authorization": "Bearer attacker.jwt.token"})
        assert res.status_code == 404, f"Expected 404, got {res.status_code}"
        assert "not found" in res.json().get("detail", "").lower()
    print(" -> PASS: User A cannot delete User B's forecast record (returns 404 Not Found).")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 14: Database Persistence Failure Prevents Forecast Response
    # --------------------------------------------------------------------------
    print("\n[TEST 14] Database Persistence Failure Produces Safe 500 Error...")
    with patch.object(main, "ENVIRONMENT", "production"), \
         patch.object(firebase_admin, "_apps", {"[DEFAULT]": MagicMock()}), \
         patch.object(firebase_auth, "verify_id_token", return_value={"uid": "user_123"}), \
         patch.object(main, "get_firestore_client", return_value=mock_db), \
         patch("main.download_and_parse_csv", return_value=MagicMock()), \
         patch("main.validate_and_prepare_sales_data", return_value=MagicMock()), \
         patch("main.train_sarima_model", return_value=MagicMock()), \
         patch("main.generate_forecast_payload", return_value=MOCK_FORECAST_RESP), \
         patch("main.insert_forecast_record", side_effect=DatabaseConnectionError("Postgres connection timeout at db.secret.supabase.co")):

        res = client.post("/forecast", 
            headers={"Authorization": "Bearer valid.jwt.token"},
            json={"sales_csv_url": SAMPLE_CSV_URL_USER1, "forecast_days": 30}
        )
        assert res.status_code == 500, f"Expected 500, got {res.status_code}: {res.text}"
        detail = res.json().get("detail", "")
        assert "secret" not in detail
        assert "supabase.co" not in detail
        assert "Failed to save forecast record" in detail
        # Confirm that forecast results are never returned to client on persistence failure
        assert "forecast" not in res.json()
        assert "summary" not in res.json()

    print(" -> PASS: Database insertion failure produces safe 500 error without leaking internal details or forecast data.")
    passed += 1

    print("\n" + "=" * 70)
    print(f"SECURITY AUDIT RESULTS: {passed}/{total} TESTS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
