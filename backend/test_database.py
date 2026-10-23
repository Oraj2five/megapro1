"""
SMART BPI — Database Connection, Readiness, & Persistence Tests
Comprehensive test suite covering:
1. Missing SUPABASE_URL configuration.
2. Missing SUPABASE_SERVICE_ROLE_KEY configuration.
3. Safe readiness check when unconfigured.
4. Missing SDK dependency handling.
5. Client initialization failure without secret leakage.
6. Successful client initialization & table reachability (readiness query succeeds).
7. Database readiness query failure when table is unreachable.
8. Successful forecast insertion with verified Firebase UID.
9. Dataset fingerprint precision:
   - 9A: Identical datasets produce identical fingerprint.
   - 9B: Sub-0.0001 difference (0.00001) produces distinct fingerprint (lossless IEEE-754).
   - 9C: Changed dates produce distinct fingerprint.
   - 9D: Reordered rows produce identical fingerprint (deterministic ordering).
   - 9E: NaN, Infinity, or missing dates raise ValueError.
10. Idempotency handling: duplicate request safely retrieves existing record without insert.
11. Idempotency: changed sales data at the same URL produces a distinct forecast record.
12. Idempotency: same sales data with different forecast horizon produces a distinct forecast.
13. Custom Idempotency Key Scoping & Conflict Detection:
   - 13A: Custom key is strictly scoped to authenticated UID.
   - 13B: Same custom key reused with different forecast horizon raises DatabaseIdempotencyConflictError.
   - 13C: Same custom key reused with different CSV URL raises DatabaseIdempotencyConflictError.
   - 13D: Same custom key with matching parameters safely returns existing record.
14. Idempotency: database lookup failure aborts persistence immediately (no silent conversion).
15. Idempotency: concurrent duplicate requests / unique constraint conflict safely resolved.
16. Forecast history retrieval with user scoping and pagination.
17. Cross-tenant isolation: User A cannot retrieve User B's forecast record.
18. Single record deletion scoped to owner.
19. Database insertion failure produces safe error without secret leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from database import (
    get_supabase_client,
    get_supabase_config,
    check_database_readiness,
    reset_supabase_client,
    compute_series_hash,
    compute_idempotency_key,
    insert_forecast_record,
    get_user_forecast_history,
    get_forecast_by_id,
    delete_forecast_by_id,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseIdempotencyConflictError,
)

SAMPLE_SERIES_1 = pd.Series(
    [100.0, 120.0, 150.0],
    index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
)

SAMPLE_SERIES_MODIFIED = pd.Series(
    [100.0, 120.0, 200.0],
    index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
)

SAMPLE_PAYLOAD = {
    "summary": {
        "historical_total_sales": 370.0,
        "historical_average_daily_sales": 123.33,
        "forecast_30_days_total": 3600.0,
        "forecast_30_days_average": 120.0,
        "forecast_6_months_total": 21600.0,
        "forecast_6_months_average": 120.0,
        "forecast_1_year_total": 43800.0,
        "forecast_1_year_average": 120.0,
        "expected_growth_percentage": 5.2,
        "forecast_start_date": "2024-01-04",
        "forecast_end_date": "2024-02-02"
    },
    "forecast": [
        {"date": "2024-01-04", "predicted_sales": 120.0, "lower_bound": 110.0, "upper_bound": 130.0}
    ]
}


def run_tests():
    print("=" * 70)
    print("RUNNING SMART BPI DATABASE READINESS & PERSISTENCE TESTS")
    print("=" * 70)
    
    passed = 0
    total = 19

    # --------------------------------------------------------------------------
    # TEST 1: Missing SUPABASE_URL
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Missing SUPABASE_URL Handling...")
    reset_supabase_client()
    with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": "fake_service_key_1234567890"}):
        try:
            get_supabase_client()
            raise AssertionError("Expected DatabaseConfigurationError was not raised")
        except DatabaseConfigurationError as e:
            assert "SUPABASE_URL" in str(e)
            assert "fake_service_key" not in str(e)
            print(" -> PASS: Correctly raised DatabaseConfigurationError for missing SUPABASE_URL.")
            passed += 1

    # --------------------------------------------------------------------------
    # TEST 2: Missing SUPABASE_SERVICE_ROLE_KEY
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Missing SUPABASE_SERVICE_ROLE_KEY Handling...")
    reset_supabase_client()
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": ""}):
        try:
            get_supabase_client()
            raise AssertionError("Expected DatabaseConfigurationError was not raised")
        except DatabaseConfigurationError as e:
            assert "SUPABASE_SERVICE_ROLE_KEY" in str(e)
            print(" -> PASS: Correctly raised DatabaseConfigurationError for missing SUPABASE_SERVICE_ROLE_KEY.")
            passed += 1

    # --------------------------------------------------------------------------
    # TEST 3: Safe Readiness Check When Unconfigured
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Safe Readiness Check (Unconfigured)...")
    reset_supabase_client()
    with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}):
        status = check_database_readiness()
        assert status["ready"] is False
        assert status["status"] in ("missing_configuration", "missing_dependency")
        assert "message" in status
        print(f" -> PASS: Safe check completed without exception. Status: {status['status']}")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 4: Missing SDK Dependency Handling
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Missing SDK Dependency Handling...")
    reset_supabase_client()
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "supabase":
            raise ImportError("No module named 'supabase'")
        return original_import(name, *args, **kwargs)

    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "fake_secret_key_1234567890"}), \
         patch.object(builtins, "__import__", side_effect=mock_import):
        status = check_database_readiness()
        assert status["ready"] is False
        assert status["sdk_installed"] is False
        assert status["status"] == "missing_dependency"
        print(" -> PASS: Readiness check cleanly caught missing SDK without uncaught errors.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 5: Initialization Failure Without Secret Leakage
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Secret Protection During Initialization Failure...")
    reset_supabase_client()
    SECRET_KEY_STRING = "super_confidential_service_role_secret_9876543210"
    
    mock_supabase_module = MagicMock()
    mock_supabase_module.create_client.side_effect = Exception(f"Connection refused to https://secret.db.co with key {SECRET_KEY_STRING}")
    
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": SECRET_KEY_STRING}), \
         patch.dict("sys.modules", {"supabase": mock_supabase_module}):
        try:
            get_supabase_client()
            raise AssertionError("Expected DatabaseConnectionError was not raised")
        except DatabaseConnectionError as e:
            assert SECRET_KEY_STRING not in str(e), "CRITICAL: Secret leaked in exception message!"
            assert "Database initialization failure" in str(e)
            print(" -> PASS: DatabaseConnectionError raised safely without leaking credentials.")
            passed += 1

    # --------------------------------------------------------------------------
    # TEST 6: Successful Client Initialization & Table Reachability Check
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Successful Client Initialization & Readiness...")
    reset_supabase_client()
    mock_client_instance = MagicMock()
    mock_client_instance.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    mock_supabase_module = MagicMock()
    mock_supabase_module.create_client.return_value = mock_client_instance
    
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "valid_service_role_key_for_testing"}), \
         patch.dict("sys.modules", {"supabase": mock_supabase_module}):
        client = get_supabase_client()
        assert client == mock_client_instance
        status = check_database_readiness()
        assert status["ready"] is True
        assert status["status"] == "ready"
        assert status["table_accessible"] is True
        print(" -> PASS: Client successfully initialized and table accessibility confirmed.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 7: Database Readiness Query Fails (Table Unreachable)
    # --------------------------------------------------------------------------
    print("\n[TEST 7] Database Readiness Query Fails (Table Unreachable)...")
    reset_supabase_client()
    mock_unreachable_client = MagicMock()
    mock_unreachable_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Postgres connection timeout to db host")
    
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "valid_service_role_key_for_testing"}), \
         patch.dict("sys.modules", {"supabase": MagicMock()}), \
         patch("database.get_supabase_client", return_value=mock_unreachable_client):
        status = check_database_readiness()
        assert status["ready"] is False
        assert status["status"] == "unreachable"
        assert status["table_accessible"] is False
        assert "timeout" not in status["message"]
        assert "db host" not in status["message"]
        print(" -> PASS: Table unreachable safely caught and returned sanitized failure status.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 8: Successful Forecast Insertion with Verified Firebase UID
    # --------------------------------------------------------------------------
    print("\n[TEST 8] Forecast Record Insertion & Firebase UID Scoping...")
    reset_supabase_client()
    
    mock_table = MagicMock()
    mock_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    mock_inserted_row = {
        "id": "uuid-forecast-12345",
        "user_id": "verified_firebase_uid_100",
        "forecast_days": 30,
        "forecast_total": 3600.0,
        "forecast_average": 120.0,
        "created_at": "2024-01-04T00:00:00Z"
    }
    mock_table.insert.return_value.execute.return_value = MagicMock(data=[mock_inserted_row])
    
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch("database.get_supabase_client", return_value=mock_client):
        res = insert_forecast_record(
            user_id="verified_firebase_uid_100",
            forecast_days=30,
            csv_source_url="https://test.supabase.co/storage/v1/object/public/service-data/user/sales.csv",
            series=SAMPLE_SERIES_1,
            forecast_payload=SAMPLE_PAYLOAD
        )
        assert res["id"] == "uuid-forecast-12345"
        assert res["user_id"] == "verified_firebase_uid_100"
        
        insert_args = mock_table.insert.call_args[0][0]
        assert insert_args["user_id"] == "verified_firebase_uid_100"
        assert insert_args["forecast_days"] == 30
        assert insert_args["forecast_total"] == 3600.0
        assert insert_args["forecast_average"] == 120.0
        assert insert_args["historical_start_date"] == "2024-01-01"
        assert insert_args["historical_end_date"] == "2024-01-03"
        assert "idempotency_key" in insert_args
        print(" -> PASS: Forecast inserted with exact Firebase UID and mapped SQL columns.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 9: Dataset Fingerprint Precision & Determinism Suite
    # --------------------------------------------------------------------------
    print("\n[TEST 9] Dataset Fingerprint Precision & Determinism Suite...")
    # 9A: Identical datasets produce identical hash
    hash_base1 = compute_series_hash(SAMPLE_SERIES_1)
    hash_base2 = compute_series_hash(SAMPLE_SERIES_1.copy())
    assert hash_base1 == hash_base2, "Identical datasets produced different hashes!"
    print("   * 9A Identical Datasets: Equal fingerprints.")

    # 9B: Sub-0.0001 difference (< 0.0001) produces distinct hash (lossless IEEE-754)
    series_tiny_diff = pd.Series(
        [100.00001, 120.0, 150.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    series_tiny_diff_alt = pd.Series(
        [100.00002, 120.0, 150.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    hash_tiny1 = compute_series_hash(series_tiny_diff)
    hash_tiny2 = compute_series_hash(series_tiny_diff_alt)
    assert hash_tiny1 != hash_tiny2, "Lossless precision failure: 0.00001 difference collided!"
    print("   * 9B Sub-0.0001 Precision: Lossless IEEE-754 differentiates tiny differences.")

    # 9C: Changed dates produce distinct hash
    series_changed_date = pd.Series(
        [100.0, 120.0, 150.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-06"])
    )
    assert compute_series_hash(SAMPLE_SERIES_1) != compute_series_hash(series_changed_date)
    print("   * 9C Changed Dates: Produces distinct fingerprint.")

    # 9D: Reordered input produces identical hash (deterministic sorting)
    series_reordered = pd.Series(
        [150.0, 100.0, 120.0],
        index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    )
    assert compute_series_hash(SAMPLE_SERIES_1) == compute_series_hash(series_reordered)
    print("   * 9D Reordered Input: Automatically sorted chronologically for identical hash.")

    # 9E: NaN, Infinity, or missing dates raise ValueError
    series_with_nan = pd.Series([100.0, np.nan, 150.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    series_with_inf = pd.Series([100.0, np.inf, 150.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    try:
        compute_series_hash(series_with_nan)
        raise AssertionError("Expected ValueError for NaN was not raised")
    except ValueError:
        pass
    try:
        compute_series_hash(series_with_inf)
        raise AssertionError("Expected ValueError for Infinity was not raised")
    except ValueError:
        pass
    print("   * 9E Input Validation: NaN and Infinity cleanly rejected.")

    # 9F: Unsupported input type raises ValueError (no generic fallback to str(series))
    try:
        compute_series_hash(12345)
        raise AssertionError("Expected ValueError for unsupported int type was not raised")
    except ValueError as e:
        assert "Unsupported sales series input type" in str(e)
        print("   * 9F Unsupported Type: Non-series input cleanly rejected with ValueError.")

    print(" -> PASS: Dataset fingerprint precision, determinism, and validation verified.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 10: Idempotency Handling (Duplicate Request Returns Existing Record)
    # --------------------------------------------------------------------------
    print("\n[TEST 10] Idempotency Handling on Duplicate Request...")
    reset_supabase_client()
    
    existing_record = {
        "id": "uuid-existing-forecast-999",
        "user_id": "verified_firebase_uid_100",
        "forecast_days": 30,
        "forecast_start_date": "2024-01-04",
        "csv_source_url": "https://test.supabase.co/storage/v1/object/public/service-data/user/sales.csv",
        "summary_metrics": {"dataset_fingerprint": compute_series_hash(SAMPLE_SERIES_1)},
        "created_at": "2024-01-04T00:00:00Z"
    }
    mock_table_idemp = MagicMock()
    mock_table_idemp.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[existing_record])
    
    mock_client_idemp = MagicMock()
    mock_client_idemp.table.return_value = mock_table_idemp

    with patch("database.get_supabase_client", return_value=mock_client_idemp):
        res = insert_forecast_record(
            user_id="verified_firebase_uid_100",
            forecast_days=30,
            csv_source_url="https://test.supabase.co/storage/v1/object/public/service-data/user/sales.csv",
            series=SAMPLE_SERIES_1,
            forecast_payload=SAMPLE_PAYLOAD
        )
        assert res["id"] == "uuid-existing-forecast-999"
        mock_table_idemp.insert.assert_not_called()
        print(" -> PASS: Duplicate request with matching idempotency key returned existing record without insert.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 11: Changed Sales Data Produces Distinct Idempotency Key
    # --------------------------------------------------------------------------
    print("\n[TEST 11] Changed Sales Data Produces Distinct Idempotency Key...")
    key_original = compute_idempotency_key(
        user_id="user_123",
        forecast_days=30,
        series=SAMPLE_SERIES_1,
        csv_source_url="https://test.supabase.co/sales.csv",
        forecast_start_date="2024-01-04"
    )
    key_modified = compute_idempotency_key(
        user_id="user_123",
        forecast_days=30,
        series=SAMPLE_SERIES_MODIFIED,
        csv_source_url="https://test.supabase.co/sales.csv",
        forecast_start_date="2024-01-04"
    )
    assert key_original != key_modified, "Idempotency key did not change when sales data changed!"
    print(" -> PASS: Changed sales data at the same URL generates a distinct idempotency key.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 12: Different Horizon Produces Distinct Idempotency Key
    # --------------------------------------------------------------------------
    print("\n[TEST 12] Different Horizon Produces Distinct Idempotency Key...")
    key_30 = compute_idempotency_key(
        user_id="user_123",
        forecast_days=30,
        series=SAMPLE_SERIES_1,
        csv_source_url="https://test.supabase.co/sales.csv",
        forecast_start_date="2024-01-04"
    )
    key_180 = compute_idempotency_key(
        user_id="user_123",
        forecast_days=180,
        series=SAMPLE_SERIES_1,
        csv_source_url="https://test.supabase.co/sales.csv",
        forecast_start_date="2024-01-04"
    )
    assert key_30 != key_180, "Idempotency key identical across different forecast horizons!"
    print(" -> PASS: Different forecast horizon produces a distinct idempotency key.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 13: Custom Idempotency Key Scoping & Conflicting Reuse Suite
    # --------------------------------------------------------------------------
    print("\n[TEST 13] Custom Idempotency Key Scoping & Conflicting Reuse...")
    # 13A: Scoped to user (User A vs User B produce distinct keys for same custom key)
    key_user_a = compute_idempotency_key(user_id="user_a", forecast_days=30, custom_idempotency_key="client_req_001")
    key_user_b = compute_idempotency_key(user_id="user_b", forecast_days=30, custom_idempotency_key="client_req_001")
    assert key_user_a != key_user_b, "Custom idempotency key collided across different users!"
    assert "user_a" in key_user_a and "user_b" in key_user_b
    print("   * 13A User Scoping: Custom keys are namespaced by authenticated UID.")

    # 13B: Conflicting reuse with different forecast horizon raises DatabaseIdempotencyConflictError
    reset_supabase_client()
    mock_conflict_table = MagicMock()
    # Existing record was for forecast_days=30
    mock_conflict_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{
            "id": "prev_f1",
            "user_id": "user_a",
            "forecast_days": 30,
            "forecast_start_date": "2024-01-04",
            "csv_source_url": "https://test.supabase.co/sales.csv",
            "summary_metrics": {"dataset_fingerprint": compute_series_hash(SAMPLE_SERIES_1)}
        }]
    )
    mock_conflict_client = MagicMock()
    mock_conflict_client.table.return_value = mock_conflict_table

    with patch("database.get_supabase_client", return_value=mock_conflict_client):
        try:
            # Client attempts to reuse same custom key for forecast_days=180
            insert_forecast_record(
                user_id="user_a",
                forecast_days=180,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=SAMPLE_PAYLOAD,
                custom_idempotency_key="client_req_001"
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for mismatched horizon!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters" in str(e)
            print("   * 13B Horizon Mismatch: Reused custom key with different horizon rejected with conflict error.")

    # 13C: Conflicting reuse with different CSV URL raises DatabaseIdempotencyConflictError
    with patch("database.get_supabase_client", return_value=mock_conflict_client):
        try:
            insert_forecast_record(
                user_id="user_a",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales_DIFFERENT.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=SAMPLE_PAYLOAD,
                custom_idempotency_key="client_req_001"
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for mismatched CSV URL!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters" in str(e)
            print("   * 13C CSV URL Mismatch: Reused custom key with different CSV rejected with conflict error.")

    # 13D: Reusing custom key with matching parameters safely returns existing record
    with patch("database.get_supabase_client", return_value=mock_conflict_client):
        matching_res = insert_forecast_record(
            user_id="user_a",
            forecast_days=30,
            csv_source_url="https://test.supabase.co/sales.csv",
            series=SAMPLE_SERIES_1,
            forecast_payload=SAMPLE_PAYLOAD,
            custom_idempotency_key="client_req_001"
        )
        assert matching_res["id"] == "prev_f1"
        print("   * 13D Identical Retry: Matching custom key safely returns existing record without insert.")

    # 13E: Conflicting reuse with different dataset content raises DatabaseIdempotencyConflictError
    with patch("database.get_supabase_client", return_value=mock_conflict_client):
        try:
            insert_forecast_record(
                user_id="user_a",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_MODIFIED,  # Modified series -> different fingerprint!
                forecast_payload=SAMPLE_PAYLOAD,
                custom_idempotency_key="client_req_001"
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for modified dataset!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters or data" in str(e)
            print("   * 13E Dataset Content Mismatch: Reused custom key with different dataset rejected with conflict error.")

    # 13F: Conflicting reuse with different forecast_start_date raises DatabaseIdempotencyConflictError
    payload_diff_start = {
        "summary": {**SAMPLE_PAYLOAD["summary"], "forecast_start_date": "2024-06-01"},
        "forecast": SAMPLE_PAYLOAD["forecast"]
    }
    with patch("database.get_supabase_client", return_value=mock_conflict_client):
        try:
            insert_forecast_record(
                user_id="user_a",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=payload_diff_start,
                custom_idempotency_key="client_req_001"
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for mismatched forecast_start_date!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters" in str(e)
            print("   * 13F Start Date Mismatch: Reused key with different forecast_start_date rejected.")

    # 13G: Fail closed: Missing dataset_fingerprint in existing record raises DatabaseIdempotencyConflictError
    mock_legacy_table = MagicMock()
    mock_legacy_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{
            "id": "legacy_f1",
            "user_id": "user_a",
            "forecast_days": 30,
            "forecast_start_date": "2024-01-04",
            "csv_source_url": "https://test.supabase.co/sales.csv",
            "summary_metrics": {}  # Missing dataset_fingerprint!
        }]
    )
    mock_legacy_client = MagicMock()
    mock_legacy_client.table.return_value = mock_legacy_table

    with patch("database.get_supabase_client", return_value=mock_legacy_client):
        try:
            insert_forecast_record(
                user_id="user_a",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=SAMPLE_PAYLOAD,
                custom_idempotency_key="client_req_001"
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for missing fingerprint!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters or data" in str(e)
            print("   * 13G Missing Fingerprint Fail-Closed: Missing dataset_fingerprint rejected without returning record.")

    print(" -> PASS: Custom idempotency key scoping and conflicting reuse verified.")
    passed += 1

    # --------------------------------------------------------------------------
    # TEST 14: Database Lookup Failure Aborts Persistence Immediately
    # --------------------------------------------------------------------------
    print("\n[TEST 14] Idempotency Database Lookup Failure Aborts Persistence...")
    reset_supabase_client()
    mock_table_lookup_err = MagicMock()
    mock_table_lookup_err.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.side_effect = Exception("DB network partition during idempotency lookup")
    
    mock_client_lookup_err = MagicMock()
    mock_client_lookup_err.table.return_value = mock_table_lookup_err

    with patch("database.get_supabase_client", return_value=mock_client_lookup_err):
        try:
            insert_forecast_record(
                user_id="user_123",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=SAMPLE_PAYLOAD
            )
            raise AssertionError("Expected DatabaseConnectionError was not raised on lookup failure")
        except DatabaseConnectionError as e:
            assert "Failed to check existing forecast in database" in str(e)
            mock_table_lookup_err.insert.assert_not_called()
            print(" -> PASS: Idempotency lookup error aborted persistence immediately without attempting insert.")
            passed += 1

    # --------------------------------------------------------------------------
    # TEST 15: Concurrent Duplicate Requests Handled Safely
    # --------------------------------------------------------------------------
    print("\n[TEST 15] Concurrent Duplicate Requests Unique Conflict Resolution...")
    reset_supabase_client()
    mock_table_race = MagicMock()
    # 1st lookup finds empty (race condition: both requests started at same time)
    mock_table_race.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    # insert() fails with unique constraint conflict on idempotency_key
    mock_table_race.insert.return_value.execute.side_effect = Exception("duplicate key value violates unique constraint 'uq_sales_forecasts_idempotency_key'")
    
    # 2nd retry lookup finds the record that was inserted by the concurrent request
    resolved_record = {
        "id": "concurrent_forecast_id_777",
        "user_id": "user_123",
        "forecast_days": 30,
        "forecast_start_date": "2024-01-04",
        "csv_source_url": "https://test.supabase.co/sales.csv",
        "summary_metrics": {"dataset_fingerprint": compute_series_hash(SAMPLE_SERIES_1)}
    }
    mock_retry_query = MagicMock()
    mock_retry_query.data = [resolved_record]
    
    # Configure retry lookup on table
    mock_table_race.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_retry_query
    
    mock_client_race = MagicMock()
    mock_client_race.table.return_value = mock_table_race

    with patch("database.get_supabase_client", return_value=mock_client_race):
        res = insert_forecast_record(
            user_id="user_123",
            forecast_days=30,
            csv_source_url="https://test.supabase.co/sales.csv",
            series=SAMPLE_SERIES_1,
            forecast_payload=SAMPLE_PAYLOAD
        )
        assert res["id"] == "concurrent_forecast_id_777"
        print(" -> PASS: Concurrent unique constraint conflict safely resolved by retrieving the existing record.")

    # 15B: Concurrent conflict with mismatched dataset raises DatabaseIdempotencyConflictError
    with patch("database.get_supabase_client", return_value=mock_client_race):
        try:
            insert_forecast_record(
                user_id="user_123",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_MODIFIED,  # Mismatched series
                forecast_payload=SAMPLE_PAYLOAD
            )
            raise AssertionError("Expected DatabaseIdempotencyConflictError was not raised for concurrent conflict with modified data!")
        except DatabaseIdempotencyConflictError as e:
            assert "different forecast parameters or data" in str(e)
            print(" -> PASS: Concurrent conflict with mismatched dataset content safely rejected with conflict error.")

    passed += 1

    # --------------------------------------------------------------------------
    # TEST 16: Forecast History Retrieval with User Scoping
    # --------------------------------------------------------------------------
    print("\n[TEST 16] Forecast History Retrieval Scoped to Authenticated User...")
    reset_supabase_client()
    
    history_records = [
        {"id": "f1", "user_id": "user_alice", "forecast_days": 30, "created_at": "2024-01-04T12:00:00Z"},
        {"id": "f2", "user_id": "user_alice", "forecast_days": 180, "created_at": "2024-01-03T12:00:00Z"}
    ]
    mock_table_hist = MagicMock()
    mock_query = mock_table_hist.select.return_value.eq.return_value.order.return_value.range.return_value
    mock_query.execute.return_value = MagicMock(data=history_records, count=2)
    
    mock_client_hist = MagicMock()
    mock_client_hist.table.return_value = mock_table_hist

    with patch("database.get_supabase_client", return_value=mock_client_hist):
        result = get_user_forecast_history(user_id="user_alice", limit=10, offset=0)
        assert len(result["history"]) == 2
        assert result["total"] == 2
        
        eq_call = mock_table_hist.select.return_value.eq.call_args
        assert eq_call[0] == ("user_id", "user_alice")
        print(" -> PASS: History query strictly scoped to authenticated user with pagination.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 17: Cross-Tenant Isolation: User A Cannot Retrieve User B's Forecast
    # --------------------------------------------------------------------------
    print("\n[TEST 17] Cross-Tenant Isolation on Single Record Retrieval...")
    reset_supabase_client()
    
    mock_table_iso = MagicMock()
    mock_table_iso.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    
    mock_client_iso = MagicMock()
    mock_client_iso.table.return_value = mock_table_iso

    with patch("database.get_supabase_client", return_value=mock_client_iso):
        record = get_forecast_by_id(user_id="user_attacker_a", forecast_id="victim_b_forecast_id")
        assert record is None, "Cross-tenant record leak detected!"
        print(" -> PASS: Cross-tenant access blocked. User A cannot view User B's forecast record.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 18: Single Record Deletion Scoped to Owner
    # --------------------------------------------------------------------------
    print("\n[TEST 18] Single Record Deletion Scoped to Owner...")
    reset_supabase_client()
    
    mock_table_del = MagicMock()
    mock_table_del.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "del_123"}])
    mock_client_del = MagicMock()
    mock_client_del.table.return_value = mock_table_del

    with patch("database.get_supabase_client", return_value=mock_client_del):
        deleted = delete_forecast_by_id(user_id="owner_user", forecast_id="del_123")
        assert deleted is True
        print(" -> PASS: Record deletion strictly scoped to owner.")
        passed += 1

    # --------------------------------------------------------------------------
    # TEST 19: Database Insertion Failure Produces Safe Error
    # --------------------------------------------------------------------------
    print("\n[TEST 19] Database Insertion Failure Error Sanitization...")
    reset_supabase_client()
    
    mock_table_err = MagicMock()
    mock_table_err.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    mock_table_err.insert.side_effect = Exception("Internal PostgreSQL server error at 10.0.0.1:5432 with secret key XYZ")
    
    mock_client_err = MagicMock()
    mock_client_err.table.return_value = mock_table_err

    with patch("database.get_supabase_client", return_value=mock_client_err):
        try:
            insert_forecast_record(
                user_id="user_123",
                forecast_days=30,
                csv_source_url="https://test.supabase.co/sales.csv",
                series=SAMPLE_SERIES_1,
                forecast_payload=SAMPLE_PAYLOAD
            )
            raise AssertionError("Expected DatabaseConnectionError was not raised")
        except DatabaseConnectionError as e:
            assert "XYZ" not in str(e), "CRITICAL: Secret leaked in error message!"
            assert "5432" not in str(e)
            assert "Failed to store forecast in database" in str(e)
            print(" -> PASS: Database error cleanly sanitized without leaking internal details or credentials.")
            passed += 1

    print("\n" + "=" * 70)
    print(f"DATABASE PERSISTENCE RESULTS: {passed}/{total} TESTS PASSED (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
