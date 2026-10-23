"""
SMART BPI — Live Supabase Forecast Persistence & Idempotency Verification
Performs real operations against public.sales_forecasts using service-role credentials:
1. Verifies database readiness and table accessibility.
2. Generates real SARIMA forecast data for a test user.
3. Inserts the forecast record into Supabase PostgreSQL.
4. Reads back the saved record directly from Supabase and validates:
   - user_id matches authenticated test UID
   - forecast_days matches horizon (30)
   - forecast_start_date matches projection start date
   - dataset_fingerprint matches lossless series hash
   - forecast_records count and KPIs match model outputs
5. Repeats the identical request to verify idempotency (same ID returned, no duplicate row).
6. Tests cross-tenant security:
   - An unauthorized user cannot retrieve the record (returns None)
   - An unauthorized user cannot delete the record (returns False)
7. Deletes the test record using the owner UID and confirms clean removal.
"""

import os
import sys
import datetime
import pandas as pd

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    check_database_readiness,
    get_supabase_client,
    insert_forecast_record,
    get_forecast_by_id,
    delete_forecast_by_id,
    compute_series_hash,
)
from forecasting import (
    train_sarima_model,
    generate_forecast_payload,
)

def run_live_verification():
    print("=" * 70)
    print("RUNNING LIVE SUPABASE FORECAST PERSISTENCE VERIFICATION")
    print("=" * 70)

    # 1. Check readiness
    readiness = check_database_readiness()
    print(f"\n[STEP 1] Database Readiness Check:")
    print(f"  - Ready: {readiness.get('ready')}")
    print(f"  - Status: {readiness.get('status')}")
    print(f"  - Table Accessible: {readiness.get('table_accessible')}")
    assert readiness["ready"] is True, f"Database not ready: {readiness}"
    assert readiness["table_accessible"] is True, "sales_forecasts table not accessible!"
    print("  -> PASS: Database is connected and public.sales_forecasts is accessible.")

    # 2. Build test sales time series (90 continuous daily rows)
    base_date = datetime.date(2024, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(90)]
    # Realistic revenue sequence
    revenues = [100.0 + (i % 7) * 15.0 + i * 0.5 for i in range(90)]
    series = pd.Series(revenues, index=pd.to_datetime(dates), name="daily_revenue")
    data_fingerprint = compute_series_hash(series)

    # 3. Train SARIMA model and generate forecast payload
    print("\n[STEP 2] Training SARIMA model and generating forecast...")
    sarima_results = train_sarima_model(series)
    forecast_payload = generate_forecast_payload(sarima_results, series, forecast_days=30)
    summary = forecast_payload["summary"]
    print(f"  - Forecast 30-Day Total: ${summary['forecast_30_days_total']:,.2f}")
    print(f"  - Forecast 30-Day Average: ${summary['forecast_30_days_average']:,.2f}")
    print(f"  - Forecast Start Date: {summary['forecast_start_date']}")
    print(f"  - Expected Growth: {summary['expected_growth_percentage']}%")
    print("  -> PASS: Model trained and forecast payload generated successfully.")

    # 4. Insert forecast record into Supabase PostgreSQL
    TEST_USER_ID = "verified_test_user_smart_bpi"
    TEST_CSV_URL = (
        "https://dzwzsnnrfstsmtpueilo.supabase.co"
        "/storage/v1/object/public/service-data"
        "/verified_test_user_smart_bpi/sales_live_test.csv"
    )

    print("\n[STEP 3] Persisting forecast to Supabase public.sales_forecasts...")
    inserted_record = insert_forecast_record(
        user_id=TEST_USER_ID,
        forecast_days=30,
        csv_source_url=TEST_CSV_URL,
        series=series,
        forecast_payload=forecast_payload
    )
    record_id = inserted_record.get("id")
    print(f"  - Inserted Record UUID: {record_id}")
    print(f"  - Created At: {inserted_record.get('created_at')}")
    assert record_id is not None, "Insert did not return a record ID!"
    print("  -> PASS: Forecast record successfully persisted to Supabase.")

    # 5. Read back the record directly from Supabase and verify fields
    print("\n[STEP 4] Reading back and verifying saved record from Supabase...")
    saved_record = get_forecast_by_id(user_id=TEST_USER_ID, forecast_id=record_id)
    assert saved_record is not None, "Failed to retrieve saved record from Supabase!"

    assert saved_record["user_id"] == TEST_USER_ID, f"User ID mismatch: {saved_record['user_id']}"
    assert saved_record["forecast_days"] == 30, f"Horizon mismatch: {saved_record['forecast_days']}"
    assert saved_record["forecast_start_date"] == summary["forecast_start_date"], "Start date mismatch"
    assert saved_record["forecast_end_date"] == summary["forecast_end_date"], "End date mismatch"
    assert float(saved_record["forecast_total"]) == float(summary["forecast_30_days_total"]), "Total mismatch"
    assert float(saved_record["forecast_average"]) == float(summary["forecast_30_days_average"]), "Average mismatch"
    assert float(saved_record["expected_growth_percentage"]) == float(summary["expected_growth_percentage"]), "Growth mismatch"
    assert saved_record["csv_source_url"] == TEST_CSV_URL, "CSV URL mismatch"

    # Verify dataset fingerprint in summary_metrics
    saved_summary = saved_record.get("summary_metrics") or {}
    assert saved_summary.get("dataset_fingerprint") == data_fingerprint, "Dataset fingerprint mismatch in summary_metrics!"

    # Verify forecast_records
    forecast_records = saved_record.get("forecast_records") or []
    assert len(forecast_records) == 30, f"Expected 30 daily predictions, got {len(forecast_records)}"
    first_pred = forecast_records[0]
    assert "date" in first_pred and "predicted_sales" in first_pred and "lower_bound" in first_pred and "upper_bound" in first_pred

    print(f"  - Confirmed User ID: {saved_record['user_id']}")
    print(f"  - Confirmed Horizon: {saved_record['forecast_days']} days")
    print(f"  - Confirmed Start Date: {saved_record['forecast_start_date']}")
    print(f"  - Confirmed Dataset Fingerprint: {saved_summary.get('dataset_fingerprint')[:16]}... (matches series)")
    print(f"  - Confirmed Daily Records Count: {len(forecast_records)}")
    print("  -> PASS: All record columns and summary metrics match expected values.")

    # 6. Verify Idempotency on repeated request
    print("\n[STEP 5] Testing idempotency on repeated identical request...")
    duplicate_run = insert_forecast_record(
        user_id=TEST_USER_ID,
        forecast_days=30,
        csv_source_url=TEST_CSV_URL,
        series=series,
        forecast_payload=forecast_payload
    )
    assert duplicate_run["id"] == record_id, f"Idempotency failed: new ID generated ({duplicate_run['id']} != {record_id})"
    
    # Confirm no duplicate rows in database
    client = get_supabase_client()
    count_check = client.table("sales_forecasts").select("id", count="exact").eq("user_id", TEST_USER_ID).execute()
    assert count_check.count == 1, f"Expected exactly 1 record for test user, found {count_check.count}"
    print(f"  - Existing Record ID returned: {duplicate_run['id']}")
    print(f"  - Exact Database Record Count: {count_check.count}")
    print("  -> PASS: Idempotency confirmed. Duplicate request returned existing record without inserting duplicate.")

    # 7. Verify Cross-Tenant Isolation
    print("\n[STEP 6] Testing cross-tenant security isolation...")
    ATTACKER_UID = "unauthorized_user_attacker_999"
    
    # Unauthorized retrieval attempt
    attacker_view = get_forecast_by_id(user_id=ATTACKER_UID, forecast_id=record_id)
    assert attacker_view is None, "SECURITY FAILURE: Unauthorized user was able to retrieve forecast record!"
    print("  - Attacker retrieval attempt returned: None (Blocked)")

    # Unauthorized deletion attempt
    attacker_del = delete_forecast_by_id(user_id=ATTACKER_UID, forecast_id=record_id)
    assert attacker_del is False, "SECURITY FAILURE: Unauthorized user was able to delete forecast record!"
    print("  - Attacker deletion attempt returned: False (Blocked)")

    # Verify the record is still intact for the owner
    owner_check = get_forecast_by_id(user_id=TEST_USER_ID, forecast_id=record_id)
    assert owner_check is not None, "Owner record was unexpectedly deleted!"
    print("  -> PASS: Cross-tenant isolation verified. Other users cannot read or delete the record.")

    # 8. Clean up test record using owner credentials
    print("\n[STEP 7] Cleaning up test record...")
    deleted = delete_forecast_by_id(user_id=TEST_USER_ID, forecast_id=record_id)
    assert deleted is True, "Failed to delete test record!"
    cleanup_check = get_forecast_by_id(user_id=TEST_USER_ID, forecast_id=record_id)
    assert cleanup_check is None, "Test record still exists after deletion!"
    print("  -> PASS: Test record safely cleaned up from Supabase database.")

    print("\n" + "=" * 70)
    print("ALL LIVE SUPABASE PERSISTENCE & IDEMPOTENCY CHECKS PASSED (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_live_verification()
