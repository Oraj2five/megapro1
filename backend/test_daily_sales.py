"""
SMART BPI — Step 4: Daily Sales Entry Focused Tests
Covers all required areas:
1. Valid daily sales entry creation and return format.
2. Validation of invalid dates (bad format, non-existent calendar date) and invalid/non-finite revenue (negative, NaN, Inf, non-numeric).
3. Duplicate sales date rejection (pre-check and database unique constraint race condition).
4. Authenticated user ownership isolation (records tagged with verified Firebase UID).
5. Cross-user read and update isolation (User B cannot read or modify User A's entries).
6. Successful database persistence, retrieval, and updates.
7. Manual sales entries being included in the forecasting input and model preparation.
8. CSV upload compatibility: clean merge of non-overlapping dates, exact deduplication of matching dates, and explicit conflict detection when CSV and manual values differ.
"""

import os
import sys
import math
import unittest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    validate_daily_sales_inputs,
    create_daily_sales,
    update_daily_sales,
    get_user_daily_sales,
    get_user_all_daily_sales,
    DailySalesConflictError,
    DatabaseConnectionError
)
from forecasting import (
    merge_sales_sources,
    validate_and_prepare_sales_data
)


class TestDailySalesModule(unittest.TestCase):

    # --------------------------------------------------------------------------
    # 1. Valid Daily Sales Entry
    # --------------------------------------------------------------------------
    def test_01_valid_daily_sales_entry(self):
        clean_date, clean_rev = validate_daily_sales_inputs("2024-03-15", 12500.556)
        self.assertEqual(clean_date, "2024-03-15")
        self.assertEqual(clean_rev, 12500.56)

        # Zero revenue is valid (non-negative)
        clean_date_zero, clean_rev_zero = validate_daily_sales_inputs("2024-03-16", 0)
        self.assertEqual(clean_date_zero, "2024-03-16")
        self.assertEqual(clean_rev_zero, 0.0)

    # --------------------------------------------------------------------------
    # 2. Invalid Date and Invalid/Non-Finite Revenue
    # --------------------------------------------------------------------------
    def test_02_invalid_date_validation(self):
        # Invalid format
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("15-03-2024", 1000)
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("not-a-date", 1000)
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("", 1000)
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs(None, 1000)

        # Invalid calendar date (February 30th)
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-02-30", 1000)
        # Month 13
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-13-01", 1000)

    def test_02_invalid_revenue_validation(self):
        # Negative revenue
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", -50.0)
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", -0.01)

        # Non-finite values
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", float("nan"))
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", float("inf"))
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", float("-inf"))

        # Non-numeric
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", "abc")
        with self.assertRaises(ValueError):
            validate_daily_sales_inputs("2024-03-15", None)

    # --------------------------------------------------------------------------
    # 3. Duplicate Sales Date and Concurrent Duplicate Submission
    # --------------------------------------------------------------------------
    def test_03_duplicate_date_precheck(self):
        mock_client = MagicMock()
        # Mock existing record found on pre-check
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": "existing-uuid", "user_id": "test_user_1", "sale_date": "2024-03-15", "daily_revenue": 5000.0}]
        )

        with patch("database.get_supabase_client", return_value=mock_client):
            with self.assertRaises(DailySalesConflictError) as ctx:
                create_daily_sales(user_id="test_user_1", sale_date="2024-03-15", daily_revenue=6000.0)
            self.assertIn("already exists", str(ctx.exception))

    def test_03_concurrent_duplicate_race_condition(self):
        mock_client = MagicMock()
        # Pre-check returns empty (simulating race before second insert hits constraint)
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        # Insert raises Postgres 23505 unique constraint violation on uq_daily_sales_user_date
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception(
            "duplicate key value violates unique constraint 'uq_daily_sales_user_date'"
        )

        with patch("database.get_supabase_client", return_value=mock_client):
            with self.assertRaises(DailySalesConflictError) as ctx:
                create_daily_sales(user_id="test_user_1", sale_date="2024-03-15", daily_revenue=6000.0)
            self.assertIn("already exists", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 4. Authenticated User Ownership Isolation
    # --------------------------------------------------------------------------
    def test_04_user_ownership_enforcement(self):
        # Empty user_id rejected
        with self.assertRaises(ValueError):
            create_daily_sales(user_id="", sale_date="2024-03-15", daily_revenue=1000.0)
        with self.assertRaises(ValueError):
            create_daily_sales(user_id=None, sale_date="2024-03-15", daily_revenue=1000.0)

        # Successful creation must pass exact authenticated user_id in payload
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_inserted = {
            "id": "uuid-123",
            "user_id": "auth_user_alpha",
            "sale_date": "2024-03-15",
            "daily_revenue": 1000.0,
            "created_at": "2024-03-15T10:00:00Z",
            "updated_at": "2024-03-15T10:00:00Z"
        }
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[mock_inserted])

        with patch("database.get_supabase_client", return_value=mock_client):
            result = create_daily_sales(user_id="auth_user_alpha", sale_date="2024-03-15", daily_revenue=1000.0)
            self.assertEqual(result["user_id"], "auth_user_alpha")
            insert_call_args = mock_client.table.return_value.insert.call_args[0][0]
            self.assertEqual(insert_call_args["user_id"], "auth_user_alpha")

    # --------------------------------------------------------------------------
    # 5. Cross-User Read and Update Isolation
    # --------------------------------------------------------------------------
    def test_05_cross_user_isolation(self):
        mock_client = MagicMock()
        # Update attempt where user_id='user_attacker' tries to edit 'entry-owned-by-victim'
        # .eq("id", entry_id).eq("user_id", user_id) yields no records because owner doesn't match
        mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

        with patch("database.get_supabase_client", return_value=mock_client):
            result = update_daily_sales(user_id="user_attacker", entry_id="entry-owned-by-victim", daily_revenue=9999.0)
            self.assertIsNone(result)

            # Confirm .eq("id", "entry-owned-by-victim") and .eq("user_id", "user_attacker") were strictly called
            first_eq = mock_client.table.return_value.update.return_value.eq
            self.assertEqual(first_eq.call_args[0], ("id", "entry-owned-by-victim"))

            second_eq = first_eq.return_value.eq
            self.assertEqual(second_eq.call_args[0], ("user_id", "user_attacker"))

    # --------------------------------------------------------------------------
    # 6. Database Persistence and Readback
    # --------------------------------------------------------------------------
    def test_06_database_persistence_and_readback(self):
        mock_client = MagicMock()
        mock_records = [
            {"id": "entry-1", "user_id": "test_user", "sale_date": "2024-03-16", "daily_revenue": 1500.0},
            {"id": "entry-2", "user_id": "test_user", "sale_date": "2024-03-15", "daily_revenue": 1200.0},
        ]
        mock_query = mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value
        mock_query.execute.return_value = MagicMock(data=mock_records, count=2)

        with patch("database.get_supabase_client", return_value=mock_client):
            history_res = get_user_daily_sales(user_id="test_user", limit=20, offset=0)
            self.assertEqual(len(history_res["entries"]), 2)
            self.assertEqual(history_res["total"], 2)
            self.assertEqual(history_res["entries"][0]["daily_revenue"], 1500.0)

    # --------------------------------------------------------------------------
    # 7. Manual Sales Entries Included in Forecasting Input
    # --------------------------------------------------------------------------
    def test_07_manual_sales_forecasting_integration(self):
        # Generate 65 continuous daily records purely from manual entries
        start_date = datetime(2024, 1, 1)
        manual_entries = []
        for i in range(65):
            cur_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            manual_entries.append({
                "sale_date": cur_date,
                "daily_revenue": 1000.0 + i * 10
            })

        # Merge with no CSV
        merged_df = merge_sales_sources(df_csv=None, manual_entries=manual_entries)
        self.assertEqual(len(merged_df), 65)

        # Validate with existing forecasting data preparer
        series = validate_and_prepare_sales_data(merged_df)
        self.assertEqual(len(series), 65)
        self.assertEqual(series.iloc[0], 1000.0)
        self.assertEqual(series.iloc[-1], 1000.0 + 64 * 10)

    # --------------------------------------------------------------------------
    # 8. CSV Upload Compatibility & Deduplication / Conflict Prevention
    # --------------------------------------------------------------------------
    def test_08_csv_merge_and_extension(self):
        # Base CSV has days 1..60
        start_date = datetime(2024, 1, 1)
        csv_rows = []
        for i in range(60):
            cur_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            csv_rows.append({"sale_date": cur_date, "daily_revenue": 500.0})
        df_csv = pd.DataFrame(csv_rows)

        # Manual entries have days 61..65 (chronological extension)
        manual_entries = []
        for i in range(60, 65):
            cur_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            manual_entries.append({"sale_date": cur_date, "daily_revenue": 600.0})

        merged_df = merge_sales_sources(df_csv=df_csv, manual_entries=manual_entries)
        self.assertEqual(len(merged_df), 65)
        # Check first and last
        self.assertEqual(merged_df.iloc[0]["daily_revenue"], 500.0)
        self.assertEqual(merged_df.iloc[-1]["daily_revenue"], 600.0)

    def test_08_csv_duplicate_deduplication(self):
        # CSV and manual entry have the same date with matching revenue (500.0)
        df_csv = pd.DataFrame([{"sale_date": "2024-01-01", "daily_revenue": 500.0}])
        manual_entries = [{"sale_date": "2024-01-01", "daily_revenue": 500.0}]

        merged_df = merge_sales_sources(df_csv=df_csv, manual_entries=manual_entries)
        # Deduplicated: not counted twice
        self.assertEqual(len(merged_df), 1)
        self.assertEqual(merged_df.iloc[0]["daily_revenue"], 500.0)

    def test_08_csv_conflict_detection(self):
        # CSV has ₹500, but manual entry has ₹600 on the same date
        df_csv = pd.DataFrame([{"sale_date": "2024-01-01", "daily_revenue": 500.0}])
        manual_entries = [{"sale_date": "2024-01-01", "daily_revenue": 600.0}]

        with self.assertRaises(DailySalesConflictError) as ctx:
            merge_sales_sources(df_csv=df_csv, manual_entries=manual_entries)
        self.assertIn("Sales data conflict on date '2024-01-01'", str(ctx.exception))
        self.assertIn("500.00", str(ctx.exception))
        self.assertIn("600.00", str(ctx.exception))


class TestDailySalesAPIEndpoints(unittest.TestCase):
    """
    Tests the FastAPI HTTP endpoints for /sales/daily:
    - POST /sales/daily (201 Created, 400 Bad Request, 409 Conflict, 401 Unauthorized)
    - GET /sales/daily (200 OK)
    - PUT /sales/daily/{id} (200 OK, 404 Not Found, 400 Bad Request, 409 Conflict)
    """

    @classmethod
    def setUpClass(cls):
        from main import app, verify_user
        from fastapi.testclient import TestClient
        cls.app = app
        cls.verify_user = verify_user
        cls.client = TestClient(app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_api_unauthorized_without_token(self):
        # Missing Authorization header -> 401
        res = self.client.post("/sales/daily", json={"sale_date": "2024-03-15", "daily_revenue": 1000.0})
        self.assertEqual(res.status_code, 401)

    def test_api_create_daily_sales_success(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        mock_record = {
            "id": "uuid-daily-101",
            "user_id": "test_tenant_uid",
            "sale_date": "2024-03-15",
            "daily_revenue": 12500.0,
            "created_at": "2024-03-15T12:00:00Z"
        }
        with patch("main.create_daily_sales", return_value=mock_record):
            res = self.client.post(
                "/sales/daily",
                json={"sale_date": "2024-03-15", "daily_revenue": 12500.0},
                headers={"Authorization": "Bearer fake_token"}
            )
            self.assertEqual(res.status_code, 201)
            data = res.json()
            self.assertEqual(data["id"], "uuid-daily-101")
            self.assertEqual(data["daily_revenue"], 12500.0)

    def test_api_create_duplicate_date_conflict(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        with patch("main.create_daily_sales", side_effect=DailySalesConflictError("A sales record for date '2024-03-15' already exists.")):
            res = self.client.post(
                "/sales/daily",
                json={"sale_date": "2024-03-15", "daily_revenue": 12500.0},
                headers={"Authorization": "Bearer fake_token"}
            )
            self.assertEqual(res.status_code, 409)
            self.assertIn("already exists", res.json()["detail"])

    def test_api_create_negative_revenue_rejected(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        res = self.client.post(
            "/sales/daily",
            json={"sale_date": "2024-03-15", "daily_revenue": -50.0},
            headers={"Authorization": "Bearer fake_token"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Negative sales values are not permitted", res.json()["detail"])

    def test_api_create_invalid_date_rejected(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        res = self.client.post(
            "/sales/daily",
            json={"sale_date": "2024-02-30", "daily_revenue": 500.0},
            headers={"Authorization": "Bearer fake_token"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid date format", res.json()["detail"])

    def test_api_get_daily_sales_list(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        mock_response = {
            "entries": [
                {"id": "entry-1", "user_id": "test_tenant_uid", "sale_date": "2024-03-15", "daily_revenue": 1000.0}
            ],
            "total": 1,
            "limit": 50,
            "offset": 0
        }
        with patch("main.get_user_daily_sales", return_value=mock_response):
            res = self.client.get(
                "/sales/daily",
                headers={"Authorization": "Bearer fake_token"}
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(len(data["entries"]), 1)
            self.assertEqual(data["total"], 1)

    def test_api_update_daily_sales_success(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "test_tenant_uid"
        mock_updated = {
            "id": "entry-1",
            "user_id": "test_tenant_uid",
            "sale_date": "2024-03-15",
            "daily_revenue": 1800.0,
            "updated_at": "2024-03-15T14:00:00Z"
        }
        with patch("main.update_daily_sales", return_value=mock_updated):
            res = self.client.put(
                "/sales/daily/entry-1",
                json={"daily_revenue": 1800.0},
                headers={"Authorization": "Bearer fake_token"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["daily_revenue"], 1800.0)

    def test_api_update_cross_user_not_found(self):
        self.app.dependency_overrides[self.verify_user] = lambda: "attacker_uid"
        # When entry not found or belongs to another user, update_daily_sales returns None
        with patch("main.update_daily_sales", return_value=None):
            res = self.client.put(
                "/sales/daily/victim-entry-id",
                json={"daily_revenue": 9999.0},
                headers={"Authorization": "Bearer fake_token"}
            )
            self.assertEqual(res.status_code, 404)
            self.assertIn("not found", res.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
