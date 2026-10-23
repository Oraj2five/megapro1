"""
SMART BPI — Model Versioning & Idempotency Tests
Verifies:
1. Same data + same horizon + same model version -> reuses existing record (idempotency hit).
2. Same data + same horizon + different model version -> computes distinct key and allows fresh forecast generation.
3. Existing historical forecast records remain accessible via get_user_forecast_history.
4. Existing verified stationary 365-day forecast is preserved without generating duplicates.
5. 30-day, 180-day, and 365-day forecasts are internally consistent when generated from the same model fit.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    compute_idempotency_key,
    compute_series_hash,
    insert_forecast_record,
    get_user_forecast_history,
    DEFAULT_MODEL_VERSION,
)
from forecasting import (
    train_sarima_model,
    generate_forecast_payload,
    FORECAST_MODEL_VERSION,
)


class TestModelVersioningAndIdempotency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Deterministic 90-day daily sales series
        base_dates = pd.date_range(start="2026-01-01", periods=90, freq="D")
        np.random.seed(42)
        weekly_pattern = np.array([1000, 1200, 1100, 1300, 1800, 2500, 2000])
        revenues = [float(weekly_pattern[i % 7] + (i * 5)) for i in range(90)]
        cls.series = pd.Series(revenues, index=base_dates, name="daily_revenue")
        cls.user_id = "test_user_mv"
        cls.csv_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/test_user_mv/sales.csv"

        # Fit SARIMA and generate base payload
        cls.model_results = train_sarima_model(cls.series)
        cls.payload_30 = generate_forecast_payload(cls.model_results, cls.series, 30)
        cls.payload_180 = generate_forecast_payload(cls.model_results, cls.series, 180)
        cls.payload_365 = generate_forecast_payload(cls.model_results, cls.series, 365)

    def test_01_same_data_same_horizon_same_version_reuses_record(self):
        """Same data + same horizon + same model version -> returns cached record without duplicate insert."""
        key_1 = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=30,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date="2026-04-01",
            model_version=DEFAULT_MODEL_VERSION,
        )
        key_2 = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=30,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date="2026-04-01",
            model_version=DEFAULT_MODEL_VERSION,
        )
        self.assertEqual(key_1, key_2, "Identical inputs and version must produce identical idempotency key")

        # Mock Supabase client returning existing record
        mock_client = MagicMock()
        existing_row = {
            "id": "rec_uuid_same_version",
            "user_id": self.user_id,
            "forecast_days": 30,
            "forecast_start_date": self.payload_30["summary"]["forecast_start_date"],
            "forecast_end_date": self.payload_30["summary"]["forecast_end_date"],
            "csv_source_url": self.csv_url,
            "idempotency_key": key_1,
            "summary_metrics": {
                "dataset_fingerprint": compute_series_hash(self.series),
                "model_version": DEFAULT_MODEL_VERSION,
            },
        }
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[existing_row]
        )

        with patch("database.get_supabase_client", return_value=mock_client):
            res = insert_forecast_record(
                user_id=self.user_id,
                forecast_days=30,
                csv_source_url=self.csv_url,
                series=self.series,
                forecast_payload=self.payload_30,
                model_version=DEFAULT_MODEL_VERSION,
            )
            self.assertEqual(res["id"], "rec_uuid_same_version")
            # Verify insert was NEVER called
            mock_client.table.return_value.insert.assert_not_called()

    def test_02_different_model_version_generates_distinct_key_and_allows_new_forecast(self):
        """Same data + same horizon + different model version -> distinct key, allowing new forecast."""
        key_v1 = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=30,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date="2026-04-01",
            model_version="sarima_v1_legacy",
        )
        key_v2 = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=30,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date="2026-04-01",
            model_version=DEFAULT_MODEL_VERSION,
        )
        self.assertNotEqual(key_v1, key_v2, "Different model versions must generate distinct idempotency keys")

        # Mock Supabase client: lookup by key_v2 finds NO record, proceeds to insert
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )
        new_row = {"id": "rec_uuid_fresh_v2", "created_at": "2026-10-23T20:00:00Z"}
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[new_row])

        with patch("database.get_supabase_client", return_value=mock_client):
            res = insert_forecast_record(
                user_id=self.user_id,
                forecast_days=30,
                csv_source_url=self.csv_url,
                series=self.series,
                forecast_payload=self.payload_30,
                model_version=DEFAULT_MODEL_VERSION,
            )
            self.assertEqual(res["id"], "rec_uuid_fresh_v2")
            mock_client.table.return_value.insert.assert_called_once()
            inserted_data = mock_client.table.return_value.insert.call_args[0][0]
            self.assertEqual(inserted_data["idempotency_key"], key_v2)
            self.assertEqual(inserted_data["summary_metrics"]["model_version"], DEFAULT_MODEL_VERSION)

    def test_03_existing_historical_forecast_records_remain_accessible(self):
        """Historical forecast records (across both legacy and new versions) remain fully queryable."""
        mock_client = MagicMock()
        mock_history_records = [
            {"id": "uuid-v2-365", "forecast_days": 365, "created_at": "2026-10-23T14:28:00Z"},
            {"id": "uuid-v1-180", "forecast_days": 180, "created_at": "2026-10-23T10:06:00Z"},
            {"id": "uuid-v1-30", "forecast_days": 30, "created_at": "2026-10-23T09:58:00Z"},
        ]
        mock_query = mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value
        mock_query.execute.return_value = MagicMock(data=mock_history_records, count=3)

        with patch("database.get_supabase_client", return_value=mock_client):
            history_result = get_user_forecast_history(user_id=self.user_id, limit=10, offset=0)
            self.assertEqual(history_result["total"], 3)
            self.assertEqual(len(history_result["history"]), 3)
            self.assertEqual(history_result["history"][0]["id"], "uuid-v2-365")
            self.assertEqual(history_result["history"][1]["id"], "uuid-v1-180")
            self.assertEqual(history_result["history"][2]["id"], "uuid-v1-30")

    def test_04_fallback_preserves_stationary_365_day_forecast_without_duplicate(self):
        """A verified stationary 365-day forecast saved under legacy unversioned key is adopted without duplicate insert."""
        start_date = self.payload_365["summary"]["forecast_start_date"]
        legacy_key = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=365,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date=start_date,
            model_version=None,
        )
        new_key = compute_idempotency_key(
            user_id=self.user_id,
            forecast_days=365,
            series=self.series,
            csv_source_url=self.csv_url,
            forecast_start_date=start_date,
            model_version=DEFAULT_MODEL_VERSION,
        )
        self.assertNotEqual(legacy_key, new_key)

        mock_client = MagicMock()
        # First query (with new_key) returns empty
        # Second query (legacy fallback) returns the existing valid stationary record
        existing_365_row = {
            "id": "rec_uuid_existing_stationary_365",
            "user_id": self.user_id,
            "forecast_days": 365,
            "forecast_start_date": start_date,
            "forecast_end_date": self.payload_365["summary"]["forecast_end_date"],
            "csv_source_url": self.csv_url,
            "idempotency_key": legacy_key,
            "summary_metrics": {
                "dataset_fingerprint": compute_series_hash(self.series),
                "forecast_1_year_total": 44880650.01,
            },
        }

        # Set up side_effect for .execute() on eq().eq().limit()
        mock_limit = mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value
        mock_limit.execute.side_effect = [
            MagicMock(data=[]),  # new key query: miss
            MagicMock(data=[existing_365_row]),  # legacy key query: hit!
        ]

        with patch("database.get_supabase_client", return_value=mock_client):
            res = insert_forecast_record(
                user_id=self.user_id,
                forecast_days=365,
                csv_source_url=self.csv_url,
                series=self.series,
                forecast_payload=self.payload_365,
                model_version=DEFAULT_MODEL_VERSION,
            )
            self.assertEqual(res["id"], "rec_uuid_existing_stationary_365")
            # Verify insert was NOT called
            mock_client.table.return_value.insert.assert_not_called()

    def test_05_three_horizons_internally_consistent_from_same_model_fit(self):
        """30-day, 180-day, and 365-day forecasts generated under the stationary model match identically on common days."""
        preds_30 = [r["predicted_sales"] for r in self.payload_30["forecast"]]
        preds_180 = [r["predicted_sales"] for r in self.payload_180["forecast"]]
        preds_365 = [r["predicted_sales"] for r in self.payload_365["forecast"]]

        self.assertEqual(len(preds_30), 30)
        self.assertEqual(len(preds_180), 180)
        self.assertEqual(len(preds_365), 365)

        # First 30 days must match across all 3 horizons
        self.assertEqual(preds_30, preds_180[:30])
        self.assertEqual(preds_30, preds_365[:30])
        # First 180 days must match between 180 and 365
        self.assertEqual(preds_180, preds_365[:180])

        # All values strictly non-negative and finite
        for p in preds_365:
            self.assertGreaterEqual(p, 0.0)
            self.assertFalse(np.isnan(p))
            self.assertFalse(np.isinf(p))

    def test_06_history_derives_dates_from_forecast_records_with_fallback(self):
        """History and detail queries derive start/end dates from forecast_records with safe fallback to summary columns."""
        mock_client = MagicMock()
        mock_history_records = [
            # Record 1: Old 30d forecast with incorrect summary end date (2027-10-23) but valid 30 records
            {
                "id": "rec_old_30d",
                "forecast_days": 30,
                "forecast_start_date": "2026-10-24",
                "forecast_end_date": "2027-10-23",  # Stale summary
                "forecast_records": [
                    {"date": "2026-10-24", "predicted_sales": 100.0},
                    {"date": "2026-11-22", "predicted_sales": 120.0},
                ],
                "created_at": "2026-10-23T09:58:00Z"
            },
            # Record 2: Older record without forecast_records (safe fallback)
            {
                "id": "rec_legacy_no_records",
                "forecast_days": 180,
                "forecast_start_date": "2026-10-24",
                "forecast_end_date": "2027-04-21",
                "forecast_records": None,
                "created_at": "2026-10-22T10:00:00Z"
            }
        ]
        mock_query = mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value
        mock_query.execute.return_value = MagicMock(data=mock_history_records, count=2)

        with patch("database.get_supabase_client", return_value=mock_client):
            history_result = get_user_forecast_history(user_id=self.user_id, limit=10, offset=0)
            items = history_result["history"]

            # Record 1: derived from forecast_records
            self.assertEqual(items[0]["forecast_start_date"], "2026-10-24")
            self.assertEqual(items[0]["forecast_end_date"], "2026-11-22")

            # Record 2: safely fell back to existing summary fields
            self.assertEqual(items[1]["forecast_start_date"], "2026-10-24")
            self.assertEqual(items[1]["forecast_end_date"], "2027-04-21")


if __name__ == "__main__":
    unittest.main()
