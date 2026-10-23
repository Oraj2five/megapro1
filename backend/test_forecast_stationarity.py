"""
SMART BPI — 365-Day Forecast Stability & Persistence Verification Test
Confirms that:
1. train_sarima_model enforces stationarity and prevents explosive autoregressive projections.
2. 30-day, 180-day, and 365-day forecasts produce realistic, bounded values.
3. Calculated forecast_total, forecast_average, and expected_growth_percentage strictly
   satisfy PostgreSQL schema limits (NUMERIC(15, 2) < 10^13, NUMERIC(6, 2) < 10^4).
4. Full insertion into Supabase PostgreSQL (mocked and schema-validated) succeeds without overflow.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting import train_sarima_model, generate_forecast_payload
from database import insert_forecast_record, compute_series_hash


class TestForecastStationarityAndPersistence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate 206 days of realistic business daily sales with seasonal pattern
        # similar to the user's dataset
        np.random.seed(42)
        base_date = pd.date_range(start="2026-04-01", periods=206, freq="D")
        
        # Base trend around 35,000 + weekly seasonality (higher on weekends) + random variation
        weekly_pattern = np.array([30000, 32000, 35000, 34000, 45000, 60000, 50000])
        revenues = []
        for i in range(206):
            base_rev = weekly_pattern[i % 7] + np.random.uniform(-5000, 10000)
            if i % 30 == 0:  # Occasional spikes
                base_rev += 40000
            revenues.append(max(10000.0, round(float(base_rev), 2)))

        cls.series = pd.Series(revenues, index=base_date, name="daily_revenue")

    def test_01_model_trains_with_stationary_roots(self):
        """Confirm SARIMA model trains with stationary roots (|Phi| < 1, |phi| < 1)."""
        results = train_sarima_model(self.series)
        params = results.params

        # Seasonal AR parameter must be strictly inside unit circle (|ar.S.L7| < 1)
        if "ar.S.L7" in params:
            self.assertLess(abs(float(params["ar.S.L7"])), 1.0, "Seasonal AR parameter must be stationary (< 1.0)")
        
        # Regular AR parameter must be strictly inside unit circle (|ar.L1| < 1)
        if "ar.L1" in params:
            self.assertLess(abs(float(params["ar.L1"])), 1.0, "AR parameter must be stationary (< 1.0)")

    def test_02_365_day_forecast_produces_reasonable_bounded_values(self):
        """Confirm 365-day forecast produces stable, non-explosive values within realistic range."""
        results = train_sarima_model(self.series)
        payload = generate_forecast_payload(results, self.series, 365)
        summary = payload["summary"]

        hist_avg = self.series.mean()
        avg_365 = summary["forecast_1_year_average"]
        tot_365 = summary["forecast_1_year_total"]

        # Daily predictions must be within 10x of historical average, not trillions
        self.assertLess(avg_365, hist_avg * 10, f"365-day average {avg_365} is unrealistically large compared to {hist_avg}")
        self.assertGreater(avg_365, 0, "365-day average must be positive")

        # Total 1-year sales must be strictly less than 10^13 (PostgreSQL NUMERIC(15, 2) maximum)
        MAX_NUMERIC_15_2 = 9_999_999_999_999.99
        self.assertLess(tot_365, MAX_NUMERIC_15_2, f"forecast_total {tot_365} exceeds NUMERIC(15, 2) limit!")
        self.assertLess(avg_365, MAX_NUMERIC_15_2, f"forecast_average {avg_365} exceeds NUMERIC(15, 2) limit!")

        # Expected growth percentage must fit in NUMERIC(6, 2) (-9999.99 to 9999.99)
        growth_pct = summary["expected_growth_percentage"]
        self.assertGreaterEqual(growth_pct, -9999.99)
        self.assertLessEqual(growth_pct, 9999.99)

    def test_03_30_and_180_day_forecasts_continue_working(self):
        """Confirm 30-day and 180-day forecast horizons produce valid, expected outputs."""
        results = train_sarima_model(self.series)

        # 30-day horizon
        payload_30 = generate_forecast_payload(results, self.series, 30)
        self.assertEqual(len(payload_30["forecast"]), 30)
        self.assertGreater(payload_30["summary"]["forecast_30_days_total"], 0)
        self.assertGreater(payload_30["summary"]["forecast_30_days_average"], 0)

        # 180-day horizon
        payload_180 = generate_forecast_payload(results, self.series, 180)
        self.assertEqual(len(payload_180["forecast"]), 180)
        self.assertGreater(payload_180["summary"]["forecast_6_months_total"], 0)
        self.assertGreater(payload_180["summary"]["forecast_6_months_average"], 0)

    def test_04_365_day_persistence_payload_validates_against_database_schema(self):
        """Simulate insertion into Supabase and verify all fields pass PostgreSQL constraints."""
        results = train_sarima_model(self.series)
        payload = generate_forecast_payload(results, self.series, 365)

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_inserted = {
            "id": "uuid-365-test",
            "user_id": "test_user_365",
            "forecast_days": 365,
            "created_at": "2026-10-23T12:00:00Z"
        }
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[mock_inserted])

        with patch("database.get_supabase_client", return_value=mock_client):
            record = insert_forecast_record(
                user_id="test_user_365",
                forecast_days=365,
                csv_source_url="https://test.supabase.co/storage/v1/object/public/service-data/test_user/sales.csv",
                series=self.series,
                forecast_payload=payload
            )
            self.assertEqual(record["id"], "uuid-365-test")

            # Inspect exact fields passed to table("sales_forecasts").insert(...)
            insert_call_args = mock_client.table.return_value.insert.call_args[0][0]
            
            # Numeric precision constraints
            MAX_NUMERIC_15_2 = 9_999_999_999_999.99
            self.assertLess(insert_call_args["forecast_total"], MAX_NUMERIC_15_2)
            self.assertGreaterEqual(insert_call_args["forecast_total"], 0)
            self.assertLess(insert_call_args["forecast_average"], MAX_NUMERIC_15_2)
            self.assertGreaterEqual(insert_call_args["forecast_average"], 0)
            self.assertLess(insert_call_args["expected_growth_percentage"], 10000.0)

            # Horizon constraint (chk_sales_forecast_horizon)
            self.assertIn(insert_call_args["forecast_days"], [30, 180, 365])
            
            # Date order constraints
            self.assertLessEqual(insert_call_args["forecast_start_date"], insert_call_args["forecast_end_date"])
            self.assertLessEqual(insert_call_args["historical_start_date"], insert_call_args["historical_end_date"])

    def test_05_forecast_date_windows_and_records_agreement(self):
        """Verify correct date windows and agreement between forecast_records and summary forecast_end_date."""
        results = train_sarima_model(self.series)

        expected_windows = {
            30: ("2026-10-24", "2026-11-22"),
            180: ("2026-10-24", "2027-04-21"),
            365: ("2026-10-24", "2027-10-23"),
        }

        for days, (expected_start, expected_end) in expected_windows.items():
            payload = generate_forecast_payload(results, self.series, days)
            summary = payload["summary"]
            records = payload["forecast"]

            # 1. Summary date window verification
            self.assertEqual(summary["forecast_start_date"], expected_start, f"Start date mismatch for {days} days")
            self.assertEqual(summary["forecast_end_date"], expected_end, f"End date mismatch for {days} days")

            # 2. Detailed forecast_records length & date window verification
            self.assertEqual(len(records), days, f"Record count mismatch for {days} days")
            self.assertEqual(records[0]["date"], expected_start, f"records[0] date mismatch for {days} days")
            self.assertEqual(records[-1]["date"], expected_end, f"records[-1] date mismatch for {days} days")

            # 3. Agreement between detailed records and summary
            self.assertEqual(summary["forecast_end_date"], records[-1]["date"], f"Summary end date must agree with records[-1] for {days} days")
            self.assertEqual(summary["forecast_start_date"], records[0]["date"], f"Summary start date must agree with records[0] for {days} days")

    def test_06_forecast_total_equals_sum_of_records_across_horizons(self):
        """Verify that 30d, 180d, and 365d forecast totals strictly equal the sum of their detailed records."""
        results = train_sarima_model(self.series)

        horizon_configs = [
            (30, "forecast_30_days_total", "forecast_30_days_average"),
            (180, "forecast_6_months_total", "forecast_6_months_average"),
            (365, "forecast_1_year_total", "forecast_1_year_average"),
        ]

        for days, total_key, avg_key in horizon_configs:
            payload = generate_forecast_payload(results, self.series, days)
            summary = payload["summary"]
            records = payload["forecast"]

            # Sum of detailed daily predicted_sales
            daily_sum = sum(item["predicted_sales"] for item in records)
            summary_total = summary[total_key]
            summary_avg = summary[avg_key]

            # 1. Total must equal the sum of daily predicted sales (allowing for cumulative rounding error of up to 0.005 per daily entry)
            max_rounding_delta = round(days * 0.005 + 0.05, 2)
            self.assertAlmostEqual(
                daily_sum,
                summary_total,
                delta=max_rounding_delta,
                msg=f"Sum of {days}-day forecast records ({daily_sum}) does not match summary total ({summary_total})"
            )

            # 2. Average must strictly equal summary_total / days
            expected_avg = round(summary_total / days, 2)
            self.assertAlmostEqual(
                summary_avg,
                expected_avg,
                delta=0.02,
                msg=f"Summary average ({summary_avg}) does not equal total / days ({expected_avg}) for {days} days"
            )

        # 3. Mathematical invariance check: verify behavior when tail daily sales decay/clip to 0.00
        # When days > 64 decay to 0, 180-day and 365-day cumulative totals remain identical,
        # but daily average strictly declines in inverse proportion to the horizon length.
        mock_results = MagicMock()
        mock_forecast = MagicMock()
        # Create 365 predictions: 50 positive values then all zeros
        mock_preds = pd.Series([50000.0] * 50 + [0.0] * 315, index=pd.date_range("2026-10-24", periods=365, freq="D"))
        mock_conf = pd.DataFrame({0: mock_preds * 0.8, 1: mock_preds * 1.2}, index=mock_preds.index)
        mock_forecast.predicted_mean = mock_preds
        mock_forecast.conf_int.return_value = mock_conf
        mock_results.get_forecast.return_value = mock_forecast

        payload_180_zero_tail = generate_forecast_payload(mock_results, self.series, 180)
        payload_365_zero_tail = generate_forecast_payload(mock_results, self.series, 365)

        tot_180 = payload_180_zero_tail["summary"]["forecast_6_months_total"]
        tot_365 = payload_365_zero_tail["summary"]["forecast_1_year_total"]
        avg_180 = payload_180_zero_tail["summary"]["forecast_6_months_average"]
        avg_365 = payload_365_zero_tail["summary"]["forecast_1_year_average"]

        # Totals match because days 181..365 contribute 0.00 additional revenue
        self.assertEqual(tot_180, tot_365, "Totals must match when tail daily values are 0.00")
        # But 365-day average is strictly lower than 180-day average: tot / 365 < tot / 180
        self.assertLess(avg_365, avg_180, "Daily average must decrease as horizon expands over zero tail")
        self.assertAlmostEqual(avg_365, round(tot_365 / 365, 2), delta=0.02)
        self.assertAlmostEqual(avg_180, round(tot_180 / 180, 2), delta=0.02)


if __name__ == "__main__":
    unittest.main()
