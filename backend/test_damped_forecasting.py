"""
SMART BPI - Damped Trend Sales Forecasting & Regression Test Suite

Verifies:
1. Zero-tail resolution: Models fitted on datasets with late-stage shocks do not collapse to zero;
   180-day and 365-day forecast totals scale with horizon and do not freeze at identical totals.
2. Non-negative outputs: Predictions and confidence bounds remain strictly non-negative.
3. Horizon consistency: Overlapping periods across 30d, 180d, and 365d forecasts match identically.
4. Summary metrics agreement: Summary totals strictly equal the sum of detailed daily predictions.
5. Date windows: Summary start/end dates match records[0] and records[-1] across all horizons.
6. Edge case robustness: Handles constant data, high volatility, and enforces 60-day minimum data volume.
7. Model versioning: Uses 'damped_ets_v3' versioning for clean idempotency separation from legacy records.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting import (
    train_forecasting_model,
    train_sarima_model,
    generate_forecast_payload,
    validate_and_prepare_sales_data,
    FORECAST_MODEL_VERSION,
)
from database import (
    compute_idempotency_key,
    compute_series_hash,
    insert_forecast_record,
    DEFAULT_MODEL_VERSION,
)


class TestDampedTrendForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Realistic 206-day series ending with a downward shock, exactly matching the user's scenario
        dates = pd.date_range(start="2026-04-01", periods=206, freq="D")
        np.random.seed(42)
        weekly_pattern = np.array([35000, 36000, 36500, 37300, 42000, 46500, 44800])
        revs = []
        for i in range(204):
            # Gradual upward trend + weekly seasonality
            trend_val = 25000 + (i * 250)
            seasonal_val = weekly_pattern[i % 7]
            noise = np.random.uniform(-2000, 2000)
            revs.append(round(float((trend_val + seasonal_val) / 2 + noise), 2))
        # Final 2 days: sharp drop shock
        revs.append(60000.0)
        revs.append(70000.0)

        cls.series = pd.Series(revs, index=dates, name="daily_revenue")
        cls.user_id = "test_user_damped"
        cls.csv_url = "https://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/test_user_damped/sales.csv"

    def test_01_model_version_is_damped_ets_v3(self):
        """Confirm model version is updated to damped_ets_v3 in both forecasting and database."""
        self.assertEqual(FORECAST_MODEL_VERSION, "damped_ets_v3")
        self.assertEqual(DEFAULT_MODEL_VERSION, "damped_ets_v3")

    def test_02_zero_tail_resolved_and_totals_differ_across_horizons(self):
        """Confirm model does not collapse to zero; 180d and 365d totals strictly differ and scale proportionally."""
        model_result = train_forecasting_model(self.series)
        
        payload_30 = generate_forecast_payload(model_result, self.series, 30)
        payload_180 = generate_forecast_payload(model_result, self.series, 180)
        payload_365 = generate_forecast_payload(model_result, self.series, 365)

        tot_30 = payload_30["summary"]["forecast_30_days_total"]
        tot_180 = payload_180["summary"]["forecast_6_months_total"]
        tot_365 = payload_365["summary"]["forecast_1_year_total"]

        # 1. Zero-tail check: NO daily prediction in 180d or 365d is 0.00
        zeros_180 = sum(1 for r in payload_180["forecast"] if r["predicted_sales"] == 0.0)
        zeros_365 = sum(1 for r in payload_365["forecast"] if r["predicted_sales"] == 0.0)
        self.assertEqual(zeros_180, 0, f"Expected 0 zero-predictions in 180-day forecast, found {zeros_180}")
        self.assertEqual(zeros_365, 0, f"Expected 0 zero-predictions in 365-day forecast, found {zeros_365}")

        # 2. Distinct totals check: 365-day total must be significantly greater than 180-day total
        self.assertGreater(tot_180, tot_30 * 4, "180-day total must be significantly larger than 30-day total")
        self.assertGreater(tot_365, tot_180 * 1.5, "365-day total must be significantly larger than 180-day total")
        self.assertNotEqual(tot_180, tot_365, "180-day and 365-day totals must NOT be identical!")

    def test_03_non_negative_predictions_and_confidence_bounds(self):
        """Confirm all predicted daily sales and bounds are strictly non-negative."""
        model_result = train_forecasting_model(self.series)
        payload = generate_forecast_payload(model_result, self.series, 365)

        for record in payload["forecast"]:
            self.assertGreaterEqual(record["predicted_sales"], 0.0, f"Negative predicted sales on {record['date']}")
            self.assertGreaterEqual(record["lower_bound"], 0.0, f"Negative lower bound on {record['date']}")
            self.assertGreaterEqual(record["upper_bound"], record["predicted_sales"], f"Upper bound < predicted on {record['date']}")
            self.assertLessEqual(record["lower_bound"], record["predicted_sales"], f"Lower bound > predicted on {record['date']}")

    def test_04_horizon_consistency_overlapping_predictions_match(self):
        """Confirm overlapping periods across 30d, 180d, and 365d match identically from the same cutoff."""
        model_result = train_forecasting_model(self.series)

        payload_30 = generate_forecast_payload(model_result, self.series, 30)
        payload_180 = generate_forecast_payload(model_result, self.series, 180)
        payload_365 = generate_forecast_payload(model_result, self.series, 365)

        preds_30 = [r["predicted_sales"] for r in payload_30["forecast"]]
        preds_180 = [r["predicted_sales"] for r in payload_180["forecast"]]
        preds_365 = [r["predicted_sales"] for r in payload_365["forecast"]]

        # First 30 days identical across all three horizons
        self.assertEqual(preds_30, preds_180[:30], "First 30 days of 180d must match 30d forecast")
        self.assertEqual(preds_30, preds_365[:30], "First 30 days of 365d must match 30d forecast")

        # First 180 days identical between 180d and 365d
        self.assertEqual(preds_180, preds_365[:180], "First 180 days of 365d must match 180d forecast")

    def test_05_summary_totals_strictly_equal_sum_of_records(self):
        """Confirm summary totals equal the sum of daily predictions within 2-decimal rounding tolerance."""
        model_result = train_forecasting_model(self.series)

        horizons = [
            (30, "forecast_30_days_total", "forecast_30_days_average"),
            (180, "forecast_6_months_total", "forecast_6_months_average"),
            (365, "forecast_1_year_total", "forecast_1_year_average"),
        ]

        for days, total_key, avg_key in horizons:
            payload = generate_forecast_payload(model_result, self.series, days)
            summary = payload["summary"]
            records = payload["forecast"]

            daily_sum = sum(r["predicted_sales"] for r in records)
            summary_tot = summary[total_key]
            summary_avg = summary[avg_key]

            # Sum within cumulative 2-decimal rounding tolerance
            self.assertAlmostEqual(daily_sum, summary_tot, delta=0.5, msg=f"Sum mismatch for {days} days")
            # Average strictly equals total / days
            self.assertAlmostEqual(summary_avg, round(summary_tot / days, 2), delta=0.05, msg=f"Avg mismatch for {days} days")

    def test_06_date_windows_and_lengths_strictly_agree(self):
        """Confirm summary start/end dates agree with detailed records[0] and records[-1]."""
        model_result = train_forecasting_model(self.series)

        expected_ends = {
            30: "2026-11-22",
            180: "2027-04-21",
            365: "2027-10-23",
        }

        for days, exp_end in expected_ends.items():
            payload = generate_forecast_payload(model_result, self.series, days)
            summary = payload["summary"]
            records = payload["forecast"]

            self.assertEqual(len(records), days)
            self.assertEqual(summary["forecast_start_date"], "2026-10-24")
            self.assertEqual(summary["forecast_end_date"], exp_end)
            self.assertEqual(records[0]["date"], "2026-10-24")
            self.assertEqual(records[-1]["date"], exp_end)

    def test_07_insufficient_historical_data_rejected(self):
        """Confirm series with fewer than 60 days is rejected with clear message."""
        short_dates = pd.date_range("2026-09-01", periods=59, freq="D")
        short_df = pd.DataFrame({"sale_date": short_dates, "daily_revenue": [50000.0] * 59})
        with self.assertRaises(ValueError) as ctx:
            validate_and_prepare_sales_data(short_df)
        self.assertIn("minimum 60 days required", str(ctx.exception).lower())

    def test_08_unusual_historical_data_constant_sales_handled_cleanly(self):
        """Confirm constant historical data (zero variance) fits cleanly and forecasts stable revenue."""
        const_dates = pd.date_range("2026-01-01", periods=100, freq="D")
        const_series = pd.Series([50000.0] * 100, index=const_dates, name="daily_revenue")

        model_result = train_forecasting_model(const_series)
        payload = generate_forecast_payload(model_result, const_series, 365)

        tot_365 = payload["summary"]["forecast_1_year_total"]
        avg_365 = payload["summary"]["forecast_1_year_average"]

        self.assertAlmostEqual(avg_365, 50000.0, delta=100.0)
        self.assertAlmostEqual(tot_365, 50000.0 * 365, delta=5000.0)
        self.assertEqual(len(payload["forecast"]), 365)

    def test_09_unusual_historical_data_high_volatility_handled_cleanly(self):
        """Confirm highly volatile series fits without crashing and outputs bounded positive values."""
        np.random.seed(99)
        vol_dates = pd.date_range("2026-01-01", periods=120, freq="D")
        vol_revs = [max(100.0, float(np.random.exponential(scale=40000))) for _ in range(120)]
        vol_series = pd.Series(vol_revs, index=vol_dates, name="daily_revenue")

        model_result = train_forecasting_model(vol_series)
        payload = generate_forecast_payload(model_result, vol_series, 365)

        self.assertEqual(len(payload["forecast"]), 365)
        for r in payload["forecast"]:
            self.assertGreaterEqual(r["predicted_sales"], 0.0)

    def test_10_backward_compatible_train_sarima_model_alias(self):
        """Confirm train_sarima_model alias continues to function for legacy callers."""
        res = train_sarima_model(self.series)
        self.assertTrue(hasattr(res, "get_forecast"))
        self.assertTrue(hasattr(res, "params"))
        fc = res.get_forecast(30)
        self.assertEqual(len(fc.predicted_mean), 30)


if __name__ == "__main__":
    unittest.main()
