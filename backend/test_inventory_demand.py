import unittest
from fastapi.testclient import TestClient

from main import app
from Inventory_demand_prediction.router import reset_inventory_data, load_verified_dataset


class TestInventoryDemandPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset inventory state before each test
        reset_inventory_data()

    def tearDown(self):
        # Reset after each test
        reset_inventory_data()

    def test_01_model_info_returns_trained_model_metadata(self):
        response = self.client.get("/inventory/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("feature_cols", data)
        self.assertIn("model_type", data)
        self.assertIn("forecast_horizon_days", data)
        self.assertEqual(data["forecast_horizon_days"], 1)

        expected_features = [
            "product_id",
            "store_id",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "day_of_week",
            "month",
            "day_of_year",
        ]
        self.assertEqual(data["feature_cols"], expected_features)
        self.assertIn("raj_business_demand_model.joblib", data.get("model_file", ""))

    def test_02_predict_endpoint_validates_required_features(self):
        # Missing lag_1 and day_of_week
        incomplete_features = {
            "product_id": "1001",
            "store_id": "1",
            "lag_7": 9.0,
            "lag_14": 7.0,
            "lag_28": 5.0,
            "rolling_mean_7": 11.0,
            "month": 12,
            "day_of_year": 365,
        }
        response = self.client.post("/inventory/predict", json={"features": incomplete_features})
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("missing_features", err["detail"])

    def test_03_predict_endpoint_succeeds_with_exact_features(self):
        features = {
            "product_id": "1001",
            "store_id": "1",
            "lag_1": 8.0,
            "lag_7": 9.0,
            "lag_14": 7.0,
            "lag_28": 5.0,
            "rolling_mean_7": 11.0,
            "day_of_week": 2,
            "month": 12,
            "day_of_year": 365,
        }
        response = self.client.post("/inventory/predict", json={"features": features})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_demand", data)
        self.assertGreaterEqual(data["predicted_demand"], 0.0)
        # Expected prediction from verified model is approximately 9.8
        self.assertAlmostEqual(data["predicted_demand"], 9.8, delta=0.5)

    def test_04_catalog_empty_state_when_no_data_loaded(self):
        response = self.client.get("/inventory/catalog")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["has_data"])
        self.assertEqual(len(data["products"]), 0)
        self.assertEqual(data["summary"]["total_skus"], 0)
        self.assertEqual(data["summary"]["total_valuation"], 0.0)
        self.assertTrue(data["dataset_available_to_load"])

    def test_05_load_verified_dataset_and_catalog_aggregation(self):
        load_res = self.client.post("/inventory/load-dataset")
        self.assertEqual(load_res.status_code, 200)
        load_data = load_res.json()
        self.assertTrue(load_data["success"])
        self.assertEqual(load_data["products_count"], 20)

        # Now query catalog
        cat_res = self.client.get("/inventory/catalog?store_id=1")
        self.assertEqual(cat_res.status_code, 200)
        cat_data = cat_res.json()

        self.assertTrue(cat_data["has_data"])
        self.assertEqual(cat_data["summary"]["total_skus"], 20)
        self.assertGreater(cat_data["summary"]["total_valuation"], 0)
        self.assertGreater(len(cat_data["products"]), 0)
        self.assertGreater(len(cat_data["suppliers"]), 0)

        # Check product fields
        p1 = cat_data["products"][0]
        self.assertIn("sku", p1)
        self.assertIn("current_stock", p1)
        self.assertIn("reorder_threshold", p1)
        self.assertIn("unit_price", p1)
        self.assertIn("supplier_lead_time_days", p1)

    def test_06_predict_product_calculates_actual_lags_and_replenishment(self):
        load_res = self.client.post("/inventory/load-dataset")
        self.assertEqual(load_res.status_code, 200)

        # Predict for Product 1001 Store 1
        req = {
            "product_id": "1001",
            "store_id": "1",
            "target_date": "2025-12-31"
        }
        res = self.client.post("/inventory/predict-product", json=req)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["product_id"], "1001")
        self.assertEqual(data["target_date"], "2025-12-31")
        self.assertAlmostEqual(data["predicted_demand"], 9.8, delta=0.5)

        # Verify features used
        features = data["features_used"]
        self.assertEqual(features["lag_1"], 8.0)
        self.assertEqual(features["lag_7"], 9.0)
        self.assertEqual(features["lag_14"], 7.0)
        self.assertEqual(features["lag_28"], 5.0)
        self.assertEqual(features["rolling_mean_7"], 11.0)
        self.assertEqual(features["day_of_week"], 2)
        self.assertEqual(features["month"], 12)
        self.assertEqual(features["day_of_year"], 365)

        # Verify evidence-based replenishment advice
        replenishment = data["replenishment"]
        self.assertIn("status", replenishment)
        self.assertIn("recommended_reorder_qty", replenishment)
        self.assertIn("reason", replenishment)
        self.assertIn("days_until_stockout", replenishment)


if __name__ == "__main__":
    unittest.main()
