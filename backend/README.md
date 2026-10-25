# SMART BPI Forecasting Backend

This is the FastAPI ML backend for the SMART BPI application. It provides time-series forecasting using SARIMA, directly pulling user sales data from Supabase storage based on their Firestore configuration.

## 1. Backend architecture
- **FastAPI**: Provides the HTTP REST API.
- **Statsmodels**: Powers the SARIMA forecasting algorithm `(1,1,1)(1,1,1,7)`.
- **Pandas**: Used for data validation, cleaning, and preparation.

## 2. Installation
Requires Python 3.10–3.13. The saved inventory model was created with
scikit-learn 1.6.1, which is pinned in `requirements.txt` so the serialized
pipeline can be loaded with a compatible version.

```bash
cd pixel-perfect/backend
python -m venv venv
```

Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Environment Variables
Copy `.env.example` to `.env` and fill in your Supabase and Firebase credentials if you need to implement strict server-side validation. Currently, the endpoint expects a public Supabase URL passed from the authenticated frontend.

**SECURITY NOTE:** For local troubleshooting *only*, you can set `ALLOW_UNVERIFIED_DEV_MODE=True` in your `.env`. This bypasses Firebase Authentication. **NEVER enable this in production.**

## 4. Running FastAPI
```bash
uvicorn main:app --reload --port 8000
```

## 5. Inventory Demand Prediction & Catalog Endpoints

The backend integrates with `raj_business_demand_model.joblib` (Extra Trees Regressor) and provides full inventory endpoints:

- **GET `/inventory/model-info`**: Inspect model architecture, training cutoff date, required 10 features, and test metrics ($R^2 = 0.688$, $\text{MAE} = 2.10$).
- **GET `/inventory/catalog?store_id=1`**: Returns catalog products with current stock, velocity, suppliers, stock movements, and valuation. Shows honest empty states when no data is loaded.
- **POST `/inventory/load-dataset`**: Loads the verified 73,000-record dataset across 5 stores and 20 products (`2024-01-01` to `2025-12-30`).
- **POST `/inventory/predict-product`**: Accepts `{"product_id": "1001", "store_id": "1"}`. Computes actual lags (`lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `day_of_week`, `month`, `day_of_year`), executes the pipeline model, and generates replenishment recommendations.
- **POST `/inventory/upload-csv`**: Upload custom multi-store sales/inventory CSV.
- **DELETE `/inventory/reset`**: Resets active catalog back to the honest empty state.

Interactive OpenAPI docs are available at `http://localhost:8000/docs`.

## 6. Frontend Environment Variable
Make sure your frontend React app (`pixel-perfect/.env`) contains:
```env
VITE_FORECAST_API_URL=http://localhost:8000
```

## 7. Running Tests
Run the backend test suite:
```bash
pytest test_inventory_demand.py -v
```


## 8. Expected CSV Format
```csv
sale_date,daily_revenue
2023-01-01,25000
2023-01-02,27000
```
Must contain at least 60 consecutive days of non-negative revenue.
