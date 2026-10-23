import io
import os
import time
import logging
import ipaddress
from urllib.parse import urlparse
from typing import Optional, Any, List, Dict

import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV Download Security Configuration
# ---------------------------------------------------------------------------
# Maximum CSV file size in bytes (default 10 MB). Configurable via MAX_CSV_BYTES or MAX_CSV_SIZE_MB.
# The limit is enforced by counting streamed bytes — Content-Length alone
# is never trusted because it can be absent, spoofed, or misleading.
_DEFAULT_MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB

def _get_max_csv_bytes() -> int:
    raw_bytes = os.getenv("MAX_CSV_BYTES", "").strip()
    if raw_bytes:
        try:
            val = int(raw_bytes)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    raw_mb = os.getenv("MAX_CSV_SIZE_MB", "").strip()
    if raw_mb:
        try:
            val = int(raw_mb)
            if val > 0:
                return val * 1024 * 1024
        except (ValueError, TypeError):
            pass

    return _DEFAULT_MAX_CSV_BYTES


# Configurable finite timeouts
_DEFAULT_CONNECT_TIMEOUT = 5.0        # seconds
_DEFAULT_READ_TIMEOUT = 15.0          # seconds
_DEFAULT_MAX_DOWNLOAD_SECONDS = 30.0  # seconds (stalled/slow download guard)

def _get_download_timeouts() -> tuple[float, float, float]:
    """Returns (connect_timeout, read_timeout, max_download_seconds) from environment or defaults."""
    connect_to = _DEFAULT_CONNECT_TIMEOUT
    read_to = _DEFAULT_READ_TIMEOUT
    max_to = _DEFAULT_MAX_DOWNLOAD_SECONDS

    raw_connect = os.getenv("CSV_CONNECT_TIMEOUT", "").strip()
    if raw_connect:
        try:
            val = float(raw_connect)
            if val > 0:
                connect_to = val
        except (ValueError, TypeError):
            pass

    raw_read = os.getenv("CSV_READ_TIMEOUT", "").strip() or os.getenv("CSV_DOWNLOAD_TIMEOUT", "").strip()
    if raw_read:
        try:
            val = float(raw_read)
            if val > 0:
                read_to = val
        except (ValueError, TypeError):
            pass

    raw_max = os.getenv("CSV_MAX_DOWNLOAD_TIME", "").strip()
    if raw_max:
        try:
            val = float(raw_max)
            if val > 0:
                max_to = val
        except (ValueError, TypeError):
            pass

    return connect_to, read_to, max_to


# Trusted storage hosts and path prefixes
_DEFAULT_SUPABASE_HOST = os.getenv("TRUSTED_SUPABASE_HOST", "dzwzsnnrfstsmtpueilo.supabase.co").strip().lower()
_DEFAULT_FIREBASE_HOST = os.getenv("TRUSTED_FIREBASE_STORAGE_HOST", "firebasestorage.googleapis.com").strip().lower()
_DEFAULT_FIREBASE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "final-business-os.firebasestorage.app").strip().lower()

_DEFAULT_STORAGE_PATH_PREFIX = os.getenv(
    "TRUSTED_STORAGE_PATH_PREFIX",
    "/storage/v1/object/public/service-data"
).strip()


class CsvDownloadError(ValueError):
    """Raised when the CSV download fails for a known, expected reason.
    The message is safe to surface directly to the API caller."""


def _is_internal_or_metadata_host(hostname: str) -> bool:
    """
    Checks if a hostname points to localhost, loopback, private RFC-1918 networks,
    link-local, or cloud instance metadata endpoints (SSRF protection).
    """
    if not hostname:
        return True

    clean_host = hostname.strip().lower()

    # Known forbidden hostnames and domain suffixes
    forbidden_hosts = {
        "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
        "metadata.google.internal", "metadata", "instance-data",
    }
    if clean_host in forbidden_hosts or clean_host.endswith(".localhost"):
        return True

    # Check IP addresses
    try:
        ip = ipaddress.ip_address(clean_host)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
        if str(ip) == "169.254.169.254":
            return True
    except ValueError:
        pass

    return False


def _validate_csv_url(url: str) -> None:
    """
    Validate that `url` is a safe, trusted storage URL before any network
    connection is opened. Rejects unsafe schemes, credentials embedded in URLs,
    requests to localhost, private/internal IP addresses, and cloud metadata endpoints.
    Preserves legitimate supported storage URLs (Supabase & Firebase Storage).

    Raises CsvDownloadError with a caller-safe message on any violation.
    """
    if not url or not isinstance(url, str):
        raise CsvDownloadError("Invalid URL: URL cannot be empty.")

    try:
        parsed = urlparse(url)
    except Exception:
        raise CsvDownloadError("Invalid URL format.")

    # 1. Scheme must be HTTPS only
    if (parsed.scheme or "").lower() != "https":
        raise CsvDownloadError(
            f"Rejected URL scheme '{parsed.scheme}'. Only HTTPS is permitted."
        )

    # 2. Port: None or 443 only
    try:
        port = parsed.port
    except ValueError:
        raise CsvDownloadError("Invalid URL: Malformed port.")
    if port and port != 443:
        raise CsvDownloadError("Invalid URL: Unexpected port.")

    # 3. No credentials embedded in the URL (userinfo component)
    if parsed.username or parsed.password:
        raise CsvDownloadError("URL must not contain embedded credentials.")

    # 4. No fragments
    if parsed.fragment:
        raise CsvDownloadError("URL must not contain fragments.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise CsvDownloadError("Invalid URL: Missing hostname.")

    # 5. Reject localhost, private/internal IP addresses, and cloud metadata endpoints
    if _is_internal_or_metadata_host(hostname):
        raise CsvDownloadError(
            f"Rejected URL host '{hostname}'. Requests to private, loopback, or metadata addresses are forbidden."
        )

    # 6. Host and path must match trusted Supabase or Firebase storage configuration
    trusted_sb_host = os.getenv("TRUSTED_SUPABASE_HOST", _DEFAULT_SUPABASE_HOST).strip().lower()
    trusted_fb_host = os.getenv("TRUSTED_FIREBASE_STORAGE_HOST", _DEFAULT_FIREBASE_HOST).strip().lower()

    if hostname == trusted_sb_host:
        prefix = _DEFAULT_STORAGE_PATH_PREFIX
        # Ensure path starts with expected storage prefix
        if not parsed.path.startswith(prefix):
            raise CsvDownloadError(
                f"Rejected URL path. Expected path prefix: '{prefix}'."
            )
    elif hostname == trusted_fb_host:
        expected_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", _DEFAULT_FIREBASE_BUCKET).strip().lower()
        if not parsed.path.startswith(f"/v0/b/{expected_bucket}/o/"):
            raise CsvDownloadError(
                "Rejected URL path. Path does not match trusted Firebase storage bucket."
            )
    else:
        raise CsvDownloadError(
            f"Rejected URL host '{hostname}'. Only '{trusted_sb_host}' is an accepted storage host."
        )


def download_and_parse_csv(url: str) -> pd.DataFrame:
    """
    Download a sales CSV from a trusted storage URL with the following security guarantees:

    - Pre-request validation: Scheme, hostname, port, and storage path are validated before
      any network connection is opened.
    - Redirect safety: allow_redirects=False unconditionally. Redirects are rejected and never followed.
    - Memory exhaustion protection: Streams chunk-by-chunk with a strict maximum byte limit.
      Aborts immediately if the byte limit is exceeded without buffering unbounded content.
    - Finite download timeouts: Enforces connect timeout, read timeout, and stalled/slow streaming watchdog.
    - Resource cleanup: On any failure, partial chunks and open sockets are immediately closed and cleared.
    - CSV validation: Parses the streamed body into a DataFrame, rejecting non-CSV or corrupted tabular data.

    Returns a pandas DataFrame on success.
    Raises CsvDownloadError with a caller-safe message on any failure.
    """
    # --- Step 1: Validate URL before touching the network ---
    _validate_csv_url(url)

    max_bytes = _get_max_csv_bytes()
    connect_to, read_to, max_download_seconds = _get_download_timeouts()

    # --- Step 2: Open a streaming connection, redirects disabled ---
    try:
        response = requests.get(
            url,
            timeout=(connect_to, read_to),
            stream=True,
            allow_redirects=False,  # reject ALL redirects unconditionally
        )
    except requests.exceptions.Timeout:
        raise CsvDownloadError("CSV download timed out. The storage server did not respond in time.")
    except requests.exceptions.ConnectionError as e:
        raise CsvDownloadError(f"CSV download connection error: {e}")
    except requests.exceptions.RequestException as e:
        raise CsvDownloadError(f"CSV download failed: {e}")

    # --- Step 3: Reject HTTP redirects explicitly ---
    if 300 <= response.status_code < 400:
        location = response.headers.get("Location", "(none)")
        response.close()
        raise CsvDownloadError(
            f"CSV download rejected: server returned a {response.status_code} redirect "
            f"to '{location}'. Redirects to untrusted hosts are not permitted."
        )

    # --- Step 4: Reject non-200 responses ---
    if response.status_code != 200:
        response.close()
        raise CsvDownloadError(
            f"CSV download failed: server returned HTTP {response.status_code}."
        )

    # --- Step 5: Early size hint check via Content-Length (not trusted as a hard limit) ---
    content_length_header = response.headers.get("Content-Length")
    if content_length_header is not None:
        try:
            declared_length = int(content_length_header)
            if declared_length > max_bytes:
                response.close()
                raise CsvDownloadError(
                    f"CSV file is too large. "
                    f"Declared size {declared_length:,} bytes exceeds the "
                    f"{max_bytes:,}-byte limit ({max_bytes // (1024*1024)} MB)."
                )
        except ValueError:
            pass

    # --- Step 6: Stream and accumulate with hard byte cap and slowloris watchdog ---
    chunks: list[bytes] = []
    total_bytes = 0
    chunk_size = 65_536  # 64 KiB per chunk
    start_time = time.monotonic()

    try:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            # Protect against stalled or slow downloads
            if time.monotonic() - start_time > max_download_seconds:
                response.close()
                raise CsvDownloadError(
                    f"CSV download timed out. The storage server did not respond in time."
                )

            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                response.close()
                raise CsvDownloadError(
                    f"CSV file is too large. "
                    f"Download aborted after {total_bytes:,} bytes "
                    f"(limit: {max_bytes:,} bytes / {max_bytes // (1024*1024)} MB)."
                )
            chunks.append(chunk)
    except CsvDownloadError:
        chunks.clear()
        raise
    except requests.exceptions.Timeout:
        chunks.clear()
        raise CsvDownloadError("CSV download timed out. The storage server did not respond in time.")
    except requests.exceptions.ChunkedEncodingError as e:
        chunks.clear()
        raise CsvDownloadError(f"CSV download interrupted (connection dropped mid-stream): {e}")
    except Exception as e:
        chunks.clear()
        raise CsvDownloadError(f"Unexpected error during CSV streaming: {e}")
    finally:
        response.close()

    if total_bytes == 0:
        chunks.clear()
        raise CsvDownloadError("CSV download returned an empty response.")

    logger.info(f"CSV downloaded: {total_bytes:,} bytes (limit: {max_bytes:,} bytes).")

    # --- Step 7: Decode and parse into a DataFrame ---
    try:
        raw_bytes = b"".join(chunks)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")
        df = pd.read_csv(io.StringIO(text))
        if df.empty or len(df.columns) < 2:
            raise CsvDownloadError("Invalid CSV format: file does not contain valid tabular sales data.")
    except CsvDownloadError:
        raise
    except Exception as e:
        raise CsvDownloadError(f"Invalid CSV format: {e}")
    finally:
        chunks.clear()

    return df


# ---------------------------------------------------------------------------
# Sales Data Merging & Data-Source Policy
# ---------------------------------------------------------------------------

try:
    from database import DailySalesConflictError
except ImportError:
    class DailySalesConflictError(ValueError):
        pass


def merge_sales_sources(
    df_csv: Optional[pd.DataFrame],
    manual_entries: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Merges CSV sales data with authenticated manual daily sales records according to
    SMART BPI Data-Source Policy:
    
    1. CSV Preservation: Existing CSV uploads and ownership validation remain the base historical dataset.
    2. Deduplication & Conflict Detection:
       - If a date exists in both CSV and manual entries:
         - If revenue matches within 0.01 tolerance, deduplicate (never count twice).
         - If revenue differs, raise DailySalesConflictError to make the conflict visible
           to the user instead of silently choosing or overwriting values.
    3. Seamless Merge:
       - Non-overlapping manual daily entries (e.g. recent dates logged after the CSV cutoff)
         are merged into the historical dataset.
    4. Data Continuity:
       - Returns a consolidated DataFrame with 'sale_date' (YYYY-MM-DD string) and 'daily_revenue' (float).
    """
    has_csv = df_csv is not None and not df_csv.empty
    has_manual = bool(manual_entries)

    if not has_csv and not has_manual:
        raise ValueError("No sales data found. Please upload a sales CSV or enter daily sales.")

    csv_records: dict[str, float] = {}
    if has_csv:
        if "sale_date" not in df_csv.columns or "daily_revenue" not in df_csv.columns:
            raise ValueError("CSV must contain 'sale_date' and 'daily_revenue' columns")
        
        # Parse and normalize CSV dates
        try:
            temp_dates = pd.to_datetime(df_csv["sale_date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            raise ValueError("Invalid date format in CSV 'sale_date' column")

        try:
            temp_revs = pd.to_numeric(df_csv["daily_revenue"])
        except Exception:
            raise ValueError("Non-numeric values found in CSV 'daily_revenue'")

        for d_str, r_val in zip(temp_dates, temp_revs):
            if pd.isna(r_val) or r_val < 0:
                raise ValueError("Invalid or negative sales values detected in CSV")
            csv_records[d_str] = round(float(r_val), 2)

    manual_records: dict[str, float] = {}
    if has_manual:
        for entry in manual_entries:
            raw_date = entry.get("sale_date")
            raw_rev = entry.get("daily_revenue")
            if not raw_date or raw_rev is None:
                continue
            d_str = str(raw_date)[:10]
            try:
                r_val = float(raw_rev)
            except (ValueError, TypeError):
                continue
            manual_records[d_str] = round(r_val, 2)

    # If only CSV is present
    if has_csv and not manual_records:
        rows = [{"sale_date": d, "daily_revenue": r} for d, r in sorted(csv_records.items())]
        return pd.DataFrame(rows)

    # If only manual entries are present
    if not has_csv and manual_records:
        rows = [{"sale_date": d, "daily_revenue": r} for d, r in sorted(manual_records.items())]
        return pd.DataFrame(rows)

    # Both CSV and manual records exist: check for conflicts and merge
    common_dates = sorted(set(csv_records.keys()) & set(manual_records.keys()))
    for date_str in common_dates:
        csv_val = csv_records[date_str]
        man_val = manual_records[date_str]
        if abs(csv_val - man_val) > 0.01:
            raise DailySalesConflictError(
                f"Sales data conflict on date '{date_str}': CSV has ₹{csv_val:,.2f} "
                f"while manual entry has ₹{man_val:,.2f}. Please resolve the discrepancy "
                f"in your daily sales or upload an updated CSV."
            )

    # Merge: CSV records + non-overlapping manual records
    combined = dict(csv_records)
    for d_str, man_val in manual_records.items():
        if d_str not in combined:
            combined[d_str] = man_val

    merged_rows = [{"sale_date": d, "daily_revenue": combined[d]} for d in sorted(combined.keys())]
    return pd.DataFrame(merged_rows)


# ---------------------------------------------------------------------------
# Sales Data Validation & Forecasting (unchanged)
# ---------------------------------------------------------------------------

def validate_and_prepare_sales_data(df: pd.DataFrame) -> pd.Series:
    # 1. Require columns
    if "sale_date" not in df.columns or "daily_revenue" not in df.columns:
        raise ValueError("CSV must contain 'sale_date' and 'daily_revenue' columns")

    # 2. Validate dates
    try:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
    except Exception:
        raise ValueError("Invalid date format in 'sale_date' column")
    
    if df['sale_date'].isnull().any():
        raise ValueError("Missing dates detected in CSV")

    # 3. Validate duplicates
    if df['sale_date'].duplicated().any():
        raise ValueError("Duplicate dates detected in CSV")

    # 4. Validate numeric sales
    try:
        df['daily_revenue'] = pd.to_numeric(df['daily_revenue'])
    except Exception:
        raise ValueError("Non-numeric values found in 'daily_revenue'")

    # 5. Validate negative sales
    if (df['daily_revenue'] < 0).any():
        raise ValueError("Negative sales values are not permitted")

    # 6. Prepare Time Series (sort and set index)
    df = df.sort_values('sale_date').set_index('sale_date')
    
    # 7. Check historical data volume
    if len(df) < 60:
        raise ValueError("Insufficient historical data: minimum 60 days required")

    # 8. Check for missing calendar dates implicitly
    # If we reindex to 'D' and missing values appear, reject.
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    if len(full_idx) != len(df):
        raise ValueError("Missing calendar dates detected in your sales CSV. Continuous daily data is required.")

    return df['daily_revenue']


# Stable identifier for the damped trend forecasting model configuration
FORECAST_MODEL_VERSION = "damped_ets_v3"


class StateSpaceForecastResult:
    """Standardized forecast result exposing predicted_mean and conf_int()."""
    def __init__(self, predicted_mean: pd.Series, conf_int_df: pd.DataFrame):
        self.predicted_mean = predicted_mean
        self._conf_int = conf_int_df

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        return self._conf_int


class ForecastModelResult:
    """
    Standardized forecast result wrapper compatible with statsmodels statespace interface.
    Provides get_forecast(steps), conf_int(alpha), and params attributes.
    """
    def __init__(self, model_fit, series: pd.Series, model_type: str = "damped_ets"):
        self._fit = model_fit
        self._series = series
        self.model_type = model_type
        self.params = dict(getattr(model_fit, "params", {}) or {})
        # Expose damping factor explicitly
        if "damping_trend" in self.params:
            self.params["damping_factor"] = self.params["damping_trend"]

    def get_forecast(self, steps: int = 365) -> StateSpaceForecastResult:
        preds = self._fit.forecast(steps)
        start_date = self._series.index[-1] + pd.Timedelta(days=1)
        pred_idx = pd.date_range(start=start_date, periods=steps, freq="D")
        preds.index = pred_idx

        # Calculate prediction intervals based on residual variance
        residuals = getattr(self._fit, "resid", None)
        if residuals is not None and len(residuals) > 1:
            sigma = float(np.std(residuals, ddof=1))
        else:
            sigma = float(self._series.std() or 1.0)

        alpha = float(self.params.get("smoothing_level", 0.3) or 0.3)
        phi = float(self.params.get("damping_trend", 0.9) or 0.9)
        # Bounded damping factor (0.1 <= phi <= 0.99)
        phi = min(max(phi, 0.1), 0.99)

        h = np.arange(1, steps + 1)
        se_h = sigma * np.sqrt(1.0 + (alpha ** 2) * np.cumsum(phi ** (2 * (h - 1))))

        z = 1.96  # 95% confidence interval
        lower = preds - z * se_h
        upper = preds + z * se_h

        # Non-negative lower bound guardrail
        lower = lower.clip(lower=0)

        ci_df = pd.DataFrame({0: lower, 1: upper}, index=pred_idx)
        return StateSpaceForecastResult(preds, ci_df)


def train_forecasting_model(series: pd.Series) -> ForecastModelResult:
    """
    Fits a robust forecasting model that preserves weekly seasonality (when supported)
    and uses a damped additive trend to prevent explosive or negative runaway projections.

    Methodology:
    1. Primary Model: Holt-Winters Exponential Smoothing with Additive Damped Trend
       and Additive Weekly Seasonality (s=7).
       - Evaluated via chronological holdout validation against SARIMAX and non-seasonal baselines.
       - Solves the zero-tail extrapolation problem by damping the trend slope (phi < 1),
         ensuring projections smoothly stabilize at a realistic run-rate rather than plunging
         into negative numbers when the dataset ends with a negative shock.
    2. Fallback 1: If weekly seasonality cannot be fit (e.g. fewer than 14 observations or
       seasonal singularity), fits a Damped Additive Trend model without seasonality.
    3. Fallback 2: If damped trend fitting fails (e.g. zero variance / constant data),
       fits Simple Exponential Smoothing (SES / level only).
    """
    has_enough_for_seasonality = len(series) >= 14

    if has_enough_for_seasonality:
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated"
            )
            fit_result = model.fit()
            return ForecastModelResult(fit_result, series, model_type="damped_ets_seasonal")
        except Exception as e:
            logger.warning(f"Seasonal damped ETS fit failed: {e}. Retrying without seasonality...")

    # Fallback 1: Damped trend without seasonality
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated"
        )
        fit_result = model.fit()
        return ForecastModelResult(fit_result, series, model_type="damped_ets_nonseasonal")
    except Exception as e:
        logger.warning(f"Non-seasonal damped ETS fit failed: {e}. Retrying with simple exponential smoothing...")

    # Fallback 2: Simple Exponential Smoothing (level only)
    try:
        model = ExponentialSmoothing(
            series,
            trend=None,
            seasonal=None,
            initialization_method="estimated"
        )
        fit_result = model.fit()
        return ForecastModelResult(fit_result, series, model_type="ses_level")
    except Exception as e:
        raise ValueError(f"All model training strategies failed: {str(e)}")


# Retain backward-compatible alias for existing callers and tests
train_sarima_model = train_forecasting_model


def generate_forecast_payload(results, series, forecast_days: int):
    try:
        # Internally generate 365 days always
        forecast_result = results.get_forecast(steps=365)
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)
    except Exception as e:
        raise ValueError(f"Prediction generation failed: {str(e)}")
        
    # Prevent negative values in predictions and bounds
    predictions = predictions.clip(lower=0)
    conf_int.iloc[:, 0] = conf_int.iloc[:, 0].clip(lower=0)
    conf_int.iloc[:, 1] = conf_int.iloc[:, 1].clip(lower=0)

    historical_average = float(series.mean())
    historical_total = float(series.sum())

    sum_30 = float(predictions.iloc[:30].sum())
    sum_180 = float(predictions.iloc[:180].sum())
    sum_365 = float(predictions.sum())

    avg_30 = float(predictions.iloc[:30].mean())
    avg_180 = float(predictions.iloc[:180].mean())
    avg_365 = float(predictions.mean())

    if historical_average > 0:
        growth_pct = ((avg_30 - historical_average) / historical_average) * 100
    else:
        growth_pct = 0.0
        
    start_date = predictions.index[0].strftime("%Y-%m-%d")
    end_date = predictions.index[forecast_days - 1].strftime("%Y-%m-%d")

    summary = {
        "historical_total_sales": round(historical_total, 2),
        "historical_average_daily_sales": round(historical_average, 2),
        "forecast_30_days_total": round(sum_30, 2),
        "forecast_30_days_average": round(avg_30, 2),
        "forecast_6_months_total": round(sum_180, 2),
        "forecast_6_months_average": round(avg_180, 2),
        "forecast_1_year_total": round(sum_365, 2),
        "forecast_1_year_average": round(avg_365, 2),
        "expected_growth_percentage": round(growth_pct, 2),
        "forecast_start_date": start_date,
        "forecast_end_date": end_date
    }

    records = []
    # Only return requested number of days
    for i in range(forecast_days):
        pred = float(predictions.iloc[i])
        lb = float(conf_int.iloc[i, 0])
        ub = float(conf_int.iloc[i, 1])
        
        # Ensure logical bounds
        if lb > pred: lb = pred
        if ub < pred: ub = pred

        records.append({
            "date": predictions.index[i].strftime("%Y-%m-%d"),
            "predicted_sales": round(pred, 2),
            "lower_bound": round(lb, 2),
            "upper_bound": round(ub, 2)
        })

    return {
        "summary": summary,
        "forecast": records
    }
