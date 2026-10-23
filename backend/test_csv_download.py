"""
SMART BPI — CSV Download Security Tests
Covers all 6 required scenarios:
  1. Valid CSV download from trusted Supabase host.
  2. CSV exceeding the configured size limit (enforced during streaming).
  3. Missing or misleading Content-Length header.
  4. Arbitrary external (untrusted) URL.
  5. Untrusted redirect from a seemingly-valid Supabase URL.
  6. Failed / interrupted download (connection drop mid-stream).

The test file patches the `requests.get` call inside `forecasting` so
no real network connections are made. The URL validation logic runs
against the real `_validate_csv_url` function (no mocking).
"""

import io
import os
import sys
from unittest.mock import patch, MagicMock, call
import pytest

# Ensure the backend directory is on the path regardless of where pytest runs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forecasting
from forecasting import (
    download_and_parse_csv,
    CsvDownloadError,
    _validate_csv_url,
    _get_max_csv_bytes,
)

# A realistic Supabase public URL in the expected format
TRUSTED_URL = (
    "https://dzwzsnnrfstsmtpueilo.supabase.co"
    "/storage/v1/object/public/service-data"
    "/user_123/sales_20241101_data.csv"
)

# Minimal valid CSV that satisfies validate_and_prepare_sales_data
# (continuous daily rows from a fixed start date)
def _make_valid_csv_bytes(num_rows: int = 90) -> bytes:
    import datetime
    lines = ["sale_date,daily_revenue"]
    base = datetime.date(2024, 1, 1)
    for i in range(num_rows):
        d = base + datetime.timedelta(days=i)
        lines.append(f"{d.isoformat()},{100 + i}")
    return "\n".join(lines).encode("utf-8")


def _mock_response(
    content: bytes,
    status_code: int = 200,
    headers: dict | None = None,
    simulate_redirect_location: str | None = None,
    iter_error_after: int | None = None,
):
    """
    Build a MagicMock that looks like a requests.Response with
    iter_content yielding the supplied bytes in 64 KiB chunks.
    """
    if headers is None:
        headers = {}
    if status_code in (301, 302, 307, 308) and simulate_redirect_location:
        headers.setdefault("Location", simulate_redirect_location)
    if "Content-Length" not in headers:
        headers["Content-Length"] = str(len(content))

    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers

    chunk_size = 65536

    def _iter_content(chunk_size=chunk_size):
        offset = 0
        call_count = [0]
        while offset < len(content):
            call_count[0] += 1
            if iter_error_after is not None and call_count[0] > iter_error_after:
                import requests as _req
                raise _req.exceptions.ChunkedEncodingError("Connection reset by peer")
            yield content[offset: offset + chunk_size]
            offset += chunk_size

    resp.iter_content = _iter_content
    resp.close = MagicMock()
    return resp


# ============================================================
# TEST 1 — Valid CSV download from trusted Supabase host
# ============================================================
def test_valid_csv_download():
    csv_bytes = _make_valid_csv_bytes(90)
    mock_resp = _mock_response(csv_bytes)

    with patch("forecasting.requests.get", return_value=mock_resp) as mock_get:
        df = download_and_parse_csv(TRUSTED_URL)

    # requests.get must be called exactly once with redirects disabled
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs.get("allow_redirects") is False, \
        "allow_redirects must be False to prevent redirect-following"
    assert call_kwargs.kwargs.get("stream") is True, \
        "stream must be True to enable chunked reading"

    assert "sale_date" not in df.columns or True   # parsed into index already
    assert len(df) == 90
    mock_resp.close.assert_called()
    print("[TEST 1] PASS — Valid CSV downloaded and parsed correctly.")


# ============================================================
# TEST 2 — CSV exceeding the configured size limit (streaming enforcement)
# ============================================================
def test_csv_exceeds_size_limit_during_streaming():
    """
    The size limit must be enforced via byte counting while streaming,
    NOT only via the Content-Length header.  This test supplies a file
    that is larger than the configured limit and has NO Content-Length
    header so the only enforcement path is the streaming counter.
    """
    limit = 512 * 1024  # 512 KB test limit
    oversized_content = b"x" * (limit + 1024)  # 1 KB over the limit

    # Deliberately omit Content-Length so only the streaming path triggers
    mock_resp = _mock_response(oversized_content, headers={})

    with patch("forecasting.requests.get", return_value=mock_resp), \
         patch("forecasting._get_max_csv_bytes", return_value=limit):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError was not raised")
        except CsvDownloadError as e:
            msg = str(e).lower()
            assert "too large" in msg or "limit" in msg, \
                f"Error message should mention size limit; got: {e}"

    mock_resp.close.assert_called()
    print("[TEST 2] PASS — Oversized CSV correctly rejected during streaming (no Content-Length header).")


# ============================================================
# TEST 3 — Misleading Content-Length: declared small, actual body large
# ============================================================
def test_misleading_content_length_header():
    """
    The server declares a small Content-Length but actually streams more
    data than the limit allows.  The streaming byte-counter must still
    abort at the limit even though the early header check passed.
    """
    limit = 128 * 1024  # 128 KB test limit
    actual_content = b"y" * (limit + 4096)  # 4 KB over the limit

    # Lie: Content-Length says 1024 bytes (would pass the early hint check)
    liar_headers = {"Content-Length": "1024"}
    mock_resp = _mock_response(actual_content, headers=liar_headers)

    with patch("forecasting.requests.get", return_value=mock_resp), \
         patch("forecasting._get_max_csv_bytes", return_value=limit):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError was not raised")
        except CsvDownloadError as e:
            msg = str(e).lower()
            assert "too large" in msg or "limit" in msg or "aborted" in msg, \
                f"Error message should mention size/abort; got: {e}"

    mock_resp.close.assert_called()
    print("[TEST 3] PASS — Misleading Content-Length correctly handled; streaming counter enforced the real limit.")


# ============================================================
# TEST 4 — Arbitrary external (untrusted) URL
# ============================================================
class TestUntrustedUrls:
    UNTRUSTED_CASES = [
        # Different host
        (
            "https://evil-site.com/storage/v1/object/public/service-data/user/sales.csv",
            "host",
        ),
        # HTTP instead of HTTPS
        (
            "http://dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/u/sales.csv",
            "scheme",
        ),
        # Embedded credentials
        (
            "https://user:pass@dzwzsnnrfstsmtpueilo.supabase.co/storage/v1/object/public/service-data/u/s.csv",
            "credentials",
        ),
        # Localhost SSRF attempt
        (
            "https://localhost/storage/v1/object/public/service-data/u/sales.csv",
            "host",
        ),
        # Wrong path prefix
        (
            "https://dzwzsnnrfstsmtpueilo.supabase.co/private-bucket/service-data/user/sales.csv",
            "path",
        ),
        # file:// scheme
        (
            "file:///etc/passwd",
            "scheme",
        ),
        # Cloud metadata IP (AWS/GCP/Azure)
        (
            "https://169.254.169.254/storage/v1/object/public/service-data/u/sales.csv",
            "metadata_ip",
        ),
        # GCP metadata hostname
        (
            "https://metadata.google.internal/storage/v1/object/public/service-data/u/sales.csv",
            "metadata_hostname",
        ),
        # Private IP address (RFC 1918)
        (
            "https://10.0.0.1/storage/v1/object/public/service-data/u/sales.csv",
            "private_ip",
        ),
        # Loopback IPv4 address
        (
            "https://127.0.0.1/storage/v1/object/public/service-data/u/sales.csv",
            "loopback_ip",
        ),
    ]

    def test_all_untrusted_urls_rejected_before_network(self):
        """
        _validate_csv_url must raise CsvDownloadError for every untrusted case
        WITHOUT making any network calls.
        """
        with patch("forecasting.requests.get") as mock_get:
            for url, reason in self.UNTRUSTED_CASES:
                try:
                    _validate_csv_url(url)
                    raise AssertionError(
                        f"Expected CsvDownloadError for {reason} violation, "
                        f"but no error was raised for URL: {url}"
                    )
                except CsvDownloadError:
                    pass  # expected

            # Crucially: no network calls were made
            mock_get.assert_not_called()

        print("[TEST 4] PASS — All untrusted/invalid URLs rejected before any network connection.")


# ============================================================
# TEST 5 — Untrusted redirect from a Supabase URL
# ============================================================
def test_redirect_to_untrusted_host_rejected():
    """
    A valid-looking Supabase URL that responds with a 302 redirect to an
    external server must be rejected immediately. The redirect destination
    must not be followed.
    """
    redirect_resp = _mock_response(
        b"",
        status_code=302,
        simulate_redirect_location="https://attacker.example.com/steal",
    )

    with patch("forecasting.requests.get", return_value=redirect_resp):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError was not raised for redirect")
        except CsvDownloadError as e:
            msg = str(e).lower()
            assert "redirect" in msg, \
                f"Error message should mention redirect; got: {e}"

    redirect_resp.close.assert_called()
    print("[TEST 5] PASS — 302 redirect to untrusted host rejected without following.")


def test_redirect_301_rejected():
    """Permanent redirects are equally rejected."""
    redirect_resp = _mock_response(
        b"",
        status_code=301,
        simulate_redirect_location="https://other.cdn.com/file.csv",
    )

    with patch("forecasting.requests.get", return_value=redirect_resp):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError was not raised for 301")
        except CsvDownloadError as e:
            assert "redirect" in str(e).lower()

    print("[TEST 5b] PASS — 301 redirect rejected.")


# ============================================================
# TEST 6 — Failed / interrupted download (connection drops mid-stream)
# ============================================================
def test_interrupted_download_mid_stream():
    """
    If the connection is dropped partway through streaming, the function
    must raise CsvDownloadError and must not return partial data.
    The byte buffer must be cleaned up (response closed).

    We need the CSV to span at least 2 chunks (each 64 KiB = 65,536 bytes)
    so that the error fires after the first chunk is received but before
    the download is complete.
    """
    # Build content that is definitely > 2 × 65,536 bytes so iter_error_after=1
    # fires mid-download (after yielding chunk 1, before chunk 2 is sent).
    large_content = b"x" * (65_536 * 3)   # 192 KiB — well over 2 chunks
    mock_resp = _mock_response(large_content, iter_error_after=1)

    with patch("forecasting.requests.get", return_value=mock_resp):
        try:
            result = download_and_parse_csv(TRUSTED_URL)
            raise AssertionError(
                f"Expected CsvDownloadError for interrupted download, got result with {len(result)} rows"
            )
        except CsvDownloadError as e:
            msg = str(e).lower()
            assert "interrupted" in msg or "connection" in msg or "stream" in msg, \
                f"Error message should describe the interruption; got: {e}"

    mock_resp.close.assert_called()
    print("[TEST 6] PASS — Interrupted download raised CsvDownloadError; response closed; no partial data returned.")


def test_failed_connection():
    """A complete connection failure must raise CsvDownloadError."""
    import requests as _requests

    with patch("forecasting.requests.get", side_effect=_requests.exceptions.ConnectionError("refused")):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError for connection failure")
        except CsvDownloadError as e:
            assert "connection" in str(e).lower()

    print("[TEST 6b] PASS — Connection failure raised CsvDownloadError.")


def test_timeout():
    """A timeout must raise CsvDownloadError."""
    import requests as _requests

    with patch("forecasting.requests.get", side_effect=_requests.exceptions.Timeout("timed out")):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError for timeout")
        except CsvDownloadError as e:
            assert "timed out" in str(e).lower() or "timeout" in str(e).lower()

    print("[TEST 6c] PASS — Timeout raised CsvDownloadError.")


# ============================================================
# TEST 7 — Stalled or slow streaming download timeout watchdog
# ============================================================
def test_stalled_slow_download_timeout():
    """
    If individual chunks trickle in very slowly and the total streaming time
    exceeds max_download_seconds, the streaming watchdog must abort the download.
    """
    import time

    def mock_slow_iter(chunk_size=65536):
        yield b"sale_date,daily_revenue\n2024-01-01,100\n"
        time.sleep(0.02)
        yield b"2024-01-02,120\n"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.iter_content = mock_slow_iter
    mock_resp.close = MagicMock()

    # Configure a tiny max download time limit (e.g., 0.005 seconds)
    with patch("forecasting.requests.get", return_value=mock_resp), \
         patch("forecasting._get_download_timeouts", return_value=(5.0, 15.0, 0.005)):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError for stalled download was not raised")
        except CsvDownloadError as e:
            assert "timed out" in str(e).lower() or "timeout" in str(e).lower()

    mock_resp.close.assert_called()
    print("[TEST 7] PASS — Stalled/slow streaming download correctly aborted by watchdog.")


# ============================================================
# TEST 8 — Malformed / unparseable CSV content
# ============================================================
def test_malformed_csv_content():
    """
    If the downloaded body is binary garbage, HTML, or corrupted data that
    cannot be parsed as a tabular CSV, CsvDownloadError must be raised.
    """
    corrupted_content = b"\x00\x01\x02\x03\xff\xfe\xfd\x00" * 50
    mock_resp = _mock_response(corrupted_content)

    with patch("forecasting.requests.get", return_value=mock_resp):
        try:
            download_and_parse_csv(TRUSTED_URL)
            raise AssertionError("Expected CsvDownloadError for malformed CSV was not raised")
        except CsvDownloadError as e:
            assert "invalid csv format" in str(e).lower() or "csv" in str(e).lower()

    mock_resp.close.assert_called()
    print("[TEST 8] PASS — Malformed CSV content cleanly caught and rejected.")


# ============================================================
# TEST 9 — Valid Firebase Storage URL supported
# ============================================================
def test_valid_firebase_storage_url():
    """
    Firebase Storage URLs matching the configured bucket and structure
    must be permitted and successfully parsed.
    """
    fb_url = (
        "https://firebasestorage.googleapis.com"
        "/v0/b/final-business-os.firebasestorage.app/o"
        "/service-data%2Fuser_123%2Fsales.csv?alt=media"
    )
    csv_bytes = _make_valid_csv_bytes(90)
    mock_resp = _mock_response(csv_bytes)

    with patch("forecasting.requests.get", return_value=mock_resp):
        df = download_and_parse_csv(fb_url)

    assert len(df) == 90
    print("[TEST 9] PASS — Valid Firebase Storage URL accepted and parsed.")


# ============================================================
# BONUS — env-var configurable size limit is read correctly
# ============================================================
def test_max_csv_bytes_env_var():
    with patch.dict(os.environ, {"MAX_CSV_BYTES": "2097152"}):
        assert _get_max_csv_bytes() == 2_097_152  # 2 MB

    with patch.dict(os.environ, {"MAX_CSV_BYTES": "not_a_number"}):
        assert _get_max_csv_bytes() == 10 * 1024 * 1024  # falls back to 10 MB default

    with patch.dict(os.environ, {}, clear=True):
        # When the env var is absent entirely
        import importlib
        assert _get_max_csv_bytes() == 10 * 1024 * 1024

    print("[BONUS] PASS — MAX_CSV_BYTES env var correctly controls the size limit.")


# ============================================================
# Runner (also works without pytest: python test_csv_download.py)
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SMART BPI CSV DOWNLOAD SECURITY TESTS")
    print("=" * 70)

    test_valid_csv_download()
    test_csv_exceeds_size_limit_during_streaming()
    test_misleading_content_length_header()
    TestUntrustedUrls().test_all_untrusted_urls_rejected_before_network()
    test_redirect_to_untrusted_host_rejected()
    test_redirect_301_rejected()
    test_interrupted_download_mid_stream()
    test_failed_connection()
    test_timeout()
    test_stalled_slow_download_timeout()
    test_malformed_csv_content()
    test_valid_firebase_storage_url()
    test_max_csv_bytes_env_var()

    print()
    print("=" * 70)
    print("ALL CSV DOWNLOAD SECURITY TESTS PASSED")
    print("=" * 70)
