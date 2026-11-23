"""Market data fetching, technical indicators, ticker resolution, and chart formatting."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import re
import threading
import time
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import requests
import yfinance as yf

from backend.config import STOCK_LIST, VALID_TICKERS
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.security.sanitizers import validate_ticker
from backend.services.runtime import AsyncSingleflightCache, runtime

logger = get_logger(__name__)

_NSE_BSE_TICKER_RE = re.compile(r"^[A-Z0-9&-]{1,20}\.(NS|BO)$")

_STOCK_HISTORY_LOCK = threading.Lock()
_STOCK_HISTORY_FALLBACK: dict[str, pd.DataFrame] = {}

_MARKET_OVERVIEW_SINGLEFLIGHT = AsyncSingleflightCache(maxsize=2, default_ttl=45.0)
_STOCK_DETAIL_SINGLEFLIGHT = AsyncSingleflightCache(maxsize=128, default_ttl=300.0)
_HISTORY_CHART_SINGLEFLIGHT = AsyncSingleflightCache(maxsize=128, default_ttl=300.0)


def _is_nse_market_hours() -> bool:
    """Return True during NSE cash market hours (Mon-Fri, 09:15-15:30 IST)."""
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if ist_now.weekday() >= 5:
        return False
    mins = ist_now.hour * 60 + ist_now.minute
    return 555 <= mins <= 930


def _to_unix_ts(val) -> int:
    """Convert a datetime, date, or YYYY-MM-DD string to an integer Unix timestamp."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return int(pd.Timestamp(val).timestamp())
    if hasattr(val, "timestamp"):
        return int(val.timestamp())
    return int(pd.Timestamp(val).timestamp())


def _parse_yahoo_chart_payload(payload: dict) -> pd.DataFrame:
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        return pd.DataFrame()
    item = results[0]
    timestamps = item.get("timestamp") or []
    quotes = ((item.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        return pd.DataFrame()

    idx = pd.to_datetime(timestamps, unit="s").normalize()
    df = pd.DataFrame(
        {
            "Open": quotes.get("open"),
            "High": quotes.get("high"),
            "Low": quotes.get("low"),
            "Close": quotes.get("close"),
            "Volume": quotes.get("volume"),
        },
        index=idx,
    )
    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def _fetch_yahoo_chart_via_requests(ticker: str, start=None, end=None, period=None, timeout: int = 20) -> pd.DataFrame:
    """Fallback OHLCV fetch via Yahoo v8 chart JSON API using strict ticker validation."""
    safe_ticker = validate_ticker(ticker)
    encoded_ticker = quote(safe_ticker, safe="")

    params: dict[str, object] = {"interval": "1d"}
    if period:
        params["range"] = period
    else:
        if start is not None:
            params["period1"] = _to_unix_ts(start)
        if end is not None:
            params["period2"] = _to_unix_ts(end)
        if "period1" not in params:
            params["range"] = "3y"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            url = f"https://{host}/v8/finance/chart/{encoded_ticker}"
            resp = requests.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=False)
            if resp.status_code != 200:
                continue
            df = _parse_yahoo_chart_payload(resp.json())
            if not df.empty:
                return df
        except Exception as exc:
            logger.debug("Yahoo v8 chart fallback (%s) failed for %s: %s", host, safe_ticker, exc)
    return pd.DataFrame()


def download_ticker_data(ticker: str, retries: int = 3, timeout: int = 25, **kwargs) -> pd.DataFrame:
    """Download Yahoo Finance OHLCV data with retries, locking, and v8 JSON fallback."""
    safe_ticker = validate_ticker(ticker)
    kwargs.setdefault("progress", False)
    last_exc = None

    for attempt in range(retries):
        try:
            with runtime.yfinance_lock:
                try:
                    data = yf.download(safe_ticker, timeout=timeout, **kwargs)
                except TypeError:
                    data = yf.download(safe_ticker, **kwargs)
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_exc = exc
            logger.debug("yfinance download attempt %d/%d failed for %s: %s", attempt + 1, retries, safe_ticker, exc)
        if attempt < retries - 1:
            time.sleep(0.4 * (attempt + 1))

    try:
        hist_kwargs = {k: v for k, v in kwargs.items() if k in ("period", "start", "end", "interval")}
        with runtime.yfinance_lock:
            try:
                data = yf.Ticker(safe_ticker).history(timeout=timeout, **hist_kwargs)
            except TypeError:
                data = yf.Ticker(safe_ticker).history(**hist_kwargs)
        if data is not None and not data.empty:
            return data
    except Exception as exc:
        last_exc = exc
        logger.debug("yfinance Ticker.history fallback failed for %s: %s", safe_ticker, exc)

    data = _fetch_yahoo_chart_via_requests(
        safe_ticker,
        start=kwargs.get("start"),
        end=kwargs.get("end"),
        period=kwargs.get("period"),
        timeout=timeout,
    )
    if data is not None and not data.empty:
        return data

    if last_exc:
        logger.warning("All download attempts failed for %s: %s", safe_ticker, last_exc)
    return pd.DataFrame()


def get_next_business_day(from_date=None) -> str:
    """Calculate the next business day (Mon-Fri) as YYYY-MM-DD."""
    base_date = from_date if from_date is not None else datetime.now()
    return (base_date + BDay(1)).strftime("%Y-%m-%d")


def clean_yfinance_dataframe(data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize MultiIndex columns returned by yfinance and drop invalid rows."""
    if data is None or data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(axis=0, how="all")
    data = data[data.index.notna()]
    return data


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA_50 and RSI_14 technical indicators with divide-by-zero protection."""
    data = clean_yfinance_dataframe(data.copy())
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    delta = data["Close"].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()

    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data


def fetch_stock_data(ticker: str, days: int = 730) -> pd.DataFrame | None:
    """Fetch cleaned historical OHLCV data for a ticker over the requested lookback days."""
    safe_ticker = validate_ticker(ticker)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = download_ticker_data(safe_ticker, start=start_date, end=end_date)
        if data is not None and not data.empty:
            data = clean_yfinance_dataframe(data)
            data.dropna(inplace=True)
            if not data.empty:
                with _STOCK_HISTORY_LOCK:
                    if len(_STOCK_HISTORY_FALLBACK) > 128:
                        _STOCK_HISTORY_FALLBACK.clear()
                    _STOCK_HISTORY_FALLBACK[safe_ticker] = data
                return data
    except Exception as exc:
        logger.warning("Error fetching stock data for %s: %s", safe_ticker, exc)

    with _STOCK_HISTORY_LOCK:
        cached = _STOCK_HISTORY_FALLBACK.get(safe_ticker)
    if cached is not None and not cached.empty:
        return cached.copy()
    return None


def _pick_yahoo_quote(quotes: list[dict], prefer_indian: bool, query: str) -> tuple[str | None, str | None]:
    if not quotes:
        return None, None
    if prefer_indian:
        indian_quote = next(
            (q for q in quotes if str(q.get("symbol", "")).endswith((".NS", ".BO"))),
            None,
        )
        if indian_quote:
            return indian_quote["symbol"], indian_quote.get("shortname", query)
    return quotes[0].get("symbol"), quotes[0].get("shortname", query)


def lookup_ticker(query: str, prefer_indian: bool = True) -> tuple[str | None, str | None]:
    """Search Yahoo Finance for a company/symbol using parameterized query arguments."""
    clean_q = " ".join(str(query or "").split())[:60]
    if not clean_q:
        return None, None
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": clean_q, "quotesCount": 5},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
            allow_redirects=False,
        )
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            return _pick_yahoo_quote(quotes, prefer_indian=prefer_indian, query=clean_q)
    except Exception as exc:
        logger.debug("Yahoo Search Error for '%s': %s", sanitize_for_log(clean_q), exc)
    return None, None


async def lookup_ticker_async(query: str, prefer_indian: bool = True) -> tuple[str | None, str | None]:
    """Non-blocking parameterized Yahoo Finance ticker lookup via pooled HTTPX client."""
    clean_q = " ".join(str(query or "").split())[:60]
    if not clean_q:
        return None, None
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": clean_q, "quotesCount": 5}
    try:
        client = runtime.try_get_http_client()
        if client is not None:
            resp = await client.get(url, params=params, timeout=5.0)
        else:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as temp_client:
                resp = await temp_client.get(url, params=params)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            return _pick_yahoo_quote(quotes, prefer_indian=prefer_indian, query=clean_q)
    except Exception as exc:
        logger.debug("Async Yahoo Search Error for '%s': %s", sanitize_for_log(clean_q), exc)
    return None, None


def resolve_ticker(stock_name: str | None) -> tuple[str | None, str | None]:
    """Resolve a human-readable stock name or ticker symbol into (display_name, ticker)."""
    if not stock_name:
        return None, None

    if stock_name in STOCK_LIST:
        return stock_name, STOCK_LIST[stock_name]

    lowered = stock_name.strip().lower()
    for name, ticker in STOCK_LIST.items():
        if name.lower() == lowered or ticker.lower() == lowered or ticker.split(".")[0].lower() == lowered:
            return name, ticker

    prefer_indian = not any(
        us_name in lowered for us_name in ("apple", "tesla", "nvidia", "google", "microsoft", "amazon", "meta")
    )
    found_ticker, found_name = lookup_ticker(stock_name, prefer_indian=prefer_indian)
    if found_ticker:
        return found_name or stock_name, found_ticker

    return stock_name, f"{stock_name.strip().upper()}.NS"


async def resolve_ticker_async(stock_name: str | None) -> tuple[str | None, str | None]:
    """Async counterpart to resolve_ticker for FastAPI endpoints."""
    if not stock_name:
        return None, None

    if stock_name in STOCK_LIST:
        return stock_name, STOCK_LIST[stock_name]

    lowered = stock_name.strip().lower()
    for name, ticker in STOCK_LIST.items():
        if name.lower() == lowered or ticker.lower() == lowered or ticker.split(".")[0].lower() == lowered:
            return name, ticker

    prefer_indian = not any(
        us_name in lowered for us_name in ("apple", "tesla", "nvidia", "google", "microsoft", "amazon", "meta")
    )
    found_ticker, found_name = await lookup_ticker_async(stock_name, prefer_indian=prefer_indian)
    if found_ticker:
        return found_name or stock_name, found_ticker

    return stock_name, f"{stock_name.strip().upper()}.NS"


def _format_series_records(df: pd.DataFrame, column_mapping: dict[str, str]) -> list[dict]:
    """Vectorized conversion of DataFrame slice into Lightweight Charts JSON records."""
    dates = df.index.strftime("%Y-%m-%d").tolist()
    src_cols = list(column_mapping.keys())
    dst_cols = list(column_mapping.values())
    col_arrays = [df[src].astype(float).round(4).tolist() for src in src_cols]

    records = []
    for row_idx, d_str in enumerate(dates):
        item = {"time": d_str}
        for col_idx, dst_name in enumerate(dst_cols):
            val = col_arrays[col_idx][row_idx]
            item[dst_name] = float(val) if np.isfinite(val) else 0.0
        records.append(item)
    return records


def get_historical_chart_data(ticker: str) -> dict | None:
    """Return 3 years of OHLC, volume, RSI-14, and SMA-50 data formatted for Lightweight Charts."""
    safe_ticker = validate_ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=(3 * 365) + 1)
    raw = download_ticker_data(safe_ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date)
    if raw is not None and not raw.empty:
        with _STOCK_HISTORY_LOCK:
            _STOCK_HISTORY_FALLBACK[safe_ticker] = raw
    else:
        with _STOCK_HISTORY_LOCK:
            raw = _STOCK_HISTORY_FALLBACK.get(safe_ticker)

    if raw is None or raw.empty:
        return None

    data = add_technical_features(raw)
    if data.empty:
        return None

    return {
        "ohlc": _format_series_records(data, {"Open": "open", "High": "high", "Low": "low", "Close": "close"}),
        "volume": _format_series_records(data, {"Volume": "value"}),
        "rsi": _format_series_records(data, {"RSI_14": "value"}),
        "sma": _format_series_records(data, {"SMA_50": "value"}),
    }


async def get_historical_chart_data_async(ticker: str) -> dict | None:
    """Cached, singleflight async wrapper for 3-year historical chart OHLCV/RSI/SMA data."""
    safe_ticker = validate_ticker(ticker)
    return await _HISTORY_CHART_SINGLEFLIGHT.get_or_compute(
        key=safe_ticker,
        factory=lambda: runtime.run_io(get_historical_chart_data, safe_ticker),
        ttl=300.0,
        stale_on_error=True,
    )


def _build_overview_item(name: str, ticker: str, price, prev, open_price) -> dict | None:
    if pd.isna(price) or pd.isna(prev) or float(prev) <= 0.0:
        return None
    price_f = float(price)
    prev_f = float(prev)
    change = price_f - prev_f
    p_change = (change / prev_f) * 100.0
    return {
        "name": name,
        "symbol": ticker,
        "price": round(price_f, 2),
        "open": round(float(open_price), 2) if not pd.isna(open_price) else 0.0,
        "close": round(prev_f, 2),
        "change": round(change, 2),
        "pChange": round(p_change, 2),
    }


def _fetch_overview_via_chart_api() -> list[dict]:
    """Concurrent Yahoo v8 chart JSON fallback for bulk NSE market overview."""

    def _fetch_one(item: tuple[str, str]) -> dict | None:
        name, ticker = item
        try:
            df = _fetch_yahoo_chart_via_requests(ticker, period="5d", timeout=10)
            if df is None or df.empty or "Close" not in df:
                return None
            closes = df["Close"].dropna()
            if len(closes) < 2:
                return None
            opens = df["Open"] if "Open" in df else closes
            return _build_overview_item(name, ticker, closes.iloc[-1], closes.iloc[-2], opens.iloc[-1])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_fetch_one, STOCK_LIST.items()))
    return [row for row in results if row is not None]


def get_market_overview() -> list[dict]:
    """Fetch bulk 5-day prices for all configured NSE stocks with v8 JSON chart fallback."""
    tickers = list(STOCK_LIST.values())
    data = None
    try:
        with runtime.yfinance_lock:
            data = yf.download(tickers, period="5d", progress=False)
    except Exception as exc:
        logger.warning("Bulk yfinance download failed for market overview: %s", exc)

    market_summary: list[dict] = []
    if data is not None and not data.empty:
        closes = data["Close"] if "Close" in data else data
        opens = data["Open"] if "Open" in data else data
        if len(closes) >= 2:
            current_closes = closes.iloc[-1]
            prev_closes = closes.iloc[-2]
            current_opens = opens.iloc[-1]

            for name, ticker in STOCK_LIST.items():
                try:
                    if ticker not in current_closes:
                        continue
                    item = _build_overview_item(
                        name,
                        ticker,
                        current_closes[ticker],
                        prev_closes[ticker],
                        current_opens[ticker],
                    )
                    if item is not None:
                        market_summary.append(item)
                except Exception:
                    continue

    if not market_summary:
        logger.info("Bulk yfinance market overview returned 0 rows; using Yahoo v8 chart API fallback.")
        market_summary = _fetch_overview_via_chart_api()

    return market_summary


async def get_market_overview_async() -> list[dict]:
    """Cached singleflight NSE market overview (45s TTL during market hours, 300s off-hours)."""
    ttl = 45.0 if _is_nse_market_hours() else 300.0

    async def _compute_non_empty() -> list[dict]:
        rows = await runtime.run_io(get_market_overview)
        if not rows:
            raise RuntimeError("NSE market overview returned zero rows.")
        return rows

    try:
        return await _MARKET_OVERVIEW_SINGLEFLIGHT.get_or_compute(
            key="nse_overview",
            factory=_compute_non_empty,
            ttl=ttl,
            stale_on_error=True,
        )
    except RuntimeError:
        return []


def _fetch_with_retry(fn, attempts: int = 3, delay: float = 0.4):
    """Run a yfinance network call with short retries for transient curl_cffi resets."""
    last_exc = None
    for idx in range(attempts):
        try:
            with runtime.yfinance_lock:
                return fn()
        except Exception as exc:
            last_exc = exc
            if idx < attempts - 1:
                time.sleep(delay * (idx + 1))
    raise last_exc


def _sanitize_json_value(val):
    if isinstance(val, (np.floating, float)):
        fval = float(val)
        return fval if np.isfinite(fval) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, dict):
        return {k: _sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_json_value(v) for v in val]
    return val


def validate_allowed_detail_ticker(ticker: str) -> str:
    """Restrict detail lookups to known NSE 100 tickers or explicit NSE/BSE equity symbols."""
    clean = validate_ticker(ticker)
    if clean not in VALID_TICKERS and not _NSE_BSE_TICKER_RE.fullmatch(clean):
        raise ValueError("Ticker must be a valid NSE/BSE equity symbol.")
    return clean


def get_stock_detail(ticker: str) -> dict:
    """Fetch fundamental info metrics and vectorized full historical price history."""
    clean_ticker = validate_allowed_detail_ticker(ticker)
    stock = yf.Ticker(clean_ticker)
    hist = _fetch_with_retry(lambda: stock.history(period="max"))

    info: dict = {}
    try:
        info = _fetch_with_retry(lambda: stock.info) or {}
    except Exception as exc:
        logger.warning("yfinance info failed for %s, using OHLCV fallback: %s", clean_ticker, exc)

    if not info and hist is not None and not hist.empty:
        recent_1y = hist.tail(252)
        close_s = hist["Close"].dropna()
        info = {
            "symbol": clean_ticker,
            "currentPrice": float(close_s.iloc[-1]) if not close_s.empty else None,
            "fiftyTwoWeekHigh": float(recent_1y["High"].max()) if "High" in recent_1y else None,
            "fiftyTwoWeekLow": float(recent_1y["Low"].min()) if "Low" in recent_1y else None,
            "fiftyDayAverage": float(close_s.tail(50).mean()) if len(close_s) >= 1 else None,
            "twoHundredDayAverage": float(close_s.tail(200).mean()) if len(close_s) >= 1 else None,
            "volume": int(hist["Volume"].iloc[-1])
            if "Volume" in hist and pd.notna(hist["Volume"].iloc[-1])
            else None,
        }

    chart_data: list[dict] = []
    if hist is not None and not hist.empty:
        hist_clean = clean_yfinance_dataframe(hist)
        close_series = pd.to_numeric(hist_clean["Close"], errors="coerce").dropna()
        if not close_series.empty:
            close_series = close_series.sort_index()
            dates = pd.to_datetime(close_series.index).strftime("%Y-%m-%d").tolist()
            vals = close_series.astype(float).tolist()
            # Deduplicate identical date strings keeping latest
            dedup: dict[str, float] = {}
            for d_str, val in zip(dates, vals):
                if np.isfinite(val):
                    dedup[d_str] = round(val, 4)
            chart_data = [{"time": k, "value": v} for k, v in dedup.items()]

    return {"info": _sanitize_json_value(info), "chart": chart_data}


async def get_stock_detail_async(ticker: str) -> dict:
    """Cached singleflight async wrapper for single-stock fundamental details."""
    clean_ticker = validate_allowed_detail_ticker(ticker)
    return await _STOCK_DETAIL_SINGLEFLIGHT.get_or_compute(
        key=clean_ticker,
        factory=lambda: runtime.run_io(get_stock_detail, clean_ticker),
        ttl=300.0,
        stale_on_error=True,
    )
