"""Market data fetching, technical indicators, ticker resolution, and chart formatting."""

from datetime import datetime, timedelta
import json
import threading
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.tseries.offsets import BDay

from backend.config import STOCK_LIST

_STOCK_DETAIL_CACHE = {}
_STOCK_DETAIL_TTL_SEC = 300
_STOCK_DETAIL_LOCK = threading.Lock()
_STOCK_HISTORY_CACHE = {}
_STOCK_HISTORY_LOCK = threading.Lock()


def _to_unix_ts(val):
    """Convert a datetime, date, or YYYY-MM-DD string to an integer Unix timestamp."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return int(pd.Timestamp(val).timestamp())
    if hasattr(val, "timestamp"):
        return int(val.timestamp())
    return int(pd.Timestamp(val).timestamp())


def _fetch_yahoo_chart_via_requests(ticker, start=None, end=None, period=None, timeout=20):
    """Fallback OHLCV fetch via Yahoo v8 chart JSON API using requests (bypasses curl_cffi timeouts)."""
    params = {"interval": "1d"}
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
            url = f"https://{host}/v8/finance/chart/{ticker}"
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            results = (payload.get("chart") or {}).get("result") or []
            if not results:
                continue
            item = results[0]
            timestamps = item.get("timestamp") or []
            quotes = ((item.get("indicators") or {}).get("quote") or [{}])[0]
            if not timestamps or not quotes:
                continue

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
            if not df.empty:
                return df
        except Exception as exc:
            print(f"Yahoo v8 chart fallback ({host}) failed for {ticker}: {exc}")
    return pd.DataFrame()


def download_ticker_data(ticker, retries=3, timeout=25, **kwargs):
    """Download Yahoo Finance OHLCV data with longer timeout, retries, and fallbacks."""
    kwargs.setdefault("progress", False)
    last_exc = None
    for attempt in range(retries):
        try:
            try:
                data = yf.download(ticker, timeout=timeout, **kwargs)
            except TypeError:
                data = yf.download(ticker, **kwargs)
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_exc = exc
            print(f"yfinance download attempt {attempt + 1}/{retries} failed for {ticker}: {exc}")
        if attempt < retries - 1:
            time.sleep(0.6 * (attempt + 1))

    if isinstance(ticker, str):
        try:
            hist_kwargs = {k: v for k, v in kwargs.items() if k in ("period", "start", "end", "interval")}
            try:
                data = yf.Ticker(ticker).history(timeout=timeout, **hist_kwargs)
            except TypeError:
                data = yf.Ticker(ticker).history(**hist_kwargs)
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_exc = exc
            print(f"yfinance Ticker.history fallback failed for {ticker}: {exc}")

        data = _fetch_yahoo_chart_via_requests(
            ticker,
            start=kwargs.get("start"),
            end=kwargs.get("end"),
            period=kwargs.get("period"),
            timeout=timeout,
        )
        if data is not None and not data.empty:
            return data

    if last_exc:
        print(f"All download attempts failed for {ticker}: {last_exc}")
    return pd.DataFrame()


def get_next_business_day(from_date=None):
    """Calculate the next business day (Mon-Fri) as YYYY-MM-DD."""
    base_date = from_date if from_date is not None else datetime.now()
    return (base_date + BDay(1)).strftime("%Y-%m-%d")


def clean_yfinance_dataframe(data):
    """Normalize MultiIndex columns returned by yfinance and drop invalid rows."""
    if data is None or data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(axis=0, how="all")
    data = data[data.index.notna()]
    return data


def add_technical_features(data):
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


def fetch_stock_data(ticker, days=730):
    """Fetch cleaned historical OHLCV data for a ticker over the requested lookback days."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = download_ticker_data(ticker, start=start_date, end=end_date)
        if data is not None and not data.empty:
            data = clean_yfinance_dataframe(data)
            data.dropna(inplace=True)
            if not data.empty:
                with _STOCK_HISTORY_LOCK:
                    _STOCK_HISTORY_CACHE[ticker] = data
                return data
    except Exception as exc:
        print(f"Error fetching stock data for {ticker}: {exc}")

    with _STOCK_HISTORY_LOCK:
        cached = _STOCK_HISTORY_CACHE.get(ticker)
    if cached is not None and not cached.empty:
        return cached.copy()
    return None


def lookup_ticker(query, prefer_indian=True):
    """Search Yahoo Finance for a company/symbol and return (ticker, shortname)."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            if quotes:
                if prefer_indian:
                    indian_quote = next(
                        (
                            q
                            for q in quotes
                            if q.get("symbol", "").endswith((".NS", ".BO"))
                        ),
                        None,
                    )
                    if indian_quote:
                        return indian_quote["symbol"], indian_quote.get("shortname", query)
                return quotes[0]["symbol"], quotes[0].get("shortname", query)
    except Exception as exc:
        print(f"Yahoo Search Error for '{query}': {exc}")
    return None, None


def resolve_ticker(stock_name):
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

    return stock_name, f"{stock_name}.NS"


def _format_series_records(df, column_mapping):
    """Format a DataFrame subset into Lightweight Charts JSON records with YYYY-MM-DD timestamps."""
    subset = df[list(column_mapping.keys())].copy()
    subset["time"] = subset.index.strftime("%Y-%m-%d")
    subset.rename(columns=column_mapping, inplace=True)
    ordered_cols = ["time"] + list(column_mapping.values())
    return json.loads(subset[ordered_cols].to_json(orient="records"))


def get_historical_chart_data(ticker):
    """Return 3 years of OHLC, volume, RSI-14, and SMA-50 data formatted for Lightweight Charts."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=(3 * 365) + 1)
    raw = download_ticker_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date)
    if raw is not None and not raw.empty:
        with _STOCK_HISTORY_LOCK:
            _STOCK_HISTORY_CACHE[ticker] = raw
    else:
        with _STOCK_HISTORY_LOCK:
            raw = _STOCK_HISTORY_CACHE.get(ticker)
    if raw is None or raw.empty:
        return None

    data = add_technical_features(raw)
    if data.empty:
        return None

    return {
        "ohlc": _format_series_records(
            data, {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        ),
        "volume": _format_series_records(data, {"Volume": "value"}),
        "rsi": _format_series_records(data, {"RSI_14": "value"}),
        "sma": _format_series_records(data, {"SMA_50": "value"}),
    }


def get_market_overview():
    """Fetch bulk 5-day prices for all configured NSE stocks and compute daily change metrics."""
    tickers = list(STOCK_LIST.values())
    data = yf.download(tickers, period="5d", progress=False)
    market_summary = []

    if data.empty:
        return market_summary

    closes = data["Close"] if "Close" in data else data
    opens = data["Open"] if "Open" in data else data
    if len(closes) < 2:
        return market_summary

    current_closes = closes.iloc[-1]
    prev_closes = closes.iloc[-2]
    current_opens = opens.iloc[-1]

    for name, ticker in STOCK_LIST.items():
        try:
            if ticker not in current_closes:
                continue
            price = current_closes[ticker]
            prev = prev_closes[ticker]
            open_price = current_opens[ticker]
            if pd.isna(price) or pd.isna(prev):
                continue

            change = price - prev
            p_change = (change / prev) * 100
            market_summary.append(
                {
                    "name": name,
                    "symbol": ticker,
                    "price": round(float(price), 2),
                    "open": round(float(open_price), 2) if not pd.isna(open_price) else 0.0,
                    "close": round(float(prev), 2),
                    "change": round(float(change), 2),
                    "pChange": round(float(p_change), 2),
                }
            )
        except Exception:
            continue

    return market_summary


def _fetch_with_retry(fn, attempts=3, delay=0.4):
    """Run a yfinance network call with short retries for transient curl_cffi resets."""
    last_exc = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if idx < attempts - 1:
                time.sleep(delay * (idx + 1))
    raise last_exc


def get_stock_detail(ticker):
    """Fetch fundamental info metrics and full historical price history for a single stock ticker."""
    now = time.time()
    cached = _STOCK_DETAIL_CACHE.get(ticker)
    if cached and (now - cached["ts"]) < _STOCK_DETAIL_TTL_SEC:
        return cached["data"]

    with _STOCK_DETAIL_LOCK:
        cached = _STOCK_DETAIL_CACHE.get(ticker)
        if cached and (time.time() - cached["ts"]) < _STOCK_DETAIL_TTL_SEC:
            return cached["data"]

        stock = yf.Ticker(ticker)
        hist = _fetch_with_retry(lambda: stock.history(period="max"))

        info = {}
        try:
            info = _fetch_with_retry(lambda: stock.info) or {}
        except Exception as exc:
            print(f"Warning: yfinance info failed for {ticker}, using OHLCV fallback: {exc}")

        if not info and not hist.empty:
            recent_1y = hist.tail(252)
            close_s = hist["Close"].dropna()
            info = {
                "symbol": ticker,
                "currentPrice": float(close_s.iloc[-1]) if not close_s.empty else None,
                "fiftyTwoWeekHigh": float(recent_1y["High"].max()) if "High" in recent_1y else None,
                "fiftyTwoWeekLow": float(recent_1y["Low"].min()) if "Low" in recent_1y else None,
                "fiftyDayAverage": float(close_s.tail(50).mean()) if len(close_s) >= 1 else None,
                "twoHundredDayAverage": float(close_s.tail(200).mean()) if len(close_s) >= 1 else None,
                "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist and pd.notna(hist["Volume"].iloc[-1]) else None,
            }

        chart_data = []
        if not hist.empty:
            hist = hist.dropna(subset=["Close"]).reset_index()
            for _, row in hist.iterrows():
                date_val = row.get("Date") or row.get("Datetime")
                if not date_val:
                    continue
                date_str = (
                    date_val.strftime("%Y-%m-%d")
                    if hasattr(date_val, "strftime")
                    else str(date_val).split(" ")[0]
                )
                chart_data.append({"time": date_str, "value": float(row["Close"])})

        payload = {"info": info, "chart": chart_data}
        if chart_data or info:
            _STOCK_DETAIL_CACHE[ticker] = {"ts": time.time(), "data": payload}
        return payload
