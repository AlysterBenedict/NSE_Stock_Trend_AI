"""Production-grade stationary, cross-sectional, and macro-regime feature pipeline.

Shared identically by Global Cross-Sectional Panel training and live API inference
with strict causality (global US/commodity drivers are lagged by 1 trading day).
"""

from datetime import datetime, timedelta
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

from backend.config import (
    GLOBAL_LAGGED_MACRO_TICKERS,
    INDIAN_MACRO_TICKERS,
    SECTOR_MAP,
    SECTOR_TO_ID,
    STOCK_LIST,
    TICKER_TO_ID,
)
from backend.security.logging_utils import get_logger
from backend.services.runtime import AsyncSingleflightCache, BoundedTTLCache, runtime

logger = get_logger(__name__)

EPS = 1e-8

# Per-stock + Indian/Global macro stationary features (95 features - DO NOT REORDER)
BASE_FEATURE_COLUMNS = [
    # Multi-horizon log returns & volatility-normalized Sharpe momentum
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_adj_ret_1d", "vol_adj_ret_5d", "vol_adj_ret_21d",
    # 52-week high/low proximity (George & Hwang momentum factor)
    "dist_52w_high", "dist_52w_low",
    # Candlestick & intraday microstructure
    "overnight_gap", "intraday_ret", "hl_range", "clv", "body_ratio",
    # Trend ratios & moving average spreads
    "close_ema10_ratio", "close_ema20_ratio", "close_ema50_ratio", "close_ema200_ratio",
    "ema10_ema50_spread", "ema20_ema200_spread", "sma50_slope_5d",
    # Momentum oscillators
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "rsi_7", "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "roc_10", "roc_21", "adx_14",
    # Volatility, skewness, downside risk & mean-reversion
    "vol_5d", "vol_10d", "vol_21d", "vol_63d", "vol_ratio_5_21",
    "garman_klass_vol_14", "atr_14_norm", "bb_pct_b", "bb_width", "zscore_20",
    "skew_21d", "downside_vol_ratio_21d",
    # Volume, flow & Amihud illiquidity
    "log_vol_change", "vol_sma20_ratio", "obv_slope_10", "mfi_14", "vwap_20_dev", "amihud_21d",
    # Indian Market Regime (^NSEI, ^INDIAVIX, ^NSEBANK, ^CNXIT)
    "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_21d", "nifty_ret_63d", "nifty_vol_21d",
    "rel_strength_5d", "rel_strength_21d", "rel_strength_63d", "beta_60d", "idio_vol_21d",
    "vix_level", "vix_change_5d", "bank_nifty_rel_5d", "it_nifty_rel_5d",
    # Lagged Global Cross-Asset Macro Drivers (strictly .shift(1) -> zero lookahead)
    "sp500_ret_1d_lag1", "sp500_ret_5d_lag1", "us10y_chg_5d_lag1",
    "usdinr_ret_5d_lag1", "brent_ret_5d_lag1", "gold_ret_5d_lag1",
    # Calendar & turn-of-month institutional flow cyclicality
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_month_end",
    # Institutional multi-horizon alpha factors
    "mom_12_1", "mom_6_1", "mom_3_1",
    "vol_126d", "sharpe_mom_63d", "sharpe_mom_126d", "sharpe_mom_252d",
    "beta_126d", "idio_vol_63d", "resid_mom_63d", "resid_mom_126d",
    "pct_pos_days_63d", "pct_pos_days_126d",
    "max_dd_63d", "vol_term_ratio",
]

# Cross-sectional percentile ranks ([-0.5, +0.5]) & sector-demeaned returns across the NSE universe
CROSS_SECTIONAL_COLUMNS = [
    "cs_rank_ret_1d",
    "cs_rank_ret_5d",
    "cs_rank_ret_21d",
    "cs_rank_ret_63d",
    "cs_rank_ret_126d",
    "cs_rank_vol_21d",
    "cs_rank_rsi_14",
    "cs_rank_vol_ratio",
    "cs_rank_dist_52w_high",
    "cs_rank_zscore_20",
    "sector_excess_ret_1d",
    "sector_excess_ret_5d",
    "sector_excess_ret_21d",
    "cs_rank_mom_12_1",
    "cs_rank_mom_6_1",
    "cs_rank_mom_3_1",
    "cs_rank_sharpe_mom_63d",
    "cs_rank_sharpe_mom_126d",
    "cs_rank_sharpe_mom_252d",
    "cs_rank_resid_mom_63d",
    "cs_rank_resid_mom_126d",
    "cs_rank_pct_pos_63d",
    "cs_rank_pct_pos_126d",
    "cs_rank_vol_63d",
    "cs_rank_vol_126d",
    "cs_rank_idio_vol_63d",
    "cs_rank_beta_126d",
    "cs_rank_max_dd_63d",
    "cs_rank_vol_term_ratio",
    "cs_rank_amihud_21d",
    "cs_rank_close_ema200",
    "sec_rank_ret_5d",
    "sec_rank_ret_21d",
    "sec_rank_mom_6_1",
    "sec_rank_vol_63d",
    "sec_rank_sharpe_mom_126d",
    "sector_mom_21d",
    "sector_mom_63d",
]

LONG_HORIZON_CANDIDATE_COLUMNS = CROSS_SECTIONAL_COLUMNS + [
    "vol_adj_ret_5d",
    "vol_adj_ret_21d",
    "sharpe_mom_63d",
    "sharpe_mom_126d",
    "sharpe_mom_252d",
    "resid_mom_63d",
    "resid_mom_126d",
    "pct_pos_days_63d",
    "pct_pos_days_126d",
    "dist_52w_high",
    "dist_52w_low",
    "max_dd_63d",
    "vol_term_ratio",
    "close_ema50_ratio",
    "close_ema200_ratio",
    "ema20_ema200_spread",
    "rel_strength_21d",
    "rel_strength_63d",
    "beta_126d",
    "downside_vol_ratio_21d",
    "skew_21d",
    "mfi_14",
    "adx_14",
    "vwap_20_dev",
]

# Canonical ordered feature list for all tabular & sequence models (133 features)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CROSS_SECTIONAL_COLUMNS

_CACHE_TTL_SECONDS = 1800
_MACRO_CACHE = BoundedTTLCache(maxsize=8, default_ttl=_CACHE_TTL_SECONDS)
_MACRO_LOCK = threading.Lock()
_UNIVERSE_STATE = {"timestamp": 0.0, "panel": None, "raw_map": None, "macro": None}
_UNIVERSE_SYNC_LOCK = threading.RLock()
_UNIVERSE_SINGLEFLIGHT = AsyncSingleflightCache(
    maxsize=2, default_ttl=_CACHE_TTL_SECONDS, min_refresh_interval=60.0
)


def _flatten_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Flatten MultiIndex columns returned by yfinance, deduplicate, and strip timezone info."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.dropna(axis=0, how="all")
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def _download_single_macro_symbol(symbol: str, start_str: str, end_str: str) -> pd.DataFrame | None:
    try:
        try:
            raw_dl = yf.download(symbol, start=start_str, end=end_str, progress=False, timeout=25)
        except TypeError:
            raw_dl = yf.download(symbol, start=start_str, end=end_str, progress=False)
        return _flatten_columns(raw_dl)
    except Exception as exc:
        logger.warning("Macro fallback download failed for %s: %s", symbol, exc)
        return None


def fetch_macro_context(start_date=None, end_date=None, use_cache: bool = True) -> pd.DataFrame:
    """Download Indian market indices + lagged global macro series into a single date-indexed DataFrame."""
    end_dt = pd.to_datetime(end_date) if end_date is not None else datetime.now()
    start_dt = pd.to_datetime(start_date) if start_date is not None else end_dt - timedelta(days=950)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    cache_key = (start_str, end_str)

    if use_cache:
        hit, cached_df, _ = _MACRO_CACHE.get_fresh(cache_key)
        if hit and cached_df is not None:
            return cached_df.copy()

    all_symbols = {**INDIAN_MACRO_TICKERS, **GLOBAL_LAGGED_MACRO_TICKERS}
    symbol_list = list(all_symbols.values())

    with _MACRO_LOCK:
        if use_cache:
            hit, cached_df, _ = _MACRO_CACHE.get_fresh(cache_key)
            if hit and cached_df is not None:
                return cached_df.copy()

        macro = pd.DataFrame()
        bulk = None
        try:
            with runtime.yfinance_lock:
                try:
                    bulk = yf.download(
                        symbol_list,
                        start=start_str,
                        end=end_str,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                        timeout=25,
                    )
                except TypeError:
                    bulk = yf.download(
                        symbol_list,
                        start=start_str,
                        end=end_str,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                    )
        except Exception as exc:
            logger.warning("Macro bulk download error, falling back to single tickers: %s", exc)
            bulk = None

        lvl0 = set(bulk.columns.get_level_values(0)) if bulk is not None and isinstance(bulk.columns, pd.MultiIndex) else set()
        for col_key, symbol in all_symbols.items():
            raw = None
            if symbol in lvl0:
                raw = _flatten_columns(bulk[symbol])
            if raw is None or raw.empty or "Close" not in raw:
                with runtime.yfinance_lock:
                    raw = _download_single_macro_symbol(symbol, start_str, end_str)
            if raw is not None and not raw.empty and "Close" in raw:
                series = raw["Close"].astype(float)
                if col_key in GLOBAL_LAGGED_MACRO_TICKERS:
                    series = series.shift(1)
                macro[f"{col_key}_close"] = series

        if use_cache and not macro.empty:
            _MACRO_CACHE.set(cache_key, macro.copy(), ttl=_CACHE_TTL_SECONDS)

        return macro


def _compute_rsi(close_series: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI normalized to [0, 1]."""
    delta = close_series.diff(1)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    return (100.0 - (100.0 / (1.0 + rs))) / 100.0


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Compute ATR(14) and normalized ADX(14) in [0, 1]."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high.diff(1)
    down_move = -low.diff(1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=close.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / (atr + EPS)
    minus_di = 100.0 * pd.Series(minus_dm, index=close.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / (atr + EPS)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr, adx / 100.0


def _aligned_log_ret(macro_df: pd.DataFrame | None, col_name: str, target_index: pd.Index, periods: int) -> pd.Series:
    """Safely reindex a macro close series onto target_index and compute log return."""
    if macro_df is None or macro_df.empty or col_name not in macro_df.columns:
        return pd.Series(0.0, index=target_index)
    aligned = macro_df[col_name].reindex(target_index).ffill()
    return np.log(aligned.clip(lower=EPS) / aligned.shift(periods).clip(lower=EPS)).fillna(0.0)


def build_single_stock_features(raw_df: pd.DataFrame, ticker: str | None = None, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute all 95 single-stock + macro stationary features (before cross-sectional ranking)."""
    df = _flatten_columns(raw_df)
    if df is None or df.empty:
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy().astype(float)
    open_p, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
    volume = df["Volume"].clip(lower=0.0)
    prev_close = close.shift(1)

    log_close = np.log(close.clip(lower=EPS))
    df["ret_1d"] = log_close.diff(1)
    df["ret_2d"] = log_close.diff(2)
    df["ret_3d"] = log_close.diff(3)
    df["ret_5d"] = log_close.diff(5)
    df["ret_10d"] = log_close.diff(10)
    df["ret_21d"] = log_close.diff(21)
    df["ret_63d"] = log_close.diff(63)
    df["ret_126d"] = log_close.diff(126)
    df["ret_252d"] = log_close.diff(252)

    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    df["vol_63d"] = df["ret_1d"].rolling(63).std()
    df["vol_ratio_5_21"] = df["vol_5d"] / (df["vol_21d"] + EPS)

    safe_vol21 = df["vol_21d"].clip(lower=0.005)
    df["vol_adj_ret_1d"] = df["ret_1d"] / safe_vol21
    df["vol_adj_ret_5d"] = df["ret_5d"] / (safe_vol21 * np.sqrt(5.0))
    df["vol_adj_ret_21d"] = df["ret_21d"] / (safe_vol21 * np.sqrt(21.0))

    high_252 = high.rolling(252, min_periods=60).max()
    low_252 = low.rolling(252, min_periods=60).min()
    df["dist_52w_high"] = (close / (high_252 + EPS)) - 1.0
    df["dist_52w_low"] = (close / (low_252 + EPS)) - 1.0

    df["overnight_gap"] = np.log(open_p.clip(lower=EPS) / prev_close.clip(lower=EPS))
    df["intraday_ret"] = np.log(close.clip(lower=EPS) / open_p.clip(lower=EPS))
    df["hl_range"] = (high - low) / (close + EPS)
    df["clv"] = ((close - low) - (high - close)) / (high - low + EPS)
    df["body_ratio"] = (close - open_p) / (high - low + EPS)

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    sma50 = close.rolling(50).mean()

    df["close_ema10_ratio"] = (close / (ema10 + EPS)) - 1.0
    df["close_ema20_ratio"] = (close / (ema20 + EPS)) - 1.0
    df["close_ema50_ratio"] = (close / (ema50 + EPS)) - 1.0
    df["close_ema200_ratio"] = (close / (ema200 + EPS)) - 1.0
    df["ema10_ema50_spread"] = (ema10 / (ema50 + EPS)) - 1.0
    df["ema20_ema200_spread"] = (ema20 / (ema200 + EPS)) - 1.0
    df["sma50_slope_5d"] = (sma50 / (sma50.shift(5) + EPS)) - 1.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_norm"] = macd / (close + EPS)
    df["macd_signal_norm"] = macd_signal / (close + EPS)
    df["macd_hist_norm"] = (macd - macd_signal) / (close + EPS)

    df["rsi_7"] = _compute_rsi(close, 7)
    df["rsi_14"] = _compute_rsi(close, 14)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14 + EPS)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()
    df["williams_r"] = (high14 - close) / (high14 - low14 + EPS)
    df["roc_10"] = (close / (close.shift(10) + EPS)) - 1.0
    df["roc_21"] = (close / (close.shift(21) + EPS)) - 1.0

    atr14, adx14 = _compute_adx(high, low, close, period=14)
    df["adx_14"] = adx14

    log_hl = np.log(high.clip(lower=EPS) / low.clip(lower=EPS))
    log_co = np.log(close.clip(lower=EPS) / open_p.clip(lower=EPS))
    gk_var = (0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)).clip(lower=0.0)
    df["garman_klass_vol_14"] = np.sqrt(gk_var.rolling(14).mean())
    df["atr_14_norm"] = atr14 / (close + EPS)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    df["bb_pct_b"] = (close - lower_bb) / (upper_bb - lower_bb + EPS)
    df["bb_width"] = (upper_bb - lower_bb) / (sma20 + EPS)
    df["zscore_20"] = (close - sma20) / (std20 + EPS)
    df["skew_21d"] = df["ret_1d"].rolling(21).skew().fillna(0.0)
    neg_sq = np.where(df["ret_1d"] < 0, df["ret_1d"] ** 2, 0.0)
    downside_vol = np.sqrt(pd.Series(neg_sq, index=df.index).rolling(21).mean())
    df["downside_vol_ratio_21d"] = downside_vol / (df["vol_21d"] + EPS)

    log_vol = np.log1p(volume)
    df["log_vol_change"] = log_vol.diff(1)
    vol_sma20 = volume.rolling(20).mean()
    df["vol_sma20_ratio"] = np.log1p(volume / (vol_sma20 + EPS))

    signed_vol = np.sign(close.diff(1).fillna(0.0)) * volume
    obv = signed_vol.cumsum()
    df["obv_slope_10"] = (obv - obv.shift(10)) / (volume.rolling(10).sum() + EPS)

    typical_price = (high + low + close) / 3.0
    raw_mf = typical_price * volume
    tp_diff = typical_price.diff(1)
    pos_sum = pd.Series(np.where(tp_diff > 0, raw_mf, 0.0), index=df.index).rolling(14).sum()
    neg_sum = pd.Series(np.where(tp_diff < 0, raw_mf, 0.0), index=df.index).rolling(14).sum()
    df["mfi_14"] = pos_sum / (pos_sum + neg_sum + EPS)

    vwap_20 = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + EPS)
    df["vwap_20_dev"] = (close / (vwap_20 + EPS)) - 1.0

    dollar_vol = (close * volume).clip(lower=1.0)
    df["amihud_21d"] = (df["ret_1d"].abs() / np.log1p(dollar_vol)).rolling(21).mean() * 100.0

    df["nifty_ret_1d"] = _aligned_log_ret(macro_df, "nifty_close", df.index, 1)
    df["nifty_ret_5d"] = _aligned_log_ret(macro_df, "nifty_close", df.index, 5)
    df["nifty_ret_21d"] = _aligned_log_ret(macro_df, "nifty_close", df.index, 21)
    df["nifty_ret_63d"] = _aligned_log_ret(macro_df, "nifty_close", df.index, 63)
    df["nifty_vol_21d"] = df["nifty_ret_1d"].rolling(21).std().fillna(0.0)

    df["rel_strength_5d"] = df["ret_5d"] - df["nifty_ret_5d"]
    df["rel_strength_21d"] = df["ret_21d"] - df["nifty_ret_21d"]
    df["rel_strength_63d"] = df["ret_63d"] - df["nifty_ret_63d"]

    cov_60 = df["ret_1d"].rolling(60).cov(df["nifty_ret_1d"])
    var_60 = df["nifty_ret_1d"].rolling(60).var()
    df["beta_60d"] = (cov_60 / (var_60 + EPS)).clip(-3.0, 4.0).fillna(1.0)
    idio_ret = df["ret_1d"] - df["beta_60d"] * df["nifty_ret_1d"]
    df["idio_vol_21d"] = idio_ret.rolling(21).std().fillna(df["vol_21d"])

    if macro_df is not None and not macro_df.empty and "vix_close" in macro_df.columns:
        aligned_vix = macro_df["vix_close"].reindex(df.index).ffill().bfill()
        df["vix_level"] = (aligned_vix / 20.0).fillna(1.0)
        df["vix_change_5d"] = np.log(
            aligned_vix.clip(lower=EPS) / aligned_vix.shift(5).clip(lower=EPS)
        ).fillna(0.0)
    else:
        df["vix_level"] = 1.0
        df["vix_change_5d"] = 0.0

    bank_ret_5d = _aligned_log_ret(macro_df, "bank_nifty_close", df.index, 5)
    it_ret_5d = _aligned_log_ret(macro_df, "it_nifty_close", df.index, 5)
    df["bank_nifty_rel_5d"] = bank_ret_5d - df["nifty_ret_5d"]
    df["it_nifty_rel_5d"] = it_ret_5d - df["nifty_ret_5d"]

    df["sp500_ret_1d_lag1"] = _aligned_log_ret(macro_df, "sp500_close", df.index, 1)
    df["sp500_ret_5d_lag1"] = _aligned_log_ret(macro_df, "sp500_close", df.index, 5)
    df["us10y_chg_5d_lag1"] = _aligned_log_ret(macro_df, "us10y_close", df.index, 5)
    df["usdinr_ret_5d_lag1"] = _aligned_log_ret(macro_df, "usdinr_close", df.index, 5)
    df["brent_ret_5d_lag1"] = _aligned_log_ret(macro_df, "brent_close", df.index, 5)
    df["gold_ret_5d_lag1"] = _aligned_log_ret(macro_df, "gold_close", df.index, 5)

    dow = df.index.dayofweek.astype(float)
    month = df.index.month.astype(float)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    df["is_month_end"] = ((df.index.day >= 27) | (df.index.day <= 3)).astype(float)

    df["mom_12_1"] = df["ret_252d"] - df["ret_21d"]
    df["mom_6_1"] = df["ret_126d"] - df["ret_21d"]
    df["mom_3_1"] = df["ret_63d"] - df["ret_5d"]

    df["vol_126d"] = df["ret_1d"].rolling(126).std()
    safe_vol63 = df["vol_63d"].clip(lower=0.005)
    safe_vol126 = df["vol_126d"].clip(lower=0.005)
    df["sharpe_mom_63d"] = df["ret_63d"] / (safe_vol63 * np.sqrt(63.0))
    df["sharpe_mom_126d"] = df["mom_6_1"] / (safe_vol126 * np.sqrt(105.0))
    df["sharpe_mom_252d"] = df["mom_12_1"] / (safe_vol126 * np.sqrt(231.0))

    cov_126 = df["ret_1d"].rolling(126).cov(df["nifty_ret_1d"])
    var_126 = df["nifty_ret_1d"].rolling(126).var()
    df["beta_126d"] = (cov_126 / (var_126 + EPS)).clip(-3.0, 4.0).fillna(1.0)
    idio_ret_126 = df["ret_1d"] - df["beta_126d"] * df["nifty_ret_1d"]
    df["idio_vol_63d"] = idio_ret_126.rolling(63).std().fillna(df["vol_63d"])
    safe_idio63 = df["idio_vol_63d"].clip(lower=0.005)
    nifty_ret_126d = _aligned_log_ret(macro_df, "nifty_close", df.index, 126)
    df["resid_mom_63d"] = (df["ret_63d"] - df["beta_126d"] * df["nifty_ret_63d"]) / (safe_idio63 * np.sqrt(63.0))
    df["resid_mom_126d"] = (df["ret_126d"] - df["beta_126d"] * nifty_ret_126d) / (safe_idio63 * np.sqrt(126.0))

    pos_day = (df["ret_1d"] > 0).astype(float)
    df["pct_pos_days_63d"] = pos_day.rolling(63).mean() - 0.5
    df["pct_pos_days_126d"] = pos_day.rolling(126).mean() - 0.5

    high_63 = close.rolling(63, min_periods=20).max()
    df["max_dd_63d"] = (close / (high_63 + EPS)) - 1.0
    df["vol_term_ratio"] = (df["vol_21d"] / (safe_vol126 + EPS)).clip(0.2, 5.0) - 1.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=BASE_FEATURE_COLUMNS, inplace=True)
    df[BASE_FEATURE_COLUMNS] = df[BASE_FEATURE_COLUMNS].clip(-10.0, 10.0)

    if ticker is not None:
        df["ticker"] = ticker
        df["stock_id"] = TICKER_TO_ID.get(ticker, len(TICKER_TO_ID))
        sector = SECTOR_MAP.get(ticker, "Other")
        df["sector"] = sector
        df["sector_id"] = SECTOR_TO_ID.get(sector, SECTOR_TO_ID["Other"])

    return df


def add_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly same-date vectorized cross-sectional ranks and sector-excess returns."""
    if panel_df.empty:
        return panel_df

    df = panel_df.copy()
    date_group = df.groupby("date")

    rank_specs = [
        ("ret_1d", "cs_rank_ret_1d"),
        ("ret_5d", "cs_rank_ret_5d"),
        ("ret_21d", "cs_rank_ret_21d"),
        ("ret_63d", "cs_rank_ret_63d"),
        ("ret_126d", "cs_rank_ret_126d"),
        ("vol_21d", "cs_rank_vol_21d"),
        ("rsi_14", "cs_rank_rsi_14"),
        ("vol_sma20_ratio", "cs_rank_vol_ratio"),
        ("dist_52w_high", "cs_rank_dist_52w_high"),
        ("zscore_20", "cs_rank_zscore_20"),
        ("mom_12_1", "cs_rank_mom_12_1"),
        ("mom_6_1", "cs_rank_mom_6_1"),
        ("mom_3_1", "cs_rank_mom_3_1"),
        ("sharpe_mom_63d", "cs_rank_sharpe_mom_63d"),
        ("sharpe_mom_126d", "cs_rank_sharpe_mom_126d"),
        ("sharpe_mom_252d", "cs_rank_sharpe_mom_252d"),
        ("resid_mom_63d", "cs_rank_resid_mom_63d"),
        ("resid_mom_126d", "cs_rank_resid_mom_126d"),
        ("pct_pos_days_63d", "cs_rank_pct_pos_63d"),
        ("pct_pos_days_126d", "cs_rank_pct_pos_126d"),
        ("vol_63d", "cs_rank_vol_63d"),
        ("vol_126d", "cs_rank_vol_126d"),
        ("idio_vol_63d", "cs_rank_idio_vol_63d"),
        ("beta_126d", "cs_rank_beta_126d"),
        ("max_dd_63d", "cs_rank_max_dd_63d"),
        ("vol_term_ratio", "cs_rank_vol_term_ratio"),
        ("amihud_21d", "cs_rank_amihud_21d"),
        ("close_ema200_ratio", "cs_rank_close_ema200"),
    ]
    src_cols = [src for src, _ in rank_specs]
    dst_cols = [dst for _, dst in rank_specs]
    df[dst_cols] = (date_group[src_cols].rank(pct=True).fillna(0.5) - 0.5).to_numpy()

    sec_group = df.groupby(["date", "sector"])
    for ret_col, dst_col in [
        ("ret_1d", "sector_excess_ret_1d"),
        ("ret_5d", "sector_excess_ret_5d"),
        ("ret_21d", "sector_excess_ret_21d"),
    ]:
        df[dst_col] = (df[ret_col] - sec_group[ret_col].transform("mean")).clip(-5.0, 5.0).fillna(0.0)

    sec_counts = sec_group["ret_1d"].transform("count").to_numpy()
    sec_specs = [
        ("ret_5d", "sec_rank_ret_5d"),
        ("ret_21d", "sec_rank_ret_21d"),
        ("mom_6_1", "sec_rank_mom_6_1"),
        ("vol_63d", "sec_rank_vol_63d"),
        ("sharpe_mom_126d", "sec_rank_sharpe_mom_126d"),
    ]
    sec_src = [src for src, _ in sec_specs]
    sec_dst = [dst for _, dst in sec_specs]
    sec_ranks = sec_group[sec_src].rank(method="average").to_numpy()
    denom = np.maximum(sec_counts - 1.0, 1.0)[:, None]
    norm_sec = np.where(sec_counts[:, None] > 1, (sec_ranks - 1.0) / denom - 0.5, 0.0)
    df[sec_dst] = np.nan_to_num(norm_sec, nan=0.0)

    for ret_col, dst_col in [("ret_21d", "sector_mom_21d"), ("ret_63d", "sector_mom_63d")]:
        sec_mean = sec_group[ret_col].transform("mean")
        mkt_med = date_group[ret_col].transform("median")
        df[dst_col] = (sec_mean - mkt_med).clip(-2.0, 2.0).fillna(0.0)

    df[CROSS_SECTIONAL_COLUMNS] = df[CROSS_SECTIONAL_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def build_panel_dataframe(stock_raw_map: dict[str, pd.DataFrame], macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the complete cross-sectional panel DataFrame sorted by (date, ticker)."""
    frames = []
    for ticker, raw_df in stock_raw_map.items():
        single = build_single_stock_features(raw_df, ticker=ticker, macro_df=macro_df)
        if single.empty:
            continue
        single.index.name = "date"
        single = single.reset_index()
        single["date"] = pd.to_datetime(single["date"])
        frames.append(single)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = add_cross_sectional_features(panel)
    panel.sort_values(["date", "ticker"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


def build_feature_dataframe(raw_df: pd.DataFrame, macro_df: pd.DataFrame | None = None, ticker: str | None = None) -> pd.DataFrame:
    """Build full FEATURE_COLUMNS for a single stock (neutral cross-sectional fallback if panel unavailable)."""
    df = build_single_stock_features(raw_df, ticker=ticker, macro_df=macro_df)
    if df.empty:
        return df
    for col in CROSS_SECTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def get_cached_ticker_latest_date(ticker: str):
    """Return the latest cached timestamp for a ticker from the universe cache, if warm."""
    with _UNIVERSE_SYNC_LOCK:
        raw_map = _UNIVERSE_STATE.get("raw_map")
        if not raw_map:
            return None
        df = raw_map.get(ticker)
        if df is None or df.empty:
            return None
        return df.index[-1]


def _refresh_live_universe_panel_sync() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Synchronously download universe OHLCV + macro and construct the cross-sectional panel."""
    tickers = sorted(set(STOCK_LIST.values()))
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=950)

    macro_df = fetch_macro_context(start_date=start_dt - timedelta(days=30), end_date=end_dt, use_cache=True)
    bulk = None
    for attempt in range(2):
        try:
            with runtime.yfinance_lock:
                try:
                    bulk = yf.download(
                        tickers,
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        group_by="ticker",
                        threads=True,
                        progress=False,
                        timeout=30,
                    )
                except TypeError:
                    bulk = yf.download(
                        tickers,
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        group_by="ticker",
                        threads=True,
                        progress=False,
                    )
            if bulk is not None and not bulk.empty:
                break
        except Exception as exc:
            logger.warning("Universe bulk download attempt %d/2 failed: %s", attempt + 1, exc)
        if attempt == 0:
            time.sleep(0.5)

    raw_map: dict[str, pd.DataFrame] = {}
    if bulk is not None and not bulk.empty:
        if isinstance(bulk.columns, pd.MultiIndex):
            lvl0 = set(bulk.columns.get_level_values(0))
            for t in tickers:
                if t in lvl0:
                    sub = bulk[t].dropna(how="all")
                    if not sub.empty:
                        raw_map[t] = sub
        else:
            raw_map[tickers[0]] = bulk.dropna(how="all")

    with _UNIVERSE_SYNC_LOCK:
        if not raw_map and _UNIVERSE_STATE["panel"] is not None:
            return _UNIVERSE_STATE["panel"], _UNIVERSE_STATE["raw_map"], _UNIVERSE_STATE["macro"]

    panel = build_panel_dataframe(raw_map, macro_df=macro_df)
    now = time.monotonic()
    with _UNIVERSE_SYNC_LOCK:
        _UNIVERSE_STATE["timestamp"] = now
        _UNIVERSE_STATE["panel"] = panel
        _UNIVERSE_STATE["raw_map"] = raw_map
        _UNIVERSE_STATE["macro"] = macro_df
    return panel, raw_map, macro_df


def get_live_universe_panel(force_refresh: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Thread-safe cached 2.5-year cross-sectional panel across all STOCK_LIST tickers."""
    with _UNIVERSE_SYNC_LOCK:
        now = time.monotonic()
        if (
            not force_refresh
            and _UNIVERSE_STATE["panel"] is not None
            and (now - _UNIVERSE_STATE["timestamp"]) < _CACHE_TTL_SECONDS
        ):
            return _UNIVERSE_STATE["panel"], _UNIVERSE_STATE["raw_map"], _UNIVERSE_STATE["macro"]

        if force_refresh and _UNIVERSE_STATE["panel"] is not None and (now - _UNIVERSE_STATE["timestamp"]) < 60.0:
            return _UNIVERSE_STATE["panel"], _UNIVERSE_STATE["raw_map"], _UNIVERSE_STATE["macro"]

    return _refresh_live_universe_panel_sync()


async def get_live_universe_panel_async(force_refresh: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Non-blocking singleflight cached cross-sectional panel loader."""
    return await _UNIVERSE_SINGLEFLIGHT.get_or_compute(
        key="live_universe_panel",
        factory=lambda: runtime.run_io(_refresh_live_universe_panel_sync),
        ttl=_CACHE_TTL_SECONDS,
        force_refresh=force_refresh,
        stale_on_error=True,
    )
