"""Vectorized 100-Stock Global Panel Alpha Radar, Sector Rotation Quadrant & Macro Regime service."""

import numpy as np
import pandas as pd

from backend.config import FORECAST_HORIZONS, SECTOR_MAP, STOCK_LIST
from backend.services.feature_engineering import get_live_universe_panel, get_live_universe_panel_async
from backend.services.model_registry import registry
from backend.services.prediction_service import batch_score_universe_horizon
from backend.services.runtime import AsyncSingleflightCache, runtime

_RADAR_SINGLEFLIGHT = AsyncSingleflightCache(
    maxsize=16, default_ttl=300.0, min_refresh_interval=60.0
)


def _safe_float(val, default: float = 0.0, digits: int | None = None) -> float:
    """Return a finite JSON-safe float."""
    try:
        num = float(val)
        if not np.isfinite(num):
            num = float(default)
    except Exception:
        num = float(default)
    return round(num, digits) if digits is not None else num


def _resolve_active_horizon(requested_horizon: int) -> tuple[int, list[int]]:
    """Resolve requested horizon if loaded, else fall back cleanly to nearest available horizon."""
    available = [h for h in FORECAST_HORIZONS if registry.get_bundle(h) is not None]
    if requested_horizon in available:
        return requested_horizon, available
    if available:
        nearest = min(available, key=lambda h: abs(h - requested_horizon))
        return nearest, available
    return requested_horizon, []


def _classify_macro_regime(latest_df: pd.DataFrame) -> dict:
    """Derive Indian + lagged Global Macro Regime summary from the live cross-sectional panel."""
    if latest_df.empty:
        return {
            "regime_label": "Neutral Market Regime",
            "tone": "secondary",
            "india_vix": 15.0,
            "vix_change_5d_pct": 0.0,
            "nifty_5d_pct": 0.0,
            "nifty_21d_pct": 0.0,
            "bank_nifty_rel_5d_pct": 0.0,
            "it_nifty_rel_5d_pct": 0.0,
            "usdinr_5d_pct": 0.0,
            "brent_5d_pct": 0.0,
            "sp500_5d_pct": 0.0,
        }

    def _med(col: str, default: float = 0.0) -> float:
        if col not in latest_df.columns:
            return default
        vals = pd.to_numeric(latest_df[col], errors="coerce").dropna()
        return float(vals.median()) if len(vals) > 0 else default

    india_vix = _safe_float(_med("vix_level", 0.75) * 20.0, 15.0, 2)
    vix_chg = _safe_float((np.exp(_med("vix_change_5d", 0.0)) - 1.0) * 100.0, 0.0, 2)
    nifty_5d = _safe_float((np.exp(_med("nifty_ret_5d", 0.0)) - 1.0) * 100.0, 0.0, 2)
    nifty_21d = _safe_float((np.exp(_med("nifty_ret_21d", 0.0)) - 1.0) * 100.0, 0.0, 2)
    bank_rel = _safe_float(_med("bank_nifty_rel_5d", 0.0) * 100.0, 0.0, 2)
    it_rel = _safe_float(_med("it_nifty_rel_5d", 0.0) * 100.0, 0.0, 2)
    usdinr = _safe_float((np.exp(_med("usdinr_ret_5d_lag1", 0.0)) - 1.0) * 100.0, 0.0, 2)
    brent = _safe_float((np.exp(_med("brent_ret_5d_lag1", 0.0)) - 1.0) * 100.0, 0.0, 2)
    sp500 = _safe_float((np.exp(_med("sp500_ret_5d_lag1", 0.0)) - 1.0) * 100.0, 0.0, 2)

    if india_vix >= 20.0 or nifty_21d <= -4.0:
        regime_label = "High-Vol Defensive"
        tone = "down"
    elif india_vix >= 16.0:
        regime_label = "Elevated Volatility"
        tone = "secondary"
    elif nifty_21d >= 1.5 and india_vix < 15.0:
        regime_label = "Low-Vol Risk-On"
        tone = "up"
    else:
        regime_label = "Balanced Trend"
        tone = "up" if nifty_21d >= 0 else "secondary"

    return {
        "regime_label": regime_label,
        "tone": tone,
        "india_vix": india_vix,
        "vix_change_5d_pct": vix_chg,
        "nifty_5d_pct": nifty_5d,
        "nifty_21d_pct": nifty_21d,
        "bank_nifty_rel_5d_pct": bank_rel,
        "it_nifty_rel_5d_pct": it_rel,
        "usdinr_5d_pct": usdinr,
        "brent_5d_pct": brent,
        "sp500_5d_pct": sp500,
    }


def _batch_score_stocks(panel_df: pd.DataFrame, horizon: int):
    """Run vectorized batch scoring via shared prediction_service helper and format leaderboard rows."""
    batch = batch_score_universe_horizon(panel_df, horizon)
    ordered_tickers = batch["tickers"]
    latest_df = batch["latest_df"]
    models_ready = batch["models_ready"]
    if not ordered_tickers:
        return [], latest_df, models_ready

    ticker_to_name = {t: name for name, t in STOCK_LIST.items()}
    pred_log_ret = batch["pred_log_ret"]
    low_log_ret = batch["low_log_ret"]
    high_log_ret = batch["high_log_ret"]
    bullish_counts = batch["bullish_counts"]
    total_model_count = batch["total_model_count"]
    recent_vols = batch["recent_vols"]

    stocks_out = []
    for i, ticker in enumerate(ordered_tickers):
        row = latest_df.iloc[i]
        price = _safe_float(row["Close"], 0.0, 2)
        if price <= 0:
            continue

        day_chg_pct = _safe_float((np.exp(float(row.get("ret_1d", 0.0))) - 1.0) * 100.0, 0.0, 2)
        pred_ret_pct = _safe_float((np.exp(pred_log_ret[i]) - 1.0) * 100.0, 0.0, 2)
        pred_price = _safe_float(price * np.exp(pred_log_ret[i]), price, 2)
        lower_price = _safe_float(price * np.exp(low_log_ret[i]), price, 2)
        upper_price = _safe_float(price * np.exp(high_log_ret[i]), price, 2)

        upside = max(upper_price - price, 0.01)
        downside = max(price - lower_price, 0.01)
        rr_ratio = _safe_float(upside / downside, 1.0, 2)

        bull = int(bullish_counts[i])
        majority = max(bull, total_model_count - bull) if total_model_count > 0 else 0
        consensus_pct = round((majority / total_model_count) * 100.0, 0) if total_model_count > 0 else 0.0

        mom_rank = _safe_float((float(row.get("cs_rank_sharpe_mom_252d", 0.0)) + 0.5) * 100.0, 50.0, 1)
        resid_mom_rank = _safe_float((float(row.get("cs_rank_resid_mom_63d", 0.0)) + 0.5) * 100.0, 50.0, 1)
        sector_rank = _safe_float((float(row.get("sec_rank_ret_21d", 0.0)) + 0.5) * 100.0, 50.0, 1)

        stocks_out.append(
            {
                "name": ticker_to_name.get(ticker, ticker.split(".")[0]),
                "ticker": ticker,
                "sector": SECTOR_MAP.get(ticker, "Other"),
                "price": price,
                "day_change_pct": day_chg_pct,
                "predicted_return_pct": pred_ret_pct,
                "predicted_price": pred_price,
                "lower_price": lower_price,
                "upper_price": upper_price,
                "rr_ratio": rr_ratio,
                "bullish_models": bull,
                "total_models": total_model_count,
                "consensus_pct": consensus_pct,
                "consensus_direction": "bullish" if bull >= (total_model_count - bull) else "bearish",
                "mom_rank": mom_rank,
                "resid_mom_rank": resid_mom_rank,
                "sector_rank": sector_rank,
                "vol_ann_pct": _safe_float(recent_vols[i] * np.sqrt(252.0) * 100.0, 20.0, 1),
                "beta": _safe_float(row.get("beta_126d", 1.0), 1.0, 2),
                "rsi_14": _safe_float(float(row.get("rsi_14", 0.5)) * 100.0, 50.0, 1),
                "dist_52w_high_pct": _safe_float(float(row.get("dist_52w_high", 0.0)) * 100.0, 0.0, 1),
            }
        )

    stocks_out.sort(key=lambda s: (s["predicted_return_pct"], s["mom_rank"]), reverse=True)
    for rank_idx, item in enumerate(stocks_out, start=1):
        item["alpha_rank"] = rank_idx

    return stocks_out, latest_df, models_ready


def _build_sector_rotation(stocks_out: list[dict], latest_df: pd.DataFrame) -> list[dict]:
    """Aggregate 21d/63d sector momentum & AI forecasts into RRG-style quadrants."""
    if not stocks_out:
        return []

    sec_stats: dict[str, dict] = {}
    for s in stocks_out:
        sec = s["sector"]
        entry = sec_stats.setdefault(sec, {"stocks": [], "rets": [], "days": []})
        entry["stocks"].append(s)
        entry["rets"].append(s["predicted_return_pct"])
        entry["days"].append(s["day_change_pct"])

    panel_sec_mom = {}
    if not latest_df.empty and "sector" in latest_df.columns:
        for sec, grp in latest_df.groupby("sector"):
            m21 = grp["sector_mom_21d"].mean() * 100.0 if "sector_mom_21d" in grp else grp["ret_21d"].mean() * 100.0
            m63 = grp["sector_mom_63d"].mean() * 100.0 if "sector_mom_63d" in grp else grp["ret_63d"].mean() * 100.0
            panel_sec_mom[sec] = (_safe_float(m21, 0.0, 2), _safe_float(m63, 0.0, 2))

    sectors_out = []
    for sec, data in sec_stats.items():
        mom_21d, mom_63d = panel_sec_mom.get(sec, (0.0, 0.0))
        if mom_21d >= 0 and mom_63d >= 0:
            quadrant = "Leading"
        elif mom_21d >= 0 and mom_63d < 0:
            quadrant = "Improving"
        elif mom_21d < 0 and mom_63d >= 0:
            quadrant = "Weakening"
        else:
            quadrant = "Lagging"

        top_stock = max(data["stocks"], key=lambda x: x["predicted_return_pct"])
        sectors_out.append(
            {
                "sector": sec,
                "stock_count": len(data["stocks"]),
                "avg_pred_ret_pct": _safe_float(np.mean(data["rets"]), 0.0, 2),
                "avg_day_change_pct": _safe_float(np.mean(data["days"]), 0.0, 2),
                "mom_21d_pct": mom_21d,
                "mom_63d_pct": mom_63d,
                "quadrant": quadrant,
                "top_pick": {
                    "name": top_stock["name"],
                    "ticker": top_stock["ticker"],
                    "predicted_return_pct": top_stock["predicted_return_pct"],
                },
            }
        )

    sectors_out.sort(key=lambda x: x["avg_pred_ret_pct"], reverse=True)
    return sectors_out


def _assemble_alpha_radar(panel_df: pd.DataFrame | None, req_horizon: int, active_horizon: int, available_horizons: list[int]) -> dict:
    if panel_df is None or panel_df.empty:
        return {
            "requested_horizon": req_horizon,
            "active_horizon": active_horizon,
            "available_horizons": available_horizons,
            "models_ready": len(available_horizons) > 0,
            "as_of": None,
            "macro_regime": _classify_macro_regime(pd.DataFrame()),
            "sectors": [],
            "stocks": [],
        }

    stocks_out, latest_df, models_ready = _batch_score_stocks(panel_df, active_horizon)
    sectors_out = _build_sector_rotation(stocks_out, latest_df)
    macro_regime = _classify_macro_regime(latest_df)

    as_of = None
    if not latest_df.empty and "date" in latest_df.columns:
        try:
            as_of = pd.Timestamp(latest_df["date"].max()).strftime("%Y-%m-%d")
        except Exception:
            as_of = None

    return {
        "requested_horizon": req_horizon,
        "active_horizon": active_horizon,
        "available_horizons": available_horizons,
        "models_ready": models_ready,
        "as_of": as_of,
        "macro_regime": macro_regime,
        "sectors": sectors_out,
        "stocks": stocks_out,
    }


async def get_alpha_radar_async(horizon: int = 21, force_refresh: bool = False) -> dict:
    """Async cached & singleflight Alpha Radar computation."""
    try:
        req_horizon = int(horizon)
    except Exception:
        req_horizon = 21

    active_horizon, available_horizons = _resolve_active_horizon(req_horizon)
    cache_key = (active_horizon, len(available_horizons))

    async def _compute():
        panel_df, _, _ = await get_live_universe_panel_async(force_refresh=force_refresh)
        return await runtime.run_cpu(
            _assemble_alpha_radar, panel_df, req_horizon, active_horizon, available_horizons
        )

    return await _RADAR_SINGLEFLIGHT.get_or_compute(
        key=cache_key,
        factory=_compute,
        ttl=300.0,
        force_refresh=force_refresh,
        stale_on_error=True,
    )


def get_alpha_radar(horizon: int = 21, force_refresh: bool = False) -> dict:
    """Synchronous helper for standalone scripts."""
    try:
        req_horizon = int(horizon)
    except Exception:
        req_horizon = 21
    active_horizon, available_horizons = _resolve_active_horizon(req_horizon)
    panel_df, _, _ = get_live_universe_panel(force_refresh=force_refresh)
    return _assemble_alpha_radar(panel_df, req_horizon, active_horizon, available_horizons)
