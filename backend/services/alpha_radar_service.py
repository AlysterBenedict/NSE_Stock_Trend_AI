"""Vectorized 100-Stock Global Panel Alpha Radar, Sector Rotation Quadrant & Macro Regime service."""

import threading
import time
import numpy as np
import pandas as pd
import torch

from backend.config import (
    BASE_ALGORITHMS,
    FORECAST_HORIZONS,
    LOOKBACK_WINDOW,
    SECTORS,
    SECTOR_MAP,
    SECTOR_TO_ID,
    SORTED_TICKERS,
    STOCK_LIST,
    TICKER_TO_ID,
)
from backend.services.feature_engineering import FEATURE_COLUMNS, get_live_universe_panel
from backend.services.model_registry import registry

_RADAR_CACHE = {}
_RADAR_LOCK = threading.Lock()
_RADAR_TTL_SECONDS = 300

_FALLBACK_HORIZON_ORDER = [21, 5, 10, 63, 1]


def _safe_float(val, default=0.0, digits=None):
    """Return a finite JSON-safe float."""
    try:
        num = float(val)
        if not np.isfinite(num):
            num = float(default)
    except Exception:
        num = float(default)
    return round(num, digits) if digits is not None else num


def _resolve_active_horizon(requested_horizon):
    """Resolve requested horizon if loaded, else fall back cleanly to nearest available horizon."""
    available = [h for h in FORECAST_HORIZONS if registry.get_bundle(h) is not None]
    if requested_horizon in available:
        return requested_horizon, available
    if available:
        nearest = min(available, key=lambda h: abs(h - requested_horizon))
        return nearest, available
    return requested_horizon, []


def _classify_macro_regime(latest_df):
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

    def _med(col, default=0.0):
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


def _batch_score_stocks(panel_df, horizon):
    """Run single vectorized batch pass across all tracked NSE stocks for the given horizon."""
    ticker_to_name = {t: name for name, t in STOCK_LIST.items()}
    grouped = {t: grp.sort_values("date") for t, grp in panel_df.groupby("ticker")}

    ordered_tickers = [t for t in sorted(grouped.keys()) if len(grouped[t]) >= LOOKBACK_WINDOW]
    if not ordered_tickers:
        ordered_tickers = sorted(grouped.keys())
    if not ordered_tickers:
        return [], pd.DataFrame(), False

    latest_rows = [grouped[t].iloc[-1] for t in ordered_tickers]
    latest_df = pd.DataFrame(latest_rows).reset_index(drop=True)

    bundle = registry.get_bundle(horizon)
    models_ready = bundle is not None

    n_stocks = len(ordered_tickers)
    pred_log_ret = np.zeros(n_stocks, dtype=np.float64)
    low_log_ret = np.zeros(n_stocks, dtype=np.float64)
    high_log_ret = np.zeros(n_stocks, dtype=np.float64)
    bullish_counts = np.zeros(n_stocks, dtype=int)
    total_model_count = 0

    recent_vols = np.clip(np.nan_to_num(latest_df["vol_21d"].astype(float).values, nan=0.018), 0.006, 0.10)
    vol_scales = recent_vols * np.sqrt(float(horizon))

    if models_ready:
        scaler = bundle["scaler"]
        models = bundle["models"]
        calibrators = bundle.get("calibrators", {})
        ens_meta = bundle.get("ensemble") or {}
        metadata = bundle.get("metadata", {})

        feat_cols = metadata.get("feature_columns", FEATURE_COLUMNS)
        selected_idx = metadata.get("selected_feature_indices")
        use_entity_ids = metadata.get("use_entity_ids", True)
        market_drift_ret = float(metadata.get("market_drift_ret", 0.0))

        # 1. Build batched tabular input (N, F_tab) & sequence tensor (N, LOOKBACK_WINDOW, F_sel)
        latest_scaled = np.clip(
            scaler.transform(latest_df[feat_cols].values.astype(np.float64)), -8.0, 8.0
        )
        latest_sel = latest_scaled[:, selected_idx] if selected_idx is not None else latest_scaled

        if use_entity_ids:
            stock_ids = np.array([TICKER_TO_ID.get(t, len(TICKER_TO_ID)) for t in ordered_tickers], dtype=np.int64)
            sector_ids = np.array(
                [SECTOR_TO_ID.get(SECTOR_MAP.get(t, "Other"), SECTOR_TO_ID["Other"]) for t in ordered_tickers],
                dtype=np.int64,
            )
            x_tab = np.hstack([latest_sel, stock_ids[:, None].astype(np.float64), sector_ids[:, None].astype(np.float64)])
        else:
            stock_ids = np.full(n_stocks, len(SORTED_TICKERS) + 2, dtype=np.int64)
            sector_ids = np.full(n_stocks, len(SECTORS) + 2, dtype=np.int64)
            x_tab = latest_sel

        # Batched sequence tensor for AttentionNet (only when every ticker has >= LOOKBACK_WINDOW rows)
        seq_list = []
        valid_seq_mask = []
        for t in ordered_tickers:
            sub = grouped[t]
            if len(sub) >= LOOKBACK_WINDOW:
                s_mat = np.clip(
                    scaler.transform(sub.tail(LOOKBACK_WINDOW)[feat_cols].values.astype(np.float64)), -8.0, 8.0
                )
                s_sel = s_mat[:, selected_idx] if selected_idx is not None else s_mat
                seq_list.append(s_sel.astype(np.float32))
                valid_seq_mask.append(True)
            else:
                seq_list.append(np.zeros((LOOKBACK_WINDOW, latest_sel.shape[1]), dtype=np.float32))
                valid_seq_mask.append(False)

        x_seq = np.stack(seq_list, axis=0)
        valid_seq_arr = np.asarray(valid_seq_mask, dtype=bool)

        cal_z_by_algo = {}
        for algo in BASE_ALGORITHMS:
            model_list = models.get(algo)
            if not model_list:
                continue

            if algo == "AttentionNet":
                raw_z = np.zeros(n_stocks, dtype=np.float64)
                if np.any(valid_seq_arr):
                    x_t = torch.tensor(x_seq[valid_seq_arr], dtype=torch.float32, device=registry.device)
                    s_t = torch.tensor(stock_ids[valid_seq_arr], dtype=torch.long, device=registry.device)
                    sec_t = torch.tensor(sector_ids[valid_seq_arr], dtype=torch.long, device=registry.device)
                    bag_preds = []
                    with torch.no_grad():
                        for m in model_list:
                            bag_preds.append(m(x_t, s_t, sec_t).detach().cpu().numpy().reshape(-1))
                    raw_z[valid_seq_arr] = np.mean(bag_preds, axis=0)
            else:
                bag_preds = [np.asarray(m.predict(x_tab), dtype=np.float64).reshape(-1) for m in model_list]
                raw_z = np.mean(bag_preds, axis=0)

            cal = calibrators.get(algo, {})
            center = float(cal.get("center", 0.0))
            bias = float(cal.get("bias", 0.0))
            gamma = float(cal.get("gamma", 1.0))
            cal_z = bias + gamma * (raw_z - center)
            cal_z_by_algo[algo] = cal_z

            algo_ret = market_drift_ret + np.clip(cal_z, -3.0, 3.0) * vol_scales
            bullish_counts += (algo_ret >= 0).astype(int)
            total_model_count += 1

        # Ensemble prediction
        ens_z = np.full(n_stocks, float(ens_meta.get("intercept", 0.0)), dtype=np.float64)
        for base_algo, weight in zip(ens_meta.get("base_models", []), ens_meta.get("weights", [])):
            if base_algo in cal_z_by_algo:
                ens_z += float(weight) * np.nan_to_num(cal_z_by_algo[base_algo], nan=0.0)

        ens_z = np.clip(ens_z, -3.0, 3.0)
        max_ret_cap = 0.15 if horizon == 1 else np.minimum(0.75, 3.5 * vol_scales)
        pred_log_ret = np.clip(market_drift_ret + ens_z * vol_scales, -max_ret_cap, max_ret_cap)

        # Quantile P10 / P90 uncertainty cones
        quantiles = bundle.get("quantiles", {})
        if "q10" in quantiles and "q90" in quantiles:
            q10_z = np.asarray(quantiles["q10"].predict(x_tab), dtype=np.float64).reshape(-1)
            q90_z = np.asarray(quantiles["q90"].predict(x_tab), dtype=np.float64).reshape(-1)
            low_log_ret = np.minimum(q10_z * vol_scales, pred_log_ret - 0.5 * vol_scales)
            high_log_ret = np.maximum(q90_z * vol_scales, pred_log_ret + 0.5 * vol_scales)
        else:
            low_log_ret = pred_log_ret - 1.645 * vol_scales
            high_log_ret = pred_log_ret + 1.645 * vol_scales
    else:
        # Factor-based fallback when models haven't been trained yet
        pred_log_ret = latest_df.get("cs_rank_sharpe_mom_252d", pd.Series(0.0, index=latest_df.index)).values * vol_scales * 0.2
        low_log_ret = pred_log_ret - 1.645 * vol_scales
        high_log_ret = pred_log_ret + 1.645 * vol_scales

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


def _build_sector_rotation(stocks_out, latest_df):
    """Aggregate 21d/63d sector momentum & AI forecasts into RRG-style quadrants."""
    if not stocks_out:
        return []

    sec_stats = {}
    for s in stocks_out:
        sec = s["sector"]
        entry = sec_stats.setdefault(sec, {"stocks": [], "rets": [], "days": []})
        entry["stocks"].append(s)
        entry["rets"].append(s["predicted_return_pct"])
        entry["days"].append(s["day_change_pct"])

    # Extract empirical 21d & 63d sector momentum from panel if present
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


def get_alpha_radar(horizon=21, force_refresh=False):
    """Return vectorized NSE 100 Alpha Radar leaderboard, Sector Rotation quadrant, and Macro Regime."""
    try:
        req_horizon = int(horizon)
    except Exception:
        req_horizon = 21

    active_horizon, available_horizons = _resolve_active_horizon(req_horizon)
    cache_key = (active_horizon, len(available_horizons))

    now = time.time()
    if not force_refresh:
        cached = _RADAR_CACHE.get(cache_key)
        if cached and (now - cached["ts"]) < _RADAR_TTL_SECONDS:
            return cached["data"]

    with _RADAR_LOCK:
        if not force_refresh:
            cached = _RADAR_CACHE.get(cache_key)
            if cached and (time.time() - cached["ts"]) < _RADAR_TTL_SECONDS:
                return cached["data"]

        panel_df, _, _ = get_live_universe_panel(force_refresh=force_refresh)
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

        payload = {
            "requested_horizon": req_horizon,
            "active_horizon": active_horizon,
            "available_horizons": available_horizons,
            "models_ready": models_ready,
            "as_of": as_of,
            "macro_regime": macro_regime,
            "sectors": sectors_out,
            "stocks": stocks_out,
        }
        _RADAR_CACHE[cache_key] = {"ts": time.time(), "data": payload}
        return payload
