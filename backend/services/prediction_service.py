"""Global Cross-Sectional Panel volatility-normalized 1-day, multi-horizon & batch prediction services."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from backend.config import (
    BASE_ALGORITHMS,
    FORECAST_HORIZONS,
    LONG_RANGE_HORIZONS,
    LOOKBACK_WINDOW,
    SECTORS,
    SECTOR_MAP,
    SECTOR_TO_ID,
    SORTED_TICKERS,
    TICKER_TO_ID,
)
from backend.security.logging_utils import get_logger
from backend.services.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_dataframe,
    fetch_macro_context,
    get_cached_ticker_latest_date,
    get_live_universe_panel,
    get_live_universe_panel_async,
)
from backend.services.market_data_service import (
    clean_yfinance_dataframe,
    download_ticker_data,
    fetch_stock_data,
    get_next_business_day,
)
from backend.services.model_registry import registry
from backend.services.runtime import runtime

logger = get_logger(__name__)

FEATURE_HUMAN_LABELS = {
    "sec_rank_ret_5d": ("5D Within-Sector Rank", "Sector Rotation"),
    "sec_rank_ret_21d": ("21D Within-Sector Rank", "Sector Rotation"),
    "sec_rank_mom_6_1": ("6M-1M Sector Momentum Rank", "Sector Rotation"),
    "sec_rank_vol_63d": ("63D Within-Sector Volatility Rank", "Volatility & Risk"),
    "sec_rank_sharpe_mom_126d": ("126D Sector Sharpe Momentum", "Sector Rotation"),
    "sector_excess_ret_1d": ("1D Sector-Excess Return", "Sector Rotation"),
    "sector_excess_ret_5d": ("5D Sector-Excess Return", "Sector Rotation"),
    "sector_excess_ret_21d": ("21D Sector-Excess Return", "Sector Rotation"),
    "sector_mom_21d": ("21D Sector Rotation Flow", "Sector Rotation"),
    "sector_mom_63d": ("63D Sector Leadership Trend", "Sector Rotation"),
    "cs_rank_sharpe_mom_252d": ("12M Risk-Adjusted Momentum Rank", "Momentum"),
    "cs_rank_sharpe_mom_126d": ("6M Risk-Adjusted Momentum Rank", "Momentum"),
    "cs_rank_sharpe_mom_63d": ("3M Risk-Adjusted Momentum Rank", "Momentum"),
    "cs_rank_mom_12_1": ("12-1M Classic Momentum Rank", "Momentum"),
    "cs_rank_mom_6_1": ("6-1M Intermediate Momentum Rank", "Momentum"),
    "cs_rank_mom_3_1": ("3-1M Short Momentum Rank", "Momentum"),
    "cs_rank_resid_mom_63d": ("3M Idiosyncratic Alpha Rank", "Momentum"),
    "cs_rank_resid_mom_126d": ("6M Idiosyncratic Alpha Rank", "Momentum"),
    "cs_rank_pct_pos_63d": ("3M Trend Consistency (FIP)", "Momentum"),
    "cs_rank_pct_pos_126d": ("6M Trend Consistency (FIP)", "Momentum"),
    "cs_rank_vol_21d": ("21D Cross-Sectional Volatility", "Volatility & Risk"),
    "cs_rank_vol_63d": ("63D Cross-Sectional Volatility", "Volatility & Risk"),
    "cs_rank_vol_126d": ("126D Cross-Sectional Volatility", "Volatility & Risk"),
    "cs_rank_idio_vol_63d": ("63D Idiosyncratic Volatility", "Volatility & Risk"),
    "cs_rank_beta_126d": ("126D Market Beta Rank", "Volatility & Risk"),
    "cs_rank_max_dd_63d": ("63D Drawdown Recovery Rank", "Volatility & Risk"),
    "cs_rank_vol_term_ratio": ("Vol Term-Structure Ratio", "Volatility & Risk"),
    "cs_rank_dist_52w_high": ("52-Week High Proximity Rank", "Momentum"),
    "cs_rank_rsi_14": ("14D Relative Strength (RSI) Rank", "Mean Reversion"),
    "cs_rank_zscore_20": ("20D Mean-Reversion Z-Score Rank", "Mean Reversion"),
    "cs_rank_amihud_21d": ("21D Liquidity Impact (Amihud)", "Liquidity & Flow"),
    "cs_rank_close_ema200": ("200D Trend Extension Rank", "Trend & Technical"),
    "cs_rank_ret_1d": ("1D Cross-Sectional Return Rank", "Mean Reversion"),
    "cs_rank_ret_5d": ("5D Cross-Sectional Return Rank", "Mean Reversion"),
    "cs_rank_ret_21d": ("21D Cross-Sectional Return Rank", "Momentum"),
    "cs_rank_ret_63d": ("63D Cross-Sectional Return Rank", "Momentum"),
    "cs_rank_ret_126d": ("126D Cross-Sectional Return Rank", "Momentum"),
    "sharpe_mom_63d": ("3M Sharpe Momentum", "Momentum"),
    "sharpe_mom_126d": ("6M Sharpe Momentum", "Momentum"),
    "sharpe_mom_252d": ("12M Sharpe Momentum", "Momentum"),
    "resid_mom_63d": ("3M Residual Momentum", "Momentum"),
    "resid_mom_126d": ("6M Residual Momentum", "Momentum"),
    "pct_pos_days_63d": ("3M Positive-Day Ratio", "Momentum"),
    "pct_pos_days_126d": ("6M Positive-Day Ratio", "Momentum"),
    "ret_1d": ("1-Day Return Impulse", "Mean Reversion"),
    "ret_5d": ("5-Day Return", "Mean Reversion"),
    "ret_21d": ("21-Day Return", "Momentum"),
    "rsi_14": ("RSI-14 Oscillator", "Mean Reversion"),
    "vol_21d": ("21D Realized Volatility", "Volatility & Risk"),
    "dist_52w_high": ("Distance from 52W High", "Momentum"),
    "dist_52w_low": ("Distance from 52W Low", "Trend & Technical"),
    "max_dd_63d": ("63D Peak Drawdown", "Volatility & Risk"),
    "beta_126d": ("126D Nifty Sensitivity (Beta)", "Volatility & Risk"),
    "vix_level": ("India VIX Regime", "Macro Regime"),
    "nifty_ret_5d": ("Nifty 50 5D Trend", "Macro Regime"),
    "sp500_ret_1d_lag1": ("S&P 500 Overnight Lead", "Macro Regime"),
    "usdinr_ret_5d_lag1": ("USD/INR 5D FX Pressure", "Macro Regime"),
    "brent_ret_5d_lag1": ("Brent Crude 5D Impulse", "Macro Regime"),
    "close_ema50_ratio": ("Price vs 50D EMA Spread", "Trend & Technical"),
    "close_ema200_ratio": ("Price vs 200D EMA Spread", "Trend & Technical"),
    "ema20_ema200_spread": ("20D / 200D EMA Trend Spread", "Trend & Technical"),
    "rel_strength_21d": ("21D Relative Strength vs Nifty", "Momentum"),
    "rel_strength_63d": ("63D Relative Strength vs Nifty", "Momentum"),
    "downside_vol_ratio_21d": ("Downside Volatility Skew", "Volatility & Risk"),
    "mfi_14": ("Money Flow Index (14D)", "Liquidity & Flow"),
    "adx_14": ("ADX-14 Trend Strength", "Trend & Technical"),
    "vwap_20_dev": ("20D VWAP Deviation", "Liquidity & Flow"),
    "zscore_20": ("20D Bollinger Z-Score", "Mean Reversion"),
}


def _safe_float(val, default: float = 0.0, digits: int | None = None) -> float:
    """Convert any scalar to finite JSON-safe float."""
    try:
        num = float(val)
        if not np.isfinite(num):
            num = float(default)
    except Exception:
        num = float(default)
    return round(num, digits) if digits is not None else num


def _format_feature_label(feat_name: str) -> tuple[str, str]:
    """Return human-readable label and category for an engineered factor name."""
    if feat_name in FEATURE_HUMAN_LABELS:
        return FEATURE_HUMAN_LABELS[feat_name]
    pretty = feat_name.replace("_", " ").title()
    return pretty, "Quant Factor"


def _compute_top_drivers(bundle, selected_feat_names, x_tab_2d, x_cont_2d, feat_df, vol_scale, horizon):
    """Compute live exact factor contributions via LightGBM TreeSHAP or Ridge fallback."""
    calibrators = bundle.get("calibrators", {})
    models = bundle.get("models", {})
    n_cont = len(selected_feat_names)
    if n_cont == 0:
        return []

    raw_contribs = None
    scale_gamma = 1.0

    lgbm_list = models.get("LightGBM")
    if lgbm_list:
        try:
            contrib_rows = []
            for m in lgbm_list:
                c = np.asarray(m.predict(x_tab_2d, pred_contrib=True), dtype=np.float64)
                if c.ndim == 2 and c.shape[1] >= n_cont:
                    contrib_rows.append(c[0, :n_cont])
            if contrib_rows:
                raw_contribs = np.mean(contrib_rows, axis=0)
                cal = calibrators.get("LightGBM", {})
                gamma = float(cal.get("gamma", 1.0))
                scale_gamma = gamma if gamma > 1e-6 else 0.25
        except Exception:
            raw_contribs = None

    if raw_contribs is None:
        ridge_list = models.get("Ridge")
        if ridge_list:
            try:
                coef_rows = []
                for m in ridge_list:
                    est = m.named_steps.get("ridge") if hasattr(m, "named_steps") else m
                    if hasattr(est, "coef_"):
                        coef = np.asarray(est.coef_, dtype=np.float64).reshape(-1)
                        if len(coef) >= n_cont:
                            coef_rows.append(coef[:n_cont] * x_cont_2d[0, :n_cont])
                if coef_rows:
                    raw_contribs = np.mean(coef_rows, axis=0)
                    cal = calibrators.get("Ridge", {})
                    gamma = float(cal.get("gamma", 1.0))
                    scale_gamma = gamma if gamma > 1e-6 else 0.25
            except Exception:
                raw_contribs = None

    if raw_contribs is None:
        return []

    pct_scale = scale_gamma * vol_scale * 100.0
    drivers = []
    for idx, feat_name in enumerate(selected_feat_names):
        imp = _safe_float(raw_contribs[idx] * pct_scale, 0.0)
        if abs(imp) < 1e-6:
            continue
        label, category = _format_feature_label(feat_name)
        raw_val = _safe_float(feat_df[feat_name].iloc[-1], 0.0, digits=4) if feat_name in feat_df else None
        drivers.append(
            {
                "feature": feat_name,
                "label": label,
                "category": category,
                "impact_pct": round(imp, 4 if horizon == 1 else 3),
                "direction": "bullish" if imp >= 0 else "bearish",
                "raw_value": raw_val,
            }
        )

    drivers.sort(key=lambda d: abs(d["impact_pct"]), reverse=True)
    return drivers[:6]


def _predict_family_raw_z(model_list, algo, x_tab_2d, x_seq_3d, stock_id, sector_id) -> float:
    """Run multi-seed bagged inference for one model family under GPU lock when using PyTorch."""
    preds = []
    if algo == "AttentionNet":
        with runtime.gpu_lock, torch.inference_mode():
            x_t = torch.tensor(x_seq_3d, dtype=torch.float32, device=registry.device)
            s_t = torch.tensor([stock_id], dtype=torch.long, device=registry.device)
            sec_t = torch.tensor([sector_id], dtype=torch.long, device=registry.device)
            for model in model_list:
                out = model(x_t, s_t, sec_t).detach().cpu().numpy().reshape(-1)[0]
                preds.append(float(out))
    else:
        for model in model_list:
            out = model.predict(x_tab_2d)
            preds.append(float(np.asarray(out).reshape(-1)[0]))

    return float(np.mean(preds))


def _apply_calibrator(raw_z: float, calibrator: dict | None) -> float:
    """Apply strict OOF linear variance calibration: z_cal = bias + gamma * (z_raw - center)."""
    if not calibrator:
        return raw_z
    center = float(calibrator.get("center", 0.0))
    bias = float(calibrator.get("bias", 0.0))
    gamma = float(calibrator.get("gamma", 1.0))
    return bias + gamma * (raw_z - center)


def _prepare_scaled_inputs(bundle, ticker: str, feat_df: pd.DataFrame):
    """Slice only the last LOOKBACK_WINDOW rows and scale once for fast inference."""
    scaler = bundle["scaler"]
    metadata = bundle.get("metadata", {})
    feat_cols = metadata.get("feature_columns", FEATURE_COLUMNS)
    selected_idx = metadata.get("selected_feature_indices")
    use_entity_ids = metadata.get("use_entity_ids", True)

    if use_entity_ids:
        stock_id = TICKER_TO_ID.get(ticker, len(TICKER_TO_ID))
        sector_id = SECTOR_TO_ID.get(SECTOR_MAP.get(ticker, "Other"), SECTOR_TO_ID["Other"])
    else:
        stock_id = len(SORTED_TICKERS) + 2
        sector_id = len(SECTORS) + 2

    tail_df = feat_df.tail(LOOKBACK_WINDOW)
    raw_matrix = tail_df[feat_cols].to_numpy(dtype=np.float64)
    scaled_full = np.clip(scaler.transform(raw_matrix), -8.0, 8.0)

    if selected_idx is not None:
        scaled_selected = scaled_full[:, selected_idx]
        selected_names = [feat_cols[i] for i in selected_idx]
    else:
        scaled_selected = scaled_full
        selected_names = list(feat_cols)

    x_cont_2d = scaled_selected[-1:, :]
    if use_entity_ids:
        x_tab_2d = np.hstack([x_cont_2d, np.array([[stock_id, sector_id]], dtype=np.float64)])
    else:
        x_tab_2d = x_cont_2d
    x_seq_3d = scaled_selected[-LOOKBACK_WINDOW:, :][np.newaxis, :, :].astype(np.float32)
    return x_cont_2d, x_tab_2d, x_seq_3d, stock_id, sector_id, selected_names


def _evaluate_horizon(
    ticker: str,
    algorithm_name: str,
    feat_df: pd.DataFrame,
    horizon: int = 1,
    collect_all_votes: bool = False,
):
    """Evaluate one horizon in a single pass, returning return, dispersion, votes, and scaled state."""
    bundle = registry.get_bundle(horizon)
    if bundle is None:
        script_name = "ml/train_one_day_models.py" if horizon == 1 else "ml/train_long_range_models.py"
        return {
            "error": f"Global Panel SOTA model (h={horizon}d) is not trained yet. Run `python {script_name}`."
        }

    models = bundle["models"]
    calibrators = bundle.get("calibrators", {})
    ens_meta = bundle.get("ensemble")
    metadata = bundle.get("metadata", {})

    x_cont_2d, x_tab_2d, x_seq_3d, stock_id, sector_id, selected_names = _prepare_scaled_inputs(
        bundle, ticker, feat_df
    )
    recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.006)
    vol_scale = recent_vol * np.sqrt(float(horizon))
    market_drift_ret = float(metadata.get("market_drift_ret", 0.0))
    max_ret_cap = 0.15 if horizon == 1 else min(0.75, 3.5 * vol_scale)

    cal_z_cache: dict[str, float] = {}
    if collect_all_votes or algorithm_name == "Ensemble":
        if algorithm_name == "Ensemble" and not ens_meta:
            return {"error": f"Global Panel Ensemble metadata (h={horizon}d) missing."}

        algos_to_run = (
            BASE_ALGORITHMS
            if collect_all_votes
            else [a for a in (ens_meta.get("base_models") or []) if a in models]
        )
        for algo in algos_to_run:
            model_list = models.get(algo)
            if not model_list:
                continue
            raw_z = _predict_family_raw_z(model_list, algo, x_tab_2d, x_seq_3d, stock_id, sector_id)
            cal_z_cache[algo] = _apply_calibrator(raw_z, calibrators.get(algo))

    if algorithm_name == "Ensemble":
        base_names = ens_meta.get("base_models", [])
        weights = np.asarray(ens_meta.get("weights", []), dtype=float)
        intercept = float(ens_meta.get("intercept", 0.0))

        cal_preds = []
        used_weights = []
        for idx, base_algo in enumerate(base_names):
            if base_algo in cal_z_cache:
                cal_preds.append(cal_z_cache[base_algo])
                used_weights.append(weights[idx] if idx < len(weights) else 0.0)

        if not cal_preds:
            return {"error": f"No Global Panel base models loaded for h={horizon}d."}

        cal_arr = np.asarray(cal_preds, dtype=float)
        w_arr = np.asarray(used_weights, dtype=float)
        pred_z = float(intercept + np.dot(w_arr, cal_arr))
        dispersion_ret = float(np.std(cal_arr) * vol_scale) if len(cal_arr) > 1 else 0.0
    else:
        if algorithm_name not in cal_z_cache:
            model_list = models.get(algorithm_name)
            if not model_list:
                return {
                    "error": (
                        f"Global Panel {algorithm_name} model (h={horizon}d) not found. "
                        "Please run the training script."
                    )
                }
            raw_z = _predict_family_raw_z(model_list, algorithm_name, x_tab_2d, x_seq_3d, stock_id, sector_id)
            cal_z_cache[algorithm_name] = _apply_calibrator(raw_z, calibrators.get(algorithm_name))
        pred_z = cal_z_cache[algorithm_name]
        dispersion_ret = None

    pred_z = float(np.clip(pred_z, -3.0, 3.0))
    pred_log_ret = float(np.clip(market_drift_ret + pred_z * vol_scale, -max_ret_cap, max_ret_cap))

    # Ensemble return for term structure when caller selected a non-Ensemble single algorithm
    is_full_ensemble = algorithm_name == "Ensemble" or collect_all_votes
    ens_log_ret = pred_log_ret
    if algorithm_name != "Ensemble" and collect_all_votes and ens_meta and cal_z_cache:
        base_names = ens_meta.get("base_models", [])
        weights = np.asarray(ens_meta.get("weights", []), dtype=float)
        intercept = float(ens_meta.get("intercept", 0.0))
        c_arr = [cal_z_cache[b] for b in base_names if b in cal_z_cache]
        w_arr = [weights[i] if i < len(weights) else 0.0 for i, b in enumerate(base_names) if b in cal_z_cache]
        if c_arr:
            ez = float(np.clip(intercept + float(np.dot(w_arr, c_arr)), -3.0, 3.0))
            ens_log_ret = float(np.clip(market_drift_ret + ez * vol_scale, -max_ret_cap, max_ret_cap))

    # Quantiles using already-scaled x_tab_2d (zero redundant scaler.transform!)
    default_low = pred_log_ret - 1.645 * vol_scale
    default_high = pred_log_ret + 1.645 * vol_scale
    quantiles = bundle.get("quantiles", {})
    if "q10" in quantiles and "q90" in quantiles:
        try:
            q10_z = float(quantiles["q10"].predict(x_tab_2d)[0])
            q90_z = float(quantiles["q90"].predict(x_tab_2d)[0])
            low_log_ret = float(min(q10_z * vol_scale, pred_log_ret - 0.5 * vol_scale))
            high_log_ret = float(max(q90_z * vol_scale, pred_log_ret + 0.5 * vol_scale))
        except Exception:
            low_log_ret, high_log_ret = default_low, default_high
    else:
        low_log_ret, high_log_ret = default_low, default_high

    return {
        "error": None,
        "bundle": bundle,
        "is_full_ensemble": is_full_ensemble,
        "pred_log_ret": pred_log_ret,
        "ens_log_ret": ens_log_ret,
        "low_log_ret": low_log_ret,
        "high_log_ret": high_log_ret,
        "dispersion": dispersion_ret,
        "cal_z_cache": cal_z_cache,
        "x_cont_2d": x_cont_2d,
        "x_tab_2d": x_tab_2d,
        "selected_names": selected_names,
        "vol_scale": vol_scale,
        "market_drift_ret": market_drift_ret,
        "max_ret_cap": max_ret_cap,
    }


def _predict_horizon_z_and_return(ticker: str, algorithm_name: str, feat_df: pd.DataFrame, horizon: int = 1):
    """Compatibility helper returning (predicted_log_return, dispersion, error)."""
    res = _evaluate_horizon(ticker, algorithm_name, feat_df, horizon=horizon, collect_all_votes=False)
    if res.get("error"):
        return None, None, res["error"]
    return res["pred_log_ret"], res["dispersion"], None


def _predict_horizon_quantiles(ticker: str, feat_df: pd.DataFrame, horizon: int, point_log_ret: float):
    """Predict calibrated P10/P90 log-return bounds using LightGBM quantile models (with vol fallback)."""
    recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.006)
    vol_scale = recent_vol * np.sqrt(float(horizon))
    default_low = point_log_ret - 1.645 * vol_scale
    default_high = point_log_ret + 1.645 * vol_scale

    bundle = registry.get_bundle(horizon)
    if bundle is None or "q10" not in bundle["quantiles"] or "q90" not in bundle["quantiles"]:
        return default_low, default_high

    try:
        _, x_tab_2d, _, _, _, _ = _prepare_scaled_inputs(bundle, ticker, feat_df)
        q10_z = float(bundle["quantiles"]["q10"].predict(x_tab_2d)[0])
        q90_z = float(bundle["quantiles"]["q90"].predict(x_tab_2d)[0])
        low_ret = min(q10_z * vol_scale, point_log_ret - 0.5 * vol_scale)
        high_ret = max(q90_z * vol_scale, point_log_ret + 0.5 * vol_scale)
        return float(low_ret), float(high_ret)
    except Exception:
        return default_low, default_high


def _build_forecast_explainability_from_eval(
    ticker: str,
    feat_df: pd.DataFrame,
    target_horizon: int,
    last_close: float,
    precomputed_by_horizon: dict[int, dict],
):
    """Assemble 7-model consensus votes, live SHAP drivers, and 5-horizon term structure reusing cached evals."""
    try:
        anchor_eval = precomputed_by_horizon.get(target_horizon)
        if anchor_eval is None or len(anchor_eval.get("cal_z_cache", {})) < len(anchor_eval["bundle"]["models"]):
            anchor_eval = _evaluate_horizon(
                ticker, "Ensemble", feat_df, horizon=target_horizon, collect_all_votes=True
            )
            if anchor_eval.get("error"):
                return None

        bundle = anchor_eval["bundle"]
        calibrators = bundle.get("calibrators", {})
        ens_meta = bundle.get("ensemble") or {}
        vol_scale = anchor_eval["vol_scale"]
        market_drift_ret = anchor_eval["market_drift_ret"]
        max_ret_cap = anchor_eval["max_ret_cap"]
        cal_z_cache = anchor_eval["cal_z_cache"]

        base_names = ens_meta.get("base_models", BASE_ALGORITHMS)
        ens_weights = ens_meta.get("weights", [])
        weight_map = {
            name: _safe_float(ens_weights[i], 0.0, 4) if i < len(ens_weights) else 0.0
            for i, name in enumerate(base_names)
        }

        model_votes = []
        bullish_count = 0
        bearish_count = 0
        for algo in BASE_ALGORITHMS:
            if algo not in cal_z_cache:
                continue
            cal = calibrators.get(algo, {})
            cal_z = float(np.clip(cal_z_cache[algo], -3.0, 3.0))
            log_ret = float(np.clip(market_drift_ret + cal_z * vol_scale, -max_ret_cap, max_ret_cap))
            ret_pct = _safe_float((np.exp(log_ret) - 1.0) * 100.0, 0.0, 3)
            proj_price = _safe_float(last_close * np.exp(log_ret), last_close, 2)
            if ret_pct >= 0:
                bullish_count += 1
            else:
                bearish_count += 1
            model_votes.append(
                {
                    "algorithm": algo,
                    "predicted_return_pct": ret_pct,
                    "predicted_price": proj_price,
                    "weight": weight_map.get(algo, 0.0),
                    "gamma": _safe_float(cal.get("gamma", 1.0), 1.0, 4),
                    "oof_ic": _safe_float(cal.get("oof_rank_ic", 0.0), 0.0, 4),
                }
            )

        total_models = len(model_votes)
        majority = max(bullish_count, bearish_count)
        agreement_pct = round((majority / total_models) * 100.0, 1) if total_models else 0.0
        direction = "bullish" if bullish_count >= bearish_count else "bearish"
        if agreement_pct >= 75.0:
            conviction_label = "High Conviction"
        elif agreement_pct >= 58.0:
            conviction_label = "Moderate Conviction"
        else:
            conviction_label = "Split Consensus"

        top_drivers = _compute_top_drivers(
            bundle,
            anchor_eval["selected_names"],
            anchor_eval["x_tab_2d"],
            anchor_eval["x_cont_2d"],
            feat_df,
            vol_scale,
            target_horizon,
        )

        term_structure = []
        for h in FORECAST_HORIZONS:
            if registry.get_bundle(h) is None:
                continue
            ev_h = precomputed_by_horizon.get(h)
            if ev_h is None or ev_h.get("error") or not ev_h.get("is_full_ensemble"):
                ev_h = _evaluate_horizon(ticker, "Ensemble", feat_df, horizon=h, collect_all_votes=False)
                if ev_h.get("error"):
                    continue
            ret_h = ev_h["ens_log_ret"]
            low_h, high_h = ev_h["low_log_ret"], ev_h["high_log_ret"]
            if ev_h["ens_log_ret"] != ev_h["pred_log_ret"]:
                low_h, high_h = _predict_horizon_quantiles(ticker, feat_df, h, ret_h)

            term_structure.append(
                {
                    "horizon": h,
                    "label": f"{h}D",
                    "predicted_return_pct": _safe_float((np.exp(ret_h) - 1.0) * 100.0, 0.0, 2),
                    "predicted_price": _safe_float(last_close * np.exp(ret_h), last_close, 2),
                    "lower_price": _safe_float(last_close * np.exp(low_h), last_close, 2),
                    "upper_price": _safe_float(last_close * np.exp(high_h), last_close, 2),
                }
            )

        return {
            "anchor_horizon": target_horizon,
            "consensus": {
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "total_models": total_models,
                "agreement_pct": agreement_pct,
                "direction": direction,
                "label": conviction_label,
            },
            "model_votes": model_votes,
            "top_drivers": top_drivers,
            "term_structure": term_structure,
        }
    except Exception:
        return None


def _prepare_stock_feature_slice(ticker: str, panel_df=None, raw_df=None, macro_df=None):
    """Extract the target stock's full cross-sectional feature history (falls back to single-stock fetch)."""
    if panel_df is not None and not panel_df.empty:
        sub = panel_df[panel_df["ticker"] == ticker].sort_values("date")
        if len(sub) >= LOOKBACK_WINDOW:
            return sub.set_index("date"), None

    try:
        live_panel, _, _ = get_live_universe_panel(force_refresh=False)
        if live_panel is not None and not live_panel.empty:
            sub = live_panel[live_panel["ticker"] == ticker].sort_values("date")
            if len(sub) >= LOOKBACK_WINDOW:
                return sub.set_index("date"), None
    except Exception as exc:
        logger.warning("Live universe panel fallback triggered for %s: %s", ticker, exc)

    data = raw_df if raw_df is not None else fetch_stock_data(ticker, days=950)
    if data is None or data.empty:
        return None, "Could not fetch latest stock data from yfinance."

    macro = macro_df
    if macro is None:
        macro = fetch_macro_context(
            start_date=data.index[0] - timedelta(days=30),
            end_date=datetime.now(),
            use_cache=True,
        )

    feat_df = build_feature_dataframe(data, macro_df=macro, ticker=ticker)
    if feat_df.empty or len(feat_df) < LOOKBACK_WINDOW:
        return None, f"Not enough feature history to predict. Need {LOOKBACK_WINDOW} days, got {len(feat_df)}."

    return feat_df, None


async def prepare_stock_feature_slice_async(ticker: str):
    """Non-blocking preparation of a ticker's cross-sectional feature slice."""
    try:
        live_panel, _, _ = await get_live_universe_panel_async(force_refresh=False)
        if live_panel is not None and not live_panel.empty:
            sub = live_panel[live_panel["ticker"] == ticker].sort_values("date")
            if len(sub) >= LOOKBACK_WINDOW:
                return sub.set_index("date"), None
    except Exception as exc:
        logger.warning("Async universe panel fallback for %s: %s", ticker, exc)

    return await runtime.run_io(_prepare_stock_feature_slice, ticker, pd.DataFrame(), None, None)


def make_one_day_prediction(
    ticker: str,
    algorithm_name: str,
    panel_df=None,
    raw_df=None,
    macro_df=None,
    feat_df=None,
    include_explainability: bool = True,
):
    """Perform a 1-day-ahead volatility-normalized Global Panel forecast."""
    try:
        if feat_df is None:
            feat_df, err = _prepare_stock_feature_slice(
                ticker, panel_df=panel_df, raw_df=raw_df, macro_df=macro_df
            )
            if err:
                return None, None, None, err

        ev1 = _evaluate_horizon(
            ticker,
            algorithm_name,
            feat_df,
            horizon=1,
            collect_all_votes=include_explainability,
        )
        if ev1.get("error"):
            return None, None, None, ev1["error"]

        pred_log_ret = ev1["pred_log_ret"]
        dispersion = ev1["dispersion"]
        low_ret = ev1["low_log_ret"]
        high_ret = ev1["high_log_ret"]

        last_close = float(feat_df["Close"].iloc[-1])
        predicted_price = last_close * float(np.exp(pred_log_ret))
        lower_price = last_close * float(np.exp(low_ret))
        upper_price = last_close * float(np.exp(high_ret))

        history_window = feat_df[["Close"]].tail(100)
        history_dates = list(history_window.index.strftime("%Y-%m-%d"))
        history_prices = [float(p) for p in history_window["Close"].values]
        next_day_str = get_next_business_day(feat_df.index[-1])

        trend_data = {
            "dates": history_dates + [next_day_str],
            "prices": history_prices + [float(predicted_price)],
            "upper_band": [None] * (len(history_dates) - 1) + [last_close, round(upper_price, 2)],
            "lower_band": [None] * (len(history_dates) - 1) + [last_close, round(lower_price, 2)],
            "history_cutoff": len(history_dates),
        }

        explainability = (
            _build_forecast_explainability_from_eval(ticker, feat_df, 1, last_close, {1: ev1})
            if include_explainability
            else None
        )

        meta = {
            "confidence_interval": {
                "lower": round(lower_price, 2),
                "upper": round(upper_price, 2),
            },
            "predicted_return_pct": round((np.exp(pred_log_ret) - 1.0) * 100.0, 3),
            "ensemble_dispersion": round(dispersion, 5) if dispersion is not None else None,
            "model_metrics": registry.get_metrics(ticker, algorithm_name, horizon=1),
            "explainability": explainability,
        }
        return float(predicted_price), trend_data, meta, None
    except Exception as exc:
        logger.exception("Error in 1-day prediction for %s", ticker)
        return None, None, None, str(exc)


def make_long_range_prediction(
    ticker: str,
    future_date_str: str,
    algorithm_name: str,
    panel_df=None,
    raw_df=None,
    macro_df=None,
    feat_df=None,
    include_explainability: bool = True,
):
    """Perform a Global Panel Direct Multi-Horizon forecast with calibrated 90% quantile cones."""
    try:
        if feat_df is None:
            feat_df, err = _prepare_stock_feature_slice(
                ticker, panel_df=panel_df, raw_df=raw_df, macro_df=macro_df
            )
            if err:
                return None, None, None, err

        last_date = feat_df.index[-1]
        future_date = pd.to_datetime(future_date_str)
        pred_dates = pd.bdate_range(start=last_date + timedelta(days=1), end=future_date)
        n_steps = len(pred_dates)
        if n_steps <= 0:
            return (
                None,
                None,
                None,
                "Future date must be at least one business day after the last trading day.",
            )

        last_close = float(feat_df["Close"].iloc[-1])
        recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.008)
        nearest_horizon = min(LONG_RANGE_HORIZONS, key=lambda h: abs(h - n_steps))

        anchor_steps = [0]
        anchor_rets = [0.0]
        anchor_low_rets = [0.0]
        anchor_high_rets = [0.0]
        dispersions = []
        eval_cache: dict[int, dict] = {}

        for h in [1] + LONG_RANGE_HORIZONS:
            if registry.get_bundle(h) is None:
                continue
            need_votes = include_explainability and (h == nearest_horizon)
            ev = _evaluate_horizon(ticker, algorithm_name, feat_df, horizon=h, collect_all_votes=need_votes)
            if ev.get("error"):
                continue
            eval_cache[h] = ev
            anchor_steps.append(h)
            anchor_rets.append(ev["pred_log_ret"])
            anchor_low_rets.append(ev["low_log_ret"])
            anchor_high_rets.append(ev["high_log_ret"])
            if ev["dispersion"] is not None:
                dispersions.append(ev["dispersion"])

        if len(anchor_steps) <= 1:
            return (
                None,
                None,
                None,
                f"Global Panel Direct Multi-Horizon models ({algorithm_name}) are not trained yet. "
                "Run `python ml/train_long_range_models.py`.",
            )

        max_anchor = anchor_steps[-1]
        step_grid = np.arange(1, n_steps + 1, dtype=float)
        interp_steps = np.minimum(step_grid, float(max_anchor))

        cum_rets = np.interp(interp_steps, anchor_steps, anchor_rets)
        cum_low = np.interp(interp_steps, anchor_steps, anchor_low_rets)
        cum_high = np.interp(interp_steps, anchor_steps, anchor_high_rets)

        if n_steps > max_anchor:
            hist_daily_ret = feat_df["ret_1d"].tail(504).dropna()
            raw_daily_mu = float(hist_daily_ret.mean()) if len(hist_daily_ret) > 20 else 0.0004
            nifty_prior_daily_mu = np.log(1.115) / 252.0
            shrunk_daily_mu = 0.35 * np.clip(raw_daily_mu, -0.0015, 0.0015) + 0.65 * nifty_prior_daily_mu

            extra_days = np.maximum(0.0, step_grid - float(max_anchor))
            damped_extra = shrunk_daily_mu * extra_days
            cone_extra = 1.645 * recent_vol * np.sqrt(extra_days)

            cum_rets = cum_rets + damped_extra
            cum_low = cum_low + damped_extra - cone_extra
            cum_high = cum_high + damped_extra + cone_extra

        future_prices = [float(last_close * np.exp(r)) for r in cum_rets]
        lower_prices = [round(float(last_close * np.exp(r)), 2) for r in cum_low]
        upper_prices = [round(float(last_close * np.exp(r)), 2) for r in cum_high]

        history_window = feat_df[["Close"]].tail(100)
        history_dates = list(history_window.index.strftime("%Y-%m-%d"))
        history_prices = [float(p) for p in history_window["Close"].values]
        future_dates = list(pred_dates.strftime("%Y-%m-%d"))

        pad_len = len(history_dates) - 1
        trend_data = {
            "dates": history_dates + future_dates,
            "prices": history_prices + future_prices,
            "upper_band": [None] * pad_len + [last_close] + upper_prices,
            "lower_band": [None] * pad_len + [last_close] + lower_prices,
            "history_cutoff": len(history_dates),
        }

        final_price = float(future_prices[-1])
        warning = (
            "Global Cross-Sectional Direct Multi-Horizon forecast with OOF variance calibration "
            "and 90% quantile uncertainty bands. Not financial advice."
        )
        explainability = (
            _build_forecast_explainability_from_eval(ticker, feat_df, nearest_horizon, last_close, eval_cache)
            if include_explainability
            else None
        )
        meta = {
            "confidence_interval": {
                "lower": lower_prices[-1],
                "upper": upper_prices[-1],
            },
            "predicted_return_pct": round(((final_price / last_close) - 1.0) * 100.0, 2),
            "ensemble_dispersion": round(float(np.mean(dispersions)), 5) if dispersions else None,
            "model_metrics": registry.get_metrics(ticker, algorithm_name, horizon=nearest_horizon),
            "warning": warning,
            "explainability": explainability,
        }
        return final_price, trend_data, meta, None
    except Exception as exc:
        logger.exception("Error in long-range prediction for %s", ticker)
        return None, None, None, str(exc)


def batch_score_universe_horizon(panel_df: pd.DataFrame, horizon: int) -> dict:
    """Run a single vectorized batch scoring pass across all tracked NSE stocks for one horizon."""
    grouped = {t: grp.sort_values("date") for t, grp in panel_df.groupby("ticker")}
    ordered_tickers = [t for t in sorted(grouped.keys()) if len(grouped[t]) >= LOOKBACK_WINDOW]
    if not ordered_tickers:
        ordered_tickers = sorted(grouped.keys())
    if not ordered_tickers:
        return {
            "tickers": [],
            "latest_df": pd.DataFrame(),
            "recent_vols": np.empty((0,), dtype=np.float64),
            "pred_log_ret": np.empty((0,), dtype=np.float64),
            "low_log_ret": np.empty((0,), dtype=np.float64),
            "high_log_ret": np.empty((0,), dtype=np.float64),
            "dispersion": np.empty((0,), dtype=np.float64),
            "bullish_counts": np.empty((0,), dtype=int),
            "total_model_count": 0,
            "models_ready": False,
        }

    latest_rows = [grouped[t].iloc[-1] for t in ordered_tickers]
    latest_df = pd.DataFrame(latest_rows).reset_index(drop=True)

    bundle = registry.get_bundle(horizon)
    models_ready = bundle is not None

    n_stocks = len(ordered_tickers)
    pred_log_ret = np.zeros(n_stocks, dtype=np.float64)
    low_log_ret = np.zeros(n_stocks, dtype=np.float64)
    high_log_ret = np.zeros(n_stocks, dtype=np.float64)
    dispersion = np.zeros(n_stocks, dtype=np.float64)
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

        latest_scaled = np.clip(
            scaler.transform(latest_df[feat_cols].to_numpy(dtype=np.float64)), -8.0, 8.0
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

        # Single stacked scaler.transform across all valid sequence windows (eliminates 99 extra calls)
        valid_positions = [i for i, t in enumerate(ordered_tickers) if len(grouped[t]) >= LOOKBACK_WINDOW]
        valid_seq_arr = np.zeros(n_stocks, dtype=bool)
        if valid_positions:
            valid_seq_arr[valid_positions] = True
            seq_raw = np.vstack(
                [grouped[ordered_tickers[i]].tail(LOOKBACK_WINDOW)[feat_cols].to_numpy(dtype=np.float64) for i in valid_positions]
            )
            seq_scaled = np.clip(scaler.transform(seq_raw), -8.0, 8.0)
            seq_sel = seq_scaled[:, selected_idx] if selected_idx is not None else seq_scaled
            x_seq_valid = seq_sel.reshape(len(valid_positions), LOOKBACK_WINDOW, -1).astype(np.float32)
        else:
            x_seq_valid = np.empty((0, LOOKBACK_WINDOW, latest_sel.shape[1]), dtype=np.float32)

        cal_z_by_algo: dict[str, np.ndarray] = {}
        for algo in BASE_ALGORITHMS:
            model_list = models.get(algo)
            if not model_list:
                continue

            if algo == "AttentionNet":
                raw_z = np.zeros(n_stocks, dtype=np.float64)
                if len(valid_positions) > 0:
                    with runtime.gpu_lock, torch.inference_mode():
                        x_t = torch.tensor(x_seq_valid, dtype=torch.float32, device=registry.device)
                        s_t = torch.tensor(stock_ids[valid_seq_arr], dtype=torch.long, device=registry.device)
                        sec_t = torch.tensor(sector_ids[valid_seq_arr], dtype=torch.long, device=registry.device)
                        bag_preds = [m(x_t, s_t, sec_t).detach().cpu().numpy().reshape(-1) for m in model_list]
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

        ens_z = np.full(n_stocks, float(ens_meta.get("intercept", 0.0)), dtype=np.float64)
        used_cal_rows = []
        for base_algo, weight in zip(ens_meta.get("base_models", []), ens_meta.get("weights", [])):
            if base_algo in cal_z_by_algo:
                arr = np.nan_to_num(cal_z_by_algo[base_algo], nan=0.0)
                ens_z += float(weight) * arr
                used_cal_rows.append(arr)

        if len(used_cal_rows) > 1:
            dispersion = np.std(np.stack(used_cal_rows, axis=0), axis=0) * vol_scales

        ens_z = np.clip(ens_z, -3.0, 3.0)
        max_ret_cap = 0.15 if horizon == 1 else np.minimum(0.75, 3.5 * vol_scales)
        pred_log_ret = np.clip(market_drift_ret + ens_z * vol_scales, -max_ret_cap, max_ret_cap)

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
        pred_log_ret = (
            latest_df.get("cs_rank_sharpe_mom_252d", pd.Series(0.0, index=latest_df.index)).to_numpy(dtype=np.float64)
            * vol_scales
            * 0.2
        )
        low_log_ret = pred_log_ret - 1.645 * vol_scales
        high_log_ret = pred_log_ret + 1.645 * vol_scales

    return {
        "tickers": ordered_tickers,
        "latest_df": latest_df,
        "recent_vols": recent_vols,
        "pred_log_ret": pred_log_ret,
        "low_log_ret": low_log_ret,
        "high_log_ret": high_log_ret,
        "dispersion": dispersion,
        "bullish_counts": bullish_counts,
        "total_model_count": total_model_count,
        "models_ready": models_ready,
    }


def determine_next_business_day_for_ticker(ticker: str) -> str:
    """Resolve the next expected business day using cached universe data or resilient 5-day fetch."""
    cached_dt = get_cached_ticker_latest_date(ticker)
    if cached_dt is not None:
        return get_next_business_day(cached_dt)

    latest_data = clean_yfinance_dataframe(download_ticker_data(ticker, period="5d"))
    if latest_data is not None and not latest_data.empty:
        return get_next_business_day(latest_data.index[-1])

    return get_next_business_day()


def run_prediction_for_algorithm(
    stock_name: str,
    ticker: str,
    future_date_str: str,
    algorithm_name: str,
    next_bday_str: str,
    feat_df=None,
):
    """Dispatch either 1-day or direct multi-horizon Global Panel forecast."""
    if future_date_str == next_bday_str:
        final_price, trend_data, meta, error = make_one_day_prediction(
            ticker, algorithm_name, feat_df=feat_df
        )
        pred_type = "1-Day Forecast (Global Cross-Sectional Panel)"
        warning = ""
    else:
        final_price, trend_data, meta, error = make_long_range_prediction(
            ticker, future_date_str, algorithm_name, feat_df=feat_df
        )
        pred_type = "Direct Multi-Horizon Forecast (Global Panel + 90% Cone)"
        warning = meta.get("warning", "") if meta else ""

    if error or final_price is None:
        return None, error

    return {
        "stock_name": stock_name,
        "ticker": ticker,
        "future_date": future_date_str,
        "algorithm_name": algorithm_name,
        "predicted_price": f"{final_price:.2f}",
        "trend_data": trend_data,
        "prediction_type": pred_type,
        "warning": warning,
        "confidence_interval": meta.get("confidence_interval"),
        "predicted_return_pct": meta.get("predicted_return_pct"),
        "model_metrics": meta.get("model_metrics"),
        "explainability": meta.get("explainability"),
    }, None


async def predict_async(stock_name: str, ticker: str, future_date_str: str, algorithm_name):
    """Non-blocking prediction dispatcher fetching the feature slice once across comparisons."""
    feat_df, err = await prepare_stock_feature_slice_async(ticker)
    if err:
        return None, err, 500

    next_bday_str = get_next_business_day(feat_df.index[-1])

    if isinstance(algorithm_name, list):
        def _run_many():
            out = {}
            for algo in algorithm_name:
                payload, algo_err = run_prediction_for_algorithm(
                    stock_name, ticker, future_date_str, algo, next_bday_str, feat_df=feat_df
                )
                out[algo] = {"error": algo_err} if algo_err else payload
            return out

        results = await runtime.run_cpu(_run_many)
        return results, None, 200

    payload, single_err = await runtime.run_cpu(
        run_prediction_for_algorithm,
        stock_name,
        ticker,
        future_date_str,
        algorithm_name,
        next_bday_str,
        feat_df,
    )
    if single_err:
        return None, single_err, 500
    return payload, None, 200
