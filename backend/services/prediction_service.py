"""Global Cross-Sectional Panel volatility-normalized 1-day & direct multi-horizon prediction services."""

from datetime import datetime, timedelta
import traceback
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
from backend.services.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_dataframe,
    fetch_macro_context,
    get_cached_ticker_latest_date,
    get_live_universe_panel,
)
from backend.services.market_data_service import (
    clean_yfinance_dataframe,
    download_ticker_data,
    fetch_stock_data,
    get_next_business_day,
)
from backend.services.model_registry import registry


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


def _safe_float(val, default=0.0, digits=None):
    """Convert any scalar to finite JSON-safe float."""
    try:
        num = float(val)
        if not np.isfinite(num):
            num = float(default)
    except Exception:
        num = float(default)
    return round(num, digits) if digits is not None else num


def _format_feature_label(feat_name):
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

    # Express factor attribution as annualized/horizon % return contribution
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


def _build_forecast_explainability(ticker, feat_df, target_horizon, last_close):
    """Build 7-model consensus votes, live SHAP factor drivers, and 5-horizon term structure."""
    try:
        bundle = registry.get_bundle(target_horizon)
        if bundle is None:
            return None

        scaler = bundle["scaler"]
        models = bundle["models"]
        calibrators = bundle["calibrators"]
        ens_meta = bundle.get("ensemble") or {}
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

        raw_matrix = feat_df[feat_cols].values.astype(np.float64)
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

        recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.006)
        vol_scale = recent_vol * np.sqrt(float(target_horizon))
        market_drift_ret = float(metadata.get("market_drift_ret", 0.0))
        max_ret_cap = 0.15 if target_horizon == 1 else min(0.75, 3.5 * vol_scale)

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
            model_list = models.get(algo)
            if not model_list:
                continue
            raw_z = _predict_family_raw_z(model_list, algo, x_tab_2d, x_seq_3d, stock_id, sector_id)
            cal = calibrators.get(algo, {})
            cal_z = float(np.clip(_apply_calibrator(raw_z, cal), -3.0, 3.0))
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
            bundle, selected_names, x_tab_2d, x_cont_2d, feat_df, vol_scale, target_horizon
        )

        term_structure = []
        for h in FORECAST_HORIZONS:
            if registry.get_bundle(h) is None:
                continue
            ret_h, _, err_h = _predict_horizon_z_and_return(ticker, "Ensemble", feat_df, horizon=h)
            if err_h is not None or ret_h is None:
                continue
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


def _predict_family_raw_z(model_list, algo, x_tab_2d, x_seq_3d, stock_id, sector_id):
    """Run multi-seed bagged inference for one model family returning raw volatility-normalized target z."""
    preds = []
    if algo == "AttentionNet":
        x_t = torch.tensor(x_seq_3d, dtype=torch.float32, device=registry.device)
        s_t = torch.tensor([stock_id], dtype=torch.long, device=registry.device)
        sec_t = torch.tensor([sector_id], dtype=torch.long, device=registry.device)
        with torch.no_grad():
            for model in model_list:
                out = model(x_t, s_t, sec_t).detach().cpu().numpy().reshape(-1)[0]
                preds.append(float(out))
    else:
        for model in model_list:
            out = model.predict(x_tab_2d)
            preds.append(float(np.asarray(out).reshape(-1)[0]))

    return float(np.mean(preds))


def _apply_calibrator(raw_z, calibrator):
    """Apply strict OOF linear variance calibration: z_cal = bias + gamma * (z_raw - center)."""
    if not calibrator:
        return raw_z
    center = float(calibrator.get("center", 0.0))
    bias = float(calibrator.get("bias", 0.0))
    gamma = float(calibrator.get("gamma", 1.0))
    return bias + gamma * (raw_z - center)


def _prepare_stock_feature_slice(ticker, panel_df=None, raw_df=None, macro_df=None):
    """Extract the target stock's full cross-sectional feature history (falls back to single-stock fetch)."""
    if panel_df is not None and not panel_df.empty:
        sub = panel_df[panel_df["ticker"] == ticker].sort_values("date")
        if len(sub) >= LOOKBACK_WINDOW:
            sub = sub.set_index("date")
            return sub, None

    # Use live 30-min cached universe panel so cross-sectional ranks are populated
    try:
        live_panel, _, _ = get_live_universe_panel(force_refresh=False)
        if live_panel is not None and not live_panel.empty:
            sub = live_panel[live_panel["ticker"] == ticker].sort_values("date")
            if len(sub) >= LOOKBACK_WINDOW:
                return sub.set_index("date"), None
    except Exception as exc:
        print(f"Warning: live universe panel fallback triggered for {ticker}: {exc}")

    # Single-stock fallback (neutral cross-sectional ranks = 0.0)
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


def _predict_horizon_z_and_return(ticker, algorithm_name, feat_df, horizon=1):
    """Run Global Panel inference for horizon h, returning (predicted_log_return, dispersion, error)."""
    bundle = registry.get_bundle(horizon)
    if bundle is None:
        script_name = "ml/train_one_day_models.py" if horizon == 1 else "ml/train_long_range_models.py"
        return (
            None,
            None,
            f"Global Panel SOTA model (h={horizon}d) is not trained yet. Run `python {script_name}`.",
        )

    scaler = bundle["scaler"]
    models = bundle["models"]
    calibrators = bundle["calibrators"]
    ens_meta = bundle["ensemble"]
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

    raw_matrix = feat_df[feat_cols].values.astype(np.float64)
    scaled_full = scaler.transform(raw_matrix)
    scaled_full = np.clip(scaled_full, -8.0, 8.0)

    scaled_selected = scaled_full[:, selected_idx] if selected_idx is not None else scaled_full

    # Match exact tabular matrix layout used during training
    if use_entity_ids:
        x_tab_2d = np.hstack([scaled_selected[-1:, :], np.array([[stock_id, sector_id]], dtype=np.float64)])
    else:
        x_tab_2d = scaled_selected[-1:, :]
    x_seq_3d = scaled_selected[-LOOKBACK_WINDOW:, :][np.newaxis, :, :].astype(np.float32)

    recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.006)
    vol_scale = recent_vol * np.sqrt(float(horizon))

    if algorithm_name == "Ensemble":
        if not ens_meta:
            return None, None, f"Global Panel Ensemble metadata (h={horizon}d) missing."

        base_names = ens_meta.get("base_models", [])
        weights = np.asarray(ens_meta.get("weights", []), dtype=float)
        intercept = float(ens_meta.get("intercept", 0.0))

        cal_preds = []
        used_weights = []
        for idx, base_algo in enumerate(base_names):
            model_list = models.get(base_algo)
            if not model_list:
                continue
            raw_z = _predict_family_raw_z(model_list, base_algo, x_tab_2d, x_seq_3d, stock_id, sector_id)
            cal_z = _apply_calibrator(raw_z, calibrators.get(base_algo))
            cal_preds.append(cal_z)
            used_weights.append(weights[idx] if idx < len(weights) else 0.0)

        if not cal_preds:
            return None, None, f"No Global Panel base models loaded for h={horizon}d."

        cal_arr = np.asarray(cal_preds, dtype=float)
        w_arr = np.asarray(used_weights, dtype=float)
        # Note: sum(w_arr) is intentionally <= 1.0 (regularized shrinkage stacker!)
        pred_z = float(intercept + np.dot(w_arr, cal_arr))
        dispersion_ret = float(np.std(cal_arr) * vol_scale) if len(cal_arr) > 1 else 0.0
    else:
        model_list = models.get(algorithm_name)
        if not model_list:
            return (
                None,
                None,
                f"Global Panel {algorithm_name} model (h={horizon}d) not found. Please run the training script.",
            )
        raw_z = _predict_family_raw_z(model_list, algorithm_name, x_tab_2d, x_seq_3d, stock_id, sector_id)
        pred_z = _apply_calibrator(raw_z, calibrators.get(algorithm_name))
        dispersion_ret = None

    # Convert market-neutral alpha z back to log return + decoupled market drift (order-preserving)
    pred_z = float(np.clip(pred_z, -3.0, 3.0))
    market_drift_ret = float(metadata.get("market_drift_ret", 0.0))
    pred_log_ret = market_drift_ret + pred_z * vol_scale
    max_ret_cap = 0.15 if horizon == 1 else min(0.75, 3.5 * vol_scale)
    pred_log_ret = float(np.clip(pred_log_ret, -max_ret_cap, max_ret_cap))
    return pred_log_ret, dispersion_ret, None


def _predict_horizon_quantiles(ticker, feat_df, horizon, point_log_ret):
    """Predict calibrated P10/P90 log-return bounds using LightGBM quantile models (with vol fallback)."""
    recent_vol = max(float(feat_df["vol_21d"].iloc[-1]), 0.006)
    vol_scale = recent_vol * np.sqrt(float(horizon))
    default_low = point_log_ret - 1.645 * vol_scale
    default_high = point_log_ret + 1.645 * vol_scale

    bundle = registry.get_bundle(horizon)
    if bundle is None or "q10" not in bundle["quantiles"] or "q90" not in bundle["quantiles"]:
        return default_low, default_high

    try:
        scaler = bundle["scaler"]
        metadata = bundle.get("metadata", {})
        feat_cols = metadata.get("feature_columns", FEATURE_COLUMNS)
        selected_idx = metadata.get("selected_feature_indices")
        use_entity_ids = metadata.get("use_entity_ids", True)

        scaled_full = np.clip(scaler.transform(feat_df[feat_cols].values.astype(np.float64)), -8.0, 8.0)
        scaled_selected = scaled_full[:, selected_idx] if selected_idx is not None else scaled_full

        if use_entity_ids:
            stock_id = TICKER_TO_ID.get(ticker, len(TICKER_TO_ID))
            sector_id = SECTOR_TO_ID.get(SECTOR_MAP.get(ticker, "Other"), SECTOR_TO_ID["Other"])
            x_tab_2d = np.hstack([scaled_selected[-1:, :], np.array([[stock_id, sector_id]], dtype=np.float64)])
        else:
            x_tab_2d = scaled_selected[-1:, :]

        q10_z = float(bundle["quantiles"]["q10"].predict(x_tab_2d)[0])
        q90_z = float(bundle["quantiles"]["q90"].predict(x_tab_2d)[0])
        low_ret = min(q10_z * vol_scale, point_log_ret - 0.5 * vol_scale)
        high_ret = max(q90_z * vol_scale, point_log_ret + 0.5 * vol_scale)
        return float(low_ret), float(high_ret)
    except Exception:
        return default_low, default_high


def make_one_day_prediction(ticker, algorithm_name, panel_df=None, raw_df=None, macro_df=None):
    """Perform a 1-day-ahead volatility-normalized Global Panel forecast."""
    try:
        feat_df, err = _prepare_stock_feature_slice(
            ticker, panel_df=panel_df, raw_df=raw_df, macro_df=macro_df
        )
        if err:
            return None, None, None, err

        pred_log_ret, dispersion, err = _predict_horizon_z_and_return(
            ticker, algorithm_name, feat_df, horizon=1
        )
        if err:
            return None, None, None, err

        last_close = float(feat_df["Close"].iloc[-1])
        predicted_price = last_close * float(np.exp(pred_log_ret))

        low_ret, high_ret = _predict_horizon_quantiles(ticker, feat_df, 1, pred_log_ret)
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

        meta = {
            "confidence_interval": {
                "lower": round(lower_price, 2),
                "upper": round(upper_price, 2),
            },
            "predicted_return_pct": round((np.exp(pred_log_ret) - 1.0) * 100.0, 3),
            "ensemble_dispersion": round(dispersion, 5) if dispersion is not None else None,
            "model_metrics": registry.get_metrics(ticker, algorithm_name, horizon=1),
            "explainability": _build_forecast_explainability(ticker, feat_df, 1, last_close),
        }
        return float(predicted_price), trend_data, meta, None
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, str(exc)


def make_long_range_prediction(ticker, future_date_str, algorithm_name, panel_df=None, raw_df=None, macro_df=None):
    """Perform a Global Panel Direct Multi-Horizon forecast with calibrated 90% quantile cones."""
    try:
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

        anchor_steps = [0]
        anchor_rets = [0.0]
        anchor_low_rets = [0.0]
        anchor_high_rets = [0.0]
        dispersions = []

        for h in [1] + LONG_RANGE_HORIZONS:
            if registry.get_bundle(h) is None:
                continue
            ret_h, disp_h, err_h = _predict_horizon_z_and_return(ticker, algorithm_name, feat_df, horizon=h)
            if err_h is not None or ret_h is None:
                continue

            low_h, high_h = _predict_horizon_quantiles(ticker, feat_df, h, ret_h)
            anchor_steps.append(h)
            anchor_rets.append(ret_h)
            anchor_low_rets.append(low_h)
            anchor_high_rets.append(high_h)
            if disp_h is not None:
                dispersions.append(disp_h)

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

        # Beyond 63 trading days, smoothly extend via Bayesian-shrunk equilibrium drift + volatility cone
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
        nearest_horizon = min(LONG_RANGE_HORIZONS, key=lambda h: abs(h - n_steps))
        warning = (
            "Global Cross-Sectional Direct Multi-Horizon forecast with OOF variance calibration "
            "and 90% quantile uncertainty bands. Not financial advice."
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
            "explainability": _build_forecast_explainability(ticker, feat_df, nearest_horizon, last_close),
        }
        return final_price, trend_data, meta, None
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, str(exc)


def determine_next_business_day_for_ticker(ticker):
    """Resolve the next expected business day using cached universe data or resilient 5-day fetch."""
    cached_dt = get_cached_ticker_latest_date(ticker)
    if cached_dt is not None:
        return get_next_business_day(cached_dt)

    latest_data = clean_yfinance_dataframe(download_ticker_data(ticker, period="5d"))
    if latest_data is not None and not latest_data.empty:
        return get_next_business_day(latest_data.index[-1])

    return get_next_business_day()


def run_prediction_for_algorithm(stock_name, ticker, future_date_str, algorithm_name, next_bday_str):
    """Dispatch either 1-day or direct multi-horizon Global Panel forecast."""
    if future_date_str == next_bday_str:
        final_price, trend_data, meta, error = make_one_day_prediction(ticker, algorithm_name)
        pred_type = "1-Day Forecast (Global Cross-Sectional Panel)"
        warning = ""
    else:
        final_price, trend_data, meta, error = make_long_range_prediction(
            ticker, future_date_str, algorithm_name
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
