"""Walk-forward strategy backtest simulator on strict OOS holdout or live panel history."""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from backend.config import (
    BASE_ALGORITHMS,
    FORECAST_HORIZONS,
    LOOKBACK_WINDOW,
    MODELS_DIR,
    SECTORS,
    SECTOR_MAP,
    SECTOR_TO_ID,
    SORTED_TICKERS,
    STOCK_LIST,
    TICKER_TO_ID,
)
from backend.services.feature_engineering import FEATURE_COLUMNS, get_live_universe_panel
from backend.services.market_data_service import resolve_ticker
from backend.services.model_registry import registry


def _sf(val, default=0.0, digits=None):
    try:
        n = float(val)
        if not np.isfinite(n):
            n = float(default)
    except Exception:
        n = float(default)
    return round(n, digits) if digits is not None else n


def _build_live_replay_frame(horizon, lookback_months):
    """Build out-of-sample panel replay frame from live universe data across non-overlapping dates."""
    bundle = registry.get_bundle(horizon)
    if bundle is None:
        return None, f"Global Panel model (h={horizon}D) is not trained yet."

    panel_df, _, _ = get_live_universe_panel(force_refresh=False)
    if panel_df is None or panel_df.empty:
        return None, "Could not load NSE universe panel data."

    df = panel_df.sort_values(["ticker", "date"]).copy()
    df["fwd_ret"] = np.log(df.groupby("ticker")["Close"].shift(-horizon) / df["Close"])
    df = df.dropna(subset=["fwd_ret"]).copy()
    if df.empty:
        return None, "Insufficient forward dates for live replay."

    unique_dates = np.sort(df["date"].unique())
    if lookback_months and int(lookback_months) > 0:
        max_days = int(lookback_months) * 21 + horizon
        if len(unique_dates) > max_days:
            cutoff_dt = unique_dates[-max_days]
            df = df[df["date"] >= cutoff_dt].copy()
            unique_dates = np.sort(df["date"].unique())

    # Subsample strictly non-overlapping rebalance dates (every h trading days)
    rebalance_dates = unique_dates[:: max(1, horizon)]
    if len(rebalance_dates) < 3:
        rebalance_dates = unique_dates[:: max(1, len(unique_dates) // 8 or 1)]

    scaler = bundle["scaler"]
    models = bundle["models"]
    calibrators = bundle.get("calibrators", {})
    ens_meta = bundle.get("ensemble") or {}
    metadata = bundle.get("metadata", {})
    feat_cols = metadata.get("feature_columns", FEATURE_COLUMNS)
    selected_idx = metadata.get("selected_feature_indices")
    use_entity_ids = metadata.get("use_entity_ids", True)
    market_drift_ret = float(metadata.get("market_drift_ret", 0.0))

    eval_df = df[df["date"].isin(rebalance_dates)].copy().reset_index(drop=True)
    if eval_df.empty:
        return None, "Insufficient rebalance rows for replay."

    scaled_full = np.clip(scaler.transform(eval_df[feat_cols].values.astype(np.float64)), -8.0, 8.0)
    scaled_sel = scaled_full[:, selected_idx] if selected_idx is not None else scaled_full

    if use_entity_ids:
        sids = np.array([TICKER_TO_ID.get(t, len(TICKER_TO_ID)) for t in eval_df["ticker"]], dtype=np.float64)
        secs = np.array(
            [SECTOR_TO_ID.get(SECTOR_MAP.get(t, "Other"), SECTOR_TO_ID["Other"]) for t in eval_df["ticker"]],
            dtype=np.float64,
        )
        x_tab = np.hstack([scaled_sel, sids[:, None], secs[:, None]])
    else:
        x_tab = scaled_sel

    vol_scales = np.clip(eval_df["vol_21d"].astype(float).values, 0.006, 0.10) * np.sqrt(float(horizon))
    max_ret_cap = 0.15 if horizon == 1 else np.minimum(0.75, 3.5 * vol_scales)

    cal_z_map = {}
    replay_out = pd.DataFrame(
        {
            "date": pd.to_datetime(eval_df["date"].values),
            "ticker": eval_df["ticker"].values,
            "Close": eval_df["Close"].values.astype(np.float64),
            "fwd_ret": eval_df["fwd_ret"].values.astype(np.float64),
        }
    )

    for algo in BASE_ALGORITHMS:
        if algo == "AttentionNet":
            # Build 60d sequences for AttentionNet replay rows
            model_list = models.get("AttentionNet")
            if not model_list:
                continue
            by_ticker = {t: grp.sort_values("date") for t, grp in df.groupby("ticker")}
            seq_rows = []
            valid_mask = []
            seq_sids = []
            seq_secs = []
            for idx_r, r in eval_df.iterrows():
                t_df = by_ticker.get(r["ticker"])
                sub = t_df[t_df["date"] <= r["date"]].tail(LOOKBACK_WINDOW) if t_df is not None else None
                if sub is not None and len(sub) >= LOOKBACK_WINDOW:
                    sm = np.clip(scaler.transform(sub[feat_cols].values.astype(np.float64)), -8.0, 8.0)
                    seq_rows.append((sm[:, selected_idx] if selected_idx is not None else sm).astype(np.float32))
                    valid_mask.append(True)
                else:
                    seq_rows.append(np.zeros((LOOKBACK_WINDOW, scaled_sel.shape[1]), dtype=np.float32))
                    valid_mask.append(False)
                seq_sids.append(TICKER_TO_ID.get(r["ticker"], len(TICKER_TO_ID)) if use_entity_ids else len(SORTED_TICKERS) + 2)
                seq_secs.append(
                    SECTOR_TO_ID.get(SECTOR_MAP.get(r["ticker"], "Other"), SECTOR_TO_ID["Other"])
                    if use_entity_ids
                    else len(SECTORS) + 2
                )
            vmask = np.asarray(valid_mask, dtype=bool)
            raw_z = np.zeros(len(eval_df), dtype=np.float64)
            if np.any(vmask):
                xt = torch.tensor(np.stack(seq_rows)[vmask], dtype=torch.float32, device=registry.device)
                st = torch.tensor(np.asarray(seq_sids, dtype=np.int64)[vmask], dtype=torch.long, device=registry.device)
                sct = torch.tensor(np.asarray(seq_secs, dtype=np.int64)[vmask], dtype=torch.long, device=registry.device)
                bp = []
                with torch.no_grad():
                    for m in model_list:
                        bp.append(m(xt, st, sct).detach().cpu().numpy().reshape(-1))
                raw_z[vmask] = np.mean(bp, axis=0)
        else:
            model_list = models.get(algo)
            if not model_list:
                continue
            bp = [np.asarray(m.predict(x_tab), dtype=np.float64).reshape(-1) for m in model_list]
            raw_z = np.mean(bp, axis=0)

        cal = calibrators.get(algo, {})
        cal_z = float(cal.get("bias", 0.0)) + float(cal.get("gamma", 1.0)) * (raw_z - float(cal.get("center", 0.0)))
        cal_z_map[algo] = cal_z
        replay_out[f"pred_{algo}"] = np.clip(market_drift_ret + np.clip(cal_z, -3.0, 3.0) * vol_scales, -max_ret_cap, max_ret_cap)

    ens_z = np.full(len(eval_df), float(ens_meta.get("intercept", 0.0)), dtype=np.float64)
    for base_algo, weight in zip(ens_meta.get("base_models", []), ens_meta.get("weights", [])):
        if base_algo in cal_z_map:
            ens_z += float(weight) * np.nan_to_num(cal_z_map[base_algo], nan=0.0)
    replay_out["pred_Ensemble"] = np.clip(
        market_drift_ret + np.clip(ens_z, -3.0, 3.0) * vol_scales, -max_ret_cap, max_ret_cap
    )
    return replay_out, None


def run_strategy_backtest(
    strategy="top_n_long",
    horizon=21,
    algorithm="Ensemble",
    top_n=10,
    cost_bps=15.0,
    initial_capital=100000.0,
    lookback_months=24,
    stock_name=None,
):
    """Run non-overlapping walk-forward backtest with realistic NSE transaction costs."""
    available_horizons = [h for h in FORECAST_HORIZONS if registry.get_bundle(h) is not None]
    try:
        h = int(horizon)
    except Exception:
        h = 21

    horizon_key = "one_day" if h == 1 else f"h{h}"
    holdout_path = os.path.join(MODELS_DIR, f"holdout_eval_{horizon_key}.joblib")

    if os.path.exists(holdout_path):
        df = joblib.load(holdout_path)
        replay_mode = "strict_oos_holdout"
    else:
        df, err = _build_live_replay_frame(h, lookback_months)
        if err:
            return None, (err, 400)
        replay_mode = "live_panel_replay"

    pred_col = f"pred_{algorithm}"
    if pred_col not in df.columns:
        pred_col = "pred_Ensemble" if "pred_Ensemble" in df.columns else [c for c in df.columns if c.startswith("pred_")][0]

    df = df.sort_values(["date", "ticker"]).copy()
    all_dates = np.sort(df["date"].unique())
    if lookback_months and int(lookback_months) > 0:
        max_dt = pd.Timestamp(all_dates[-1]) - pd.DateOffset(months=int(lookback_months))
        df_win = df[df["date"] >= max_dt]
        if len(df_win["date"].unique()) >= 4:
            df = df_win
            all_dates = np.sort(df["date"].unique())

    rebalance_dates = all_dates[:: max(1, h)] if replay_mode == "strict_oos_holdout" else all_dates
    if len(rebalance_dates) < 2:
        return None, ("Not enough rebalance dates for the selected lookback window.", 400)

    target_ticker = None
    if strategy == "single_stock":
        _, target_ticker = resolve_ticker(stock_name or "Infosys")
        if not target_ticker:
            return None, (f"Invalid stock_name: {stock_name}", 400)

    ticker_to_name = {t: nm for nm, t in STOCK_LIST.items()}
    cost_rate = max(float(cost_bps), 0.0) / 10000.0
    rf_period = np.log(1.065) * (float(h) / 252.0)

    strat_eq = float(initial_capital)
    bench_eq = float(initial_capital)
    strat_peak = strat_eq
    bench_peak = bench_eq

    equity_curve = []
    drawdown_curve = []
    period_rets = []
    bench_rets = []
    turnovers = []
    recent_rebalances = []

    prev_weights = {}
    date_groups = {d: grp for d, grp in df.groupby("date")}

    for r_date in rebalance_dates:
        grp = date_groups.get(r_date)
        if grp is None or len(grp) < 4:
            continue

        new_weights = {}
        bench_log = float(grp["fwd_ret"].mean())

        if strategy == "single_stock":
            row = grp[grp["ticker"] == target_ticker]
            if row.empty:
                continue
            actual_r = float(row["fwd_ret"].iloc[0])
            pred_r = float(row[pred_col].iloc[0])
            bench_log = actual_r
            if pred_r > 0:
                new_weights[target_ticker] = 1.0
                gross_r = np.exp(actual_r) - 1.0
            else:
                gross_r = np.exp(rf_period) - 1.0
        elif strategy == "long_short_quintile":
            ordered = grp.sort_values(pred_col, ascending=False)
            q_n = max(2, len(ordered) // 5)
            longs = ordered.iloc[:q_n]
            shorts = ordered.iloc[-q_n:]
            for t in longs["ticker"]:
                new_weights[t] = 0.5 / q_n
            for t in shorts["ticker"]:
                new_weights[t] = -0.5 / q_n
            long_r = float(np.mean(np.exp(longs["fwd_ret"].values) - 1.0))
            short_r = float(np.mean(np.exp(shorts["fwd_ret"].values) - 1.0))
            gross_r = 0.5 * (long_r - short_r) + 0.5 * (np.exp(rf_period) - 1.0)
        else:
            # Top-N Long-Only Rotation
            ordered = grp.sort_values(pred_col, ascending=False)
            k = min(max(2, int(top_n)), len(ordered))
            picks = ordered.iloc[:k]
            for t in picks["ticker"]:
                new_weights[t] = 1.0 / k
            gross_r = float(np.mean(np.exp(picks["fwd_ret"].values) - 1.0))

        all_keys = set(prev_weights.keys()) | set(new_weights.keys())
        turnover = 0.5 * sum(abs(new_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in all_keys)
        net_r = gross_r - (turnover * cost_rate)
        bench_simple_r = float(np.exp(bench_log) - 1.0)

        strat_eq *= max(0.05, 1.0 + net_r)
        bench_eq *= max(0.05, 1.0 + bench_simple_r)

        strat_peak = max(strat_peak, strat_eq)
        bench_peak = max(bench_peak, bench_eq)
        strat_dd = ((strat_eq - strat_peak) / strat_peak) * 100.0
        bench_dd = ((bench_eq - bench_peak) / bench_peak) * 100.0

        d_str = pd.Timestamp(r_date).strftime("%Y-%m-%d")
        equity_curve.append({"time": d_str, "strategy": _sf(strat_eq, initial_capital, 2), "benchmark": _sf(bench_eq, initial_capital, 2)})
        drawdown_curve.append({"time": d_str, "strategy_dd": _sf(strat_dd, 0.0, 2), "benchmark_dd": _sf(bench_dd, 0.0, 2)})

        period_rets.append(net_r)
        bench_rets.append(bench_simple_r)
        turnovers.append(turnover * 100.0)
        prev_weights = new_weights

        top_picks_labels = [
            ticker_to_name.get(t, t.split(".")[0])
            for t, w in sorted(new_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
            if w > 0
        ]
        recent_rebalances.append(
            {
                "date": d_str,
                "period_return_pct": _sf(net_r * 100.0, 0.0, 2),
                "benchmark_return_pct": _sf(bench_simple_r * 100.0, 0.0, 2),
                "turnover_pct": _sf(turnover * 100.0, 0.0, 1),
                "top_holdings": top_picks_labels or ["Cash / T-Bill"],
            }
        )

    if len(period_rets) < 2:
        return None, ("Too few valid periods to evaluate backtest.", 400)

    n_periods = len(period_rets)
    years_elapsed = max((n_periods * h) / 252.0, 0.1)
    periods_per_year = 252.0 / float(h)

    tot_ret = ((strat_eq / initial_capital) - 1.0) * 100.0
    bench_tot_ret = ((bench_eq / initial_capital) - 1.0) * 100.0
    cagr = (((strat_eq / initial_capital) ** (1.0 / years_elapsed)) - 1.0) * 100.0
    bench_cagr = (((bench_eq / initial_capital) ** (1.0 / years_elapsed)) - 1.0) * 100.0

    r_arr = np.asarray(period_rets, dtype=np.float64)
    sharpe = (np.mean(r_arr) - (np.exp(rf_period) - 1.0)) / max(np.std(r_arr), 1e-6) * np.sqrt(periods_per_year)
    downside = r_arr[r_arr < 0]
    down_std = np.std(downside) if len(downside) > 1 else np.std(r_arr)
    sortino = (np.mean(r_arr) - (np.exp(rf_period) - 1.0)) / max(down_std, 1e-6) * np.sqrt(periods_per_year)

    max_dd = min(pt["strategy_dd"] for pt in drawdown_curve) if drawdown_curve else 0.0
    bench_max_dd = min(pt["benchmark_dd"] for pt in drawdown_curve) if drawdown_curve else 0.0
    win_rate = float(np.mean(r_arr > 0) * 100.0)

    return {
        "replay_mode": replay_mode,
        "strategy": strategy,
        "horizon": h,
        "algorithm": algorithm,
        "available_horizons": available_horizons,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "metrics": {
            "initial_capital": _sf(initial_capital, 100000.0, 2),
            "final_capital": _sf(strat_eq, initial_capital, 2),
            "benchmark_final_capital": _sf(bench_eq, initial_capital, 2),
            "total_return_pct": _sf(tot_ret, 0.0, 2),
            "benchmark_return_pct": _sf(bench_tot_ret, 0.0, 2),
            "alpha_pct": _sf(tot_ret - bench_tot_ret, 0.0, 2),
            "cagr_pct": _sf(cagr, 0.0, 2),
            "benchmark_cagr_pct": _sf(bench_cagr, 0.0, 2),
            "sharpe_ratio": _sf(sharpe, 0.0, 2),
            "sortino_ratio": _sf(sortino, 0.0, 2),
            "max_drawdown_pct": _sf(abs(max_dd), 0.0, 2),
            "benchmark_max_drawdown_pct": _sf(abs(bench_max_dd), 0.0, 2),
            "win_rate_pct": _sf(win_rate, 0.0, 1),
            "avg_turnover_pct": _sf(np.mean(turnovers), 0.0, 1),
            "num_rebalances": n_periods,
        },
        "recent_rebalances": recent_rebalances[-8:][::-1],
    }, None
