"""Institutional portfolio projection using 10,000-path GARCH(1,1) Student-t Monte Carlo."""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.config import DEFAULT_ALGORITHM, SECTOR_MAP
from backend.services.feature_engineering import get_live_universe_panel
from backend.services.market_data_service import fetch_stock_data, resolve_ticker
from backend.services.model_registry import registry
from backend.services.prediction_service import make_long_range_prediction

NUM_SIMULATION_PATHS = 10000
RISK_FREE_RATE_ANNUAL = 0.068  # India 10Y G-Sec / T-bill proxy (~6.8%)
EQUITY_PRIOR_DRIFT_ANNUAL = 0.115  # Long-term NSE Nifty equity prior (~11.5%)


def _fit_garch_parameters(log_returns):
    """Fit GARCH(1,1) with Student-t innovations; gracefully fall back to EWMA + fat-tail t(df=6)."""
    pct_returns = log_returns.values * 100.0
    sample_var = float(np.var(pct_returns))
    fallback = {
        "omega": 0.05 * sample_var,
        "alpha": 0.08,
        "beta": 0.87,
        "nu": 6.0,
        "last_var": sample_var,
        "model_name": "EWMA-Student-t",
    }

    try:
        from arch import arch_model

        am = arch_model(pct_returns, mean="Constant", vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        params = res.params
        omega = max(float(params.get("omega", fallback["omega"])), 1e-6)
        alpha = float(np.clip(params.get("alpha[1]", 0.08), 0.01, 0.35))
        beta = float(np.clip(params.get("beta[1]", 0.85), 0.50, 0.96))
        # Ensure covariance stationarity (alpha + beta < 0.99)
        if alpha + beta >= 0.99:
            scale = 0.985 / (alpha + beta)
            alpha *= scale
            beta *= scale
        nu = float(np.clip(params.get("nu", 6.0), 3.2, 30.0))
        last_var = float(res.conditional_volatility[-1] ** 2)
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "nu": nu,
            "last_var": max(last_var, 1e-6),
            "model_name": "GARCH(1,1)-Student-t",
        }
    except Exception:
        return fallback


def calculate_portfolio_projection(stock_name, principal, future_date_str):
    """Simulate 10,000 GARCH(1,1)-t Monte Carlo paths and compute P10/P50/P90 & institutional risk metrics."""
    _, ticker = resolve_ticker(stock_name)
    if not ticker:
        return None, ("Stock not found.", 404)

    hist_data = fetch_stock_data(ticker, days=1095)
    if hist_data is None or len(hist_data) < 60:
        return None, ("Not enough historical data.", 404)

    future_date = datetime.strptime(future_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    if future_date <= today:
        return None, ("Future date must be in the future.", 400)

    calendar_days = min((future_date - today).days, 365)
    trading_steps = max(1, int(round(calendar_days * (252.0 / 365.0))))

    close = hist_data["Close"].astype(float)
    current_price = float(close.iloc[-1])
    units_bought = principal / current_price

    log_returns = np.log(close / close.shift(1)).dropna()
    sample_mu_daily = float(log_returns.mean())
    sample_sigma_daily = float(log_returns.std())

    # James-Stein / Bayesian shrinkage of noisy sample drift toward long-term NSE equity prior
    prior_mu_daily = np.log(1.0 + EQUITY_PRIOR_DRIFT_ANNUAL) / 252.0
    sample_years = len(log_returns) / 252.0
    shrinkage_weight = min(0.65, sample_years / (sample_years + 4.0))
    shrunk_mu_daily = shrinkage_weight * np.clip(sample_mu_daily, -0.0025, 0.0025) + (
        1.0 - shrinkage_weight
    ) * prior_mu_daily

    garch = _fit_garch_parameters(log_returns)
    omega, alpha, beta, nu = garch["omega"], garch["alpha"], garch["beta"], garch["nu"]

    # Vectorized 10,000-path GARCH(1,1) Student-t simulation
    rng = np.random.default_rng(seed=2026)
    t_scale = np.sqrt((nu - 2.0) / nu)  # Unit-variance standardized Student-t shocks
    z = rng.standard_t(df=nu, size=(trading_steps, NUM_SIMULATION_PATHS)) * t_scale

    var_t = np.full(NUM_SIMULATION_PATHS, garch["last_var"], dtype=np.float64)
    sim_log_returns = np.empty((trading_steps, NUM_SIMULATION_PATHS), dtype=np.float64)

    for step in range(trading_steps):
        sigma_pct = np.sqrt(np.maximum(var_t, 1e-6))
        shock_pct = sigma_pct * z[step]
        step_ret = shrunk_mu_daily + (shock_pct / 100.0)
        sim_log_returns[step] = step_ret
        var_t = omega + alpha * (shock_pct ** 2) + beta * var_t

    cum_log_returns = np.cumsum(sim_log_returns, axis=0)
    # Path values: shape (trading_steps + 1, NUM_SIMULATION_PATHS)
    path_values = np.vstack(
        [
            np.full((1, NUM_SIMULATION_PATHS), principal, dtype=np.float64),
            principal * np.exp(cum_log_returns),
        ]
    )

    p10_curve = np.percentile(path_values, 10, axis=1)
    p50_curve = np.percentile(path_values, 50, axis=1)
    p90_curve = np.percentile(path_values, 90, axis=1)

    terminal_values = path_values[-1]
    final_value = float(p50_curve[-1])
    p10_value = float(p10_curve[-1])
    p90_value = float(p90_curve[-1])

    roi = ((final_value - principal) / principal) * 100.0
    years = calendar_days / 365.0
    cagr = (((final_value / principal) ** (1.0 / years)) - 1.0) * 100.0 if years > 0 else 0.0

    # Institutional risk metrics
    annualized_vol_pct = float(np.sqrt(garch["last_var"]) / 100.0 * np.sqrt(252.0) * 100.0)
    if annualized_vol_pct <= 0:
        annualized_vol_pct = sample_sigma_daily * np.sqrt(252.0) * 100.0

    expected_annual_return = (np.exp(shrunk_mu_daily * 252.0) - 1.0)
    ann_vol_dec = max(annualized_vol_pct / 100.0, 1e-6)
    sharpe_ratio = (expected_annual_return - RISK_FREE_RATE_ANNUAL) / ann_vol_dec

    downside_returns = log_returns[log_returns < 0]
    downside_vol_ann = max(float(downside_returns.std() * np.sqrt(252.0)), 1e-6) if len(downside_returns) > 5 else ann_vol_dec
    sortino_ratio = (expected_annual_return - RISK_FREE_RATE_ANNUAL) / downside_vol_ann

    # Value at Risk (95%) & Conditional VaR (Expected Shortfall) in INR loss terms
    pnl = terminal_values - principal
    var_95_threshold = float(np.percentile(pnl, 5))
    var_95 = max(0.0, -var_95_threshold)
    tail_losses = pnl[pnl <= var_95_threshold]
    cvar_95 = max(0.0, -float(tail_losses.mean())) if len(tail_losses) > 0 else var_95

    # Expected Median Max Drawdown across paths
    running_max = np.maximum.accumulate(path_values, axis=0)
    drawdowns = (path_values - running_max) / np.maximum(running_max, 1e-6)
    max_drawdown_pct = float(np.median(np.min(drawdowns, axis=0)) * 100.0)

    prob_profit = float(np.mean(terminal_values > principal) * 100.0)

    if annualized_vol_pct > 32:
        risk_score = "High"
    elif annualized_vol_pct > 18:
        risk_score = "Medium"
    else:
        risk_score = "Low"

    # Map trading-step dates cleanly between today and target future_date
    date_offsets = np.linspace(0, calendar_days, trading_steps + 1)
    dates = [(today + timedelta(days=int(round(d)))).strftime("%Y-%m-%d") for d in date_offsets]

    # Deduplicate same-day timestamps if calendar_days < trading_steps
    chart_data, p10_chart, p90_chart = [], [], []
    seen_dates = set()
    for idx, d_str in enumerate(dates):
        if d_str in seen_dates:
            continue
        seen_dates.add(d_str)
        chart_data.append({"time": d_str, "value": round(float(p50_curve[idx]), 2), "type": "Projected"})
        p10_chart.append({"time": d_str, "value": round(float(p10_curve[idx]), 2)})
        p90_chart.append({"time": d_str, "value": round(float(p90_curve[idx]), 2)})

    return {
        "stock": stock_name,
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "units": round(units_bought, 2),
        "final_value": round(final_value, 2),
        "p10_value": round(p10_value, 2),
        "p50_value": round(final_value, 2),
        "p90_value": round(p90_value, 2),
        "roi": round(roi, 2),
        "cagr": round(cagr, 2),
        "risk_score": risk_score,
        "risk_value": round(annualized_vol_pct, 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2),
        "sortino_ratio": round(float(sortino_ratio), 2),
        "max_drawdown_pct": round(abs(max_drawdown_pct), 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "probability_of_profit": round(prob_profit, 1),
        "simulation_paths": NUM_SIMULATION_PATHS,
        "volatility_model": garch["model_name"],
        "chart_data": chart_data,
        "p10_chart_data": p10_chart,
        "p90_chart_data": p90_chart,
    }, None


def _sf(val, default=0.0, digits=None):
    try:
        n = float(val)
        if not np.isfinite(n):
            n = float(default)
    except Exception:
        n = float(default)
    return round(n, digits) if digits is not None else n


def _optimize_weights(mu_vec, cov_mat, mode="ai_bl", custom_weights=None):
    """Solve portfolio weights under AI Black-Litterman Max Sharpe, Min-Variance, Risk-Parity, or Custom."""
    n = len(mu_vec)
    equal_w = np.full(n, 1.0 / n, dtype=np.float64)

    if mode == "custom" and custom_weights and len(custom_weights) == n:
        cw = np.maximum(np.asarray(custom_weights, dtype=np.float64), 0.0)
        if np.sum(cw) > 1e-8:
            return cw / np.sum(cw)
        return equal_w

    max_cap = 0.55 if n >= 3 else 0.90
    min_floor = 0.02 if n >= 3 else 0.05
    bounds = [(min_floor, max_cap)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    if mode == "min_var":
        res = minimize(
            lambda w: float(w @ cov_mat @ w),
            x0=equal_w,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        return res.x if res.success else equal_w

    if mode == "risk_parity":
        def rp_obj(w):
            port_var = max(float(w @ cov_mat @ w), 1e-10)
            rc = w * (cov_mat @ w)
            target = port_var / n
            return float(np.sum((rc - target) ** 2))

        res = minimize(rp_obj, x0=equal_w, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else equal_w

    # Default: AI Black-Litterman Max Forecast Sharpe Ratio
    def neg_sharpe(w):
        ret = float(w @ mu_vec)
        vol = float(np.sqrt(max(w @ cov_mat @ w, 1e-10)))
        return -(ret - RISK_FREE_RATE_ANNUAL) / vol

    res = minimize(neg_sharpe, x0=equal_w, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else equal_w


def optimize_multi_stock_portfolio(
    stock_names, principal, future_date_str, mode="ai_bl", custom_weights=None
):
    """Optimize a 2-10 NSE basket and run Cholesky-correlated GARCH(1,1)-t Monte Carlo + Stress Tests."""
    if not isinstance(stock_names, list) or len(stock_names) < 2 or len(stock_names) > 10:
        return None, ("Select between 2 and 10 stocks to optimize a portfolio.", 400)

    seen_tickers = []
    basket = []
    for name in stock_names:
        resolved_name, ticker = resolve_ticker(name)
        if not ticker:
            return None, (f"Stock not found: {name}", 404)
        if ticker not in seen_tickers:
            seen_tickers.append(ticker)
            basket.append((resolved_name or name, ticker))

    if len(basket) < 2:
        return None, ("Please select at least 2 distinct NSE stocks.", 400)

    future_date = datetime.strptime(future_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    if future_date <= today:
        return None, ("Future date must be in the future.", 400)

    calendar_days = min((future_date - today).days, 365)
    trading_steps = max(1, int(round(calendar_days * (252.0 / 365.0))))
    years = max(calendar_days / 365.0, 1.0 / 252.0)

    panel_df, raw_map, macro_df = get_live_universe_panel(force_refresh=False)

    close_series = {}
    raw_frames = {}
    for s_name, ticker in basket:
        df = raw_map.get(ticker) if raw_map else None
        if df is None or len(df) < 80:
            df = fetch_stock_data(ticker, days=1095)
        if df is None or len(df) < 60 or "Close" not in df:
            return None, (f"Insufficient historical price data for {s_name} ({ticker}).", 404)
        raw_frames[ticker] = df
        s = df["Close"].astype(float).dropna()
        s = s[~s.index.duplicated(keep="last")]
        close_series[ticker] = s

    price_df = pd.DataFrame(close_series).dropna()
    if len(price_df) < 60:
        return None, ("Insufficient overlapping trading days across selected stocks.", 400)

    log_ret_df = np.log(price_df / price_df.shift(1)).dropna()
    n = len(basket)
    tickers = [t for _, t in basket]
    names = [nm for nm, _ in basket]

    # Shrinkage annualized covariance matrix (85% empirical + 15% diagonal prior)
    cov_daily = log_ret_df[tickers].cov().values.astype(np.float64)
    cov_ann_raw = cov_daily * 252.0
    cov_ann = 0.85 * cov_ann_raw + 0.15 * np.diag(np.diag(cov_ann_raw))
    cov_ann = np.nan_to_num(cov_ann, nan=0.04)

    corr_mat = log_ret_df[tickers].corr().values.astype(np.float64)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    np.fill_diagonal(corr_mat, 1.0)

    # AI Black-Litterman expected annual returns
    ai_horizon_avail = any(registry.get_bundle(h) is not None for h in [1, 5, 10, 21, 63])
    mu_ann = np.zeros(n, dtype=np.float64)
    garch_params = []
    indiv_vols = np.zeros(n, dtype=np.float64)

    for i, (s_name, ticker) in enumerate(basket):
        sr = log_ret_df[ticker]
        sample_mu = float(sr.mean())
        sample_years = len(sr) / 252.0
        shrink_w = min(0.55, sample_years / (sample_years + 4.0))
        prior_mu = np.log(1.0 + EQUITY_PRIOR_DRIFT_ANNUAL) / 252.0
        bayesian_ann = np.exp((shrink_w * np.clip(sample_mu, -0.002, 0.002) + (1.0 - shrink_w) * prior_mu) * 252.0) - 1.0

        ai_ann = bayesian_ann
        if ai_horizon_avail:
            try:
                target_ai_dt = (today + timedelta(days=90)).strftime("%Y-%m-%d")
                f_price, _, _, err = make_long_range_prediction(
                    ticker,
                    target_ai_dt,
                    DEFAULT_ALGORITHM,
                    panel_df=panel_df,
                    raw_df=raw_frames[ticker],
                    macro_df=macro_df,
                )
                spot = float(price_df[ticker].iloc[-1])
                if not err and f_price and spot > 0:
                    raw_63d = (f_price / spot) ** (252.0 / 63.0) - 1.0
                    ai_ann = float(np.clip(raw_63d, -0.25, 0.45))
            except Exception:
                ai_ann = bayesian_ann

        mu_ann[i] = float(np.clip(0.60 * ai_ann + 0.40 * bayesian_ann, -0.20, 0.40))
        gp = _fit_garch_parameters(sr)
        garch_params.append(gp)
        ann_v = max(float(np.sqrt(gp["last_var"]) / 100.0 * np.sqrt(252.0)), float(sr.std() * np.sqrt(252.0)))
        indiv_vols[i] = max(ann_v, 0.08)

    w_ai = _optimize_weights(mu_ann, cov_ann, mode="ai_bl")
    w_minvar = _optimize_weights(mu_ann, cov_ann, mode="min_var")
    w_rp = _optimize_weights(mu_ann, cov_ann, mode="risk_parity")
    w_eq = np.full(n, 1.0 / n, dtype=np.float64)

    if mode == "min_var":
        w_active = w_minvar
    elif mode == "risk_parity":
        w_active = w_rp
    elif mode == "custom":
        w_active = _optimize_weights(mu_ann, cov_ann, mode="custom", custom_weights=custom_weights)
    else:
        w_active = w_ai

    def _port_stats(w):
        ret = float(w @ mu_ann)
        vol = float(np.sqrt(max(w @ cov_ann @ w, 1e-10)))
        sharpe = (ret - RISK_FREE_RATE_ANNUAL) / max(vol, 1e-6)
        return {"vol_pct": _sf(vol * 100.0, 0.0, 2), "ret_pct": _sf(ret * 100.0, 0.0, 2), "sharpe": _sf(sharpe, 0.0, 2)}

    # Efficient Frontier curve (18 target returns from min-var to max individual return)
    r_min = _port_stats(w_minvar)["ret_pct"] / 100.0
    r_max = max(float(np.max(mu_ann)), r_min + 0.01)
    frontier_curve = []
    bounds = [(0.02 if n >= 3 else 0.05, 0.55 if n >= 3 else 0.90)] * n
    for r_tgt in np.linspace(r_min, r_max * 0.92, 18):
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, rt=r_tgt: float(w @ mu_ann) - rt},
        ]
        res = minimize(lambda w: float(w @ cov_ann @ w), x0=w_minvar, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            st = _port_stats(res.x)
            frontier_curve.append(st)

    # Seeded random portfolios cloud for visual Efficient Frontier context
    rng_ef = np.random.default_rng(2026)
    cloud = []
    for _ in range(180):
        rw = rng_ef.dirichlet(np.ones(n))
        cloud.append(_port_stats(rw))

    # 5,000-path Correlated Cholesky GARCH(1,1)-t simulation
    n_paths = 5000
    rng_mc = np.random.default_rng(2026)
    try:
        chol = np.linalg.cholesky(corr_mat + 1e-5 * np.eye(n))
    except np.linalg.LinAlgError:
        chol = np.eye(n)

    daily_mu = np.log1p(np.maximum(mu_ann, -0.5)) / 252.0
    var_state = np.array([max(gp["last_var"], 1e-4) for gp in garch_params], dtype=np.float64)
    var_paths = np.broadcast_to(var_state[:, None], (n, n_paths)).copy()
    cum_asset_log = np.zeros((n, n_paths), dtype=np.float64)
    port_paths = np.empty((trading_steps + 1, n_paths), dtype=np.float64)
    port_paths[0] = principal

    omegas = np.array([gp["omega"] for gp in garch_params], dtype=np.float64)[:, None]
    alphas = np.array([gp["alpha"] for gp in garch_params], dtype=np.float64)[:, None]
    betas = np.array([gp["beta"] for gp in garch_params], dtype=np.float64)[:, None]
    avg_nu = float(np.clip(np.mean([gp["nu"] for gp in garch_params]), 4.0, 30.0))
    t_scale = np.sqrt((avg_nu - 2.0) / avg_nu)

    for step in range(trading_steps):
        ind_shocks = rng_mc.standard_t(df=avg_nu, size=(n, n_paths)) * t_scale
        corr_shocks = chol @ ind_shocks
        sigma_pct = np.sqrt(np.maximum(var_paths, 1e-6))
        shock_pct = sigma_pct * corr_shocks
        step_log = daily_mu[:, None] + (shock_pct / 100.0)
        cum_asset_log += step_log
        var_paths = omegas + alphas * (shock_pct ** 2) + betas * var_paths
        # Weighted basket value from correlated asset paths
        port_paths[step + 1] = principal * np.sum(w_active[:, None] * np.exp(cum_asset_log), axis=0)

    p10_curve = np.percentile(port_paths, 10, axis=1)
    p50_curve = np.percentile(port_paths, 50, axis=1)
    p90_curve = np.percentile(port_paths, 90, axis=1)
    terminal_vals = port_paths[-1]

    final_val = float(p50_curve[-1])
    roi = ((final_val - principal) / principal) * 100.0
    cagr = (((final_val / principal) ** (1.0 / years)) - 1.0) * 100.0
    prob_profit = float(np.mean(terminal_vals > principal) * 100.0)

    active_stats = _port_stats(w_active)
    weighted_avg_vol_pct = float(np.sum(w_active * indiv_vols) * 100.0)
    div_saved_pct = max(0.0, weighted_avg_vol_pct - active_stats["vol_pct"])

    pnl = terminal_vals - principal
    var_threshold = float(np.percentile(pnl, 5))
    var_95 = max(0.0, -var_threshold)
    tails = pnl[pnl <= var_threshold]
    cvar_95 = max(0.0, -float(tails.mean())) if len(tails) > 0 else var_95

    run_max = np.maximum.accumulate(port_paths, axis=0)
    dd_paths = (port_paths - run_max) / np.maximum(run_max, 1e-6)
    max_dd_pct = abs(float(np.median(np.min(dd_paths, axis=0)) * 100.0))

    date_offsets = np.linspace(0, calendar_days, trading_steps + 1)
    dates = [(today + timedelta(days=int(round(d)))).strftime("%Y-%m-%d") for d in date_offsets]
    chart_data, p10_chart, p90_chart = [], [], []
    seen_dates = set()
    for idx, d_str in enumerate(dates):
        if d_str in seen_dates:
            continue
        seen_dates.add(d_str)
        chart_data.append({"time": d_str, "value": _sf(p50_curve[idx], principal, 2), "type": "Projected"})
        p10_chart.append({"time": d_str, "value": _sf(p10_curve[idx], principal, 2)})
        p90_chart.append({"time": d_str, "value": _sf(p90_curve[idx], principal, 2)})

    # Marginal risk contribution breakdown
    port_var = max(float(w_active @ cov_ann @ w_active), 1e-10)
    marginal_rc = (w_active * (cov_ann @ w_active)) / port_var

    allocations = []
    for i, (s_name, ticker) in enumerate(basket):
        allocations.append(
            {
                "name": s_name,
                "ticker": ticker,
                "sector": SECTOR_MAP.get(ticker, "Other"),
                "weight_pct": _sf(w_active[i] * 100.0, 0.0, 1),
                "allocated_inr": _sf(principal * w_active[i], 0.0, 2),
                "expected_return_pct": _sf(mu_ann[i] * 100.0, 0.0, 2),
                "ann_vol_pct": _sf(indiv_vols[i] * 100.0, 0.0, 1),
                "risk_contribution_pct": _sf(marginal_rc[i] * 100.0, 0.0, 1),
            }
        )
    allocations.sort(key=lambda x: x["weight_pct"], reverse=True)

    # Stress Tests: empirical basket 21d/63d worst windows + beta-scaled macro shocks
    port_hist_ret = log_ret_df[tickers].values @ w_active
    roll_21 = pd.Series(port_hist_ret).rolling(21).sum().dropna()
    roll_63 = pd.Series(port_hist_ret).rolling(63).sum().dropna()
    worst_21_pct = _sf((np.exp(float(roll_21.min())) - 1.0) * 100.0, -8.0, 2) if len(roll_21) > 0 else -8.0
    worst_63_pct = _sf((np.exp(float(roll_63.min())) - 1.0) * 100.0, -14.0, 2) if len(roll_63) > 0 else -14.0

    # Estimate basket beta vs Nifty if available in macro_df
    port_beta = 1.0
    if macro_df is not None and "nifty_close" in macro_df.columns:
        ns = macro_df["nifty_close"].dropna()
        n_ret = np.log(ns / ns.shift(1)).dropna()
        aligned = pd.concat([pd.Series(port_hist_ret, index=log_ret_df.index), n_ret], axis=1).dropna()
        if len(aligned) > 40 and aligned.iloc[:, 1].var() > 1e-10:
            port_beta = float(np.clip(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(), 0.4, 1.8))

    stress_tests = [
        {
            "scenario": "COVID-Style Systemic Crash (-25% Nifty)",
            "impact_pct": _sf(-25.0 * port_beta, -25.0, 1),
            "projected_value": _sf(principal * (1.0 + (-0.25 * port_beta)), principal * 0.75, 0),
        },
        {
            "scenario": "Rate-Hike / FII Outflow Shock (-12% Nifty)",
            "impact_pct": _sf(-12.0 * port_beta, -12.0, 1),
            "projected_value": _sf(principal * (1.0 + (-0.12 * port_beta)), principal * 0.88, 0),
        },
        {
            "scenario": "Worst Historical 21-Day Window (Empirical)",
            "impact_pct": worst_21_pct,
            "projected_value": _sf(principal * (1.0 + worst_21_pct / 100.0), principal * 0.92, 0),
        },
        {
            "scenario": "Worst Historical 63-Day Drawdown (Empirical)",
            "impact_pct": worst_63_pct,
            "projected_value": _sf(principal * (1.0 + worst_63_pct / 100.0), principal * 0.86, 0),
        },
    ]

    return {
        "mode": mode,
        "principal": _sf(principal, 0.0, 2),
        "final_value": _sf(final_val, principal, 2),
        "p10_value": _sf(p10_curve[-1], principal, 2),
        "p90_value": _sf(p90_curve[-1], principal, 2),
        "roi": _sf(roi, 0.0, 2),
        "cagr": _sf(cagr, 0.0, 2),
        "probability_of_profit": _sf(prob_profit, 50.0, 1),
        "portfolio_vol_pct": active_stats["vol_pct"],
        "sharpe_ratio": active_stats["sharpe"],
        "var_95": _sf(var_95, 0.0, 2),
        "cvar_95": _sf(cvar_95, 0.0, 2),
        "max_drawdown_pct": _sf(max_dd_pct, 0.0, 2),
        "diversification_benefit_pct": _sf(div_saved_pct, 0.0, 2),
        "portfolio_beta": _sf(port_beta, 1.0, 2),
        "allocations": allocations,
        "correlation_matrix": {
            "labels": names,
            "tickers": [t.split(".")[0] for t in tickers],
            "values": [[_sf(corr_mat[r, c], 0.0, 2) for c in range(n)] for r in range(n)],
        },
        "efficient_frontier": {
            "curve": frontier_curve,
            "cloud": cloud,
            "markers": {
                "active": active_stats,
                "ai_bl": _port_stats(w_ai),
                "min_var": _port_stats(w_minvar),
                "equal_weight": _port_stats(w_eq),
            },
        },
        "stress_tests": stress_tests,
        "chart_data": chart_data,
        "p10_chart_data": p10_chart,
        "p90_chart_data": p90_chart,
        "simulation_paths": n_paths,
        "ai_models_used": ai_horizon_avail,
    }, None
