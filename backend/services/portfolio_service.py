"""Institutional portfolio projection using 10,000-path GARCH(1,1) Student-t Monte Carlo."""

from datetime import datetime, timedelta
import numpy as np

from backend.services.market_data_service import fetch_stock_data, resolve_ticker

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
