"""Vectorized Risk-Adjusted Multi-Factor AI Investment Engine backed by the Global Cross-Sectional Panel."""

import asyncio
from datetime import timedelta
import re

import numpy as np
import pandas as pd

from backend.config import DEFAULT_ALGORITHM, FORECAST_HORIZONS, LONG_RANGE_HORIZONS, STOCK_LIST
from backend.security.logging_utils import get_logger
from backend.services.feature_engineering import get_live_universe_panel_async
from backend.services.llm_client import async_chat_completion, strip_emojis
from backend.services.model_registry import registry
from backend.services.prediction_service import (
    batch_score_universe_horizon,
    make_long_range_prediction,
)
from backend.services.runtime import runtime
from backend.services.sentiment_service import analyze_stock_sentiment_async

logger = get_logger(__name__)


def _zscore_normalize(values) -> np.ndarray:
    """Compute clipped z-scores ([-2.5, 2.5]) mapped to [0, 100]."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1 or np.std(arr) < 1e-9:
        return np.full(len(arr), 50.0)
    z = np.clip((arr - np.mean(arr)) / np.std(arr), -2.5, 2.5)
    return 50.0 + 20.0 * z


def _clean_markdown(text: str) -> str:
    """Remove stray emojis and collapse extra spaces after Markdown headings."""
    cleaned = strip_emojis(text)
    return re.sub(r"(#+\s+)\s+", r"\1", cleaned).strip()


async def _build_investment_advice_async(top_stocks: list[dict]) -> str:
    """Generate a concise Markdown strategic memo via LM Studio with deterministic fallback."""
    if not top_stocks:
        return "No qualified candidates available. Ensure Global Panel SOTA models are trained."

    prompt = "You are a Senior Quantitative Investment Strategist. Review these top 3 multi-factor NSE picks:\n"
    for stock in top_stocks:
        prompt += (
            f"- {stock['name']} ({stock['ticker']}): Alpha Score {stock['alpha_score']}/100, "
            f"Forecast {stock['profit_pct']:+.1f}%, Ann. Volatility {stock['volatility']:.1f}%, "
            f"Consensus {stock['consensus_score']}/100, FinBERT Sentiment {stock['sentiment_score']}/100.\n"
        )
    prompt += (
        "\nProvide a professional investment memo in strict Markdown format (do not use any emojis):\n\n"
        "### Strategic Analysis\n"
        "* [Point 1: Risk-adjusted return vs volatility balance]\n"
        "* [Point 2: Model consensus and FinBERT sentiment alignment]\n\n"
        "### Final Verdict\n"
        "**[Verdict: Aggressive Growth / Balanced / Conservative]**\n"
        "[One sentence justification].\n\n"
        "Keep it concise, professional, under 130 words, and strictly free of emojis."
    )

    try:
        raw = await async_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=220,
            timeout=25.0,
        )
        return _clean_markdown(raw)
    except Exception:
        leader = top_stocks[0]
        return (
            "### Strategic Analysis\n"
            f"* **{leader['name']}** leads with an Alpha Score of **{leader['alpha_score']}/100**, "
            f"combining a {leader['profit_pct']:+.1f}% forecast with {leader['volatility']:.1f}% annualized volatility.\n"
            f"* Ensemble consensus stands at **{leader['consensus_score']}/100** with FinBERT sentiment at "
            f"**{leader['sentiment_score']}/100**.\n\n"
            "### Final Verdict\n"
            "**Balanced Quantitative Allocation**\n"
            "Diversify across the top 3 risk-adjusted candidates rather than concentrating in a single ticker."
        )


def _score_universe_candidates_sync(panel_df: pd.DataFrame, raw_map: dict, macro_df: pd.DataFrame, withdrawal_date_str: str):
    """Vectorized 100-stock scoring across loaded horizons using at most 5 batch passes."""
    if panel_df is None or panel_df.empty or not raw_map:
        return []

    nifty_3m_ret = 0.0
    if macro_df is not None and "nifty_close" in macro_df.columns and len(macro_df["nifty_close"].dropna()) > 63:
        nifty_s = macro_df["nifty_close"].dropna()
        nifty_3m_ret = float((nifty_s.iloc[-1] / nifty_s.iloc[-63]) - 1.0)

    daily_changes = []
    for _, ticker in STOCK_LIST.items():
        df = raw_map.get(ticker)
        if df is None or len(df) < 120 or "Close" not in df:
            continue
        c_last = float(df["Close"].iloc[-1])
        c_prev = float(df["Close"].iloc[-2])
        if not pd.isna(c_last) and not pd.isna(c_prev) and c_prev > 0:
            daily_changes.append({"ticker": ticker, "pct": ((c_last - c_prev) / c_prev) * 100.0})
    daily_changes.sort(key=lambda x: x["pct"], reverse=True)
    market_ranks = {item["ticker"]: idx + 1 for idx, item in enumerate(daily_changes)}

    available_horizons = [h for h in ([1] + LONG_RANGE_HORIZONS) if registry.get_bundle(h) is not None]
    if not available_horizons:
        return []

    # 5 vectorized batch forward passes cover all 100 stocks
    horizon_batches = {h: batch_score_universe_horizon(panel_df, h) for h in available_horizons}

    # Build lookup ticker -> (index inside batch) per horizon
    batch_ticker_idx: dict[int, dict[str, int]] = {
        h: {t: idx for idx, t in enumerate(b["tickers"])} for h, b in horizon_batches.items()
    }

    future_dt = pd.to_datetime(withdrawal_date_str)
    panel_grouped = {t: grp.sort_values("date") for t, grp in panel_df.groupby("ticker")}

    candidates = []
    for name, ticker in STOCK_LIST.items():
        df = raw_map.get(ticker)
        p_sub = panel_grouped.get(ticker)
        if df is None or len(df) < 120 or p_sub is None or p_sub.empty:
            continue

        last_date = pd.to_datetime(p_sub["date"].iloc[-1])
        n_steps = len(pd.bdate_range(start=last_date + timedelta(days=1), end=future_dt))
        if n_steps <= 0:
            continue

        anchor_steps = [0]
        anchor_rets = [0.0]
        anchor_lows = [0.0]
        anchor_highs = [0.0]
        dispersions = []

        for h in available_horizons:
            idx_in_batch = batch_ticker_idx[h].get(ticker)
            if idx_in_batch is None:
                continue
            b = horizon_batches[h]
            anchor_steps.append(h)
            anchor_rets.append(float(b["pred_log_ret"][idx_in_batch]))
            anchor_lows.append(float(b["low_log_ret"][idx_in_batch]))
            anchor_highs.append(float(b["high_log_ret"][idx_in_batch]))
            dispersions.append(float(b["dispersion"][idx_in_batch]))

        if len(anchor_steps) <= 1:
            continue

        max_anchor = anchor_steps[-1]
        clamped_step = min(float(n_steps), float(max_anchor))
        target_ret = float(np.interp(clamped_step, anchor_steps, anchor_rets))
        target_low = float(np.interp(clamped_step, anchor_steps, anchor_lows))
        target_high = float(np.interp(clamped_step, anchor_steps, anchor_highs))

        if n_steps > max_anchor:
            hist_ret = p_sub["ret_1d"].tail(504).dropna()
            raw_daily_mu = float(hist_ret.mean()) if len(hist_ret) > 20 else 0.0004
            nifty_prior_daily_mu = np.log(1.115) / 252.0
            shrunk_daily_mu = 0.35 * np.clip(raw_daily_mu, -0.0015, 0.0015) + 0.65 * nifty_prior_daily_mu
            recent_vol = max(float(p_sub["vol_21d"].iloc[-1]), 0.008)
            extra_days = float(n_steps - max_anchor)
            damped_extra = shrunk_daily_mu * extra_days
            cone_extra = 1.645 * recent_vol * np.sqrt(extra_days)
            target_ret += damped_extra
            target_low += damped_extra - cone_extra
            target_high += damped_extra + cone_extra

        curr_price = float(df["Close"].iloc[-1])
        final_price = curr_price * float(np.exp(target_ret))
        profit_pct = ((final_price - curr_price) / curr_price) * 100.0

        recent_close = df["Close"].tail(126).dropna()
        daily_ret = recent_close.pct_change().dropna()
        ann_vol = max(float(daily_ret.std() * np.sqrt(252.0) * 100.0), 5.0)
        downside_ret = daily_ret[daily_ret < 0]
        downside_vol = (
            max(float(downside_ret.std() * np.sqrt(252.0) * 100.0), 5.0)
            if len(downside_ret) > 5
            else ann_vol
        )

        forecast_sharpe = profit_pct / ann_vol
        stock_3m_ret = float((recent_close.iloc[-1] / recent_close.iloc[max(0, len(recent_close) - 63)]) - 1.0)
        rel_strength_pct = (stock_3m_ret - nifty_3m_ret) * 100.0

        dispersion = float(np.mean(dispersions)) if dispersions else None
        consensus_score = (
            float(np.clip(100.0 - dispersion * 2500.0, 20.0, 98.0))
            if dispersion is not None
            else 70.0
        )

        candidates.append(
            {
                "name": name,
                "ticker": ticker,
                "current_price": round(curr_price, 2),
                "predicted_price": round(final_price, 2),
                "profit_pct": round(profit_pct, 2),
                "volatility": round(ann_vol, 2),
                "downside_vol": round(downside_vol, 2),
                "forecast_sharpe": round(forecast_sharpe, 3),
                "rel_strength_pct": round(rel_strength_pct, 2),
                "consensus_score": round(consensus_score, 1),
                "nse_rank": market_ranks.get(ticker, 99),
                "confidence_interval": {
                    "lower": round(curr_price * float(np.exp(target_low)), 2),
                    "upper": round(curr_price * float(np.exp(target_high)), 2),
                },
            }
        )

    if not candidates:
        return []

    sharpe_norm = _zscore_normalize([c["forecast_sharpe"] for c in candidates])
    consensus_norm = _zscore_normalize([c["consensus_score"] for c in candidates])
    stability_norm = _zscore_normalize([-c["downside_vol"] for c in candidates])
    rs_norm = _zscore_normalize([c["rel_strength_pct"] for c in candidates])

    for idx, cand in enumerate(candidates):
        cand["_quant_score"] = (
            0.41 * sharpe_norm[idx]
            + 0.24 * consensus_norm[idx]
            + 0.175 * stability_norm[idx]
            + 0.175 * rs_norm[idx]
        )

    candidates.sort(key=lambda x: x["_quant_score"], reverse=True)
    return candidates[:8]


def _attach_top_trend_charts(top_picks: list[dict], withdrawal_date_str: str, panel_df: pd.DataFrame, raw_map: dict, macro_df: pd.DataFrame) -> None:
    """Populate chart trend_data only for the final top 3 candidates without wasted SHAP computation."""
    for pick in top_picks:
        ticker = pick["ticker"]
        final_price, trend_data, meta, err = make_long_range_prediction(
            ticker,
            withdrawal_date_str,
            DEFAULT_ALGORITHM,
            panel_df=panel_df,
            raw_df=raw_map.get(ticker),
            macro_df=macro_df,
            include_explainability=False,
        )
        if not err and final_price is not None and trend_data is not None:
            pick["trend_data"] = trend_data
            if meta and meta.get("confidence_interval"):
                pick["confidence_interval"] = meta["confidence_interval"]
        else:
            pick["trend_data"] = None


async def run_investment_engine_async(principal: float, withdrawal_date_str: str) -> dict:
    """Async vectorized multi-factor NSE 100 ranker + concurrent FinBERT sentiment + AI memo."""
    logger.info("Running Global Panel Investment Engine: principal=%.2f date=%s", principal, withdrawal_date_str)
    panel_df, raw_map, macro_df = await get_live_universe_panel_async(force_refresh=False)

    shortlist = await runtime.run_cpu(
        _score_universe_candidates_sync, panel_df, raw_map, macro_df, withdrawal_date_str
    )

    if not shortlist:
        has_models = any(registry.get_bundle(h) is not None for h in FORECAST_HORIZONS)
        msg = (
            "No Global Panel models found. Please run `python ml/train_long_range_models.py` first."
            if not has_models
            else "No qualified candidates matched the target withdrawal date."
        )
        return {"top_stocks": [], "ai_advice": msg}

    # Concurrent FinBERT sentiment scoring across the 8 shortlisted candidates
    sem = asyncio.Semaphore(4)

    async def _bounded_sent(cand: dict):
        async with sem:
            return await analyze_stock_sentiment_async(cand["name"], cand["ticker"])

    sent_results = await asyncio.gather(*(_bounded_sent(c) for c in shortlist), return_exceptions=True)
    for cand, sent_res in zip(shortlist, sent_results):
        if isinstance(sent_res, Exception) or not isinstance(sent_res, dict):
            cand["sentiment_score"] = 50
            cand["sentiment_summary"] = ""
        else:
            cand["sentiment_score"] = sent_res.get("score", 50)
            cand["sentiment_summary"] = sent_res.get("summary", "")
        cand["alpha_score"] = round(0.85 * cand["_quant_score"] + 0.15 * cand["sentiment_score"], 1)
        cand.pop("_quant_score", None)

    shortlist.sort(key=lambda x: x["alpha_score"], reverse=True)
    top_picks = shortlist[:3]

    await runtime.run_cpu(_attach_top_trend_charts, top_picks, withdrawal_date_str, panel_df, raw_map, macro_df)
    ai_advice = await _build_investment_advice_async(top_picks)

    return {
        "top_stocks": top_picks,
        "ai_advice": ai_advice,
    }
