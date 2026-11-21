"""Risk-adjusted multi-factor AI Investment Engine backed by the Global Cross-Sectional Panel."""

import re
import numpy as np
import pandas as pd

from backend.config import DEFAULT_ALGORITHM, STOCK_LIST
from backend.services.feature_engineering import get_live_universe_panel
from backend.services.llm_client import chat_completion, strip_emojis
from backend.services.prediction_service import make_long_range_prediction
from backend.services.sentiment_service import analyze_stock_sentiment


def _zscore_normalize(values):
    """Compute clipped z-scores ([-2.5, 2.5]) mapped to [0, 100]."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1 or np.std(arr) < 1e-9:
        return np.full(len(arr), 50.0)
    z = np.clip((arr - np.mean(arr)) / np.std(arr), -2.5, 2.5)
    return 50.0 + 20.0 * z


def _clean_markdown(text):
    """Remove any stray emojis and collapse extra spaces after Markdown headings."""
    cleaned = strip_emojis(text)
    return re.sub(r"(#+\s+)\s+", r"\1", cleaned).strip()


def _build_investment_advice(top_stocks):
    """Generate a concise Markdown strategic memo via LM Studio with deterministic fallback."""
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
        raw = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=220,
            timeout=25,
        )
        return _clean_markdown(raw)
    except Exception:
        if not top_stocks:
            return "No qualified candidates available. Ensure Global Panel SOTA models are trained."
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


def run_investment_engine(principal, withdrawal_date_str):
    """Rank all 58 NSE stocks via 5-factor Risk-Adjusted Alpha Score using the cached live panel."""
    print(f"--- Running Global Panel Investment Engine: Principal={principal}, Date={withdrawal_date_str} ---")
    panel_df, raw_map, macro_df = get_live_universe_panel(force_refresh=False)

    nifty_3m_ret = 0.0
    if macro_df is not None and "nifty_close" in macro_df.columns and len(macro_df["nifty_close"].dropna()) > 63:
        nifty_s = macro_df["nifty_close"].dropna()
        nifty_3m_ret = float((nifty_s.iloc[-1] / nifty_s.iloc[-63]) - 1.0)

    # 1. Compute daily market gain ranks across the 58-stock universe
    daily_changes = []
    for _, ticker in STOCK_LIST.items():
        df = raw_map.get(ticker) if raw_map else None
        if df is None or len(df) < 120 or "Close" not in df:
            continue
        c_last = float(df["Close"].iloc[-1])
        c_prev = float(df["Close"].iloc[-2])
        if not pd.isna(c_last) and not pd.isna(c_prev) and c_prev > 0:
            daily_changes.append({"ticker": ticker, "pct": ((c_last - c_prev) / c_prev) * 100.0})

    daily_changes.sort(key=lambda x: x["pct"], reverse=True)
    market_ranks = {item["ticker"]: idx + 1 for idx, item in enumerate(daily_changes)}

    # 2. Score all stocks using Global Panel Ensemble predictions
    candidates = []
    for name, ticker in STOCK_LIST.items():
        df = raw_map.get(ticker) if raw_map else None
        if df is None or len(df) < 120:
            continue

        final_price, trend_data, meta, err = make_long_range_prediction(
            ticker,
            withdrawal_date_str,
            DEFAULT_ALGORITHM,
            panel_df=panel_df,
            raw_df=df,
            macro_df=macro_df,
        )
        if err or final_price is None:
            continue

        curr_price = float(df["Close"].iloc[-1])
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

        dispersion = meta.get("ensemble_dispersion")
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
                "confidence_interval": meta.get("confidence_interval"),
                "trend_data": trend_data,
            }
        )

    if not candidates:
        return {
            "top_stocks": [],
            "ai_advice": "No Global Panel models found. Please run `python ml/train_long_range_models.py` first.",
        }

    # 3. 4-Factor Quant Score to shortlist top 8 for FinBERT sentiment scoring
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
    shortlist = candidates[:8]

    # 4. FinBERT Sentiment on top 8 -> Final 5-Factor Risk-Adjusted Alpha Score
    for cand in shortlist:
        sent_data = analyze_stock_sentiment(cand["name"], cand["ticker"])
        cand["sentiment_score"] = sent_data.get("score", 50)
        cand["sentiment_summary"] = sent_data.get("summary", "")
        cand["alpha_score"] = round(0.85 * cand["_quant_score"] + 0.15 * cand["sentiment_score"], 1)
        cand.pop("_quant_score", None)

    shortlist.sort(key=lambda x: x["alpha_score"], reverse=True)
    top_picks = shortlist[:3]

    return {
        "top_stocks": top_picks,
        "ai_advice": _build_investment_advice(top_picks),
    }
