"""AI assistant services for TrendAI market insights and FinTeach financial education."""

from datetime import datetime, timedelta
import json
import traceback
import numpy as np
import yfinance as yf

from backend.config import FORECAST_HORIZONS, SECTOR_MAP, STOCK_LIST
from backend.services.feature_engineering import get_live_universe_panel
from backend.services.llm_client import chat_completion
from backend.services.market_data_service import clean_yfinance_dataframe, lookup_ticker
from backend.services.model_registry import registry
from backend.services.prediction_service import (
    _predict_horizon_quantiles,
    _predict_horizon_z_and_return,
    _prepare_stock_feature_slice,
)
from backend.services.sentiment_service import analyze_stock_sentiment


INSIGHTS_SYSTEM_PROMPTS = {
    "PRICE": (
        "You are a price check assistant. "
        "Output ONLY the User's requested current price(s) based on the live dossier provided. "
        "Format: 'The current price of [Stock] is ₹[Price].' (Repeat for each stock). "
        "Do NOT include the date or time."
    ),
    "PREDICT": (
        "You are an expert quantitative forecaster powered by the NSE 100 Global Panel. "
        "Cite the exact multi-horizon Global Panel model forecasts (1D/5D/21D/63D), volatility, and FinBERT sentiment provided in the quant dossier. "
        "If comparing stocks, declare which has stronger risk-adjusted upside and why."
    ),
    "ADVICE": (
        "You are a disciplined quantitative investment advisor. "
        "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing Price, Volatility, RSI-14, AI Forecast, and FinBERT Sentiment. "
        "After the table, synthesize factor momentum, tail risk, and sentiment catalysts from the dossier. "
        "CRITICAL: End with a '**Final Verdict**' section stating key risks and clearer conviction."
    ),
    "ANALYZE": (
        "You are TrendAI, an institutional quantitative equity analyst. "
        "Always ground your answer in the live NSE Global Panel quant dossier, multi-horizon forecasts, and FinBERT news catalysts provided below. "
        "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing them. "
        "Provide structured bullet points covering Price & Trend, Quant Model Signal, Risk Regime, and News Sentiment."
    ),
    "GENERAL": (
        "You are TrendAI, a knowledgeable financial educator. "
        "The user's query is general or theoretical. "
        "Explain financial concepts (e.g., derivatives, P/E ratio, inflation, Sharpe ratio) clearly and simply. "
        "Provide educational investment guidance if asked. "
        "Do NOT fabricate specific stock prices. Focus on concepts and strategy."
    ),
}


def answer_general_knowledge(user_question):
    """Answer general financial literacy questions using the FinTeach educator persona."""
    system_prompt = (
        "You are FinTeach, a helpful and knowledgeable financial educator. "
        "The user is asking a general or theoretical financial question. "
        "Explain concepts clearly, simply, and professionally. "
        "Use formatting (bolding, lists) to make it easy to read. "
        "Do NOT provide specific stock recommendations or real-time prices unless explicitly asked to use general knowledge. "
        "Focus on definitions, strategies, and economic principles."
    )
    return chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7,
        max_tokens=800,
    )


def _extract_companies_and_intent(user_question):
    """Detect known NSE stocks, heuristic intent, and extract additional tickers via LLM."""
    detected_stocks = []
    user_text_lower = user_question.lower()

    for name, tick in STOCK_LIST.items():
        if name.lower() in user_text_lower or tick.split(".")[0].lower() in user_text_lower:
            if (name, tick) not in detected_stocks:
                detected_stocks.append((name, tick))

    if "price" in user_text_lower and "predict" not in user_text_lower:
        intent = "PRICE"
    elif "predict" in user_text_lower or "future" in user_text_lower:
        intent = "PREDICT"
    elif "invest" in user_text_lower or "buy" in user_text_lower or "compare" in user_text_lower or "vs" in user_text_lower:
        intent = "ADVICE"
    else:
        intent = "ANALYZE"

    extraction_prompt = (
        "Analyze the user's question to extract:\n"
        "1. A list of ALL SPECIFIC company names/tickers mentioned (e.g., ['Apple', 'TCS']).\n"
        "   - Do NOT extract general terms like 'stocks', 'market', 'mutual funds', 'bonds' as queries.\n"
        "2. The likely region (INDIA or GLOBAL).\n"
        "3. The User Intent (PRICE, PREDICT, ADVICE, ANALYZE, GENERAL).\n"
        'Output valid JSON ONLY: {"queries": ["name1", "name2"], "region": "...", "intent": "..."}'
    )

    try:
        raw_content = chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON entity extractor. Output valid JSON only."},
                {"role": "user", "content": f"{extraction_prompt}\nUser Query: '{user_question}'"},
            ],
            temperature=0.0,
            max_tokens=150,
            timeout=6,
        ).strip()

        if raw_content.startswith("```"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_content)
        llm_queries = parsed.get("queries", [])
        region = parsed.get("region", "INDIA")
        intent = parsed.get("intent", intent).upper()

        current_tickers = [t[1] for t in detected_stocks]
        for query in llm_queries:
            already_found = any(
                query.lower() in existing_name.lower() or existing_name.lower() in query.lower()
                for existing_name, _ in detected_stocks
            )
            if not already_found:
                found_tick, found_name = lookup_ticker(
                    query, prefer_indian=(region.upper() == "INDIA")
                )
                if found_tick and found_tick not in current_tickers:
                    detected_stocks.append((found_name, found_tick))
                    current_tickers.append(found_tick)
    except Exception as exc:
        print(f"LLM entity extraction skipped/failed: {exc}")

    return detected_stocks, intent


def _resolve_stock_by_name(raw_name):
    """Resolve an equity name or symbol string into a (name, ticker) pair."""
    if not raw_name or str(raw_name).strip().lower() in ("market assistant", "none", "null"):
        return None
    clean = str(raw_name).strip()
    clean_lower = clean.lower()
    for s_name, s_tick in STOCK_LIST.items():
        if s_name.lower() == clean_lower or s_tick.lower() == clean_lower or s_tick.split(".")[0].lower() == clean_lower:
            return s_name, s_tick
    found_tick, found_name = lookup_ticker(clean, prefer_indian=True)
    if found_tick:
        return found_name or clean, found_tick
    return None


def _build_quant_dossier(detected_stocks):
    """Assemble live Global Panel factors, multi-horizon forecasts, and FinBERT sentiment per stock."""
    context_parts = []
    structured_dossiers = []
    panel_df = None
    try:
        panel_df, _, _ = get_live_universe_panel(force_refresh=False)
    except Exception as exc:
        print(f"Quant dossier universe panel fallback: {exc}")

    include_sentiment = len(detected_stocks) <= 2

    for s_name, s_ticker in detected_stocks[:4]:
        dossier = {
            "name": s_name,
            "ticker": s_ticker,
            "sector": SECTOR_MAP.get(s_ticker, "NSE Equity"),
            "spot_price": None,
            "ret_5d_pct": None,
            "ret_21d_pct": None,
            "ann_vol_pct": None,
            "rsi_14": None,
            "mom_rank": None,
            "sector_excess_21d_pct": None,
            "vix_level": None,
            "forecasts": [],
            "sentiment": None,
            "history_summary": "",
        }

        try:
            feat_df, _ = _prepare_stock_feature_slice(s_ticker, panel_df=panel_df)
            if feat_df is not None and not feat_df.empty:
                latest = feat_df.iloc[-1]
                close_col = "Close" if "Close" in feat_df.columns else None
                spot = float(latest[close_col]) if close_col else None
                if spot is not None and np.isfinite(spot):
                    dossier["spot_price"] = round(spot, 2)

                if "ret_5d" in latest and np.isfinite(latest["ret_5d"]):
                    dossier["ret_5d_pct"] = round((np.exp(float(latest["ret_5d"])) - 1.0) * 100.0, 2)
                if "ret_21d" in latest and np.isfinite(latest["ret_21d"]):
                    dossier["ret_21d_pct"] = round((np.exp(float(latest["ret_21d"])) - 1.0) * 100.0, 2)
                if "vol_21d" in latest and np.isfinite(latest["vol_21d"]):
                    dossier["ann_vol_pct"] = round(float(latest["vol_21d"]) * np.sqrt(252.0) * 100.0, 1)
                if "rsi_14" in latest and np.isfinite(latest["rsi_14"]):
                    dossier["rsi_14"] = round(float(latest["rsi_14"]), 1)
                if "cs_rank_ret_21d" in latest and np.isfinite(latest["cs_rank_ret_21d"]):
                    dossier["mom_rank"] = int(round((float(latest["cs_rank_ret_21d"]) + 0.5) * 100.0))
                if "sector_excess_ret_21d" in latest and np.isfinite(latest["sector_excess_ret_21d"]):
                    dossier["sector_excess_21d_pct"] = round(float(latest["sector_excess_ret_21d"]) * 100.0, 2)
                if "vix_level" in latest and np.isfinite(latest["vix_level"]):
                    dossier["vix_level"] = round(float(latest["vix_level"]) * 20.0, 2)

                if close_col:
                    closes = feat_df[close_col].dropna().values[-10:]
                    dates = feat_df.index[-10:]
                    dossier["history_summary"] = ", ".join(
                        f"{d.strftime('%Y-%m-%d')}: {float(p):.2f}" for d, p in zip(dates, closes)
                    )

                for h in FORECAST_HORIZONS:
                    if registry.get_bundle(h) is None:
                        continue
                    try:
                        ret_h, _, err_h = _predict_horizon_z_and_return(s_ticker, "Ensemble", feat_df, horizon=h)
                        if err_h is None and ret_h is not None and dossier["spot_price"]:
                            low_h, high_h = _predict_horizon_quantiles(s_ticker, feat_df, h, ret_h)
                            dossier["forecasts"].append(
                                {
                                    "horizon": h,
                                    "return_pct": round((np.exp(ret_h) - 1.0) * 100.0, 2),
                                    "target_price": round(dossier["spot_price"] * np.exp(ret_h), 2),
                                    "lower_price": round(dossier["spot_price"] * np.exp(low_h), 2),
                                    "upper_price": round(dossier["spot_price"] * np.exp(high_h), 2),
                                }
                            )
                    except Exception:
                        pass
        except Exception as exc:
            print(f"Quant dossier feature error for {s_name}: {exc}")

        if dossier["spot_price"] is None:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                df = yf.download(s_ticker, start=start_date, end=end_date, progress=False)
                df = clean_yfinance_dataframe(df)
                if df is not None and not df.empty:
                    closes = df["Close"].dropna().values
                    dates = df.index
                    if len(closes) > 0:
                        dossier["spot_price"] = round(float(closes[-1]), 2)
                        if len(closes) >= 22:
                            dossier["ret_21d_pct"] = round(((closes[-1] / closes[-22]) - 1.0) * 100.0, 2)
                        dossier["history_summary"] = ", ".join(
                            f"{d.strftime('%Y-%m-%d')}: {float(p):.2f}" for d, p in zip(dates[-10:], closes[-10:])
                        )
            except Exception as exc:
                print(f"Fallback price download failed for {s_name}: {exc}")

        if include_sentiment:
            try:
                sent = analyze_stock_sentiment(s_name, s_ticker)
                if isinstance(sent, dict):
                    top_headlines = [
                        f"[{n.get('sentiment_label', 'Neutral')}] {n.get('title', '')}"
                        for n in (sent.get("news") or [])[:3]
                        if n.get("title")
                    ]
                    dossier["sentiment"] = {
                        "score": sent.get("score", 50),
                        "label": sent.get("label", "Neutral"),
                        "summary": sent.get("summary", ""),
                        "headlines": top_headlines,
                    }
            except Exception as exc:
                print(f"Sentiment dossier skipped for {s_name}: {exc}")

        structured_dossiers.append(dossier)

        lines = [f"--- Quant Dossier: {s_name} ({s_ticker}) | Sector: {dossier['sector']} ---"]
        if dossier["spot_price"] is not None:
            lines.append(f"Spot Price: INR {dossier['spot_price']:.2f}")
        metrics_bits = []
        if dossier["ret_5d_pct"] is not None:
            metrics_bits.append(f"5D Ret: {dossier['ret_5d_pct']:+.2f}%")
        if dossier["ret_21d_pct"] is not None:
            metrics_bits.append(f"21D Ret: {dossier['ret_21d_pct']:+.2f}%")
        if dossier["ann_vol_pct"] is not None:
            metrics_bits.append(f"Ann Vol: {dossier['ann_vol_pct']:.1f}%")
        if dossier["rsi_14"] is not None:
            metrics_bits.append(f"RSI-14: {dossier['rsi_14']:.1f}")
        if dossier["mom_rank"] is not None:
            metrics_bits.append(f"NSE Momentum Rank: {dossier['mom_rank']}/100")
        if dossier["sector_excess_21d_pct"] is not None:
            metrics_bits.append(f"21D Sector Excess: {dossier['sector_excess_21d_pct']:+.2f}%")
        if dossier["vix_level"] is not None:
            metrics_bits.append(f"India VIX: {dossier['vix_level']:.1f}")
        if metrics_bits:
            lines.append("Technical & Factor Metrics: " + " | ".join(metrics_bits))

        if dossier["forecasts"]:
            fc_str = " | ".join(
                f"{f['horizon']}D: {f['return_pct']:+.2f}% (Target INR {f['target_price']:.2f}, 90% CI {f['lower_price']:.2f}-{f['upper_price']:.2f})"
                for f in dossier["forecasts"]
            )
            lines.append(f"Global Panel Ensemble Forecasts: {fc_str}")

        if dossier["sentiment"]:
            s_info = dossier["sentiment"]
            lines.append(
                f"FinBERT News Sentiment: {s_info['score']}/100 ({s_info['label']}) - {s_info['summary']}"
            )
            if s_info["headlines"]:
                lines.append("Recent Catalysts: " + " ; ".join(s_info["headlines"]))

        if dossier["history_summary"]:
            lines.append(f"Recent 10 Closes: {dossier['history_summary']}")

        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts), structured_dossiers


def _build_deterministic_quant_brief(structured_dossiers, intent, active_context=None):
    """Build a factual Markdown quantitative research memo when local LM Studio is offline."""
    if not structured_dossiers:
        tab_label = (active_context or {}).get("tab") or "Market Overview"
        return (
            f"### TrendAI Quant Desk ({tab_label})\n\n"
            "Local LLM synthesis (LM Studio) is currently offline, and no specific equity ticker was detected in your prompt.\n\n"
            "- Mention an NSE stock (e.g., **Infosys**, **HDFC Bank**, **Reliance**) to pull live Global Panel forecasts, factor ranks, and FinBERT headline catalysts.\n"
            "- Open the **Alpha Radar** tab to inspect all 100 NSE equities scored side-by-side."
        )

    sections = []
    if len(structured_dossiers) >= 2:
        sections.append("### Quantitative Equity Comparison")
        sections.append("| Stock | Spot (INR) | 21D Ret | Ann Vol | RSI-14 | 21D AI Signal | FinBERT |")
        sections.append("|---|---:|---:|---:|---:|---:|---|")
        for d in structured_dossiers:
            fc21 = next((f for f in d["forecasts"] if f["horizon"] == 21), d["forecasts"][0] if d["forecasts"] else None)
            fc_txt = f"{fc21['return_pct']:+.2f}%" if fc21 else "N/A"
            sent_txt = f"{d['sentiment']['score']}/100 ({d['sentiment']['label']})" if d["sentiment"] else "N/A"
            ret21_txt = f"{d['ret_21d_pct']:+.2f}%" if d["ret_21d_pct"] is not None else "N/A"
            vol_txt = f"{d['ann_vol_pct']:.1f}%" if d["ann_vol_pct"] is not None else "N/A"
            rsi_txt = f"{d['rsi_14']}" if d["rsi_14"] is not None else "N/A"
            sections.append(
                f"| **{d['name']}** | {d['spot_price'] or 'N/A'} | {ret21_txt} | {vol_txt} | {rsi_txt} | {fc_txt} | {sent_txt} |"
            )

    for d in structured_dossiers:
        sections.append(f"### {d['name']} ({d['ticker']}) · {d['sector']}")
        bullets = []
        if d["spot_price"] is not None:
            bullets.append(f"- **Spot Price:** INR {d['spot_price']:.2f}")
        if d["ret_21d_pct"] is not None or d["ann_vol_pct"] is not None:
            r5 = f"{d['ret_5d_pct']:+.2f}%" if d["ret_5d_pct"] is not None else "N/A"
            r21 = f"{d['ret_21d_pct']:+.2f}%" if d["ret_21d_pct"] is not None else "N/A"
            vol_s = f"{d['ann_vol_pct']:.1f}%" if d["ann_vol_pct"] is not None else "N/A"
            rsi_s = f"{d['rsi_14']:.1f}" if d["rsi_14"] is not None else "N/A"
            bullets.append(
                f"- **Momentum & Volatility:** 5D `{r5}`, 21D `{r21}`, Annualized Vol `{vol_s}`, RSI-14 `{rsi_s}`"
            )
        if d["mom_rank"] is not None:
            pos_str = f"- **Cross-Sectional Positioning:** NSE Momentum Rank `{d['mom_rank']}/100`"
            if d["sector_excess_21d_pct"] is not None:
                pos_str += f" | 21D Sector Excess `{d['sector_excess_21d_pct']:+.2f}%`"
            bullets.append(pos_str)
        if d["forecasts"]:
            fc_line = ", ".join(
                f"**{f['horizon']}D:** {f['return_pct']:+.2f}% (INR {f['target_price']:.2f})"
                for f in d["forecasts"]
            )
            bullets.append(f"- **Global Panel Forecasts:** {fc_line}")
        if d["sentiment"]:
            s_info = d["sentiment"]
            bullets.append(
                f"- **FinBERT News Regime:** `{s_info['score']}/100 ({s_info['label']})` — {s_info['summary']}"
            )
        if bullets:
            sections.extend(bullets)

    if len(structured_dossiers) >= 2 and intent == "ADVICE":
        ranked = sorted(
            structured_dossiers,
            key=lambda x: next((f["return_pct"] for f in x["forecasts"] if f["horizon"] == 21), x["ret_21d_pct"] or 0.0),
            reverse=True,
        )
        lead = ranked[0]
        sections.append(
            f"**Final Verdict:** Quantitatively, **{lead['name']}** exhibits the stronger composite signal in this snapshot; always pair position sizing with the 90% volatility confidence band."
        )

    sections.append("_Note: Generated directly from live Global Panel data & FinBERT signals (LM Studio offline fallback)._")
    return "\n\n".join(sections)


def generate_ai_insights(stock_name=None, trend_data=None, user_question=None, active_context=None):
    """Generate contextual AI stock insights grounded in live Global Panel dossiers + FinBERT."""
    try:
        detected_stocks = []
        intent = "ANALYZE"

        if user_question:
            detected_stocks, intent = _extract_companies_and_intent(user_question)
        elif stock_name and trend_data:
            resolved = _resolve_stock_by_name(stock_name)
            detected_stocks = [resolved] if resolved else [(stock_name, "DASHBOARD_CONTEXT")]

        if not detected_stocks:
            context_candidate = (active_context or {}).get("stock_name") or stock_name
            q_lower = (user_question or "").lower()
            deictic_terms = (
                "this stock",
                "this equity",
                "this forecast",
                "explain",
                "bull",
                "bear",
                "risk",
                "catalyst",
                "sentiment",
                "driver",
            )
            if context_candidate and (intent != "GENERAL" or any(tok in q_lower for tok in deictic_terms)):
                resolved = _resolve_stock_by_name(context_candidate)
                if resolved:
                    detected_stocks.append(resolved)

        market_context, structured_dossiers = _build_quant_dossier(detected_stocks)

        screen_context_block = ""
        if isinstance(active_context, dict) and active_context:
            parts = []
            if active_context.get("tab"):
                parts.append(f"Active Screen: {active_context['tab']}")
            if active_context.get("forecastSummary"):
                parts.append(f"Visible Forecast: {active_context['forecastSummary']}")
            if parts:
                screen_context_block = "User Screen Context:\n" + "\n".join(parts) + "\n\n"

        if stock_name and trend_data and not user_question:
            cutoff = trend_data.get("history_cutoff", 100)
            history_dates = trend_data["dates"][:cutoff]
            history_prices = trend_data["prices"][:cutoff]
            data_summary = ", ".join(
                f"{d}: {p}" for d, p in zip(history_dates[-5:], history_prices[-5:])
            )
            intent = "ANALYZE"
            final_prompt = (
                f"Analyze {stock_name} based on this prediction trend: {data_summary}\n\n"
                f"{screen_context_block}"
                f"Live Quantitative & News Dossier:\n{market_context or 'Spot snapshot unavailable.'}\n\n"
                "Provide key support/resistance, model drivers, and risks."
            )
        else:
            if not detected_stocks:
                intent = "GENERAL"
                final_prompt = f"{screen_context_block}User Question: '{user_question}'"
            else:
                if user_question and ("compare" in user_question.lower() or "vs" in user_question.lower()):
                    intent = "ADVICE"
                final_prompt = (
                    f"User Question: '{user_question}'\n\n"
                    f"{screen_context_block}"
                    f"Live Quantitative & News Dossier:\n{market_context or 'Spot snapshot unavailable.'}\n\n"
                    "Answer accurately using the real metrics, Global Panel forecasts, and FinBERT sentiment above."
                )

        system_prompt = INSIGHTS_SYSTEM_PROMPTS.get(intent, INSIGHTS_SYSTEM_PROMPTS["ANALYZE"])
        try:
            reply = chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.6,
                max_tokens=1200,
                timeout=35,
            )
            return reply, 200
        except Exception as llm_exc:
            print(f"LM Studio unavailable for TrendAI insights ({llm_exc}); returning deterministic quant brief.")
            brief = _build_deterministic_quant_brief(
                structured_dossiers,
                intent=intent,
                active_context=active_context,
            )
            return brief, 200
    except Exception as exc:
        traceback.print_exc()
        print(f"Error generating insights: {exc}")
        return (
            "I'm having trouble assembling market insights right now. Please try again shortly.",
            500,
        )
