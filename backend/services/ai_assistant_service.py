"""AI assistant services for TrendAI market insights and FinTeach financial education."""

from datetime import datetime, timedelta
import json
import traceback
import yfinance as yf

from backend.config import STOCK_LIST
from backend.services.llm_client import chat_completion
from backend.services.market_data_service import clean_yfinance_dataframe, lookup_ticker


INSIGHTS_SYSTEM_PROMPTS = {
    "PRICE": (
        "You are a price check assistant. "
        "Output ONLY the User's requested current price(s). "
        "Format: 'The current price of [Stock] is [Price].' (Repeat for each stock). "
        "Do NOT include the date or time."
    ),
    "PREDICT": (
        "You are an expert forecaster. "
        "Estimate the future price based on the trends provided. "
        "If comparing, determine which looks stronger. "
        "Briefly explain the trend basis."
    ),
    "ADVICE": (
        "You are a conservative financial advisor. "
        "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing them. "
        "Table Columns: Metric (e.g., Price, Trend, Volatility), [Stock 1 Name], [Stock 2 Name]... "
        "After the table, provide a brief analysis for each. "
        "CRITICAL: You MUST end with a '**Final Verdict**' section explicitly stating which is the better buy."
    ),
    "ANALYZE": (
        "You are TrendAI. "
        "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing them. "
        "For trend/forecast queries: Provide a professional technical analysis (support, resistance, momentum). "
        "For general/fundamental queries (e.g., 'What does it do?', 'Why is it falling?'): "
        "Answer using your knowledge and recent price data. "
        "Keep the tone professional and helpful."
    ),
    "GENERAL": (
        "You are TrendAI, a knowledgeable financial educator. "
        "The user's query is general or theoretical. "
        "Explain financial concepts (e.g., derivatives, P/E ratio, inflation) clearly and simply. "
        "Provide educational investment guidance if asked (e.g., 'Principles of value investing'). "
        "Do NOT make up specific stock data. Focus on concepts and strategy."
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


def _build_market_context(detected_stocks):
    """Fetch last 30 trading days of Close prices for each detected stock."""
    context_parts = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    for s_name, s_ticker in detected_stocks:
        try:
            df = yf.download(s_ticker, start=start_date, end=end_date, progress=False)
            df = clean_yfinance_dataframe(df)
            if df is None or df.empty:
                continue

            closes = df["Close"].dropna().values
            dates = df.index
            if len(closes) == 0:
                continue

            current_price = closes[-1]
            history_str = ", ".join(
                f"{d.strftime('%Y-%m-%d')}: {p:.2f}" for d, p in zip(dates[-30:], closes[-30:])
            )
            context_parts.append(
                f"--- Data for {s_name} ({s_ticker}) ---\n"
                f"Current Price: {current_price:.2f}\n"
                f"Price History (Last 30 days): {history_str}\n"
            )
        except Exception as exc:
            print(f"Error fetching chat context for {s_name}: {exc}")

    return "\n".join(context_parts)


def generate_ai_insights(stock_name=None, trend_data=None, user_question=None):
    """Generate contextual AI stock insights or answer financial queries via LM Studio."""
    try:
        detected_stocks = []
        intent = "ANALYZE"

        if user_question:
            detected_stocks, intent = _extract_companies_and_intent(user_question)
        elif stock_name and trend_data:
            detected_stocks = [(stock_name, "DASHBOARD_CONTEXT")]

        if stock_name and trend_data and not user_question:
            cutoff = trend_data.get("history_cutoff", 100)
            history_dates = trend_data["dates"][:cutoff]
            history_prices = trend_data["prices"][:cutoff]
            data_summary = ", ".join(
                f"{d}: {p}" for d, p in zip(history_dates[-5:], history_prices[-5:])
            )
            combined_context = (
                f"Stock: {stock_name}\n"
                f"Current Price of {stock_name}: {history_prices[-1]:.2f}\n"
                f"Trend: {data_summary}"
            )
        else:
            if not detected_stocks:
                intent = "GENERAL"
                combined_context = ""
            else:
                combined_context = _build_market_context(detected_stocks)
                if not combined_context and intent != "GENERAL":
                    return f"I identified {detected_stocks}, but couldn't fetch data for them.", 200

        if user_question and ("compare" in user_question.lower() or "vs" in user_question.lower()):
            intent = "ADVICE"

        system_prompt = INSIGHTS_SYSTEM_PROMPTS.get(intent, INSIGHTS_SYSTEM_PROMPTS["GENERAL"])
        context_str = combined_context or "No real-time market data available for this query."
        final_prompt = (
            f"User Question: '{user_question}'\n\n"
            f"Market Data Context:\n{context_str}\n\n"
            "Based on this data (if relevant) and your knowledge, answer the user's question."
        )

        reply = chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        return reply, 200
    except Exception as exc:
        traceback.print_exc()
        print(f"Error generating insights: {exc}")
        return (
            "I'm having trouble connecting to my brain right now. Please ensure the AI model is running.",
            500,
        )
