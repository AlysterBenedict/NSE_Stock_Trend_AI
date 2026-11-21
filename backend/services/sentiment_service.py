"""Financial news sentiment analysis powered by ProsusAI/finbert + recency decay + optional LLM summary."""

from datetime import datetime, timezone
import math
import torch
import yfinance as yf

from backend.config import FINBERT_MODEL_NAME
from backend.services.attention_net import get_torch_device
from backend.services.llm_client import chat_completion

_FINBERT_STATE = {"tokenizer": None, "model": None, "device": None, "failed": False}


def _get_finbert():
    """Lazy-load ProsusAI/finbert onto CUDA (FP16 on RTX 4060) or CPU."""
    if _FINBERT_STATE["model"] is not None:
        return _FINBERT_STATE["tokenizer"], _FINBERT_STATE["model"], _FINBERT_STATE["device"]
    if _FINBERT_STATE["failed"]:
        return None, None, None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = get_torch_device()
        print(f"--- Loading FinBERT ({FINBERT_MODEL_NAME}) on {device} ---")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        if device.type == "cuda":
            model = model.half()
        model.to(device)
        model.eval()

        _FINBERT_STATE["tokenizer"] = tokenizer
        _FINBERT_STATE["model"] = model
        _FINBERT_STATE["device"] = device
        return tokenizer, model, device
    except Exception as exc:
        print(f"Warning: FinBERT could not be loaded ({exc}). Falling back to LM Studio.")
        _FINBERT_STATE["failed"] = True
        return None, None, None


def _parse_age_days(pub_time):
    """Compute article age in days for exponential recency decay weighting."""
    if not pub_time:
        return 3.0
    try:
        now = datetime.now(timezone.utc)
        if isinstance(pub_time, (int, float)):
            dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(pub_time).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 3.0


def _extract_news_items(raw_news, max_items=12):
    """Extract normalized title, summary snippet, link, and publish time from yfinance news."""
    articles = []
    seen_titles = set()

    for item in raw_news:
        title = item.get("title")
        summary = item.get("summary") or ""
        link = item.get("link")
        pub_time = item.get("providerPublishTime")

        if "content" in item and isinstance(item["content"], dict):
            content = item["content"]
            title = title or content.get("title")
            summary = summary or content.get("summary") or content.get("description") or ""
            if not link:
                canonical = content.get("canonicalUrl")
                if isinstance(canonical, dict):
                    link = canonical.get("url")
                elif isinstance(canonical, str):
                    link = canonical
            if not link:
                link = (content.get("clickThroughUrl") or {}).get("url")
            if not pub_time:
                pub_time = content.get("pubDate") or 0

        if title and title not in seen_titles:
            seen_titles.add(title)
            articles.append(
                {
                    "title": title.strip(),
                    "summary": summary.strip(),
                    "link": link,
                    "time": pub_time,
                }
            )
            if len(articles) >= max_items:
                break

    return articles


def _score_articles_with_finbert(articles):
    """Batch-score articles with FinBERT and return per-article scores + recency-weighted aggregate."""
    tokenizer, model, device = _get_finbert()
    if model is None:
        return None

    texts = [
        f"{a['title']}. {a['summary'][:240]}" if a["summary"] else a["title"]
        for a in articles
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    # Map label indices dynamically from FinBERT config (positive / negative / neutral)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = next((i for i, lbl in id2label.items() if "pos" in lbl), 0)
    neg_idx = next((i for i, lbl in id2label.items() if "neg" in lbl), 1)
    neu_idx = next((i for i, lbl in id2label.items() if "neu" in lbl), 2)

    weighted_score_sum = 0.0
    total_weight = 0.0
    enriched_news = []

    for article, prob_vec in zip(articles, probs):
        p_pos = float(prob_vec[pos_idx])
        p_neg = float(prob_vec[neg_idx])
        p_neu = float(prob_vec[neu_idx])

        article_score = int(round(50.0 + 50.0 * (p_pos - p_neg)))
        if article_score >= 60:
            article_label = "Bullish"
        elif article_score <= 40:
            article_label = "Bearish"
        else:
            article_label = "Neutral"

        age_days = _parse_age_days(article["time"])
        decay_weight = math.exp(-0.15 * age_days) * (1.0 - 0.35 * p_neu)

        weighted_score_sum += article_score * decay_weight
        total_weight += decay_weight

        enriched_news.append(
            {
                "title": article["title"],
                "link": article["link"],
                "time": article["time"],
                "sentiment_score": article_score,
                "sentiment_label": article_label,
                "confidence": round(max(p_pos, p_neg, p_neu) * 100.0, 1),
            }
        )

    agg_score = int(round(weighted_score_sum / total_weight)) if total_weight > 0 else 50
    if agg_score >= 60:
        agg_label = "Bullish"
    elif agg_score <= 40:
        agg_label = "Bearish"
    else:
        agg_label = "Neutral"

    return agg_score, agg_label, enriched_news


def _build_narrative_summary(stock_name, agg_score, agg_label, enriched_news):
    """Use LM Studio for an executive 1-sentence synthesis, falling back to deterministic FinBERT summary."""
    top_headlines = "\n".join(
        f"- [{n['sentiment_label']} {n['sentiment_score']}/100] {n['title']}"
        for n in enriched_news[:6]
    )
    try:
        prompt = (
            f"Stock: {stock_name} | FinBERT Aggregate Score: {agg_score}/100 ({agg_label})\n"
            f"Scored Headlines:\n{top_headlines}\n\n"
            "Write ONE concise, professional sentence summarizing the primary catalyst driving this sentiment."
        )
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are a concise financial news analyst. Respond in one sentence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=90,
            timeout=15,
        ).strip()
    except Exception:
        bullish_ct = sum(1 for n in enriched_news if n["sentiment_label"] == "Bullish")
        bearish_ct = sum(1 for n in enriched_news if n["sentiment_label"] == "Bearish")
        top_title = enriched_news[0]["title"] if enriched_news else ""
        return (
            f"FinBERT scored {len(enriched_news)} recent articles ({bullish_ct} bullish, "
            f"{bearish_ct} bearish), led by: \"{top_title}\"."
        )


def analyze_stock_sentiment(stock_name, ticker_symbol):
    """Fetch up to 12 recent news items and compute FinBERT recency-weighted sentiment."""
    print(f"--- Analyzing FinBERT Sentiment: {stock_name} ({ticker_symbol}) ---")
    raw_news = []

    try:
        raw_news = yf.Ticker(ticker_symbol).news or []
    except Exception as exc:
        print(f"Ticker news fetch failed for {ticker_symbol}: {exc}")

    if len(raw_news) < 5:
        try:
            search_query = stock_name.replace(".NS", "")
            search_news = yf.Search(search_query, news_count=12).news or []
            raw_news = list(raw_news) + list(search_news)
        except Exception as exc:
            print(f"Search news fallback failed for {stock_name}: {exc}")

    articles = _extract_news_items(raw_news, max_items=12)
    if not articles:
        return {
            "score": 50,
            "label": "Neutral",
            "summary": "No recent news found.",
            "engine": "FinBERT",
            "news": [],
        }

    finbert_result = _score_articles_with_finbert(articles)
    if finbert_result is not None:
        agg_score, agg_label, enriched_news = finbert_result
        summary = _build_narrative_summary(stock_name, agg_score, agg_label, enriched_news)
        return {
            "score": agg_score,
            "label": agg_label,
            "summary": summary,
            "engine": "ProsusAI/finbert",
            "news": enriched_news,
        }

    # Graceful fallback if transformers/torch are unavailable
    formatted_news = [
        {"title": a["title"], "link": a["link"], "time": a["time"]} for a in articles[:8]
    ]
    return {
        "score": 50,
        "label": "Neutral",
        "summary": "FinBERT unavailable; install torch & transformers for calibrated financial sentiment.",
        "engine": "Fallback",
        "news": formatted_news,
    }
