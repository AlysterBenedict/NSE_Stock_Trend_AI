"""Async FinBERT news sentiment analysis endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import SentimentRequest
from backend.security.auth import require_auth
from backend.security.middleware import rate_limit
from backend.security.sanitizers import validate_ticker
from backend.services.market_data_service import resolve_ticker_async
from backend.services.sentiment_service import analyze_stock_sentiment_async

router = APIRouter(tags=["sentiment"])


@router.post("/get-sentiment")
async def get_sentiment_endpoint(
    payload: SentimentRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("predict", limit=20, window_seconds=60)),
):
    """Analyze market sentiment from recent news headlines for a given stock."""
    display_name, ticker_symbol = await resolve_ticker_async(payload.stock_name)
    if not ticker_symbol:
        raise HTTPException(status_code=404, detail="Could not resolve ticker symbol.")
    try:
        safe_ticker = validate_ticker(ticker_symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await analyze_stock_sentiment_async(display_name or payload.stock_name, safe_ticker)
