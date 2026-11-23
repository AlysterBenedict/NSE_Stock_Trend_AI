"""Async prediction, historical chart data, and walk-forward backtest endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.config import STOCK_LIST
from backend.schemas import BacktestRequest, PredictRequest
from backend.security.auth import require_auth
from backend.security.logging_utils import get_logger
from backend.security.middleware import rate_limit
from backend.services.backtest_service import run_strategy_backtest_async
from backend.services.market_data_service import get_historical_chart_data_async
from backend.services.prediction_service import predict_async

logger = get_logger(__name__)

router = APIRouter(tags=["prediction"])


@router.post("/predict")
async def predict_endpoint(
    payload: PredictRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("predict", limit=25, window_seconds=60)),
):
    """Handle single-algorithm or multi-algorithm comparison price predictions."""
    ticker = STOCK_LIST[payload.stock_name]
    future_date_str = payload.future_date.strftime("%Y-%m-%d")
    logger.info("Prediction request for %s target=%s algo=%s", ticker, future_date_str, payload.algorithm_name)

    result, err, status_code = await predict_async(
        payload.stock_name, ticker, future_date_str, payload.algorithm_name
    )
    if err:
        raise HTTPException(status_code=status_code, detail=err)
    return result


@router.get("/historical-data")
async def historical_data_endpoint(
    stock_name: str = Query(..., min_length=1, max_length=80),
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("read", limit=60, window_seconds=60)),
):
    """Provide 3 years of OHLC, volume, RSI-14, and SMA-50 series for financial charts."""
    ticker = STOCK_LIST.get(stock_name)
    if not ticker:
        raise HTTPException(status_code=400, detail="Invalid stock_name")

    logger.debug("Fetching historical data for %s", ticker)
    chart_data = await get_historical_chart_data_async(ticker)
    if not chart_data:
        raise HTTPException(status_code=502, detail=f"Could not fetch data for {ticker}")
    return chart_data


@router.post("/run-backtest")
async def run_backtest_endpoint(
    payload: BacktestRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("heavy", limit=10, window_seconds=60)),
):
    """Execute non-overlapping walk-forward strategy backtest with NSE cost assumptions."""
    result, error_info = await run_strategy_backtest_async(
        strategy=payload.strategy,
        horizon=payload.horizon,
        algorithm=payload.algorithm,
        top_n=payload.top_n,
        cost_bps=payload.cost_bps,
        initial_capital=payload.initial_capital,
        lookback_months=payload.lookback_months,
        stock_name=payload.stock_name,
    )
    if error_info:
        msg, status_code = error_info
        raise HTTPException(status_code=status_code, detail=msg)
    return result
