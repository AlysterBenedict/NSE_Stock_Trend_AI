"""Async market overview, live Alpha Radar, and individual stock detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.config import FORECAST_HORIZONS
from backend.security.auth import require_auth
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.security.middleware import rate_limit
from backend.services.alpha_radar_service import get_alpha_radar_async
from backend.services.market_data_service import (
    get_market_overview_async,
    get_stock_detail_async,
    validate_allowed_detail_ticker,
)

logger = get_logger(__name__)

router = APIRouter(tags=["market"])


@router.get("/get-market-data")
async def get_market_data_endpoint(
    ticker: str | None = Query(default=None, max_length=25),
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("read", limit=60, window_seconds=60)),
):
    """Return bulk market overview or detailed stock metrics when ?ticker= is provided."""
    if ticker is not None:
        try:
            safe_ticker = validate_allowed_detail_ticker(ticker)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("Fetching detailed metrics for %s", sanitize_for_log(safe_ticker))
        return await get_stock_detail_async(safe_ticker)

    logger.debug("Fetching cached NSE market overview")
    return await get_market_overview_async()


@router.get("/alpha-radar")
async def alpha_radar_endpoint(
    horizon: int = Query(default=21),
    refresh: bool = Query(default=False),
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("read", limit=40, window_seconds=60)),
):
    """Return vectorized NSE 100 Alpha Radar leaderboard, Sector Rotation quadrant, and Macro Regime."""
    if horizon not in FORECAST_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon. Supported values: {FORECAST_HORIZONS}.",
        )
    return await get_alpha_radar_async(horizon=horizon, force_refresh=refresh)
