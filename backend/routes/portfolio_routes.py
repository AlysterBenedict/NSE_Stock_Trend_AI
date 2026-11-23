"""Async portfolio projection, multi-stock optimizer, and AI Investment Engine endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import (
    InvestmentEngineRequest,
    OptimizePortfolioRequest,
    PortfolioProjectionRequest,
)
from backend.security.auth import require_auth
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.security.middleware import rate_limit
from backend.services.investment_engine_service import run_investment_engine_async
from backend.services.portfolio_service import (
    calculate_portfolio_projection_async,
    optimize_multi_stock_portfolio_async,
)

logger = get_logger(__name__)

router = APIRouter(tags=["portfolio"])


@router.post("/calculate-portfolio")
async def calculate_portfolio_endpoint(
    payload: PortfolioProjectionRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("predict", limit=20, window_seconds=60)),
):
    """Calculate projected portfolio value, ROI, CAGR, and Monte Carlo risk metrics."""
    logger.info(
        "Portfolio projection: stock=%s principal=%.2f date=%s",
        sanitize_for_log(payload.stock_name),
        payload.principal,
        payload.future_date.isoformat(),
    )
    result, error_info = await calculate_portfolio_projection_async(
        payload.stock_name, payload.principal, payload.future_date.strftime("%Y-%m-%d")
    )
    if error_info:
        msg, status_code = error_info
        raise HTTPException(status_code=status_code, detail=msg)
    return result


@router.post("/investment-engine")
async def investment_engine_endpoint(
    payload: InvestmentEngineRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("heavy", limit=8, window_seconds=60)),
):
    """Rank top NSE investment candidates and return an AI investment memo."""
    return await run_investment_engine_async(
        payload.principal, payload.withdrawal_date.strftime("%Y-%m-%d")
    )


@router.post("/optimize-portfolio")
async def optimize_portfolio_endpoint(
    payload: OptimizePortfolioRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("heavy", limit=10, window_seconds=60)),
):
    """Optimize a multi-stock NSE basket and run correlated GARCH Monte Carlo + stress tests."""
    result, error_info = await optimize_multi_stock_portfolio_async(
        payload.stocks,
        payload.principal,
        payload.future_date.strftime("%Y-%m-%d"),
        mode=payload.mode,
        custom_weights=payload.custom_weights,
    )
    if error_info:
        msg, status_code = error_info
        raise HTTPException(status_code=status_code, detail=msg)
    return result
