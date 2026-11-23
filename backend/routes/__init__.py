"""FastAPI router registration for NeuroStock Async REST API."""

from fastapi import FastAPI

from backend.routes.ai_routes import router as ai_router
from backend.routes.market_routes import router as market_router
from backend.routes.portfolio_routes import router as portfolio_router
from backend.routes.prediction_routes import router as prediction_router
from backend.routes.sentiment_routes import router as sentiment_router


def register_routers(app: FastAPI) -> None:
    """Mount all feature APIRouters on the FastAPI application."""
    app.include_router(prediction_router)
    app.include_router(market_router)
    app.include_router(ai_router)
    app.include_router(sentiment_router)
    app.include_router(portfolio_router)
