"""Flask Blueprint registration for NeuroStock REST API."""

from backend.routes.ai_routes import ai_bp
from backend.routes.market_routes import market_bp
from backend.routes.portfolio_routes import portfolio_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.sentiment_routes import sentiment_bp


def register_blueprints(app):
    """Register all feature route blueprints on the Flask application."""
    app.register_blueprint(prediction_bp)
    app.register_blueprint(market_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(sentiment_bp)
    app.register_blueprint(portfolio_bp)
