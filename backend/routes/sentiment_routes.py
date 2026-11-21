"""News sentiment analysis endpoint."""

from flask import Blueprint, jsonify, request

from backend.services.market_data_service import resolve_ticker
from backend.services.sentiment_service import analyze_stock_sentiment

sentiment_bp = Blueprint("sentiment", __name__)


@sentiment_bp.route("/get-sentiment", methods=["POST"])
def get_sentiment_endpoint():
    """Analyze market sentiment from recent news headlines for a given stock."""
    try:
        content = request.json or {}
        stock_name = content.get("stock_name")
        if not stock_name:
            return jsonify({"error": "Missing stock_name"}), 400

        display_name, ticker_symbol = resolve_ticker(stock_name)
        result = analyze_stock_sentiment(display_name, ticker_symbol)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
