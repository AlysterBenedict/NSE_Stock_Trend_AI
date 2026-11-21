"""Market overview and individual stock fundamental detail endpoints."""

import traceback
from flask import Blueprint, jsonify, request

from backend.services.market_data_service import get_market_overview, get_stock_detail

market_bp = Blueprint("market", __name__)


@market_bp.route("/get-market-data", methods=["GET"])
def get_market_data_endpoint():
    """Return bulk market overview or detailed stock metrics when ?ticker= is provided."""
    try:
        specific_ticker = request.args.get("ticker")
        if specific_ticker:
            print(f"Fetching detailed metrics for {specific_ticker}...")
            return jsonify(get_stock_detail(specific_ticker))

        print("Fetching market overview...")
        return jsonify(get_market_overview())
    except Exception as exc:
        print(f"Market Data Error: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
