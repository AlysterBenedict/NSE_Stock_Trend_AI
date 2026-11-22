"""Market overview, live Alpha Radar, and individual stock fundamental detail endpoints."""

import traceback
from flask import Blueprint, jsonify, request

from backend.services.alpha_radar_service import get_alpha_radar
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


@market_bp.route("/alpha-radar", methods=["GET"])
def alpha_radar_endpoint():
    """Return vectorized NSE 100 Alpha Radar leaderboard, Sector Rotation quadrant, and Macro Regime."""
    try:
        horizon = request.args.get("horizon", default=21, type=int)
        force_refresh = request.args.get("refresh", default="false").lower() in ("1", "true", "yes")
        return jsonify(get_alpha_radar(horizon=horizon, force_refresh=force_refresh))
    except Exception as exc:
        print(f"Alpha Radar Error: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
