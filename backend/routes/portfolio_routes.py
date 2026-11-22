"""Portfolio projection and AI Investment Engine endpoints."""

import traceback
from flask import Blueprint, jsonify, request

from backend.services.investment_engine_service import run_investment_engine
from backend.services.portfolio_service import (
    calculate_portfolio_projection,
    optimize_multi_stock_portfolio,
)

portfolio_bp = Blueprint("portfolio", __name__)


@portfolio_bp.route("/calculate-portfolio", methods=["POST"])
def calculate_portfolio_endpoint():
    """Calculate projected portfolio value, ROI, CAGR, and risk score."""
    try:
        content = request.json or {}
        stock_name = content.get("stock_name")
        principal = float(content.get("principal", 0))
        future_date_str = content.get("future_date")

        if not stock_name or principal <= 0 or not future_date_str:
            return jsonify({"error": "Invalid input. Check stock, principal, and date."}), 400

        print(f"--- Calculating Portfolio: {stock_name}, {principal}, {future_date_str} ---")
        result, error_info = calculate_portfolio_projection(stock_name, principal, future_date_str)
        if error_info:
            msg, status_code = error_info
            return jsonify({"error": msg}), status_code
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@portfolio_bp.route("/investment-engine", methods=["POST"])
def investment_engine_endpoint():
    """Rank top NSE investment candidates and return an AI investment memo."""
    try:
        content = request.json or {}
        principal = float(content.get("principal", 100000))
        withdrawal_date_str = content.get("withdrawal_date")

        if not withdrawal_date_str:
            return jsonify({"error": "Missing withdrawal_date"}), 400

        return jsonify(run_investment_engine(principal, withdrawal_date_str))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@portfolio_bp.route("/optimize-portfolio", methods=["POST"])
def optimize_portfolio_endpoint():
    """Optimize a multi-stock NSE basket and run correlated GARCH Monte Carlo + stress tests."""
    try:
        content = request.json or {}
        stocks = content.get("stocks") or []
        principal = float(content.get("principal", 0))
        future_date_str = content.get("future_date")
        mode = content.get("mode", "ai_bl")
        custom_weights = content.get("custom_weights")

        if not stocks or principal <= 0 or not future_date_str:
            return jsonify({"error": "Provide 2-10 stocks, positive principal, and future_date."}), 400

        result, error_info = optimize_multi_stock_portfolio(
            stocks, principal, future_date_str, mode=mode, custom_weights=custom_weights
        )
        if error_info:
            msg, status_code = error_info
            return jsonify({"error": msg}), status_code
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
