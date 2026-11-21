"""Prediction and historical chart data endpoints."""

import traceback
from flask import Blueprint, jsonify, request

from backend.config import ALGORITHMS, STOCK_LIST
from backend.services.market_data_service import get_historical_chart_data
from backend.services.prediction_service import (
    determine_next_business_day_for_ticker,
    run_prediction_for_algorithm,
)

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    """Handle single-algorithm or multi-algorithm comparison price predictions."""
    content = request.json or {}
    stock_name = content.get("stock_name")
    future_date_str = content.get("future_date")
    algorithm_name = content.get("algorithm_name")

    if not all([stock_name, future_date_str, algorithm_name]):
        return jsonify({"error": "Missing stock_name, future_date, or algorithm_name"}), 400

    ticker = STOCK_LIST.get(stock_name)
    if not ticker:
        return jsonify({"error": "Invalid stock_name"}), 400

    try:
        next_bday_str = determine_next_business_day_for_ticker(ticker)
        if not next_bday_str:
            return jsonify({"error": "Could not fetch latest stock data."}), 500
    except Exception as exc:
        return jsonify({"error": f"yfinance error: {exc}"}), 500

    if isinstance(algorithm_name, list):
        print(f"Received comparison request for {ticker} on {future_date_str} using {algorithm_name}")
        results = {}
        for algo in algorithm_name:
            if algo not in ALGORITHMS:
                results[algo] = {"error": f"Invalid algorithm: {algo}"}
                continue
            payload, error = run_prediction_for_algorithm(
                stock_name, ticker, future_date_str, algo, next_bday_str
            )
            results[algo] = {"error": error} if error else payload
        return jsonify(results)

    if algorithm_name not in ALGORITHMS:
        return jsonify({"error": "Invalid algorithm_name"}), 400

    print(f"Received prediction request for {ticker} on {future_date_str} using {algorithm_name}")
    payload, error = run_prediction_for_algorithm(
        stock_name, ticker, future_date_str, algorithm_name, next_bday_str
    )
    if error:
        return jsonify({"error": error}), 500
    return jsonify(payload)


@prediction_bp.route("/historical-data", methods=["GET"])
def historical_data_endpoint():
    """Provide 3 years of OHLC, volume, RSI-14, and SMA-50 series for financial charts."""
    stock_name = request.args.get("stock_name")
    if not stock_name:
        return jsonify({"error": "Missing stock_name parameter"}), 400

    ticker = STOCK_LIST.get(stock_name)
    if not ticker:
        return jsonify({"error": "Invalid stock_name"}), 400

    print(f"Fetching historical data for {ticker}...")
    try:
        chart_data = get_historical_chart_data(ticker)
        if not chart_data:
            return jsonify({"error": f"Could not fetch data for {ticker}"}), 500
        return jsonify(chart_data)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
