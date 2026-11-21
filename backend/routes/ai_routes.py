"""AI insights chatbot and financial education endpoints."""

from flask import Blueprint, jsonify, request

from backend.services.ai_assistant_service import (
    answer_general_knowledge,
    generate_ai_insights,
)

ai_bp = Blueprint("ai", __name__)


@ai_bp.route("/get-general-knowledge", methods=["POST"])
def get_general_knowledge_endpoint():
    """Answer general financial literacy questions for the FinTeach tutor."""
    try:
        content = request.json or {}
        user_question = content.get("user_question")
        if not user_question:
            return jsonify({"error": "Missing user_question"}), 400

        print(f"--- Financial Teacher Query: {user_question} ---")
        return jsonify({"answer": answer_general_knowledge(user_question)})
    except Exception as exc:
        print(f"Error in Financial Teacher: {exc}")
        return jsonify({"error": str(exc)}), 500


@ai_bp.route("/get-ai-insights", methods=["POST"])
def get_ai_insights_endpoint():
    """Generate contextual AI stock insights or answer financial questions."""
    content = request.json or {}
    stock_name = content.get("stock_name")
    trend_data = content.get("trend_data")
    user_question = content.get("user_question")

    if not user_question and not stock_name:
        return jsonify({"error": "Missing user_question or stock context"}), 400

    insights, status_code = generate_ai_insights(
        stock_name=stock_name,
        trend_data=trend_data,
        user_question=user_question,
    )
    return jsonify({"insights": insights}), status_code
