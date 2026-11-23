"""Async AI insights chatbot and financial education endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse

from backend.schemas import AiInsightsRequest, GeneralKnowledgeRequest
from backend.security.auth import require_auth
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.security.middleware import rate_limit
from backend.services.ai_assistant_service import (
    answer_general_knowledge_async,
    generate_ai_insights_async,
)

logger = get_logger(__name__)

router = APIRouter(tags=["ai"])


@router.post("/get-general-knowledge")
async def get_general_knowledge_endpoint(
    payload: GeneralKnowledgeRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("ai", limit=20, window_seconds=60)),
):
    """Answer general financial literacy questions for the FinTeach tutor."""
    logger.info("Financial Teacher query (%d chars)", len(payload.user_question))
    try:
        answer = await answer_general_knowledge_async(payload.user_question)
        return {"answer": answer}
    except Exception as exc:
        logger.warning("FinTeach LLM unavailable: %s", sanitize_for_log(exc))
        return ORJSONResponse(
            {"error": "Financial tutor is temporarily unavailable. Ensure LM Studio is running."},
            status_code=503,
        )


@router.post("/get-ai-insights")
async def get_ai_insights_endpoint(
    payload: AiInsightsRequest,
    _claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("ai", limit=20, window_seconds=60)),
):
    """Generate contextual AI stock insights or answer quantitative equity questions."""
    insights, status_code = await generate_ai_insights_async(
        stock_name=payload.stock_name,
        trend_data=payload.trend_data.model_dump() if payload.trend_data else None,
        user_question=payload.user_question,
        active_context=payload.active_context.model_dump() if payload.active_context else None,
    )
    return ORJSONResponse({"insights": insights}, status_code=status_code)
