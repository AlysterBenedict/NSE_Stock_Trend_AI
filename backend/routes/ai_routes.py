"""Async AI insights chatbot, financial education, and user LLM connection settings endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse

from backend.config import (
    LOCAL_MODEL_DISPLAY_NAME,
    LOCAL_MODEL_NAME,
    NVIDIA_MODEL_DISPLAY_NAME,
    get_nvidia_api_key_env,
    get_nvidia_base_url,
    get_nvidia_model_name,
)
from backend.schemas import AiInsightsRequest, GeneralKnowledgeRequest, LlmKeyUpdateRequest
from backend.security.auth import require_auth
from backend.security.key_vault import key_vault
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.security.middleware import rate_limit
from backend.services.ai_assistant_service import (
    answer_general_knowledge_async,
    generate_ai_insights_async,
)
from backend.services.llm_client import check_lm_studio_health_async

logger = get_logger(__name__)

router = APIRouter(tags=["ai"])


def _build_llm_status_payload(user_id: str, lm_online: bool) -> dict:
    meta = key_vault.get_user_key_metadata(user_id)
    has_api_key = bool(meta["has_api_key"])
    has_env_key = bool(get_nvidia_api_key_env())
    nvidia_online = has_api_key or has_env_key
    llm_configured = lm_online or nvidia_online

    if lm_online:
        active_provider = "lm_studio"
        active_model_display = LOCAL_MODEL_DISPLAY_NAME
    elif nvidia_online:
        active_provider = "nvidia"
        active_model_display = NVIDIA_MODEL_DISPLAY_NAME
    else:
        active_provider = "none"
        active_model_display = "LLM Offline"

    return {
        "has_api_key": has_api_key,
        "masked_key": meta["masked_key"],
        "updated_at": meta["updated_at"],
        "lm_studio_available": lm_online,
        "nvidia_llm_online": nvidia_online,
        "llm_configured": llm_configured,
        "active_provider": active_provider,
        "active_model_display": active_model_display,
        "local_model": LOCAL_MODEL_NAME,
        "local_model_display": LOCAL_MODEL_DISPLAY_NAME,
        "nvidia_model": get_nvidia_model_name(),
        "nvidia_model_display": NVIDIA_MODEL_DISPLAY_NAME,
        "nvidia_base_url": get_nvidia_base_url(),
    }


@router.get("/llm-settings")
async def get_llm_settings_endpoint(
    claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("read", limit=60, window_seconds=60)),
):
    """Return user-scoped NVIDIA API key metadata and active LLM routing status."""
    user_id = claims.get("sub") or "dev-local"
    lm_online = await check_lm_studio_health_async()
    return _build_llm_status_payload(user_id, lm_online)


@router.post("/llm-settings")
async def save_llm_settings_endpoint(
    payload: LlmKeyUpdateRequest,
    claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("ai", limit=20, window_seconds=60)),
):
    """Encrypt and save the authenticated user's NVIDIA API key in the per-user vault."""
    user_id = claims.get("sub") or "dev-local"
    key_vault.save_user_api_key(user_id, payload.api_key)
    lm_online = await check_lm_studio_health_async()
    logger.info("Updated encrypted NVIDIA API key for user %s", sanitize_for_log(user_id, max_len=24))
    return {"status": "saved", **_build_llm_status_payload(user_id, lm_online)}


@router.delete("/llm-settings")
async def delete_llm_settings_endpoint(
    claims: dict = Depends(require_auth),
    _rl: None = Depends(rate_limit("ai", limit=20, window_seconds=60)),
):
    """Remove the authenticated user's saved NVIDIA API key."""
    user_id = claims.get("sub") or "dev-local"
    key_vault.delete_user_api_key(user_id)
    lm_online = await check_lm_studio_health_async()
    logger.info("Removed encrypted NVIDIA API key for user %s", sanitize_for_log(user_id, max_len=24))
    return {"status": "deleted", **_build_llm_status_payload(user_id, lm_online)}


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
            {
                "error": (
                    "Financial tutor is temporarily unavailable. Ensure LM Studio is running "
                    "or configure your NVIDIA API key."
                )
            },
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
