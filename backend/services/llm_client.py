"""Async pooled client for LM Studio (primary) with NVIDIA NIM fallback routing."""

import re
from urllib.parse import urlparse

import httpx

from backend.config import (
    LOCAL_MODEL_NAME,
    get_lm_studio_api_url,
    get_nvidia_api_key_env,
    get_nvidia_api_url,
    get_nvidia_model_name,
)
from backend.security.auth import current_user_id_ctx
from backend.security.key_vault import key_vault
from backend.security.logging_utils import get_logger, sanitize_for_log
from backend.services.runtime import runtime

logger = get_logger(__name__)

_NO_EMOJI_INSTRUCTION = (
    "Strict rule: Never use emojis, emoticons, or decorative Unicode symbols anywhere in your response."
)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F1E6-\U0001F1FF\uFE0F\u200D]+"
)


def strip_emojis(text: str | None) -> str:
    """Remove decorative emoji codepoints from LLM output."""
    return _EMOJI_RE.sub("", text or "")


def _with_no_emoji_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure the no-emoji directive is present in the message list sent to the LLM."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0].get('content', '').rstrip()}\n{_NO_EMOJI_INSTRUCTION}".strip()
    else:
        msgs.insert(0, {"role": "system", "content": _NO_EMOJI_INSTRUCTION})
    return msgs


def _get_lm_studio_models_url() -> str:
    api_url = get_lm_studio_api_url().rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")] + "/models"
    parsed = urlparse(api_url)
    return f"{parsed.scheme}://{parsed.netloc}/v1/models"


def _extract_completion_content(body: object) -> str:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM provider returned empty completion choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("LLM provider response missing assistant content.")
    return strip_emojis(content)


async def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float | httpx.Timeout,
) -> httpx.Response:
    client = runtime.try_get_http_client()
    if client is not None:
        return await client.post(url, json=payload, headers=headers, timeout=timeout)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as temp_client:
        return await temp_client.post(url, json=payload, headers=headers)


async def check_lm_studio_health_async(timeout: float = 1.2) -> bool:
    """Probe whether the local LM Studio inference server is reachable."""
    try:
        models_url = _get_lm_studio_models_url()
        probe_timeout = httpx.Timeout(timeout, connect=min(1.0, float(timeout)))
        client = runtime.try_get_http_client()
        if client is not None:
            response = await client.get(models_url, timeout=probe_timeout)
        else:
            async with httpx.AsyncClient(timeout=probe_timeout, follow_redirects=False) as temp_client:
                response = await temp_client.get(models_url)
        return response.status_code == 200
    except Exception:
        return False


async def async_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
    model: str = LOCAL_MODEL_NAME,
    timeout: float = 45.0,
    user_id: str | None = None,
) -> str:
    """Query local LM Studio first; fall back to NVIDIA NIM only if LM Studio is unavailable."""
    formatted_messages = _with_no_emoji_system(messages)
    lm_url = get_lm_studio_api_url()
    lm_payload = {
        "messages": formatted_messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    lm_timeout = httpx.Timeout(timeout, connect=min(2.5, float(timeout)))

    try:
        lm_response = await _post_json(
            lm_url,
            lm_payload,
            headers={"Content-Type": "application/json"},
            timeout=lm_timeout,
        )
        lm_response.raise_for_status()
        content = _extract_completion_content(lm_response.json())
        logger.debug("Completed local LM Studio LLM call (%d chars).", len(content))
        return content
    except Exception as lm_exc:
        effective_user_id = user_id or current_user_id_ctx.get("dev-local")
        nvidia_key = key_vault.get_user_api_key(effective_user_id) or get_nvidia_api_key_env()
        if not nvidia_key:
            raise

        nvidia_url = get_nvidia_api_url()
        nvidia_model = get_nvidia_model_name()
        logger.info(
            "Local LM Studio unavailable (%s); routing completion to NVIDIA NIM (%s).",
            _safe_error_hint(lm_exc),
            sanitize_for_log(nvidia_model),
        )
        nvidia_payload = {
            "messages": formatted_messages,
            "model": nvidia_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        nvidia_response = await _post_json(
            nvidia_url,
            nvidia_payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {nvidia_key}",
            },
            timeout=timeout,
        )
        nvidia_response.raise_for_status()
        content = _extract_completion_content(nvidia_response.json())
        logger.debug("Completed NVIDIA NIM fallback LLM call (%d chars).", len(content))
        return content


def _safe_error_hint(exc: Exception) -> str:
    return sanitize_for_log(type(exc).__name__)
