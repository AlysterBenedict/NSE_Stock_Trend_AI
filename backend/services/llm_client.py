"""Async pooled client for communicating with the local LM Studio inference server."""

import re

import httpx

from backend.config import LOCAL_MODEL_NAME, get_lm_studio_api_url
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
    """Ensure the no-emoji directive is present in the message list sent to LM Studio."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0].get('content', '').rstrip()}\n{_NO_EMOJI_INSTRUCTION}".strip()
    else:
        msgs.insert(0, {"role": "system", "content": _NO_EMOJI_INSTRUCTION})
    return msgs


async def async_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
    model: str = LOCAL_MODEL_NAME,
    timeout: float = 45.0,
) -> str:
    """Send an asynchronous chat completion request to LM Studio using the pooled HTTPX client."""
    url = get_lm_studio_api_url()
    payload = {
        "messages": _with_no_emoji_system(messages),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    client = runtime.try_get_http_client()
    if client is not None:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    else:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as temp_client:
            response = await temp_client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM provider returned empty completion choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("LLM provider response missing assistant content.")
    logger.debug("Completed LLM call (%d chars).", len(content))
    return strip_emojis(content)


def _safe_error_hint(exc: Exception) -> str:
    return sanitize_for_log(type(exc).__name__)
