"""Shared client for communicating with the local LM Studio inference server."""

import re
import requests

from backend.config import LM_STUDIO_API_URL, LOCAL_MODEL_NAME

_NO_EMOJI_INSTRUCTION = "Strict rule: Never use emojis, emoticons, or decorative Unicode symbols anywhere in your response."
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F1E6-\U0001F1FF\uFE0F\u200D]+"
)


def strip_emojis(text):
    """Remove any emoji codepoints from LLM output."""
    return _EMOJI_RE.sub("", text or "")


def _with_no_emoji_system(messages):
    """Ensure the no-emoji rule is present in the message list sent to LM Studio."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0].get('content', '').rstrip()}\n{_NO_EMOJI_INSTRUCTION}".strip()
    else:
        msgs.insert(0, {"role": "system", "content": _NO_EMOJI_INSTRUCTION})
    return msgs


def chat_completion(
    messages,
    temperature=0.7,
    max_tokens=800,
    model=LOCAL_MODEL_NAME,
    timeout=60,
):
    """Send a chat completion request to LM Studio and return the assistant message content."""
    payload = {
        "messages": _with_no_emoji_system(messages),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        LM_STUDIO_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return strip_emojis(content)
