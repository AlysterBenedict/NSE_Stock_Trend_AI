"""Input, ticker, URL, SSRF, and LLM prompt-injection sanitizers."""

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

TICKER_RE = re.compile(r"^[A-Za-z0-9._^=-]{1,20}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ALL_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_LOCAL_LLM_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})


def validate_ticker(raw: object) -> str:
    """Validate and normalize a market ticker symbol, rejecting traversal or URL injection characters."""
    if not isinstance(raw, str):
        raise ValueError("Ticker must be a string.")
    cleaned = raw.strip().upper()
    if not cleaned or ".." in cleaned or not TICKER_RE.fullmatch(cleaned):
        raise ValueError("Invalid ticker symbol format.")
    return cleaned


def is_safe_http_url(url: Any) -> bool:
    """Return True only if url is a well-formed http:// or https:// URL without control characters."""
    if not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate or len(candidate) > 2048 or _ALL_CONTROL_RE.search(candidate):
        return False
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


def sanitize_http_url(url: Any) -> str | None:
    """Return the stripped URL if it uses http/https safely, otherwise None."""
    if not is_safe_http_url(url):
        return None
    return str(url).strip()


def validate_llm_url(url: str, allow_remote: bool = False) -> str:
    """Validate LM Studio / LLM API endpoint URL against scheme and SSRF host rules."""
    if not isinstance(url, str):
        raise ValueError("LM_STUDIO_API_URL must be a string.")
    cleaned = url.strip()
    if not is_safe_http_url(cleaned):
        raise ValueError("LM_STUDIO_API_URL must be a valid http:// or https:// URL.")

    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("LM_STUDIO_API_URL must include a hostname.")

    if not allow_remote:
        is_local = hostname in _LOCAL_LLM_HOSTS
        if not is_local:
            try:
                ip_obj = ipaddress.ip_address(hostname)
                is_local = ip_obj.is_loopback
            except ValueError:
                is_local = False
        if not is_local:
            raise ValueError(
                "Remote LM_STUDIO_API_URL is blocked when ALLOW_REMOTE_LLM_URL is false."
            )

    return cleaned


def sanitize_prompt_text(text: Any, max_len: int = 1000) -> str:
    """Remove dangerous control characters, normalize whitespace, and enforce max length."""
    if text is None:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub(" ", str(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def wrap_untrusted(tag: str, text: str, max_len: int = 2000) -> str:
    """Isolate untrusted user or external text inside explicit XML-style boundaries."""
    safe_tag = re.sub(r"[^a-zA-Z0-9_-]", "", tag) or "untrusted_data"
    cleaned = sanitize_prompt_text(text, max_len=max_len)
    # Prevent tag-breakout injection inside untrusted content
    cleaned = re.sub(rf"</?\s*{re.escape(safe_tag)}\s*>", "[tag-filtered]", cleaned, flags=re.IGNORECASE)
    return f"<{safe_tag}>\n{cleaned}\n</{safe_tag}>"
