"""Structured logging and CRLF/control-character log-injection sanitization."""

from contextvars import ContextVar
import logging
import os
import re
import sys

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class _RequestIdFilter(logging.Filter):
    """Attach the active request correlation ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        return True


def configure_logging(level_name: str | None = None) -> None:
    """Configure root logging once with structured formatting and correlation IDs."""
    raw_level = (level_name or os.environ.get("LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, raw_level, logging.INFO)

    root = logging.getLogger()
    if getattr(root, "_neurostock_configured", False):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | req=%(request_id)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_RequestIdFilter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root._neurostock_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger."""
    configure_logging()
    return logging.getLogger(name)


def sanitize_for_log(value: object, max_len: int = 160) -> str:
    """Strip CR/LF/ANSI/control characters and truncate to prevent log forging."""
    if value is None:
        return ""
    text = str(value)
    text = _ANSI_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text
