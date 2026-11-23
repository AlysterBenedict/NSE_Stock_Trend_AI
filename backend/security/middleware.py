"""Pure ASGI security headers, payload size guard, request correlation IDs, and rate limiting."""

from collections import OrderedDict, deque
import json
import math
import threading
import time
import uuid

from fastapi import HTTPException, Request, Response, status

from backend.config import get_max_request_body_bytes, is_hsts_enabled, is_proxy_trusted
from backend.security.logging_utils import get_logger, request_id_ctx, sanitize_for_log
from backend.security.sanitizers import REQUEST_ID_RE

logger = get_logger(__name__)


class _PayloadTooLargeError(Exception):
    """Raised internally when a streamed request body exceeds MAX_REQUEST_BODY_BYTES."""


def _extract_client_ip(scope: dict) -> str:
    if is_proxy_trusted():
        for raw_name, raw_val in scope.get("headers") or ():
            if raw_name.lower() == b"x-forwarded-for":
                first_ip = raw_val.decode("latin-1").split(",")[0].strip()
                if first_ip:
                    return sanitize_for_log(first_ip, max_len=64)
    client = scope.get("client")
    if client and len(client) >= 1 and client[0]:
        return sanitize_for_log(client[0], max_len=64)
    return "unknown"


def _extract_client_ip_from_request(request: Request) -> str:
    if is_proxy_trusted():
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return sanitize_for_log(forwarded, max_len=64)
    if request.client and request.client.host:
        return sanitize_for_log(request.client.host, max_len=64)
    return "unknown"


def _build_security_headers(req_id: str, has_cache_control: bool) -> list[tuple[bytes, bytes]]:
    headers = [
        (b"x-request-id", req_id.encode("ascii")),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
        (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'"),
    ]
    if not has_cache_control:
        headers.append((b"cache-control", b"no-store"))
    if is_hsts_enabled():
        headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
    return headers


async def _send_json_error(send, status_code: int, message: str, req_id: str) -> None:
    body = json.dumps({"error": message, "request_id": req_id}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *_build_security_headers(req_id, has_cache_control=False),
    ]
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


class SecurityAndContextMiddleware:
    """Pure ASGI middleware for security headers, body size caps, correlation IDs, and access logs."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        req_headers = dict(scope.get("headers") or ())
        incoming_id = req_headers.get(b"x-request-id", b"").decode("latin-1").strip()
        req_id = incoming_id if REQUEST_ID_RE.fullmatch(incoming_id) else uuid.uuid4().hex
        ctx_token = request_id_ctx.set(req_id)

        method = scope.get("method", "GET")
        path = sanitize_for_log(scope.get("path", "/"), max_len=200)
        client_ip = _extract_client_ip(scope)
        max_bytes = get_max_request_body_bytes()
        start_time = time.perf_counter()
        status_code_holder = [500]

        content_len_raw = req_headers.get(b"content-length")
        if content_len_raw is not None:
            try:
                content_len = int(content_len_raw.decode("ascii").strip())
                if content_len < 0:
                    await _send_json_error(send, 400, "Invalid Content-Length header.", req_id)
                    request_id_ctx.reset(ctx_token)
                    return
                if content_len > max_bytes:
                    await _send_json_error(
                        send, 413, f"Request payload exceeds {max_bytes} byte limit.", req_id
                    )
                    request_id_ctx.reset(ctx_token)
                    return
            except ValueError:
                await _send_json_error(send, 400, "Invalid Content-Length header.", req_id)
                request_id_ctx.reset(ctx_token)
                return

        received_bytes = 0

        async def bounded_receive():
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body") or b""
                received_bytes += len(chunk)
                if received_bytes > max_bytes:
                    raise _PayloadTooLargeError()
            return message

        async def hardened_send(message):
            if message.get("type") == "http.response.start":
                status_code_holder[0] = int(message.get("status", 200))
                filtered = [
                    (k, v)
                    for k, v in (message.get("headers") or [])
                    if k.lower() not in (b"server", b"x-request-id")
                ]
                has_cc = any(k.lower() == b"cache-control" for k, _ in filtered)
                filtered.extend(_build_security_headers(req_id, has_cache_control=has_cc))
                message = {**message, "headers": filtered}
            await send(message)

        try:
            await self.app(scope, bounded_receive, hardened_send)
        except _PayloadTooLargeError:
            status_code_holder[0] = 413
            await _send_json_error(
                send, 413, f"Request payload exceeds {max_bytes} byte limit.", req_id
            )
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "%s %s -> %d (%.1fms) ip=%s",
                method,
                path,
                status_code_holder[0],
                elapsed_ms,
                client_ip,
            )
            request_id_ctx.reset(ctx_token)


class SlidingWindowRateLimiter:
    """Thread-safe bounded sliding-window rate limiter protecting API and compute endpoints."""

    def __init__(self, max_keys: int = 10000) -> None:
        self.max_keys = max(128, int(max_keys))
        self._buckets: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    def check(self, bucket: str, key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
        now = time.monotonic()
        cutoff = now - float(window_seconds)
        composite = (bucket, key)

        with self._lock:
            dq = self._buckets.get(composite)
            if dq is None:
                dq = deque()
                self._buckets[composite] = dq
            else:
                self._buckets.move_to_end(composite)

            while dq and dq[0] <= cutoff:
                dq.popleft()

            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)

            if len(dq) >= limit:
                retry_after = max(1, int(math.ceil(dq[0] + float(window_seconds) - now)))
                return False, 0, retry_after

            dq.append(now)
            remaining = max(0, limit - len(dq))
            return True, remaining, 0


rate_limiter = SlidingWindowRateLimiter()


def rate_limit(bucket: str, limit: int, window_seconds: int = 60):
    """Return a FastAPI dependency enforcing per-user / per-IP sliding-window rate limits."""

    async def _dependency(request: Request, response: Response) -> None:
        user_id = getattr(request.state, "user_id", None)
        if user_id and user_id != "dev-local":
            key = f"u:{sanitize_for_log(user_id, max_len=128)}"
        else:
            key = f"ip:{_extract_client_ip_from_request(request)}"

        allowed, remaining, retry_after = rate_limiter.check(bucket, key, limit, window_seconds)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait before retrying.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

    return _dependency
