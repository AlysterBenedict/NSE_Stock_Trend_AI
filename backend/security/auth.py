"""Stateless Firebase ID Token (RS256 JWT) verification with zero private keys on disk."""

import asyncio
from contextvars import ContextVar
import re
import time
from typing import Any, Awaitable, Callable

from cryptography import x509
from fastapi import HTTPException, Request, status
import httpx
import jwt
from jwt import InvalidTokenError

from backend.config import get_firebase_project_id, is_auth_required
from backend.security.logging_utils import get_logger, sanitize_for_log

logger = get_logger(__name__)

FIREBASE_CERTS_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/"
    "securetoken@system.gserviceaccount.com"
)
_CACHE_CONTROL_MAX_AGE_RE = re.compile(r"max-age=(\d+)", re.IGNORECASE)
_BEARER_RE = re.compile(r"^Bearer\s+([^\s]+)$", re.IGNORECASE)
_SAFE_SUB_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
current_user_id_ctx: ContextVar[str] = ContextVar("current_user_id", default="dev-local")


def _parse_max_age(headers: httpx.Headers | dict[str, str], default: int = 1800) -> int:
    cc = headers.get("cache-control", "")
    match = _CACHE_CONTROL_MAX_AGE_RE.search(cc)
    if not match:
        return default
    try:
        val = int(match.group(1))
        return max(60, min(val, 86400))
    except ValueError:
        return default


class FirebaseTokenVerifier:
    """Verifies Firebase Auth ID tokens against Google's official rotating X.509 certificates."""

    def __init__(
        self,
        certs_url: str = FIREBASE_CERTS_URL,
        http_client_getter: Callable[[], httpx.AsyncClient | None] | None = None,
        cert_fetcher: Callable[[], Awaitable[tuple[dict[str, str], int]]] | None = None,
    ) -> None:
        self.certs_url = certs_url
        self._http_client_getter = http_client_getter
        self._custom_cert_fetcher = cert_fetcher
        self._keys: dict[str, Any] = {}
        self._expires_at: float = 0.0
        self._last_refresh_at: float = 0.0
        self._min_refresh_interval: float = 60.0
        self._loop_locks: dict[int, asyncio.Lock] = {}

    def _get_lock(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        lock = self._loop_locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            self._loop_locks[loop_id] = lock
        return lock

    def reset_cache(self) -> None:
        """Clear cached public keys (intended for tests or manual rotation reset)."""
        self._keys = {}
        self._expires_at = 0.0
        self._last_refresh_at = 0.0

    async def _download_certs(self) -> tuple[dict[str, str], int]:
        if self._custom_cert_fetcher is not None:
            return await self._custom_cert_fetcher()

        client = self._http_client_getter() if self._http_client_getter else None
        if client is not None:
            resp = await client.get(self.certs_url, timeout=8.0)
            resp.raise_for_status()
            return resp.json(), _parse_max_age(resp.headers)

        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as temp_client:
            resp = await temp_client.get(self.certs_url)
            resp.raise_for_status()
            return resp.json(), _parse_max_age(resp.headers)

    async def _get_public_key(self, kid: str) -> Any | None:
        now = time.monotonic()
        if self._keys and now < self._expires_at and kid in self._keys:
            return self._keys[kid]

        async with self._get_lock():
            now = time.monotonic()
            if self._keys and now < self._expires_at and kid in self._keys:
                return self._keys[kid]

            # Prevent unknown-kid DoS floods against Google's JWKS endpoint
            if self._keys and (now - self._last_refresh_at) < self._min_refresh_interval and now < self._expires_at:
                return self._keys.get(kid)

            raw_certs, ttl = await self._download_certs()
            parsed_keys: dict[str, Any] = {}
            for cert_kid, pem in raw_certs.items():
                if not isinstance(cert_kid, str) or not isinstance(pem, str):
                    continue
                cert = x509.load_pem_x509_certificate(pem.encode("utf-8"))
                parsed_keys[cert_kid] = cert.public_key()

            if not parsed_keys:
                raise RuntimeError("No valid Firebase signing certificates loaded.")

            self._keys = parsed_keys
            self._last_refresh_at = time.monotonic()
            self._expires_at = self._last_refresh_at + float(ttl)
            return self._keys.get(kid)

    async def verify_id_token(self, token: str, project_id: str) -> dict[str, Any]:
        """Validate a Firebase ID token and return its verified claims."""
        if not token or len(token) > 16384:
            raise InvalidTokenError("Token is missing or exceeds maximum allowed length.")

        header = jwt.get_unverified_header(token)
        alg = header.get("alg")
        if alg != "RS256":
            raise InvalidTokenError("Unsupported JWT signing algorithm.")

        kid = header.get("kid")
        if not isinstance(kid, str) or not kid.strip():
            raise InvalidTokenError("Missing JWT key identifier.")

        public_key = await self._get_public_key(kid)
        if public_key is None:
            raise InvalidTokenError("Unknown JWT signing key.")

        expected_issuer = f"https://securetoken.google.com/{project_id}"
        claims = jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=expected_issuer,
            options={
                "require": ["exp", "iat", "sub", "aud", "iss"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
            leeway=10,
        )

        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub.strip() or len(sub) > 128:
            raise InvalidTokenError("Invalid token subject claim.")

        now_epoch = time.time()
        auth_time = claims.get("auth_time")
        if auth_time is not None:
            if not isinstance(auth_time, (int, float)) or auth_time > (now_epoch + 10):
                raise InvalidTokenError("Invalid auth_time claim.")

        return claims


def _get_runtime_http_client() -> httpx.AsyncClient | None:
    try:
        from backend.services.runtime import runtime

        return runtime.try_get_http_client()
    except Exception:
        return None


token_verifier = FirebaseTokenVerifier(http_client_getter=_get_runtime_http_client)


async def require_auth(request: Request) -> dict[str, Any]:
    """FastAPI dependency enforcing stateless Firebase RS256 Bearer authentication."""
    if not is_auth_required():
        resolved_sub = "dev-local"
        auth_header = request.headers.get("authorization", "")
        match = _BEARER_RE.match(auth_header.strip())
        if match:
            try:
                unverified = jwt.decode(
                    match.group(1),
                    options={"verify_signature": False, "verify_exp": False},
                )
                candidate = unverified.get("sub") or unverified.get("user_id")
                if isinstance(candidate, str) and _SAFE_SUB_RE.fullmatch(candidate.strip()):
                    resolved_sub = candidate.strip()
            except Exception:
                resolved_sub = "dev-local"
        request.state.user_id = resolved_sub
        current_user_id_ctx.set(resolved_sub)
        return {"uid": resolved_sub, "sub": resolved_sub, "auth_bypass": True}

    project_id = get_firebase_project_id()
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is enabled but FIREBASE_PROJECT_ID is not configured.",
        )

    auth_header = request.headers.get("authorization", "")
    match = _BEARER_RE.match(auth_header.strip())
    if not match:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_token = match.group(1)
    try:
        claims = await token_verifier.verify_id_token(raw_token, project_id)
    except InvalidTokenError as exc:
        logger.warning("Rejected authentication token: %s", sanitize_for_log(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.error("Auth verification backend error: %s", sanitize_for_log(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication verification temporarily unavailable.",
        ) from exc

    request.state.user_id = claims["sub"]
    current_user_id_ctx.set(claims["sub"])
    return claims
