"""Pytest fixtures for ephemeral RS256 Firebase JWT verification and FastAPI TestClient."""

from datetime import datetime, timedelta, timezone
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient
import jwt
import pytest

from backend.app import create_app
from backend.security.auth import token_verifier
from backend.security.middleware import rate_limiter


def _generate_ephemeral_rsa_cert():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "neurostock-test")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=2))
        .sign(private_key, hashes.SHA256())
    )
    pem_cert = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    return private_key, pem_cert


@pytest.fixture
def rsa_auth_env(monkeypatch):
    """Configure an isolated Firebase RS256 JWT test environment with ephemeral keys."""
    private_key, pem_cert = _generate_ephemeral_rsa_cert()
    other_private_key, _ = _generate_ephemeral_rsa_cert()

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "neurostock-test-proj")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("VERIFY_MODEL_INTEGRITY", "false")
    monkeypatch.setenv("NEUROSTOCK_SKIP_MODEL_LOAD", "1")

    rate_limiter.clear()
    token_verifier.reset_cache()

    async def fake_cert_fetcher():
        return {"kid-1": pem_cert}, 1800

    monkeypatch.setattr(token_verifier, "_custom_cert_fetcher", fake_cert_fetcher)

    def sign_token(
        sub: str = "user-alice-123",
        aud: str = "neurostock-test-proj",
        iss: str = "https://securetoken.google.com/neurostock-test-proj",
        exp_offset: int = 3600,
        iat_offset: int = 0,
        kid: str = "kid-1",
        signing_key=None,
        auth_time_offset: int | None = -60,
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": sub,
            "user_id": sub,
            "aud": aud,
            "iss": iss,
            "iat": now + iat_offset,
            "exp": now + exp_offset,
        }
        if auth_time_offset is not None:
            payload["auth_time"] = now + auth_time_offset
        return jwt.encode(
            payload,
            signing_key or private_key,
            algorithm="RS256",
            headers={"kid": kid} if kid else None,
        )

    yield {
        "private_key": private_key,
        "other_private_key": other_private_key,
        "pem_cert": pem_cert,
        "sign_token": sign_token,
    }

    token_verifier.reset_cache()
    rate_limiter.clear()


@pytest.fixture
def auth_client(rsa_auth_env):
    """FastAPI TestClient with RS256 auth enabled."""
    app = create_app(load_models=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, rsa_auth_env
