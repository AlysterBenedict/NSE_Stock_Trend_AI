# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in NeuroStock, please **do not** open a public GitHub issue with exploit details.

Instead, report it privately to **Alyster Benedict** via GitHub Security Advisories ("Report a vulnerability" under the repository Security tab) with:
- A description of the vulnerability and affected files/endpoints
- Steps to reproduce or a minimal proof-of-concept
- Potential impact assessment

You can expect an acknowledgment within 48 hours and a coordinated fix prior to public disclosure.

## Built-In Security Controls

1. **Authentication & Authorization**: Every API endpoint (except `/health` and `/ready`) verifies Firebase RS256 ID tokens statelessly against Google's official X.509 certificate metadata endpoint (`AUTH_REQUIRED=true` by default when `FIREBASE_PROJECT_ID` is set or in production).
2. **Model Artifact Integrity**: Serialized ML artifacts in `models/` are verified against `models/manifest.sha256.json` (SHA-256 + canonical path containment) prior to deserialization (`torch.load(..., weights_only=True)`, `joblib.load`, `CatBoostRegressor.load_model`).
3. **Strict Input Validation & Bounds**: All request payloads are validated using Pydantic v2 schemas (finite float checks, ISO date caps, allowlisted tickers/algorithms/modes) and bounded request body size (`256 KB` default).
4. **Prompt Injection & SSRF Mitigations**: Untrusted user prompts and external Yahoo Finance news headlines are sanitized, length-bounded, and isolated inside explicit XML data delimiters; external URLs and `LM_STUDIO_API_URL` are strictly scheme/host-validated.
5. **Browser & Transport Hardening**: Strict CORS origin allowlist, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, CSP, sliding-window per-user/per-IP rate limiting, DOM-safe PDF export without `document.write`, and `http:`/`https:` link sanitization.

## Public Deployment Checklist

Before deploying publicly or pushing forks:
- Copy `frontend/.env.example` to `frontend/.env` and `backend/.env.example` to `backend/.env` with your own Firebase Project ID and keys (never commit `.env` files).
- Restrict your Firebase Web API key in Google Cloud Console to your authorized HTTP referrers and Identity Toolkit / Firebase Auth APIs only.
- Enable Firebase App Check and enforce TLS via a reverse proxy (`ENABLE_HSTS=true`, `APP_ENV=production`).
