"""Async FastAPI ASGI application factory and Uvicorn entry point for NeuroStock."""

from contextlib import asynccontextmanager
import os
import sys
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
import uvicorn

# Allow running directly as `python backend/app.py` from the repository root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.config import (  # noqa: E402
    LOCAL_MODEL_NAME,
    get_allowed_origins,
    get_app_env,
    get_host,
    get_lm_studio_api_url,
    get_log_level,
    get_port,
    is_api_docs_enabled,
    is_auth_required,
    validate_runtime_security_config,
)
from backend.routes import register_routers  # noqa: E402
from backend.security.logging_utils import (  # noqa: E402
    configure_logging,
    get_logger,
    request_id_ctx,
)
from backend.security.middleware import SecurityAndContextMiddleware  # noqa: E402
from backend.services.model_registry import registry  # noqa: E402
from backend.services.runtime import runtime  # noqa: E402

logger = get_logger(__name__)


def _configure_narrow_warnings() -> None:
    """Silence only benign third-party pandas/sklearn deprecation noise without hiding security warnings."""
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"^(yfinance|pandas|sklearn|seaborn).*")


def create_app(load_models: bool = True) -> FastAPI:
    """Create and configure the asynchronous FastAPI application."""
    configure_logging(get_log_level())
    _configure_narrow_warnings()
    validate_runtime_security_config()

    if not is_auth_required():
        logger.warning(
            "SECURITY WARNING: AUTH_REQUIRED is disabled (APP_ENV=%s). "
            "Enable FIREBASE_PROJECT_ID and AUTH_REQUIRED=true before exposing beyond localhost.",
            get_app_env(),
        )

    docs_enabled = is_api_docs_enabled()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "Starting NeuroStock ASGI server (env=%s, llm=%s, model=%s)",
            get_app_env(),
            get_lm_studio_api_url(),
            LOCAL_MODEL_NAME,
        )
        await runtime.startup()
        if load_models:
            await runtime.run_io(registry.load_all)
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(
        title="NeuroStock Quantitative API",
        version="3.0.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # Registration order: last registered wraps outermost.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
        max_age=600,
    )
    app.add_middleware(SecurityAndContextMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first_msg = errors[0].get("msg", "Invalid request payload.") if errors else "Invalid request payload."
        if first_msg.startswith("Value error, "):
            first_msg = first_msg[len("Value error, ") :]
        return ORJSONResponse({"error": first_msg}, status_code=400)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        return ORJSONResponse(
            {"error": str(exc.detail)},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, _exc: Exception):
        logger.exception("Unhandled exception while processing request")
        return ORJSONResponse(
            {
                "error": "Internal server error.",
                "request_id": request_id_ctx.get("-"),
            },
            status_code=500,
        )

    @app.get("/health", tags=["ops"])
    async def health_endpoint():
        """Lightweight liveness probe."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def readiness_endpoint():
        """Readiness probe confirming model registry state and loaded forecast horizons."""
        is_ready = bool(registry._loaded)
        payload = {
            "status": "ready" if is_ready else "degraded",
            "models_loaded": is_ready,
            "horizons": sorted(registry.horizons.keys()),
        }
        return ORJSONResponse(payload, status_code=200 if is_ready else 503)

    register_routers(app)
    return app


_SKIP_MODELS = os.environ.get("NEUROSTOCK_SKIP_MODEL_LOAD", "").strip().lower() in ("1", "true", "yes")
app = create_app(load_models=not _SKIP_MODELS)

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=get_host(),
        port=get_port(),
        log_level=get_log_level().lower(),
        server_header=False,
    )
