"""Hugging Face Spaces entrypoint mounting Gradio onto the NeuroStock FastAPI app."""

import logging
import os
import socket

# Avoid initializing ZeroGPU CUDA in the main FastAPI process on Hugging Face Spaces
os.environ.setdefault("NEUROSTOCK_FORCE_CPU", "1")

import gradio as gr
import spaces
import uvicorn

from backend.app import app as fastapi_app

logger = logging.getLogger(__name__)

_NEUROSTOCK_API_PATHS = frozenset(
    {
        "/health",
        "/ready",
        "/predict",
        "/historical-data",
        "/run-backtest",
        "/get-market-data",
        "/alpha-radar",
        "/calculate-portfolio",
        "/investment-engine",
        "/optimize-portfolio",
        "/get-sentiment",
        "/llm-settings",
        "/get-general-knowledge",
        "/get-ai-insights",
    }
)


@spaces.GPU
def status_probe() -> str:
    """ZeroGPU-compatible probe required by Hugging Face ZeroGPU Spaces at startup."""
    return "NeuroStock Quantitative API is online."


with gr.Blocks(title="NeuroStock API") as demo:
    gr.Markdown(
        "# NeuroStock Quantitative API\n"
        "FastAPI + Uvicorn backend is running.\n\n"
        "- Health check: `/gradio_api/health`\n"
        "- Readiness probe: `/gradio_api/ready`"
    )
    probe_btn = gr.Button("Check Backend Status")
    probe_out = gr.Textbox(label="Status")
    probe_btn.click(fn=status_probe, outputs=probe_out)

gr.mount_gradio_app(fastapi_app, demo, path="/")


class GradioApiPrefixMiddleware:
    """Route /gradio_api/<backend_endpoint> requests forwarded by Gradio 6 SSR to FastAPI."""

    def __init__(self, asgi_app) -> None:
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path.startswith("/gradio_api/"):
                stripped = path[len("/gradio_api") :]
                if stripped in _NEUROSTOCK_API_PATHS:
                    scope = {
                        **scope,
                        "path": stripped,
                        "raw_path": stripped.encode("latin-1"),
                    }
        await self.asgi_app(scope, receive, send)


app = GradioApiPrefixMiddleware(fastapi_app)


def _find_open_port(preferred: int = 7860) -> int:
    for candidate in range(preferred, preferred + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", candidate))
                return candidate
            except OSError:
                continue
    return preferred


if __name__ == "__main__":
    raw_port = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or "7860"
    port = _find_open_port(int(raw_port))
    logger.info("Starting NeuroStock Space server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, server_header=False)
