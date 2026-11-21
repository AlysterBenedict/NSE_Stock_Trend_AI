"""Flask application factory and server entry point for NeuroStock."""

import os
import sys
import warnings

from flask import Flask
from flask_cors import CORS

# Allow running directly as `python backend/app.py` from the repository root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.config import LM_STUDIO_API_URL, LOCAL_MODEL_NAME  # noqa: E402
from backend.routes import register_blueprints  # noqa: E402
from backend.services.model_registry import registry  # noqa: E402


def create_app(load_models=True):
    """Create and configure the Flask application."""
    warnings.filterwarnings("ignore")

    app = Flask(__name__)
    CORS(app)

    print(f"--- AI insights will be routed to: {LM_STUDIO_API_URL} using model {LOCAL_MODEL_NAME} ---")
    if load_models:
        registry.load_all()

    register_blueprints(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
