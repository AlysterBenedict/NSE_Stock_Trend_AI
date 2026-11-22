"""Global Cross-Sectional Panel 1-Day-Ahead Training Pipeline (100 NSE Stocks, 2005-Present).

Usage:
    python ml/train_one_day_models.py
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.config import MODELS_DIR_ONE_DAY, START_DATE, TRAINABLE_STOCK_LIST  # noqa: E402
from ml.training_utils import (  # noqa: E402
    build_global_training_panel,
    configure_training_environment,
    train_global_panel_for_horizon,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Global Cross-Sectional Panel 1-day-ahead SOTA models across all 100 NSE stocks."
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Historical start date for OHLCV download (default: {START_DATE}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = configure_training_environment(MODELS_DIR_ONE_DAY)
    panel = build_global_training_panel(TRAINABLE_STOCK_LIST, start_date=args.start_date)
    train_global_panel_for_horizon(
        panel_base=panel,
        horizon=1,
        models_dir=MODELS_DIR_ONE_DAY,
        prefix="global",
        device=device,
        train_quantiles=True,
    )
    print("\n--- GLOBAL PANEL 1-DAY TRAINING COMPLETE ---")
