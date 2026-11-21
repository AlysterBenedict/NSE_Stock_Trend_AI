"""Global Cross-Sectional Panel Direct Multi-Horizon Training Pipeline (h = 5, 10, 21, 63d).

Usage:
    # Train all 4 multi-day horizons across all 58 NSE stocks (resumable):
    python ml/train_long_range_models.py --resume

    # Or train specific horizons:
    python ml/train_long_range_models.py --horizons 5 21
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.config import (  # noqa: E402
    LONG_RANGE_HORIZONS,
    MODELS_DIR_LONG_RANGE,
    START_DATE,
    TRAINABLE_STOCK_LIST,
)
from ml.training_utils import (  # noqa: E402
    build_global_training_panel,
    configure_training_environment,
    train_global_panel_for_horizon,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Global Cross-Sectional Panel Direct Multi-Horizon models + P10/P50/P90 quantiles."
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=LONG_RANGE_HORIZONS,
        help=f"Target trading-day horizons (default: {LONG_RANGE_HORIZONS}).",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Historical start date for OHLCV download (default: {START_DATE}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip horizons that already have Global Panel production artifacts saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = configure_training_environment(MODELS_DIR_LONG_RANGE)

    target_horizons = list(args.horizons)
    if args.resume:
        target_horizons = [
            h
            for h in target_horizons
            if not os.path.exists(os.path.join(MODELS_DIR_LONG_RANGE, f"global_h{h}_Ensemble_meta.joblib"))
        ]
        print(f"Resume mode active: remaining horizons = {target_horizons}")

    if not target_horizons:
        print("All requested multi-horizon Global Panel models are already trained.")
        sys.exit(0)

    panel = build_global_training_panel(TRAINABLE_STOCK_LIST, start_date=args.start_date)
    for h in target_horizons:
        train_global_panel_for_horizon(
            panel_base=panel,
            horizon=h,
            models_dir=MODELS_DIR_LONG_RANGE,
            prefix=f"global_h{h}",
            device=device,
            train_quantiles=True,
        )

    print("\n--- GLOBAL PANEL DIRECT MULTI-HORIZON TRAINING COMPLETE ---")
