"""Global Cross-Sectional Panel model, calibrator, and benchmark metrics registry."""

import json
import os
import joblib

from backend.config import (
    BASE_ALGORITHMS,
    LONG_RANGE_HORIZONS,
    METRICS_SUMMARY_PATH,
    MODELS_DIR_LONG_RANGE,
    MODELS_DIR_ONE_DAY,
)
from backend.services.attention_net import get_torch_device, load_attention_net_bundle


class ModelRegistry:
    """Loads Global Cross-Sectional Panel models, multi-seed bags, OOF calibrators, and quantile cones."""

    def __init__(self):
        self.device = None
        # Horizon -> dict with keys: scaler, models, calibrators, ensemble, quantiles, feature_indices
        self.horizons = {}
        self.metrics_summary = {}
        self._loaded = False

    def load_all(self):
        """Load Global Panel models for h=1 (one_day) and h in {5, 10, 21, 63} (long_range)."""
        if self._loaded:
            return

        self.device = get_torch_device()
        self.horizons = {}
        print(f"--- Loading Global Panel SOTA models (PyTorch device: {self.device}) ---")
        self._load_metrics_summary()

        # Load 1-day global panel
        bundle_1d = self._load_horizon_bundle(MODELS_DIR_ONE_DAY, "global")
        if bundle_1d is not None:
            self.horizons[1] = bundle_1d
            print(f"--- Loaded Global Panel h=1d ({len(bundle_1d['models'])} base families) ---")

        # Load direct multi-horizon global panels
        for h in LONG_RANGE_HORIZONS:
            bundle_h = self._load_horizon_bundle(MODELS_DIR_LONG_RANGE, f"global_h{h}")
            if bundle_h is not None:
                self.horizons[h] = bundle_h
                print(f"--- Loaded Global Panel h={h}d ({len(bundle_h['models'])} base families) ---")

        self._loaded = True
        print("--- Model registry initialization complete. ---")

    def _load_metrics_summary(self):
        if os.path.exists(METRICS_SUMMARY_PATH):
            try:
                with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as fh:
                    self.metrics_summary = json.load(fh)
            except Exception as exc:
                print(f"Warning: failed to read metrics summary ({METRICS_SUMMARY_PATH}): {exc}")

    def _load_single_family(self, directory, prefix, algo):
        """Load one algorithm family (supports multi-seed bagged lists, .pt bundles, .cbm lists, or .joblib)."""
        if algo == "AttentionNet":
            path = os.path.join(directory, f"{prefix}_AttentionNet.pt")
            if os.path.exists(path):
                return load_attention_net_bundle(path, device=self.device)
            return None

        if algo == "CatBoost":
            # Support multi-seed .cbm files: {prefix}_CatBoost_seed0.cbm, ...
            from catboost import CatBoostRegressor

            seed_models = []
            for seed_idx in range(10):
                seed_path = os.path.join(directory, f"{prefix}_CatBoost_seed{seed_idx}.cbm")
                if os.path.exists(seed_path):
                    m = CatBoostRegressor()
                    m.load_model(seed_path)
                    seed_models.append(m)
                else:
                    break
            if seed_models:
                return seed_models

            single_cbm = os.path.join(directory, f"{prefix}_CatBoost.cbm")
            if os.path.exists(single_cbm):
                m = CatBoostRegressor()
                m.load_model(single_cbm)
                return [m]
            return None

        path = os.path.join(directory, f"{prefix}_{algo}.joblib")
        if os.path.exists(path):
            loaded = joblib.load(path)
            return loaded if isinstance(loaded, list) else [loaded]
        return None

    def _load_horizon_bundle(self, directory, prefix):
        """Load scaler, selected feature indices, base models, calibrators, stacker, and quantiles."""
        scaler_path = os.path.join(directory, f"{prefix}_feature_scaler.joblib")
        if not os.path.exists(scaler_path):
            return None

        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            print(f"Warning: failed loading scaler ({scaler_path}): {exc}")
            return None

        models = {}
        for algo in BASE_ALGORITHMS:
            try:
                loaded = self._load_single_family(directory, prefix, algo)
                if loaded:
                    models[algo] = loaded
            except Exception as exc:
                print(f"Warning: failed loading {prefix}_{algo}: {exc}")

        calibrators = {}
        cal_path = os.path.join(directory, f"{prefix}_calibrators.joblib")
        if os.path.exists(cal_path):
            try:
                calibrators = joblib.load(cal_path)
            except Exception as exc:
                print(f"Warning: failed loading calibrators ({cal_path}): {exc}")

        ensemble = None
        ens_path = os.path.join(directory, f"{prefix}_Ensemble_meta.joblib")
        if os.path.exists(ens_path):
            try:
                ensemble = joblib.load(ens_path)
            except Exception as exc:
                print(f"Warning: failed loading ensemble meta ({ens_path}): {exc}")

        quantiles = {}
        for q_name in ("q10", "q50", "q90"):
            q_path = os.path.join(directory, f"{prefix}_{q_name}.joblib")
            if os.path.exists(q_path):
                try:
                    quantiles[q_name] = joblib.load(q_path)
                except Exception:
                    pass

        meta_path = os.path.join(directory, f"{prefix}_metadata.joblib")
        metadata = {}
        if os.path.exists(meta_path):
            try:
                metadata = joblib.load(meta_path)
            except Exception:
                metadata = {}

        if not models:
            return None

        return {
            "scaler": scaler,
            "models": models,
            "calibrators": calibrators,
            "ensemble": ensemble,
            "quantiles": quantiles,
            "metadata": metadata,
        }

    def get_bundle(self, horizon=1):
        """Return loaded Global Panel bundle for the specified trading-day horizon."""
        return self.horizons.get(horizon)

    def get_metrics(self, ticker, algorithm_name, horizon=1):
        """Return per-ticker out-of-sample metrics if present, else GLOBAL_PANEL metrics."""
        horizon_key = "one_day" if horizon == 1 else f"h{horizon}"
        ticker_metrics = (
            self.metrics_summary.get(ticker, {})
            .get(horizon_key, {})
            .get(algorithm_name)
        )
        if ticker_metrics:
            return ticker_metrics
        return (
            self.metrics_summary.get("GLOBAL_PANEL", {})
            .get(horizon_key, {})
            .get(algorithm_name)
        )


registry = ModelRegistry()
