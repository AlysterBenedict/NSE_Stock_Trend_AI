"""Global Cross-Sectional Panel model, calibrator, and benchmark metrics registry with SHA-256 verification."""

import json
import os
import threading

import joblib

from backend.config import (
    BASE_ALGORITHMS,
    LONG_RANGE_HORIZONS,
    METRICS_SUMMARY_PATH,
    MODELS_DIR,
    MODELS_DIR_LONG_RANGE,
    MODELS_DIR_ONE_DAY,
)
from backend.security.integrity import ModelIntegrityError, verify_artifact_path_and_hash
from backend.security.logging_utils import get_logger
from backend.services.attention_net import get_torch_device, load_attention_net_bundle

logger = get_logger(__name__)


class ModelRegistry:
    """Loads integrity-verified Global Cross-Sectional Panel models, calibrators, and quantile cones."""

    def __init__(self) -> None:
        self.device = None
        self.horizons: dict[int, dict] = {}
        self.metrics_summary: dict = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load_all(self) -> None:
        """Load Global Panel models for h=1 (one_day) and h in {5, 10, 21, 63} (long_range)."""
        with self._lock:
            if self._loaded:
                return

            self.device = get_torch_device()
            self.horizons = {}
            logger.info("Loading Global Panel SOTA models (PyTorch device: %s)", self.device)
            self._load_metrics_summary()

            bundle_1d = self._load_horizon_bundle(MODELS_DIR_ONE_DAY, "global")
            if bundle_1d is not None:
                self.horizons[1] = bundle_1d
                logger.info("Loaded Global Panel h=1d (%d base families)", len(bundle_1d["models"]))

            for h in LONG_RANGE_HORIZONS:
                bundle_h = self._load_horizon_bundle(MODELS_DIR_LONG_RANGE, f"global_h{h}")
                if bundle_h is not None:
                    self.horizons[h] = bundle_h
                    logger.info("Loaded Global Panel h=%dd (%d base families)", h, len(bundle_h["models"]))

            self._loaded = True
            logger.info("Model registry initialization complete (%d horizons loaded).", len(self.horizons))

    def _verified_joblib_load(self, path: str):
        safe_path = verify_artifact_path_and_hash(path, models_dir=MODELS_DIR)
        return joblib.load(safe_path)

    def _load_metrics_summary(self) -> None:
        if os.path.exists(METRICS_SUMMARY_PATH):
            try:
                safe_path = verify_artifact_path_and_hash(METRICS_SUMMARY_PATH, models_dir=MODELS_DIR)
                with open(safe_path, "r", encoding="utf-8") as fh:
                    self.metrics_summary = json.load(fh)
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed to read metrics summary (%s): %s", METRICS_SUMMARY_PATH, exc)

    def _load_single_family(self, directory: str, prefix: str, algo: str):
        """Load one algorithm family after path containment and SHA-256 verification."""
        if algo == "AttentionNet":
            path = os.path.join(directory, f"{prefix}_AttentionNet.pt")
            if os.path.exists(path):
                safe_path = verify_artifact_path_and_hash(path, models_dir=MODELS_DIR)
                return load_attention_net_bundle(safe_path, device=self.device)
            return None

        if algo == "CatBoost":
            from catboost import CatBoostRegressor

            seed_models = []
            for seed_idx in range(10):
                seed_path = os.path.join(directory, f"{prefix}_CatBoost_seed{seed_idx}.cbm")
                if os.path.exists(seed_path):
                    safe_path = verify_artifact_path_and_hash(seed_path, models_dir=MODELS_DIR)
                    model = CatBoostRegressor()
                    model.load_model(safe_path)
                    seed_models.append(model)
                else:
                    break
            if seed_models:
                return seed_models

            single_cbm = os.path.join(directory, f"{prefix}_CatBoost.cbm")
            if os.path.exists(single_cbm):
                safe_path = verify_artifact_path_and_hash(single_cbm, models_dir=MODELS_DIR)
                model = CatBoostRegressor()
                model.load_model(safe_path)
                return [model]
            return None

        path = os.path.join(directory, f"{prefix}_{algo}.joblib")
        if os.path.exists(path):
            loaded = self._verified_joblib_load(path)
            return loaded if isinstance(loaded, list) else [loaded]
        return None

    def _load_horizon_bundle(self, directory: str, prefix: str):
        """Load scaler, selected feature indices, base models, calibrators, stacker, and quantiles."""
        scaler_path = os.path.join(directory, f"{prefix}_feature_scaler.joblib")
        if not os.path.exists(scaler_path):
            return None

        try:
            scaler = self._verified_joblib_load(scaler_path)
        except ModelIntegrityError:
            raise
        except Exception as exc:
            logger.warning("Failed loading scaler (%s): %s", scaler_path, exc)
            return None

        models = {}
        for algo in BASE_ALGORITHMS:
            try:
                loaded = self._load_single_family(directory, prefix, algo)
                if loaded:
                    models[algo] = loaded
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed loading %s_%s: %s", prefix, algo, exc)

        calibrators = {}
        cal_path = os.path.join(directory, f"{prefix}_calibrators.joblib")
        if os.path.exists(cal_path):
            try:
                calibrators = self._verified_joblib_load(cal_path)
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed loading calibrators (%s): %s", cal_path, exc)

        ensemble = None
        ens_path = os.path.join(directory, f"{prefix}_Ensemble_meta.joblib")
        if os.path.exists(ens_path):
            try:
                ensemble = self._verified_joblib_load(ens_path)
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed loading ensemble meta (%s): %s", ens_path, exc)

        quantiles = {}
        for q_name in ("q10", "q50", "q90"):
            q_path = os.path.join(directory, f"{prefix}_{q_name}.joblib")
            if os.path.exists(q_path):
                try:
                    quantiles[q_name] = self._verified_joblib_load(q_path)
                except ModelIntegrityError:
                    raise
                except Exception as exc:
                    logger.warning("Failed loading quantile %s (%s): %s", q_name, q_path, exc)

        meta_path = os.path.join(directory, f"{prefix}_metadata.joblib")
        metadata = {}
        if os.path.exists(meta_path):
            try:
                metadata = self._verified_joblib_load(meta_path)
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed loading metadata (%s): %s", meta_path, exc)
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

    def get_bundle(self, horizon: int = 1):
        """Return loaded Global Panel bundle for the specified trading-day horizon."""
        return self.horizons.get(horizon)

    def get_metrics(self, ticker: str, algorithm_name: str, horizon: int = 1):
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
