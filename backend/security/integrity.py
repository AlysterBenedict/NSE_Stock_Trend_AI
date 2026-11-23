"""Cryptographic SHA-256 integrity manifest and canonical path containment for model artifacts."""

import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone

from backend.config import MODELS_DIR, is_model_integrity_required
from backend.security.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.sha256.json"
MANIFEST_PATH = os.path.join(MODELS_DIR, MANIFEST_FILENAME)
ARTIFACT_EXTENSIONS = frozenset({".joblib", ".pt", ".cbm", ".json"})

_MANIFEST_LOCK = threading.Lock()
_MANIFEST_CACHE = {"path": None, "mtime": None, "files": {}}


class ModelIntegrityError(RuntimeError):
    """Raised when a model artifact fails path containment or SHA-256 verification."""


def compute_sha256(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file using streaming 64 KiB reads."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while chunk := fh.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_relpath(real_path: str, real_root: str) -> str:
    rel = os.path.relpath(real_path, real_root)
    return rel.replace("\\", "/")


def generate_model_manifest(models_dir: str = MODELS_DIR) -> dict:
    """Hash all model and benchmark artifacts under models_dir and atomically write manifest.sha256.json."""
    real_root = os.path.realpath(models_dir)
    if not os.path.isdir(real_root):
        raise FileNotFoundError(f"Models directory does not exist: {real_root}")

    entries: dict[str, dict[str, object]] = {}
    for root, _, filenames in os.walk(real_root):
        for fname in sorted(filenames):
            if fname == MANIFEST_FILENAME or fname.endswith(".tmp"):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ARTIFACT_EXTENSIONS:
                continue
            full_path = os.path.realpath(os.path.join(root, fname))
            if os.path.commonpath([real_root, full_path]) != real_root:
                continue
            rel = _normalized_relpath(full_path, real_root)
            entries[rel] = {
                "sha256": compute_sha256(full_path),
                "bytes": os.path.getsize(full_path),
            }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(entries),
        "files": dict(sorted(entries.items())),
    }

    manifest_path = os.path.join(real_root, MANIFEST_FILENAME)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp_path, manifest_path)

    with _MANIFEST_LOCK:
        _MANIFEST_CACHE["path"] = None
        _MANIFEST_CACHE["mtime"] = None
        _MANIFEST_CACHE["files"] = {}

    logger.info("Generated SHA-256 model integrity manifest (%d artifacts).", len(entries))
    return manifest


def _load_manifest_files(models_dir: str) -> dict[str, dict[str, object]]:
    real_root = os.path.realpath(models_dir)
    manifest_path = os.path.join(real_root, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise ModelIntegrityError(
            f"Missing model integrity manifest: {manifest_path}. "
            "Run generate_model_manifest() before loading models."
        )

    mtime = os.path.getmtime(manifest_path)
    with _MANIFEST_LOCK:
        if (
            _MANIFEST_CACHE["path"] == manifest_path
            and _MANIFEST_CACHE["mtime"] == mtime
            and _MANIFEST_CACHE["files"]
        ):
            return _MANIFEST_CACHE["files"]

        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        files = data.get("files")
        if not isinstance(files, dict) or not files:
            raise ModelIntegrityError("Model integrity manifest is empty or malformed.")

        _MANIFEST_CACHE["path"] = manifest_path
        _MANIFEST_CACHE["mtime"] = mtime
        _MANIFEST_CACHE["files"] = files
        return files


def verify_artifact_path_and_hash(
    filepath: str,
    models_dir: str = MODELS_DIR,
    verify_hash: bool | None = None,
) -> str:
    """Verify canonical path containment inside models_dir and optional SHA-256 manifest match."""
    real_root = os.path.realpath(models_dir)
    real_path = os.path.realpath(filepath)

    if os.path.commonpath([real_root, real_path]) != real_root:
        raise ModelIntegrityError(f"Path escapes trusted models directory: {filepath}")
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"Model artifact not found: {real_path}")

    should_verify = is_model_integrity_required() if verify_hash is None else bool(verify_hash)
    if not should_verify:
        return real_path

    rel = _normalized_relpath(real_path, real_root)
    files = _load_manifest_files(real_root)
    entry = files.get(rel)
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise ModelIntegrityError(f"Artifact missing from SHA-256 manifest: {rel}")

    expected = entry["sha256"].lower()
    actual = compute_sha256(real_path).lower()
    if not hmac.compare_digest(actual, expected):
        raise ModelIntegrityError(
            f"SHA-256 integrity mismatch for {rel} (possible tampering or corruption)."
        )
    return real_path


def verify_all_manifest_entries(models_dir: str = MODELS_DIR) -> int:
    """Verify every artifact listed in manifest.sha256.json and return verified file count."""
    real_root = os.path.realpath(models_dir)
    files = _load_manifest_files(real_root)
    for rel in files:
        verify_artifact_path_and_hash(os.path.join(real_root, rel), models_dir=real_root, verify_hash=True)
    return len(files)
