"""Per-user AES-256-GCM encrypted vault for NVIDIA API keys."""

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import threading

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from backend.security.logging_utils import get_logger

logger = get_logger(__name__)

_NVIDIA_API_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{20,256}$")
_DEFAULT_VAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".llm_key_vault.json",
)


def validate_nvidia_api_key(raw: object) -> str:
    """Validate and normalize an NVIDIA NIM API key without logging its value."""
    if not isinstance(raw, str):
        raise ValueError("Invalid NVIDIA API key format.")
    cleaned = raw.strip()
    if not cleaned or any(ch.isspace() for ch in cleaned):
        raise ValueError("Invalid NVIDIA API key format.")
    if not _NVIDIA_API_KEY_RE.fullmatch(cleaned):
        raise ValueError(
            "Invalid NVIDIA API key format. Expected 20-256 alphanumeric/dash/underscore characters."
        )
    return cleaned


def mask_api_key(api_key: str) -> str:
    """Return a non-sensitive masked preview of an API key."""
    if not api_key or len(api_key) < 8:
        return "••••••••"
    if api_key.startswith("nvapi-"):
        return f"nvapi-••••••••{api_key[-4:]}"
    return f"{api_key[:4]}••••••••{api_key[-4:]}"


def _hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()


def _derive_user_cipher_key(master_secret: bytes, salt: bytes, user_hash: str) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=f"neurostock-llm-key:{user_hash}".encode("utf-8"),
    )
    return hkdf.derive(master_secret)


class EncryptedKeyVault:
    """Thread-safe, per-user AES-256-GCM encrypted storage for user LLM API keys."""

    def __init__(self, vault_path: str | None = None) -> None:
        self._explicit_vault_path = vault_path
        self._lock = threading.Lock()

    @property
    def vault_path(self) -> str:
        if self._explicit_vault_path:
            return self._explicit_vault_path
        env_path = os.environ.get("LLM_VAULT_PATH", "").strip()
        return env_path or _DEFAULT_VAULT_PATH

    @property
    def _local_master_key_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.vault_path)), ".llm_vault.key")

    def _get_master_secret(self) -> bytes:
        env_secret = os.environ.get("LLM_KEY_ENCRYPTION_SECRET", "").strip()
        if env_secret:
            return hashlib.sha256(env_secret.encode("utf-8")).digest()

        key_path = self._local_master_key_path
        if os.path.isfile(key_path):
            try:
                with open(key_path, "rb") as fh:
                    raw = fh.read().strip()
                decoded = base64.urlsafe_b64decode(raw)
                if len(decoded) == 32:
                    return decoded
            except Exception:
                pass

        os.makedirs(os.path.dirname(os.path.abspath(key_path)), exist_ok=True)
        secret_bytes = os.urandom(32)
        encoded = base64.urlsafe_b64encode(secret_bytes)
        tmp_path = f"{key_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(encoded)
        os.replace(tmp_path, key_path)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        return secret_bytes

    def _read_vault_unlocked(self) -> dict:
        path = self.vault_path
        if not os.path.isfile(path):
            return {"version": 1, "entries": {}}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                return data
        except Exception as exc:
            logger.warning("Could not parse LLM vault file (%s); initializing empty view.", type(exc).__name__)
        return {"version": 1, "entries": {}}

    def _write_vault_unlocked(self, data: dict) -> None:
        path = self.vault_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _decrypt_entry_unlocked(self, user_hash: str, entry: dict) -> str | None:
        try:
            salt = base64.b64decode(entry["salt"])
            nonce = base64.b64decode(entry["nonce"])
            ciphertext = base64.b64decode(entry["ciphertext"])
            master_secret = self._get_master_secret()
            cipher_key = _derive_user_cipher_key(master_secret, salt, user_hash)
            plaintext = AESGCM(cipher_key).decrypt(nonce, ciphertext, user_hash.encode("utf-8"))
            return validate_nvidia_api_key(plaintext.decode("utf-8"))
        except Exception:
            return None

    def save_user_api_key(self, user_id: str, raw_key: str) -> dict:
        """Validate, encrypt with AES-256-GCM + per-user HKDF, and persist an API key."""
        if not user_id or not user_id.strip():
            raise ValueError("Missing user identity.")
        cleaned_key = validate_nvidia_api_key(raw_key)
        user_hash = _hash_user_id(user_id)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        updated_at = datetime.now(timezone.utc).isoformat()
        masked = mask_api_key(cleaned_key)

        with self._lock:
            master_secret = self._get_master_secret()
            cipher_key = _derive_user_cipher_key(master_secret, salt, user_hash)
            ciphertext = AESGCM(cipher_key).encrypt(
                nonce,
                cleaned_key.encode("utf-8"),
                user_hash.encode("utf-8"),
            )
            vault = self._read_vault_unlocked()
            vault["entries"][user_hash] = {
                "salt": base64.b64encode(salt).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
                "masked_key": masked,
                "updated_at": updated_at,
            }
            self._write_vault_unlocked(vault)

        return {"has_api_key": True, "masked_key": masked, "updated_at": updated_at}

    def get_user_api_key(self, user_id: str | None) -> str | None:
        """Decrypt and return the saved NVIDIA API key for a user, or None if absent/invalid."""
        if not user_id or not user_id.strip():
            return None
        user_hash = _hash_user_id(user_id)
        with self._lock:
            vault = self._read_vault_unlocked()
            entry = vault["entries"].get(user_hash)
            if not isinstance(entry, dict):
                return None
            return self._decrypt_entry_unlocked(user_hash, entry)

    def get_user_key_metadata(self, user_id: str | None) -> dict:
        """Return safe key metadata (never the secret) after verifying the stored ciphertext."""
        empty = {"has_api_key": False, "masked_key": None, "updated_at": None}
        if not user_id or not user_id.strip():
            return empty
        user_hash = _hash_user_id(user_id)
        with self._lock:
            vault = self._read_vault_unlocked()
            entry = vault["entries"].get(user_hash)
            if not isinstance(entry, dict):
                return empty
            decrypted = self._decrypt_entry_unlocked(user_hash, entry)
            if not decrypted:
                return empty
            return {
                "has_api_key": True,
                "masked_key": mask_api_key(decrypted),
                "updated_at": entry.get("updated_at"),
            }

    def delete_user_api_key(self, user_id: str | None) -> bool:
        """Remove a user's stored NVIDIA API key from the encrypted vault."""
        if not user_id or not user_id.strip():
            return False
        user_hash = _hash_user_id(user_id)
        with self._lock:
            vault = self._read_vault_unlocked()
            if user_hash not in vault["entries"]:
                return False
            del vault["entries"][user_hash]
            self._write_vault_unlocked(vault)
            return True


key_vault = EncryptedKeyVault()
