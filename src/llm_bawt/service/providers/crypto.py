"""Symmetric encryption for provider credentials at rest.

Provider credentials (OAuth tokens, API keys, GitHub App user tokens) are
persisted in the ``runtime_settings`` table. Nothing else in that table is
encrypted, so this module introduces a small Fernet-based envelope: the value we
store is ``enc:v1:<fernet-ciphertext>``. Anything without the ``enc:v1:`` prefix
is treated as plaintext (back-compat / never-encrypted values).

KEY SOURCE (in priority order):
  1. env ``LLM_BAWT_SECRET_KEY`` — a urlsafe-base64 32-byte Fernet key.
  2. a persisted key file ``~/.config/llm-bawt/secret.key`` (mode 0600). If
     absent it is generated once and written, so restarts stay stable.

IMPORTANT for tenants: the key MUST survive repaves or stored credentials become
unreadable. ``tenant.sh`` bakes ``LLM_BAWT_SECRET_KEY`` into the generated
``.env`` for exactly this reason — the file fallback only covers single-host dev.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFIX = "enc:v1:"
_ENV_KEY = "LLM_BAWT_SECRET_KEY"

# Cached Fernet instance (lazy — cryptography import is deferred).
_fernet = None


def _key_file() -> Path:
    return Path.home() / ".config" / "llm-bawt" / "secret.key"


def _load_or_create_key() -> bytes:
    """Resolve the Fernet key from env, else a persisted file, else generate."""
    from cryptography.fernet import Fernet

    env = os.getenv(_ENV_KEY)
    if env:
        return env.strip().encode()

    path = _key_file()
    if path.exists():
        return path.read_text().strip().encode()

    key = Fernet.generate_key()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key.decode())
        path.chmod(0o600)
        logger.warning(
            "Generated a new provider-credential encryption key at %s. "
            "For tenants, set %s in the environment so it survives repaves.",
            path,
            _ENV_KEY,
        )
    except Exception as e:  # noqa: BLE001 — still usable in-process this run
        logger.warning("Could not persist encryption key to %s: %s", path, e)
    return key


def _get_fernet():
    global _fernet
    if _fernet is None:
        from cryptography.fernet import Fernet

        _fernet = Fernet(_load_or_create_key())
    return _fernet


def encrypt(plaintext: str) -> str:
    """Return ``enc:v1:<ciphertext>`` for a plaintext string."""
    if plaintext is None:
        raise ValueError("cannot encrypt None")
    token = _get_fernet().encrypt(plaintext.encode()).decode()
    return f"{_PREFIX}{token}"


def decrypt(stored: str) -> str:
    """Reverse :func:`encrypt`. Values without the prefix are returned as-is."""
    if stored is None:
        raise ValueError("cannot decrypt None")
    if not stored.startswith(_PREFIX):
        return stored  # plaintext / legacy value
    token = stored[len(_PREFIX) :].encode()
    return _get_fernet().decrypt(token).decode()


def is_encrypted(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith(_PREFIX)
