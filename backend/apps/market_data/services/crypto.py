"""Fernet-based encryption for broker credentials.

Local/dev path. The prod path is AWS Secrets Manager via BrokerLink.credential_arn.

Key derivation: SHA-256 of DJANGO_SECRET_KEY, urlsafe-base64 encoded — Fernet
requires a 32-byte urlsafe-base64 key. Rotating DJANGO_SECRET_KEY invalidates
all stored blobs, which is the intended security property.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings


def _derive_key() -> bytes:
    secret = settings.SECRET_KEY
    if not secret:
        raise RuntimeError("DJANGO_SECRET_KEY is empty — refusing to derive crypto key")
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_FERNET: Fernet | None = None


def _fernet() -> Fernet:
    global _FERNET
    if _FERNET is None:
        _FERNET = Fernet(_derive_key())
    return _FERNET


def encrypt_credentials(creds: dict[str, Any]) -> bytes:
    """Serialise a credentials dict to an encrypted blob suitable for BrokerLink.credential_blob."""
    payload = json.dumps(creds, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _fernet().encrypt(payload)


def decrypt_credentials(blob: bytes | memoryview | None) -> dict[str, Any]:
    """Inverse of encrypt_credentials. Returns {} for empty blob, raises on tamper."""
    if not blob:
        return {}
    if isinstance(blob, memoryview):
        blob = bytes(blob)
    try:
        plaintext = _fernet().decrypt(blob)
    except InvalidToken as e:
        raise InvalidToken(
            "Failed to decrypt broker credentials — DJANGO_SECRET_KEY may have rotated."
        ) from e
    return json.loads(plaintext.decode("utf-8"))
