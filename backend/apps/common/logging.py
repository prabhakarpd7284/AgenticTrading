import json
import logging
import re
from datetime import datetime, timezone


# ── Credential redaction ───────────────────────────────────────────────────
# The Angel One SmartApi SDK logs the FULL HTTP request (headers + body) on
# every request error via logzero's logger ("logzero_default"), leaking the
# API key (X-PrivateKey), the session JWT (Authorization: Bearer ...) and the
# login body (clientcode/password/totp) into our log files in plaintext.
# logzero attaches its own stderr handler with propagate=False, so Django's
# LOGGING dictConfig cannot reach it — the fix is a filter installed directly
# on the logzero logger (see install_credential_redaction below + the
# apps.common AppConfig.ready hook). ~30 of our own modules also log via
# logzero, so we REDACT in place and RETURN TRUE rather than dropping records.
_SECRET_DETECT = re.compile(
    r"X-PrivateKey|Bearer\s+[A-Za-z0-9._\-]{8,}"
    r"|'(?:password|totp|clientcode)'\s*:|jwtToken|refreshToken|feedToken",
    re.IGNORECASE,
)
_REDACTORS = [
    (re.compile(r"(X-PrivateKey'?\s*:\s*'?)[^',}\s]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE), r"\1[REDACTED]"),
    (
        re.compile(
            r"('(?:password|totp|clientcode|jwtToken|refreshToken|feedToken)'\s*:\s*')[^']*",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
]


class CredentialRedactionFilter(logging.Filter):
    """Redact (do not drop) log records that expose broker credentials.

    Attached to logzero's 'logzero_default' logger, which the Angel One
    SmartApi SDK uses to dump full request headers+body on errors
    (smartConnect.py:221/246). We REWRITE the message to a safe form and
    RETURN TRUE so the error signal survives on every handler (stderr ->
    logs/celery.log, and the logzero.logfile() FileHandler ->
    logs/<date>/app.log). Do NOT change this to `return False`: that would
    silently drop ALL broker-error log lines and blind incident triage.
    Our own logzero lines (which never contain these markers) pass through
    verbatim via the fast-path return.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not _SECRET_DETECT.search(msg):
            return True
        for pat, repl in _REDACTORS:
            msg = pat.sub(repl, msg)
        record.msg = msg
        record.args = ()
        return True


def install_credential_redaction() -> None:
    """Attach CredentialRedactionFilter to logzero's logger. Idempotent.

    Called from apps.common.AppConfig.ready() (and safe to call again) because
    logzero installs its handlers at import time / on websocket construction,
    out of reach of dictConfig.
    """
    try:
        from logzero import logger as _lz  # type: ignore
    except Exception:
        return
    if any(isinstance(f, CredentialRedactionFilter) for f in _lz.filters):
        return
    _lz.addFilter(CredentialRedactionFilter())


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Structlog binds extra fields onto record.__dict__
        for k, v in record.__dict__.items():
            if k in payload or k.startswith("_"):
                continue
            if k in {"args", "msg", "levelname", "name", "exc_info", "exc_text",
                     "stack_info", "pathname", "filename", "module", "lineno",
                     "funcName", "created", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process"}:
                continue
            payload[k] = v
        return json.dumps(payload, default=str)
