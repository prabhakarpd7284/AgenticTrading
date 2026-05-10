import json
import logging
from datetime import datetime, timezone


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
