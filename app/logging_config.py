import json
import logging
import os
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for structured logs.

    Automatically includes request_id from the middleware context if available.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "message": record.getMessage(),
        }

        # Include request_id from context if available
        try:
            from app.middleware.request_id import get_request_id  # noqa: PLC0415 - lazy import

            request_id = get_request_id()
            if request_id:
                base["request_id"] = request_id
        except ImportError:
            pass  # Middleware not loaded yet during early startup

        # include any custom extras if present
        for key, value in record.__dict__.items():
            if key in base or key.startswith("_"):
                continue
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info", "lineno", "pathname", "filename"):
                continue
            try:
                json.dumps(value)  # test serializability
                base[key] = value
            except TypeError:
                base[key] = str(value)
        # Attach optional extras
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            base["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    # Force our JSON handler even if uvicorn already configured handlers.
    logging.basicConfig(level=log_level, handlers=[handler], force=True)
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
