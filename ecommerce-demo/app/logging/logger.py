import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ecommerce.log"


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "ecommerce-demo",
            "event": getattr(
                record,
                "event",
                "application_event",
            ),
            "method": getattr(record, "method", None),
            "path": getattr(record, "path", None),
            "status_code": getattr(
                record,
                "status_code",
                None,
            ),
            "response_time_ms": getattr(
                record,
                "response_time_ms",
                None,
            ),
            "message": record.getMessage(),
            "context": getattr(record, "context", None),
        }

        # Include traceback details when an exception is logged.
        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_data)


def get_logger(name: str = "ecommerce-demo") -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger