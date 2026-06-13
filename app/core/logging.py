"""
ASTRA – Structured logging setup
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a readable format."""
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers on reload
    if not root.handlers:
        root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
