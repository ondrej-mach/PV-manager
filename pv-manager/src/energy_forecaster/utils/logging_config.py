import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_level(level_name: Optional[str], fallback: int) -> int:
    if not level_name:
        return fallback
    name = level_name.strip().upper()
    level = logging.getLevelName(name)
    if isinstance(level, int):
        return level
    return fallback


def configure_logging(default_level: str = "INFO", env_var: str = "LOG_LEVEL") -> None:
    """Configure global logging once per process.

    Args:
        default_level: Textual logging level (INFO, DEBUG, etc.).
        env_var: Environment variable to override the level.
    """
    root = logging.getLogger()
    if root.handlers:
        # Logging already configured; only adjust the level.
        root.setLevel(_resolve_level(os.getenv(env_var), root.level))
        return

    level = _resolve_level(os.getenv(env_var) or default_level, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("charset_normalizer").setLevel(logging.WARNING)
