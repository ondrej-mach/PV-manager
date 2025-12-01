"""PV Manager Home Assistant add-on package."""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Configure logging to show application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set specific loggers
logging.getLogger("pv_manager").setLevel(logging.INFO)
logging.getLogger("energy_forecaster").setLevel(logging.INFO)

# Ensure the legacy library under src/ is importable when the add-on runs in Docker.
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
os.environ.setdefault("PYTHONPATH", str(_SRC_PATH))
if str(_SRC_PATH) not in sys.path:
    sys.path.append(str(_SRC_PATH))

__all__ = ["create_application"]

from .addon import create_application  # noqa: E402  (import after path fix)
