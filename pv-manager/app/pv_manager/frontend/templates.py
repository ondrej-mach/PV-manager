from __future__ import annotations

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "static" / "index.html"


def render_index() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")
