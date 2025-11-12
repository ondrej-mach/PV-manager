from __future__ import annotations

import os

import uvicorn

from pv_manager import create_application


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}


def main() -> None:
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8099"))
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    reload_enabled = _env_flag("UVICORN_RELOAD")

    if reload_enabled:
        uvicorn.run(
            "pv_manager:create_application",
            host=host,
            port=port,
            log_level=log_level,
            reload=True,
            factory=True,
        )
    else:
        app = create_application()
        uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
