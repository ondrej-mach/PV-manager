from __future__ import annotations

import os

import uvicorn

from pv_manager import create_application



import logging

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}

# Configure logging with timestamps
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)




import json

def _load_ha_options():
    """Load configuration from Home Assistant options.json and set env vars."""
    options_path = "/data/options.json"
    if not os.path.exists(options_path):
        return

    try:
        with open(options_path, encoding="utf-8") as f:
            options = json.load(f)
        
        # Map specific options to environment variables
        if token := options.get("entsoe_token"):
            os.environ["ENTSOE_TOKEN"] = str(token).strip()
            logging.info("Loaded ENTSOE_TOKEN from options.json")
            
    except Exception as e:
        logging.warning(f"Failed to load options.json: {e}")


def main() -> None:
    _load_ha_options()
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
        
        # Define log config to enforce timestamps on Uvicorn
        log_config = uvicorn.config.LOGGING_CONFIG.copy()
        log_config["formatters"]["default"]["fmt"] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        log_config["formatters"]["access"]["fmt"] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

        uvisettings = {
            "host": host,
            "port": port,
            "log_level": log_level,
            "proxy_headers": True,
            "forwarded_allow_ips": "*",
            "log_config": log_config,
        }
        
        uvicorn.run(app, **uvisettings)


if __name__ == "__main__":
    main()
