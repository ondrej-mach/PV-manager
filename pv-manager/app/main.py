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
    # DEBUG: Enable faulthandler to dump stack on Ctrl+\ (SIGQUIT)
    import faulthandler
    import signal
    import sys
    faulthandler.enable()
    if hasattr(signal, "SIGQUIT"):
        faulthandler.register(signal.SIGQUIT)
    
    print(f"DEBUG: Pre-Import SIGINT: {signal.getsignal(signal.SIGINT)}")
    
    # Force import cvxpy to see if it changes signals
    try:
        import cvxpy
        print("DEBUG: cvxpy imported.")
    except ImportError:
        print("DEBUG: cvxpy not found.")
        
    print(f"DEBUG: Post-Import SIGINT: {signal.getsignal(signal.SIGINT)}")

    _load_ha_options()
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8099"))
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    reload_enabled = _env_flag("UVICORN_RELOAD")

    if reload_enabled:
        print("DEBUG: Running in reload mode")
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
        uvicorn.run(app, host=host, port=port, log_level=log_level, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
