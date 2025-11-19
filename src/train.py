import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
import pandas as pd

# Add the src directory (and repo root) to Python path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)
repo_root = os.path.dirname(src_path)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from energy_forecaster.services.training_service import run_training_pipeline
from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.utils.logging_config import configure_logging
from app.pv_manager.settings import load_settings

# Configuration
LOOKBACK_DAYS = 730  # 2 years of historical data
MODELS_DIR = "trained_models"


FALLBACK_LAT = 49.6
FALLBACK_LON = 15.6
FALLBACK_TZ = "Europe/Prague"

# Home Assistant entities to monitor (fallback when settings.json missing)
DEFAULT_ENTITIES = [
    ("sensor.house_consumption", "mean"),
    ("sensor.pv_power", "mean"),
]


configure_logging()
logger = logging.getLogger(__name__)


def _load_inverter_config():
    try:
        settings = load_settings()
    except Exception as exc:
        logger.warning("[TRAIN] Failed to load settings.json; falling back to defaults: %s", exc)
        return None, None, None
    inverter = settings.inverter
    return list(inverter.stat_ids()), dict(inverter.rename_map()), dict(inverter.scales())

def main():
    # Set up development environment
    if not os.getenv("SUPERVISOR_TOKEN"):  # If not running as an addon
        # Use local development settings
        token = os.getenv("HASS_TOKEN")
        url = os.getenv("HASS_WS_URL") or "ws://homeassistant.lan:8123/api/websocket"
        if not token:
            logger.error("HASS_TOKEN environment variable is not set")
            return
    else:
        token = None  # Will use SUPERVISOR_TOKEN
        url = None   # Will use default supervisor URL
    
    # Initialize Home Assistant client
    ha = HomeAssistant(token=token, ws_url=url)
    stat_ids, rename_map, scales = _load_inverter_config()
    active_entities = stat_ids or DEFAULT_ENTITIES
    
    logger.info("Current environment:")
    logger.info("HASS_WS_URL: %s", ha.ws_url)
    logger.info("Using token: %s", "supervisor" if os.getenv("SUPERVISOR_TOKEN") else "custom")
    
    # Get location from Home Assistant or use fallback
    logger.info("Fetching HA configuration from Home Assistant…")
    lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)
    logger.info("Resolved coordinates: %s, %s | timezone=%s", lat, lon, tz)

    logger.info("[TRAIN] Using HA sensors: %s", active_entities)
    logger.info(
        "[TRAIN] Starting training pipeline with target lookback %s days (this may take a while)",
        LOOKBACK_DAYS,
    )
    logger.info("[TRAIN] The actual HA stats window will be reported by the training pipeline")

    try:
        # Run the training pipeline reusing the existing HomeAssistant instance
        results = run_training_pipeline(
            stat_ids=active_entities,
            lookback_days=LOOKBACK_DAYS,
            lat=lat,
            lon=lon,
            tz=tz,
            save_dir=MODELS_DIR,
            ha=ha,
            rename_map=rename_map,
            scales=scales,
        )

        logger.info("[TRAIN] Training completed successfully!")
        logger.info(
            "[TRAIN] Training window: %s to %s | samples house=%s pv=%s",
            results["window"][0],
            results["window"][1],
            results.get("house_samples"),
            results.get("pv_samples"),
        )
        logger.info("[TRAIN] Models saved to: %s", MODELS_DIR)

        logger.info("[TRAIN] House model features:")
        for feat in results['house_features']:
            logger.info("  - %s", feat)

        logger.info("[TRAIN] PV model features:")
        for feat in results['pv_features']:
            logger.info("  - %s", feat)

    except Exception:
        logger.exception("Error during training")
    finally:
        # Ensure we always close the connection
        try:
            asyncio.run(ha.close())
        except Exception:
            pass

if __name__ == "__main__":
    main()
