import logging
import os
import sys

# Suppress ResourceWarnings at Python level before any imports
import warnings
warnings.simplefilter("ignore", ResourceWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

from datetime import datetime, timezone
import pandas as pd

# Ensure src is on path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)
repo_root = os.path.dirname(src_path)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.services.prediction_service import run_prediction_pipeline
from energy_forecaster.features.data_prep import PV_COL, TARGET_COL
from energy_forecaster.utils.logging_config import configure_logging
from app.pv_manager.settings import load_settings

MODELS_DIR = os.getenv("MODELS_DIR", "trained_models")
HORIZON_HOURS = 24  # Always predict 24 hours ahead
INTERVAL_MINUTES = 15  # Predict at 15-minute intervals (96 predictions per 24h)
USE_MOCK_WEATHER = os.getenv("USE_MOCK_WEATHER", "0").lower() in ("true", "1", "yes")
DEBUG_DUMP_DIR = os.getenv("DEBUG_DIR")

# Havlíčkův Brod coordinates as fallback
FALLBACK_LAT = 49.6069
FALLBACK_LON = 15.5808
FALLBACK_TZ = "Europe/Prague"

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
        logger.warning("[PRED] Failed to load settings.json; falling back to defaults: %s", exc)
        return None, None, None
    inverter = settings.inverter
    return list(inverter.stat_ids()), dict(inverter.rename_map()), dict(inverter.scales())


def main():
    # Token/URL discovery (same approach as training)
    if not os.getenv("SUPERVISOR_TOKEN"):
        token = os.getenv("HASS_TOKEN")
        url = os.getenv("HASS_WS_URL") or "ws://192.168.1.11:8123/api/websocket"
        if not token:
            logger.error("HASS_TOKEN environment variable is not set")
            return
    else:
        token = None
        url = None

    ha = HomeAssistant(token=token, ws_url=url)
    stat_ids, rename_map, scales = _load_inverter_config()
    active_entities = stat_ids or DEFAULT_ENTITIES

    logger.info("Current environment:")
    logger.info("HASS_WS_URL: %s", ha.ws_url)
    logger.info("Using token: %s", "supervisor" if os.getenv("SUPERVISOR_TOKEN") else "custom")
    logger.info("Using HA sensors: %s", active_entities)

    logger.info("Fetching HA configuration from Home Assistant…")
    lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)
    logger.info("Resolved coordinates: %s, %s | timezone=%s", lat, lon, tz)

    logger.info(
        "[PRED] Running prediction pipeline for next %sh at %s-minute intervals (weather=%s)",
        HORIZON_HOURS,
        INTERVAL_MINUTES,
        "MOCK" if USE_MOCK_WEATHER else "REAL",
    )
    if DEBUG_DUMP_DIR:
        logger.info("[PRED] Debug dumps enabled under %s", DEBUG_DUMP_DIR)
    try:
        results = run_prediction_pipeline(
            ha=ha,
            lat=lat,
            lon=lon,
            tz=tz,
            models_dir=MODELS_DIR,
            horizon_hours=HORIZON_HOURS,
            interval_minutes=INTERVAL_MINUTES,
            use_mock_weather=USE_MOCK_WEATHER,
            return_features=True,
            entities=active_entities,
            rename_map=rename_map,
            scales=scales,
            debug_dir=DEBUG_DUMP_DIR,
        )

        pv_pred = results["pv_pred"]
        house_pred = results["house_pred"]
        ha_recent = results["ha_recent"]
        features = results.get("features", {})

        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            "[PRED] ✓ Predictions complete: PV=%s rows (%.2f–%.2f kW) | House=%s rows (%.2f–%.2f kW)",
            len(pv_pred),
            pv_pred[PV_COL].min(),
            pv_pred[PV_COL].max(),
            len(house_pred),
            house_pred[TARGET_COL].min(),
            house_pred[TARGET_COL].max(),
        )

        # Persist raw data for diagnostics
        pv_pred.to_csv(os.path.join(output_dir, "pv_pred.csv"))
        house_pred.to_csv(os.path.join(output_dir, "house_pred.csv"))
        ha_recent.to_csv(os.path.join(output_dir, "ha_recent.csv"))
        pv_feats = features.get("pv_X")
        house_feats = features.get("house_X")
        weather_feats = features.get("weather")
        if isinstance(pv_feats, pd.DataFrame):
            pv_feats.to_csv(os.path.join(output_dir, "pv_features.csv"))
        if isinstance(house_feats, pd.DataFrame):
            house_feats.to_csv(os.path.join(output_dir, "house_features.csv"))
        if isinstance(weather_feats, pd.DataFrame):
            weather_feats.to_csv(os.path.join(output_dir, "weather_forecast.csv"))
        logger.info("[PRED] Saved diagnostic CSVs to output/")

    finally:
        import asyncio
        import time
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ha.close())
            loop.run_until_complete(asyncio.sleep(0.1))  # Allow cleanup
            loop.close()
            time.sleep(0.05)  # Final cleanup
        except Exception:
            pass


if __name__ == "__main__":
    main()
