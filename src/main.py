import os
import sys
import asyncio
from datetime import datetime, timezone
import pandas as pd

# Add the src directory to Python path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from energy_forecaster.services.training_service import run_training_pipeline
from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.open_meteo import fetch_openmeteo_archive

# Configuration
LOOKBACK_DAYS = 730  # 2 years of historical data
MODELS_DIR = "trained_models"

# Havlíčkův Brod coordinates as fallback
FALLBACK_LAT = 49.6069
FALLBACK_LON = 15.5808
FALLBACK_TZ = "Europe/Prague"

# Home Assistant entities to monitor
ENTITIES = [
    ("sensor.house_consumption", "mean"),  # House power consumption
    ("sensor.pv_power", "mean"),          # Solar PV production
]

def main():
    # Set up development environment
    if not os.getenv("SUPERVISOR_TOKEN"):  # If not running as an addon
        # Use local development settings
        token = os.getenv("HASS_TOKEN")
        url = "ws://homeassistant.lan:8123/api/websocket"
        if not token:
            print("Please set HASS_TOKEN environment variable")
            return
    else:
        token = None  # Will use SUPERVISOR_TOKEN
        url = None   # Will use default supervisor URL
    
    # Initialize Home Assistant client
    ha = HomeAssistant(token=token, url=url)
    
    print("Current environment:")
    print(f"HASS_WS_URL: {ha.url}")
    print(f"Using token: {'supervisor' if os.getenv('SUPERVISOR_TOKEN') else 'custom'}")
    
    # Get location from Home Assistant or use fallback
    print("\nFetching HA configuration...")
    lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)
    print(f"Using coordinates: {lat}, {lon} and timezone: {tz}")

    print(f"\n[MAIN] Starting training pipeline with target lookback {LOOKBACK_DAYS} days (this may take a while).")
    print("[MAIN] The actual available HA stats window and counts will be shown by the training pipeline.")

    try:
        # Run the training pipeline reusing the existing HomeAssistant instance
        results = run_training_pipeline(
            stat_ids=ENTITIES,
            lookback_days=LOOKBACK_DAYS,
            lat=lat,
            lon=lon,
            tz=tz,
            save_dir=MODELS_DIR,
            ha=ha,
        )

        print("\n[MAIN] Training completed successfully!")
        print(f"[MAIN] Models saved to: {MODELS_DIR}")
        print(f"[MAIN] Training window: {results['window'][0]} to {results['window'][1]}")
        print(f"[MAIN] House training samples: {results.get('house_samples')} | PV training samples: {results.get('pv_samples')}")

        print("\n[MAIN] House model features:")
        for feat in results['house_features']:
            print(f"  - {feat}")

        print("\n[MAIN] PV model features:")
        for feat in results['pv_features']:
            print(f"  - {feat}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        # Ensure we always close the connection
        try:
            asyncio.run(ha.close())
        except Exception:
            pass

if __name__ == "__main__":
    main()