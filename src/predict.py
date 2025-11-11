import os
import sys

# Suppress ResourceWarnings at Python level before any imports
import warnings
warnings.simplefilter("ignore", ResourceWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is on path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.services.prediction_service import run_prediction_pipeline
from energy_forecaster.features.data_prep import PV_COL, TARGET_COL

MODELS_DIR = os.getenv("MODELS_DIR", "trained_models")
HORIZON_HOURS = 24  # Always predict 24 hours ahead
INTERVAL_MINUTES = 15  # Predict at 15-minute intervals (96 predictions per 24h)
USE_MOCK_WEATHER = os.getenv("USE_MOCK_WEATHER", "0").lower() in ("true", "1", "yes")

# Havlíčkův Brod coordinates as fallback
FALLBACK_LAT = 49.6069
FALLBACK_LON = 15.5808
FALLBACK_TZ = "Europe/Prague"


def main():
    # Token/URL discovery (same approach as training)
    if not os.getenv("SUPERVISOR_TOKEN"):
        token = os.getenv("HASS_TOKEN")
        url = os.getenv("HASS_WS_URL") or "ws://192.168.1.11:8123/api/websocket"
        if not token:
            print("Please set HASS_TOKEN environment variable")
            return
    else:
        token = None
        url = None

    ha = HomeAssistant(token=token, url=url)

    print("Current environment:")
    print(f"HASS_WS_URL: {ha.url}")
    print(f"Using token: {'supervisor' if os.getenv('SUPERVISOR_TOKEN') else 'custom'}")

    print("\nFetching HA configuration...")
    lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)
    print(f"Using coordinates: {lat}, {lon} and timezone: {tz}")

    print(f"\n[PRED] Running prediction pipeline for next {HORIZON_HOURS} hours at {INTERVAL_MINUTES}-minute intervals...")
    print(f"[PRED] Using {'MOCK' if USE_MOCK_WEATHER else 'REAL'} weather forecast")
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
        )

        pv_pred = results["pv_pred"]
        house_pred = results["house_pred"]
        ha_recent = results["ha_recent"]

        print(f"\n[PRED] ✓ Predictions complete:")
        print(f"  - PV: {len(pv_pred)} timesteps, range {pv_pred[PV_COL].min():.2f}-{pv_pred[PV_COL].max():.2f} kW")
        print(f"  - House: {len(house_pred)} timesteps, range {house_pred[TARGET_COL].min():.2f}-{house_pred[TARGET_COL].max():.2f} kW")

        # Plot PV: last 48h actual vs next horizon predicted
        fig1, ax1 = plt.subplots(figsize=(12,4))
        recent_pv = ha_recent.get("pv_power_kw")
        if recent_pv is not None:
            recent_window = recent_pv.loc[pd.Timestamp.now(tz="UTC") - pd.Timedelta("48h"):]
            ax1.plot(recent_window.index, recent_window.values, label="PV actual", color="#1f77b4", linewidth=1.5)
        ax1.plot(pv_pred.index, pv_pred["pv_power_kw"].values, label="PV predicted", color="#ff7f0e", linestyle="--", linewidth=2)
        ax1.set_title(f"PV Power: Actual (last 48h) vs Predicted (next {HORIZON_HOURS}h at {INTERVAL_MINUTES}min intervals)")
        ax1.set_ylabel("kW")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig("output/predictions_pv.png")
        print("[PRED] Saved: output/predictions_pv.png")

        # Plot House load
        fig2, ax2 = plt.subplots(figsize=(12,4))
        recent_cons = ha_recent.get("power_consumption_kw")
        if recent_cons is not None:
            recent_window = recent_pv.loc[pd.Timestamp.now(tz="UTC") - pd.Timedelta("48h"):]
            ax2.plot(recent_window.index, recent_window.values, label="Load actual", color="#1f77b4", linewidth=1.5)
        ax2.plot(house_pred.index, house_pred["power_consumption_kw"].values, label="Load predicted", color="#2ca02c", linestyle=":", linewidth=2)
        ax2.set_title(f"House Load: Actual (last 48h) vs Predicted (next {HORIZON_HOURS}h at {INTERVAL_MINUTES}min intervals)")
        ax2.set_ylabel("kW")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig("output/predictions_house.png")
        print("[PRED] Saved: output/predictions_house.png")

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
