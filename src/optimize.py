import logging
import os
import sys

# Suppress ResourceWarnings at Python level before any imports
import warnings
warnings.simplefilter("ignore", ResourceWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

import pandas as pd

src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)
repo_root = os.path.dirname(src_path)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.services.prediction_service import run_prediction_pipeline
from energy_forecaster.features.data_prep import PV_COL, TARGET_COL
from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz
from energy_forecaster.utils.logging_config import configure_logging
from app.pv_manager.settings import load_settings


MODELS_DIR = os.getenv("MODELS_DIR", "trained_models")
HORIZON_HOURS = 24
INTERVAL_MINUTES = 15
USE_MOCK_WEATHER = os.getenv("USE_MOCK_WEATHER", "0").lower() in ("true", "1", "yes")
DEBUG_DUMP_DIR = os.getenv("DEBUG_DIR")

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
        logger.warning("[OPT] Failed to load settings.json; falling back to defaults: %s", exc)
        return None, None, None
    inverter = settings.inverter
    return list(inverter.stat_ids()), dict(inverter.rename_map()), dict(inverter.scales())


def main():
    # HA connection
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

    try:
        lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)
        logger.info("[OPT] Using HA sensors: %s", active_entities)

        # Predictions
        results = run_prediction_pipeline(
            ha=ha,
            lat=lat,
            lon=lon,
            tz=tz,
            models_dir=MODELS_DIR,
            horizon_hours=HORIZON_HOURS,
            interval_minutes=INTERVAL_MINUTES,
            use_mock_weather=USE_MOCK_WEATHER,
            entities=active_entities,
            rename_map=rename_map,
            scales=scales,
            debug_dir=DEBUG_DUMP_DIR,
        )
        pv_pred = results["pv_pred"][PV_COL]
        load_pred = results["house_pred"][TARGET_COL]

        # Align indexes (house_pred may be shorter if lags missing)
        idx = pv_pred.index.intersection(load_pred.index)
        pv_pred = pv_pred.reindex(idx)
        load_pred = load_pred.reindex(idx)

        # Fetch DA prices covering the prediction window
        # Use HA tz and derive country code
        tz_use = tz or "Europe/Prague"
        country = os.getenv("ENTSOE_COUNTRY_CODE") or guess_country_code_from_tz(tz_use)
        start_local = (idx[0].tz_convert(tz_use) if idx.tz is not None else idx[0].tz_localize("UTC").tz_convert(tz_use)).floor("D")
        end_local = (idx[-1].tz_convert(tz_use) if idx.tz is not None else idx[-1].tz_localize("UTC").tz_convert(tz_use)).ceil("D") + pd.Timedelta(days=1)
        try:
            prices_h = fetch_day_ahead_prices_country(country_code=country, start=start_local, end=end_local, tz=tz_use)
        except ModuleNotFoundError:
            logger.warning("[OPT] entsoe-py not available. Skipping optimization; saving predictions only.")
            os.makedirs("output", exist_ok=True)
            pred_idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            pred_df = pd.DataFrame(index=pred_idx_utc, data={
                "pv_kw": pv_pred.values,
                "load_kw": load_pred.values,
            })
            pred_df.to_csv("output/predictions_only.csv")
            return
        # Upsample to 15-min by forward-filling within hour
        prices_15 = prices_h.reindex(pd.date_range(start=prices_h.index.min(), end=prices_h.index.max(), freq=f"{INTERVAL_MINUTES}min", tz="UTC")).ffill()
        # Reindex to prediction index (converted to UTC)
        pred_idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        price_on_pred = prices_15.reindex(pred_idx_utc).ffill().bfill()["price_eur_per_kwh"]

        # Build S
        dt_h = INTERVAL_MINUTES / 60.0
        S = pd.DataFrame(index=pred_idx_utc, data={
            "pv_kw": pv_pred.values,
            "load_kw": load_pred.values,
            "dt_h": dt_h,
            "price_eur_per_kwh": price_on_pred.values,
        })

        # Import solver lazily to avoid hard failure if cvxpy is missing at import time
        try:
            from energy_forecaster.services.optimal_control import solve_lp_optimal_control, BatteryConfig
            cfg = BatteryConfig()
            opt = solve_lp_optimal_control(S, cfg)
        except ModuleNotFoundError:
            logger.warning("[OPT] cvxpy not available (install failed?). Skipping optimization step.")
            # Save predictions only and exit gracefully
            os.makedirs("output", exist_ok=True)
            pred_df = pd.DataFrame(index=pred_idx_utc, data={
                "pv_kw": pv_pred.values,
                "load_kw": load_pred.values,
                "dt_h": dt_h,
                "price_eur_per_kwh": price_on_pred.values,
            })
            pred_df.to_csv("output/predictions_only.csv")
            return

        # Summaries
        net_trade = float(opt["cost_import_eur"].sum() - opt["rev_export_eur"].sum())
        throughput = (opt["import_kwh"] + opt["export_kwh"]).sum()
        logger.info(
            "[OPT] Summary: timesteps=%s import=%.2f kWh export=%.2f kWh net_trade=%.2f € | throughput=%.2f kWh",
            len(opt),
            float(opt["import_kwh"].sum()),
            float(opt["export_kwh"].sum()),
            net_trade,
            throughput,
        )

        # Save combined predictions + optimization
        combined = opt.copy()
        # The opt already includes pv_kw, load_kw. Ensure price column name is consistent
        if "sell_price_eur_per_kwh" in combined.columns:
            combined.rename(columns={"sell_price_eur_per_kwh": "price_eur_per_kwh"}, inplace=True)
        os.makedirs("output", exist_ok=True)
        combined.to_csv("output/prediction_optimization.csv")
        logger.info("[OPT] Saved: output/prediction_optimization.csv")

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
