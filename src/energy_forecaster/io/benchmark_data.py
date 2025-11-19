"""Download and cache historical data for benchmarking."""
from __future__ import annotations

import logging
import os
from typing import Tuple
import pandas as pd

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.open_meteo import fetch_openmeteo_archive
from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz


BENCHMARK_DIR = os.getenv("BENCHMARK_DIR", "benchmark_data")
# Use a full year to capture seasonal variations (summer = high PV, winter = low PV)
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-12-31"  # Full year of data


logger = logging.getLogger(__name__)


def download_benchmark_data(
    ha: HomeAssistant,
    lat: float,
    lon: float,
    tz: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force_refresh: bool = False,
) -> None:
    """Download historical HA stats, weather, and prices; save to CSVs.

    Args:
        ha: HomeAssistant instance
        lat, lon, tz: location parameters
        start, end: date strings YYYY-MM-DD
        force_refresh: if True, re-download even if CSVs exist
    """
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    ha_path = os.path.join(BENCHMARK_DIR, "ha_stats.csv")
    wx_path = os.path.join(BENCHMARK_DIR, "weather.csv")
    price_path = os.path.join(BENCHMARK_DIR, "prices.csv")

    # Download HA stats
    if force_refresh or not os.path.exists(ha_path):
        logger.info("[BENCH] Downloading HA stats from %s to %s…", start, end)
        ENTITIES = [("sensor.house_consumption", "mean"), ("sensor.pv_power", "mean")]
        start_ts = pd.Timestamp(start, tz=tz)
        end_ts = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
        
        # Fetch async
        import asyncio
        ha_data = asyncio.run(
            ha.fetch_statistics_window_async(ENTITIES, start_ts, end_ts, period="hour")
        )
        ha_data.to_csv(ha_path)
        logger.info("[BENCH] Saved HA stats to %s", ha_path)
    else:
        logger.info("[BENCH] HA stats already exist at %s", ha_path)

    # Download weather
    if force_refresh or not os.path.exists(wx_path):
        logger.info("[BENCH] Downloading weather from %s to %s…", start, end)
        start_ts = pd.Timestamp(start, tz=tz)
        end_ts = pd.Timestamp(end, tz=tz)
        wx_data = fetch_openmeteo_archive(start_ts, end_ts, lat, lon, tz)
        wx_data.to_csv(wx_path)
        logger.info("[BENCH] Saved weather to %s", wx_path)
    else:
        logger.info("[BENCH] Weather already exists at %s", wx_path)

    # Download prices
    if force_refresh or not os.path.exists(price_path):
        logger.info("[BENCH] Downloading prices from %s to %s…", start, end)
        country = guess_country_code_from_tz(tz)
        start_ts = pd.Timestamp(start, tz=tz)
        end_ts = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
        try:
            prices = fetch_day_ahead_prices_country(country, start_ts, end_ts, tz)
            prices.to_csv(price_path)
            logger.info("[BENCH] Saved prices to %s", price_path)
        except Exception as e:
            logger.error("[BENCH] Price download failed: %s", e)
            logger.info("[BENCH] Creating mock prices…")
            # Create hourly mock prices
            idx = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
            mock_prices = pd.DataFrame({"price_eur_per_kwh": 0.10}, index=idx)
            mock_prices.to_csv(price_path)
            logger.info("[BENCH] Saved mock prices to %s", price_path)
    else:
        logger.info("[BENCH] Prices already exist at %s", price_path)


def load_benchmark_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached benchmark data from CSVs.

    Returns:
        (ha_stats, weather, prices) as DataFrames with proper datetime indexes
    """
    ha_path = os.path.join(BENCHMARK_DIR, "ha_stats.csv")
    wx_path = os.path.join(BENCHMARK_DIR, "weather.csv")
    price_path = os.path.join(BENCHMARK_DIR, "prices.csv")

    if not all(os.path.exists(p) for p in [ha_path, wx_path, price_path]):
        raise FileNotFoundError(
            f"Benchmark data not found in {BENCHMARK_DIR}. "
            "Run download_benchmark_data() first or delete the directory to re-download."
        )

    ha_stats = pd.read_csv(ha_path, index_col=0, parse_dates=True)
    weather = pd.read_csv(wx_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # Ensure UTC timezone
    if ha_stats.index.tz is None:
        ha_stats.index = ha_stats.index.tz_localize("UTC")
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize("UTC")
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")

    return ha_stats, weather, prices
