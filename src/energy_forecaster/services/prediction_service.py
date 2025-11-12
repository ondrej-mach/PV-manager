from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
from joblib import load

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.open_meteo import fetch_openmeteo_forecast
from energy_forecaster.features.data_prep import (
    assemble_forecast_features,
    TARGET_COL,
    PV_COL,
)


def load_models(models_dir: str) -> Tuple[object, object]:
    house_path = os.path.join(models_dir, "house_consumption.joblib")
    pv_path = os.path.join(models_dir, "pv_power.joblib")
    house_model = load(house_path)
    pv_model = load(pv_path)
    return house_model, pv_model


def run_prediction_pipeline(
    ha: HomeAssistant,
    lat: float,
    lon: float,
    tz: str,
    models_dir: str,
    horizon_hours: int = 24,
    interval_minutes: int = 15,
    use_mock_weather: bool = False,
    entities: Optional[List[Tuple[str, str]]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    scales: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run end-to-end prediction using recent HA data and Open-Meteo forecast.

    Args:
        interval_minutes: prediction interval (15 for 15-minute, 60 for hourly)
        use_mock_weather: use mock weather data instead of API (for testing)

    Returns dict with keys:
      - pv_pred: DataFrame with column PV_COL of predictions
      - house_pred: DataFrame with column TARGET_COL of predictions
      - ha_recent: normalized recent HA data (KW columns) for plotting
    """
    # Fetch recent HA data for lags at 5-min resolution if available
    ENTITIES = entities or [("sensor.house_consumption", "mean"), ("sensor.pv_power", "mean")]

    # For lags we need at least 72 hours of history
    ha_recent_raw = ha.fetch_last_hours_sync(ENTITIES, hours=max(72, horizon_hours + 72), period="5minute")

    # Weather forecast (hourly) for the horizon
    wx_forecast = fetch_openmeteo_forecast(lat=lat, lon=lon, tz=tz, hours=horizon_hours, use_mock=use_mock_weather)

    # Upsample weather to match requested interval
    if interval_minutes < 60:
        # Create 15-min timeline
        periods = (horizon_hours * 60) // interval_minutes
        future_index = pd.date_range(
            start=wx_forecast.index[0],
            periods=periods,
            freq=f"{interval_minutes}min"
        )
        # Interpolate weather linearly to 15-min
        wx_upsampled = wx_forecast.reindex(future_index).interpolate(method="linear")
    else:
        wx_upsampled = wx_forecast
        future_index = wx_forecast.index

    # Build features
    feats = assemble_forecast_features(
        ha_recent_raw,
        wx_upsampled,
        tz=tz,
        rename_map=rename_map,
        scales=scales,
    )
    pv_X = feats["pv_X"]
    house_X_template = feats["house_X"]
    ha_norm = feats["ha_norm"]

    # Load models
    house_model, pv_model = load_models(models_dir)

    # Predict PV first
    pv_pred_values = pv_model.predict(pv_X.values)
    pv_pred = pd.DataFrame({PV_COL: pv_pred_values}, index=pv_X.index)
    pv_pred[PV_COL] = pv_pred[PV_COL].clip(lower=0.0)

    # Fill PV into house features
    house_X = house_X_template.copy()
    house_X[PV_COL] = pv_pred[PV_COL]

    # Restrict to rows where lags are available
    house_X_clean = house_X.dropna()
    house_pred_idx = house_X_clean.index
    if len(house_pred_idx) == 0:
        return {
            "pv_pred": pv_pred,
            "house_pred": pd.DataFrame(columns=[TARGET_COL], index=pv_pred.index),
            "ha_recent": ha_norm,
        }

    house_pred_values = house_model.predict(house_X_clean.values)
    house_pred = pd.DataFrame({TARGET_COL: house_pred_values}, index=house_pred_idx)
    house_pred[TARGET_COL] = house_pred[TARGET_COL].clip(lower=0.0)

    return {
        "pv_pred": pv_pred,
        "house_pred": house_pred,
        "ha_recent": ha_norm,
    }
