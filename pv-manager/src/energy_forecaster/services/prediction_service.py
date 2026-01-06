from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import os
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from xgboost import XGBRegressor

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.open_meteo import fetch_openmeteo_forecast
from energy_forecaster.features.data_prep import (
    assemble_forecast_features,
    TARGET_COL,
    PV_COL,
)

_LOGGER = logging.getLogger(__name__)


def _align_features(model: Any, frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reorder DataFrame columns to match the model's training order.

    Raises if the model expects columns that are missing from the frame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {label} features but received {type(frame)}")

    expected = getattr(model, "feature_names_in_", None)
    if expected is None and hasattr(model, "steps") and model.steps:
        expected = getattr(model.steps[-1][1], "feature_names_in_", None)
    if expected is None:
        expected = getattr(model, "_feature_names_in_", None)
    if expected is None and hasattr(model, "steps") and model.steps:
        expected = getattr(model.steps[-1][1], "_feature_names_in_", None)
    if expected is None:
        return frame

    expected_list = [str(name) for name in expected]
    missing = [name for name in expected_list if name not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{label} model expects feature(s) {missing} which are missing from the generated frame."
        )

    return frame.loc[:, expected_list]


def _prepare_debug_path(debug_dir: Optional[str]) -> Optional[Path]:
    if not debug_dir:
        return None
    try:
        path = Path(debug_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as exc:  # pragma: no cover - debug helper
        _LOGGER.warning("Unable to create debug directory %s: %s", debug_dir, exc)
        return None


def _dump_debug_frame(debug_path: Optional[Path], name: str, frame: Optional[pd.DataFrame]) -> None:
    if debug_path is None or frame is None:
        return
    try:
        safe_name = name.lower().replace(" ", "_")
        target = debug_path / f"{safe_name}.csv"
        if isinstance(frame, pd.Series):
            frame.to_frame(name=name).to_csv(target)
        elif isinstance(frame, pd.DataFrame):
            frame.to_csv(target)
        else:
            pd.DataFrame(frame).to_csv(target)
    except Exception as exc:  # pragma: no cover - debug helper
        _LOGGER.debug("Failed to dump debug frame %s: %s", name, exc)


def load_models(models_dir: str) -> Tuple[object, object]:
    house_path = os.path.join(models_dir, "house_consumption.joblib")
    pv_path_json = os.path.join(models_dir, "pv_power.json")
    pv_path_joblib = os.path.join(models_dir, "pv_power.joblib")

    house_model = load(house_path)
    
    if os.path.exists(pv_path_json):
        pv_model = XGBRegressor()
        pv_model.load_model(pv_path_json)
    elif os.path.exists(pv_path_joblib):
        pv_model = load(pv_path_joblib)
    else:
        raise FileNotFoundError(f"PV model not found in {models_dir}")
        
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
    return_features: bool = False,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run end-to-end prediction using recent HA data and Open-Meteo forecast.

    Args:
    interval_minutes: prediction interval (15 for 15-minute, 60 for hourly)
    use_mock_weather: use mock weather data instead of API (for testing)
    debug_dir: optional directory path to dump intermediate DataFrames as CSVs

    Returns dict with keys:
      - pv_pred: DataFrame with column PV_COL of predictions
      - house_pred: DataFrame with column TARGET_COL of predictions
      - ha_recent: normalized recent HA data (KW columns) for plotting
    """
    debug_path = _prepare_debug_path(debug_dir)

    # Load models first - fail fast if missing
    house_model, pv_model = load_models(models_dir)

    # Fetch recent HA data for lags at high resolution
    ENTITIES = entities or [("sensor.house_consumption", "mean"), ("sensor.pv_power", "mean")]

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    max_consumption_lag_hours = 72  # house features rely on 24/48/72h consumption lags
    history_hours = max(max_consumption_lag_hours * 2, max_consumption_lag_hours + horizon_hours)
    ha_recent_raw = ha.fetch_history_sync(ENTITIES, hours=history_hours, period="15minute")
    timings["ha_fetch"] = time.perf_counter() - t0
    _dump_debug_frame(debug_path, "ha_recent_raw", ha_recent_raw)

    # Weather forecast (interpolated to requested interval) for the horizon
    t0 = time.perf_counter()
    wx_forecast = fetch_openmeteo_forecast(
        lat=lat,
        lon=lon,
        tz=tz,
        hours=horizon_hours,
        use_mock=use_mock_weather,
        interval_minutes=interval_minutes,
    )
    timings["wx_fetch"] = time.perf_counter() - t0
    _dump_debug_frame(debug_path, "weather_forecast_raw", wx_forecast)

    target_freq = f"{interval_minutes}min"
    periods = max(1, math.ceil((horizon_hours * 60) / max(1, interval_minutes)))
    future_index = pd.date_range(start=wx_forecast.index[0], periods=periods, freq=target_freq)
    wx_upsampled = wx_forecast.reindex(future_index).interpolate(method="time").ffill().bfill()

    _dump_debug_frame(debug_path, "weather_forecast_resampled", wx_upsampled)

    # Build features
    t0 = time.perf_counter()
    feats = assemble_forecast_features(
        ha_recent_raw,
        wx_upsampled,
        tz=tz,
        rename_map=rename_map,
        scales=scales,
        freq_minutes=interval_minutes,
    )
    timings["feature_build"] = time.perf_counter() - t0
    pv_X = feats["pv_X"]
    house_X_template = feats["house_X"]
    ha_norm = feats["ha_norm"]
    _dump_debug_frame(debug_path, "ha_normalized", ha_norm)
    _dump_debug_frame(debug_path, "pv_features", pv_X)
    _dump_debug_frame(debug_path, "house_features_template", house_X_template)

    # Models loaded at start

    # Predict PV first
    t0 = time.perf_counter()
    expected_pv_features = getattr(pv_model, "n_features_in_", pv_X.shape[1])
    if pv_X.shape[1] != expected_pv_features:
        raise RuntimeError(
            f"PV model expects {expected_pv_features} features but received {pv_X.shape[1]}. "
            "Retrain models to pick up the latest weather inputs."
        )

    pv_X_aligned = _align_features(pv_model, pv_X, "PV").astype(float)
    pv_pred_values = pv_model.predict(pv_X_aligned)
    timings["pv_predict"] = time.perf_counter() - t0
    pv_pred = pd.DataFrame({PV_COL: pv_pred_values}, index=pv_X.index)
    pv_pred[PV_COL] = pv_pred[PV_COL].clip(lower=0.0)
    _dump_debug_frame(debug_path, "pv_predictions", pv_pred)

    # Enforce zero generation when the irradiance forecast says there is no daylight.
    daylight_mask = None
    if "shortwave_radiation" in pv_X.columns:
        daylight_mask = pv_X["shortwave_radiation"] > 0.0
        if daylight_mask.sum() > 0:
            pv_pred.loc[~daylight_mask, PV_COL] = 0.0
        else:
            _LOGGER.debug("No positive irradiance values in forecast; skipping daylight clamp.")

    # Fill PV into house features
    house_X = house_X_template.copy()
    house_X[PV_COL] = pv_pred[PV_COL]
    _dump_debug_frame(debug_path, "house_features_with_pv", house_X)

    # Restrict to rows where lags are available
    house_X_clean = house_X.dropna()
    house_X_aligned = _align_features(house_model, house_X_clean, "House consumption").astype(float)
    _dump_debug_frame(debug_path, "house_features_clean", house_X_clean)
    expected_house_features = getattr(house_model, "n_features_in_", house_X_aligned.shape[1])
    if house_X_aligned.shape[1] != expected_house_features:
        raise RuntimeError(
            f"House model expects {expected_house_features} features but received {house_X_aligned.shape[1]}. "
            "Retrain models to stay in sync with feature engineering."
        )
    house_pred_idx = house_X_aligned.index
    if len(house_pred_idx) == 0:
        _LOGGER.info(
            "Prediction pipeline timings: HA %.2fs | weather %.2fs | features %.2fs | pv %.2fs | house (skipped); rows: ha=%d weather=%d",
            timings.get("ha_fetch", 0.0),
            timings.get("wx_fetch", 0.0),
            timings.get("feature_build", 0.0),
            timings.get("pv_predict", 0.0),
            len(ha_recent_raw),
            len(wx_forecast),
        )
        return {
            "pv_pred": pv_pred,
            "house_pred": pd.DataFrame(columns=[TARGET_COL], index=pv_pred.index),
            "ha_recent": ha_norm,
        }

    t0 = time.perf_counter()
    house_pred_values = house_model.predict(house_X_aligned)
    timings["house_predict"] = time.perf_counter() - t0
    house_pred = pd.DataFrame({TARGET_COL: house_pred_values}, index=house_pred_idx)
    house_pred[TARGET_COL] = house_pred[TARGET_COL].clip(lower=0.0)
    _dump_debug_frame(debug_path, "house_predictions", house_pred)

    _LOGGER.debug(
        "Prediction pipeline timings: HA %.2fs | weather %.2fs | features %.2fs | pv %.2fs | house %.2fs; rows: ha=%d weather=%d preds=%d",
        timings.get("ha_fetch", 0.0),
        timings.get("wx_fetch", 0.0),
        timings.get("feature_build", 0.0),
        timings.get("pv_predict", 0.0),
        timings.get("house_predict", 0.0),
        len(ha_recent_raw),
        len(wx_forecast),
        len(pv_pred),
    )

    result: Dict[str, pd.DataFrame] = {
        "pv_pred": pv_pred,
        "house_pred": house_pred,
        "ha_recent": ha_norm,
    }
    if return_features:
        result["features"] = {
            "pv_X": pv_X,
            "house_X": house_X,
            "ha_norm": ha_norm,
            "weather": wx_upsampled,
        }
    return result
