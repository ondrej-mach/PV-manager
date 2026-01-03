
from typing import List, Tuple, Optional, Dict, Any
import logging
import os
import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from joblib import dump

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.open_meteo import fetch_openmeteo_archive
from energy_forecaster.features.data_prep import (
    assemble_training_frames,
    TARGET_COL,
    PV_COL,
)


logger = logging.getLogger(__name__)

def _set_feature_metadata(model, columns):
    cols = list(columns)
    count = len(cols)
    targets = []
    if hasattr(model, "steps") and model.steps:
        # Try the final estimator inside a Pipeline (e.g. ElasticNet)
        targets.append(model.steps[-1][1])
    targets.append(model)

    for target in targets:
        try:
            setattr(target, "feature_names_in_", cols)
            setattr(target, "n_features_in_", count)
            return
        except AttributeError:
            continue

    # Fall back to private attributes so downstream code can still read the metadata
    setattr(model, "_feature_names_in_", cols)
    setattr(model, "_n_features_in_", count)

def default_house_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            alphas=100,  # integer keeps API forward-compatible without using deprecated n_alphas
            l1_ratio=[.1, .5, .7, .9, .95, .99],
            cv=5,
            random_state=42,
        )),
    ])

def default_pv_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )

def run_training_pipeline(
    stat_ids: List[Tuple[str, str]],
    lookback_days: int,
    lat: float,
    lon: float,
    tz: str = "Europe/Prague",
    save_dir: Optional[str] = None,
    house_model = None,
    pv_model = None,
    ha: Optional[HomeAssistant] = None,
    rename_map: Optional[Dict[str, str]] = None,
    scales: Optional[Dict[str, float]] = None,
    freq_minutes: int = 60,
    history_db: Any = None,
    deferrable_entity_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.info("[TRAIN] Fetching Home Assistant statistics…")
    if ha is None:
        ha = HomeAssistant()  # Use default supervisor connection if none provided
    freq_minutes = max(1, int(freq_minutes))
    stats_period = "hour"
    ha_raw = ha.fetch_statistics_sync(stat_ids, days=lookback_days, chunk_days=30, period=stats_period)
    if ha_raw.empty:
        raise RuntimeError("No HA statistics returned.")

    # Apply deferrable load subtraction if configured
    if history_db and deferrable_entity_ids and rename_map:
        target_entity = next((k for k, v in rename_map.items() if v == TARGET_COL), None)
        if target_entity and target_entity in ha_raw.columns:
            logger.info("[TRAIN] Subtracting deferrable loads (%s) from %s", deferrable_entity_ids, target_entity)
            start_dt, end_dt = ha_raw.index.min(), ha_raw.index.max()
            
            # Fetch deferrable history
            deferrable_df = history_db.get_history(start_dt, end_dt, deferrable_entity_ids)
            
            if not deferrable_df.empty:
                # Align to HA stats frequency (hour)
                # Use mean for power (since stats are mean)
                # Resample using the same index as ha_raw to ensure alignment
                aligned_deferrable = deferrable_df.resample("1h").mean().reindex(ha_raw.index).fillna(0.0)
                total_deferrable = aligned_deferrable.sum(axis=1)
                
                # Subtract and clip to 0
                original_sum = ha_raw[target_entity].sum()
                ha_raw[target_entity] = (ha_raw[target_entity] - total_deferrable).clip(lower=0.0)
                new_sum = ha_raw[target_entity].sum()
                
                logger.info("[TRAIN] Subtracted total %.2f kWh equivalent (Power logic). Reduced baseline by %.1f%%", 
                            (original_sum - new_sum), 
                            ((original_sum - new_sum) / original_sum * 100) if original_sum > 0 else 0)
            else:
                logger.info("[TRAIN] No deferrable history found in DB for this period.")

    # Report raw HA stats window and counts so user knows how much data we actually have
    start, end = ha_raw.index.min(), ha_raw.index.max()
    logger.info("[TRAIN] HA statistics received: rows=%s cols=%s", len(ha_raw), len(ha_raw.columns))
    logger.info("[TRAIN] HA stats window: %s -> %s", start, end)

    logger.info("[TRAIN] Fetching weather archive from Open-Meteo…")
    wx = fetch_openmeteo_archive(start, end, lat=lat, lon=lon, tz=tz, interval_minutes=freq_minutes)
    logger.info(
        "[TRAIN] Weather data: rows=%s cols=%s window=%s -> %s",
        len(wx),
        len(wx.columns),
        wx.index.min() if len(wx) else "N/A",
        wx.index.max() if len(wx) else "N/A",
    )

    logger.info("[TRAIN] Assembling training frames and computing features…")
    frames = assemble_training_frames(
        ha_raw,
        wx,
        tz=tz,
        rename_map=rename_map,
        scales=scales,
        freq_minutes=freq_minutes,
    )
    df_house = frames["house"]
    df_pv    = frames["pv"]
    logger.info("[TRAIN] After feature engineering: house=%s pv=%s", df_house.shape, df_pv.shape)

    house_model = house_model or default_house_model()
    pv_model    = pv_model or default_pv_model()

    # Fit house model
    house_feature_cols = list(df_house.drop(columns=[TARGET_COL]).columns)
    Xh = df_house[house_feature_cols]
    yh = df_house[TARGET_COL]
    logger.info("[TRAIN] Training house model: samples=%s features=%s", len(yh), Xh.shape[1])
    t0 = time.perf_counter()
    house_model.fit(Xh, yh)
    t1 = time.perf_counter()
    logger.info("[TRAIN] House model trained in %.1fs", t1 - t0)
    # _set_feature_metadata(house_model, house_feature_cols)

    # Fit pv model
    pv_feature_cols = list(df_pv.drop(columns=[PV_COL]).columns)
    Xp = df_pv[pv_feature_cols]
    yp = df_pv[PV_COL]
    logger.info("[TRAIN] Training PV model: samples=%s features=%s", len(yp), Xp.shape[1])
    t0 = time.perf_counter()
    pv_model.fit(Xp, yp)
    t1 = time.perf_counter()
    logger.info("[TRAIN] PV model trained in %.1fs", t1 - t0)
    # _set_feature_metadata(pv_model, pv_feature_cols)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        house_path = os.path.join(save_dir, "house_consumption.joblib")
        pv_path = os.path.join(save_dir, "pv_power.json")
        dump(house_model, house_path)
        pv_model.save_model(pv_path)
    logger.info("[TRAIN] Models saved: %s, %s", house_path, pv_path)

    return {
        "house_model": house_model,
        "pv_model": pv_model,
        "house_features": house_feature_cols,
        "pv_features": pv_feature_cols,
        "window": (start, end),
        "house_samples": len(yh),
        "pv_samples": len(yp),
    }
