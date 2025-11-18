
from typing import List, Tuple, Optional, Dict, Any
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
from energy_forecaster.features.data_prep import assemble_training_frames, TARGET_COL, PV_COL


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
) -> Dict[str, Any]:
    print("[TRAIN] Fetching Home Assistant statistics...")
    if ha is None:
        ha = HomeAssistant()  # Use default supervisor connection if none provided
    ha_raw = ha.fetch_statistics_sync(stat_ids, days=lookback_days, chunk_days=30)
    if ha_raw.empty:
        raise RuntimeError("No HA statistics returned.")

    # Report raw HA stats window and counts so user knows how much data we actually have
    start, end = ha_raw.index.min(), ha_raw.index.max()
    print(f"[TRAIN] HA statistics received: rows={len(ha_raw)}, cols={len(ha_raw.columns)}")
    print(f"[TRAIN] HA stats window: {start} to {end}")

    print("[TRAIN] Fetching weather archive from Open-Meteo...")
    wx = fetch_openmeteo_archive(start, end, lat=lat, lon=lon, tz=tz)
    print(f"[TRAIN] Weather data: rows={len(wx)}, cols={len(wx.columns)}; window={wx.index.min()} to {wx.index.max() if len(wx)>0 else 'N/A'}")

    print("[TRAIN] Assembling training frames and computing features...")
    frames = assemble_training_frames(ha_raw, wx, tz=tz, rename_map=rename_map, scales=scales)
    df_house = frames["house"]
    df_pv    = frames["pv"]
    print(f"[TRAIN] After feature engineering: house={df_house.shape}, pv={df_pv.shape}")

    house_model = house_model or default_house_model()
    pv_model    = pv_model or default_pv_model()

    # Fit house model
    house_feature_cols = list(df_house.drop(columns=[TARGET_COL]).columns)
    Xh = df_house[house_feature_cols].values
    yh = df_house[TARGET_COL].values
    print(f"[TRAIN] Training house consumption model on {len(yh)} samples and {Xh.shape[1]} features...")
    t0 = time.perf_counter()
    house_model.fit(Xh, yh)
    t1 = time.perf_counter()
    print(f"[TRAIN] House model trained in {t1-t0:.1f}s")
    _set_feature_metadata(house_model, house_feature_cols)

    # Fit pv model
    pv_feature_cols = list(df_pv.drop(columns=[PV_COL]).columns)
    Xp = df_pv[pv_feature_cols].values
    yp = df_pv[PV_COL].values
    print(f"[TRAIN] Training PV model on {len(yp)} samples and {Xp.shape[1]} features...")
    t0 = time.perf_counter()
    pv_model.fit(Xp, yp)
    t1 = time.perf_counter()
    print(f"[TRAIN] PV model trained in {t1-t0:.1f}s")
    _set_feature_metadata(pv_model, pv_feature_cols)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        house_path = os.path.join(save_dir, "house_consumption.joblib")
        pv_path = os.path.join(save_dir, "pv_power.joblib")
        dump(house_model, house_path)
        dump(pv_model, pv_path)
        print(f"[TRAIN] Models saved: {house_path}, {pv_path}")

    return {
        "house_model": house_model,
        "pv_model": pv_model,
    "house_features": house_feature_cols,
    "pv_features": pv_feature_cols,
        "window": (start, end),
        "house_samples": len(yh),
        "pv_samples": len(yp),
    }
