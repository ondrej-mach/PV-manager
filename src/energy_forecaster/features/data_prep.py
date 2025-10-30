
import pandas as pd
import numpy as np
from typing import Optional, Iterable, Tuple, List, Dict

WEATHER_FEATURES: List[str] = [
    "relative_humidity_2m",
    "dew_point_2m",
    "rain",
    "precipitation",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "weather_code",
    "pressure_msl",
    "wind_speed_10m",
    "wind_speed_100m",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "temperature_2m",
]

OPTIONAL_WEATHER: List[str] = ["vapour_pressure_deficit", "et0_fao_evapotranspiration"]

TIME_FEATURES: List[str] = [
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "holiday", "is_weekend",
    "hour_sin", "hour_cos",
]

TARGET_COL = "power_consumption_kw"
PV_COL     = "pv_power_kw"

def add_time_features(df: pd.DataFrame, tz: str = "Europe/Prague") -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    # handle both tz-aware and tz-naive indices
    if getattr(idx, 'tz', None) is None:
        idx_local = idx.tz_localize("UTC").tz_convert(tz)
    else:
        # already tz-aware
        idx_local = idx.tz_convert(tz)
    month = idx_local.month
    dow   = idx_local.dayofweek
    hour  = idx_local.hour
    out["month_sin"] = np.sin(2*np.pi*(month/12.0))
    out["month_cos"] = np.cos(2*np.pi*(month/12.0))
    out["dow_sin"]   = np.sin(2*np.pi*(dow/7.0))
    out["dow_cos"]   = np.cos(2*np.pi*(dow/7.0))
    out["hour_sin"]  = np.sin(2*np.pi*(hour/24.0))
    out["hour_cos"]  = np.cos(2*np.pi*(hour/24.0))
    out["is_weekend"] = (dow >= 5).astype(int)
    out["holiday"] = 0
    return out

def add_hdd_cdd_temp_sq(df: pd.DataFrame, base_c: float=18.0) -> pd.DataFrame:
    out = df.copy()
    if "temperature_2m" not in out.columns:
        raise KeyError("temperature_2m required to compute HDD/CDD/temp_sq")
    T = pd.to_numeric(out["temperature_2m"], errors="coerce")
    out["HDD"] = np.maximum(0.0, base_c - T)
    out["CDD"] = np.maximum(0.0, T - base_c)
    out["temp_sq"] = T**2
    return out

def add_consumption_lags(df: pd.DataFrame, lags=(24,48,72)) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"house_consumption_lag{L}"] = out[TARGET_COL].shift(L)
    return out

def add_pv_lags(df: pd.DataFrame, lags=(24,)) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"pv_power_lag{L}"] = out[PV_COL].shift(L)
    return out

def normalize_ha(df: pd.DataFrame, rename: Dict[str,str]) -> pd.DataFrame:
    out = df.rename(columns=rename)
    out = out.resample("h").mean()
    for c in [TARGET_COL, PV_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def assemble_training_frames(ha_raw: pd.DataFrame, wx: pd.DataFrame, tz: str="Europe/Prague") -> Dict[str, pd.DataFrame]:
    rename = {"sensor.house_consumption": TARGET_COL, "sensor.pv_power": PV_COL}
    ha = normalize_ha(ha_raw, rename=rename)
    wx_aligned = wx.reindex(ha.index).ffill()
    base = pd.concat([ha, wx_aligned], axis=1)
    base = add_hdd_cdd_temp_sq(base)
    base = add_time_features(base, tz=tz)
    house = add_consumption_lags(base, lags=(24,48,72))
    house_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        *TIME_FEATURES,
        PV_COL, "house_consumption_lag24","house_consumption_lag48","house_consumption_lag72",
        "HDD","CDD","temp_sq",
    ]
    house_train = house[house_feats + [TARGET_COL]].dropna()
    pv = add_pv_lags(base, lags=(24,))
    pv_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        *TIME_FEATURES,
        "pv_power_lag24",
        "HDD","CDD","temp_sq",
    ]
    pv_train = pv[pv_feats + [PV_COL]].dropna()
    return {"house": house_train, "pv": pv_train}
