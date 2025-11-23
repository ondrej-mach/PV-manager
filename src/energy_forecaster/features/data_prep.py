
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

_LOGGER = logging.getLogger(__name__)

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
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "temperature_2m",
]

OPTIONAL_WEATHER: List[str] = [
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]

TIME_FEATURES: List[str] = [
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "holiday", "is_weekend",
    "hour_sin", "hour_cos",
]

TARGET_COL = "power_consumption_kw"
PV_COL = "pv_power_kw"
DEFAULT_INTERVAL_MINUTES = 15

DEFAULT_RENAME_MAP = {"sensor.house_consumption": TARGET_COL, "sensor.pv_power": PV_COL}
DEFAULT_SCALES = {TARGET_COL: 0.001, PV_COL: 0.001}

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

def _steps_per_hour(freq_minutes: int) -> int:
    freq_minutes = max(1, int(freq_minutes))
    return max(1, int(round(60 / freq_minutes)))


def add_consumption_lags(
    df: pd.DataFrame,
    lag_hours: Tuple[int, ...] = (24, 48, 72),
    freq_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> pd.DataFrame:
    out = df.copy()
    sph = _steps_per_hour(freq_minutes)
    for hours in lag_hours:
        shift_steps = hours * sph
        out[f"house_consumption_lag{hours}"] = out[TARGET_COL].shift(shift_steps)
    return out


def add_pv_lags(
    df: pd.DataFrame,
    lag_hours: Tuple[int, ...] = (24,),
    freq_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> pd.DataFrame:
    out = df.copy()
    sph = _steps_per_hour(freq_minutes)
    for hours in lag_hours:
        shift_steps = hours * sph
        out[f"pv_power_lag{hours}"] = out[PV_COL].shift(shift_steps)
    return out

def normalize_ha(
    df: pd.DataFrame,
    rename: Dict[str, str],
    scales: Optional[Dict[str, float]] = None,
    freq_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Normalize HA data: rename columns, resample to the requested interval, convert using provided scales."""
    out = df.rename(columns=rename)

    # Guard against empty frames coming back with a RangeIndex when Home Assistant has no data yet.
    if not isinstance(out.index, pd.DatetimeIndex):
        if out.empty:
            out = out.copy()
            out.index = pd.DatetimeIndex([], tz="UTC")
        else:
            coerced_index = pd.to_datetime(out.index, utc=True, errors="coerce")
            if coerced_index.isna().any():
                raise TypeError(
                    "normalize_ha expected a DatetimeIndex; failed to coerce non-empty index to timestamps."
                )
            out = out.copy()
            out.index = coerced_index

    out = out.sort_index()
    freq_minutes = max(1, int(freq_minutes))
    target_rule = f"{freq_minutes}min"
    resampled = out.resample(target_rule).mean()
    if len(out.index) >= 2:
        src_minutes = int((out.index[1] - out.index[0]).total_seconds() / 60)
    else:
        src_minutes = freq_minutes
    if freq_minutes < src_minutes:
        resampled = resampled.interpolate(method="time")
    out = resampled.ffill().bfill()
    scales = scales or {}
    for c, factor in scales.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * factor

    # Clamp obvious sensor noise on PV readings (tiny non-zero values overnight).
    if PV_COL in out.columns:
        out[PV_COL] = out[PV_COL].clip(lower=0.0)
        noise_mask = out[PV_COL] < 0.02  # 20 W noise floor
        if noise_mask.any():
            out.loc[noise_mask, PV_COL] = 0.0
    return out

def assemble_training_frames(
    ha_raw: pd.DataFrame,
    wx: pd.DataFrame,
    tz: str = "Europe/Prague",
    rename_map: Optional[Dict[str, str]] = None,
    scales: Optional[Dict[str, float]] = None,
    freq_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, pd.DataFrame]:
    rename = rename_map or DEFAULT_RENAME_MAP
    ha = normalize_ha(ha_raw, rename=rename, scales=scales or DEFAULT_SCALES, freq_minutes=freq_minutes)
    missing_required = [col for col in (TARGET_COL, PV_COL) if col not in ha.columns]
    if missing_required:
        _LOGGER.warning("Home Assistant statistics are missing required column(s) %s. Filling with zeros.", missing_required)
        for col in missing_required:
            ha[col] = 0.0
    wx_aligned = wx.reindex(ha.index).ffill().bfill()
    base = pd.concat([ha, wx_aligned], axis=1)
    base = add_hdd_cdd_temp_sq(base)
    base = add_time_features(base, tz=tz)
    house = add_consumption_lags(base, lag_hours=(24,48,72), freq_minutes=freq_minutes)
    house_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
        *TIME_FEATURES,
        PV_COL, "house_consumption_lag24","house_consumption_lag48","house_consumption_lag72",
        "HDD","CDD","temp_sq",
    ]
    house_train = house[house_feats + [TARGET_COL]].dropna()
    pv = add_pv_lags(base, lag_hours=(24,), freq_minutes=freq_minutes)
    pv_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
        *TIME_FEATURES,
        "pv_power_lag24",
        "HDD","CDD","temp_sq",
    ]
    pv_train = pv[pv_feats + [PV_COL]].dropna()
    return {"house": house_train, "pv": pv_train}

def assemble_forecast_features(
    ha_recent: pd.DataFrame,
    wx_future: pd.DataFrame,
    tz: str = "Europe/Prague",
    rename_map: Optional[Dict[str, str]] = None,
    scales: Optional[Dict[str, float]] = None,
    freq_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, pd.DataFrame]:
    """Build feature matrices for forecasting using recent HA history and future weather.

    ha_recent: DataFrame with recent history for sensors (can be 5min or hourly resolution).
    wx_future: DataFrame of weather forecast indexed by UTC timestamps (future, any resolution).
    Returns dict with keys 'pv_X' and 'house_X' (feature frames aligned to wx_future index).
    
    Note: Lags are computed in hours from the recent history, mapped to the future timeline.
    """
    rename = rename_map or DEFAULT_RENAME_MAP
    ha = normalize_ha(ha_recent, rename=rename, scales=scales or DEFAULT_SCALES, freq_minutes=freq_minutes)
    missing_required = [col for col in (TARGET_COL, PV_COL) if col not in ha.columns]
    if missing_required:
        _LOGGER.warning("Home Assistant history is missing required column(s) %s. Filling with zeros.", missing_required)
        for col in missing_required:
            ha[col] = 0.0

    # Resample to match the future timeline frequency if needed
    try:
        future_freq = pd.infer_freq(wx_future.index)
    except ValueError:
        future_freq = None
    if future_freq is None:
        # Infer from first two timestamps
        if len(wx_future) > 1:
            delta = (wx_future.index[1] - wx_future.index[0]).total_seconds() / 60
            future_freq_min = int(delta)
        else:
            future_freq_min = 60  # default hourly
    else:
        # Parse frequency like '15min', '1h', etc.
        import re
        match = re.search(r'(\d+)', str(future_freq))
        if 'min' in str(future_freq).lower():
            future_freq_min = int(match.group(1)) if match else 15
        elif 'h' in str(future_freq).lower():
            future_freq_min = 60 * (int(match.group(1)) if match else 1)
        else:
            future_freq_min = 60

    # Compute engineered time/weather features on future timeline
    base_future = wx_future.copy()
    # Ensure all expected weather columns are present even if the forecast API omits some of them
    for col in WEATHER_FEATURES:
        if col not in base_future.columns:
            base_future[col] = 0.0
    base_future = add_hdd_cdd_temp_sq(base_future)
    base_future = add_time_features(base_future, tz=tz)

    # Helper to map lag from history to future index
    def lag_series(series: pd.Series, lag_hours: int) -> pd.Series:
        """Shift series forward by lag_hours and reindex to future."""
        s = series.copy()
        s.index = s.index + pd.Timedelta(hours=lag_hours)
        return s.reindex(base_future.index, method='ffill')

    # Build PV lag 24h for future index from recent history
    pv_hist = ha[PV_COL] if PV_COL in ha.columns else None
    if isinstance(pv_hist, pd.Series):
        pv_hist = pv_hist.sort_index()
        pv_lag24 = lag_series(pv_hist, 24).fillna(0.0)
    else:
        pv_lag24 = pd.Series(0.0, index=base_future.index, dtype=float)

    pv_frame = base_future.copy()
    pv_frame["pv_power_lag24"] = pv_lag24
    pv_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
        *TIME_FEATURES,
        "pv_power_lag24",
        "HDD","CDD","temp_sq",
    ]
    pv_X = pv_frame[pv_feats]

    # House features need predicted PV for the same timestep and consumption lags
    cons_hist = ha[TARGET_COL] if TARGET_COL in ha.columns else None

    lag24 = lag_series(cons_hist, 24) if isinstance(cons_hist, pd.Series) else pd.Series(0.0, index=base_future.index, dtype=float)
    lag48 = lag_series(cons_hist, 48) if isinstance(cons_hist, pd.Series) else pd.Series(0.0, index=base_future.index, dtype=float)
    lag72 = lag_series(cons_hist, 72) if isinstance(cons_hist, pd.Series) else pd.Series(0.0, index=base_future.index, dtype=float)

    house_frame = base_future.copy()
    # placeholder for PV; caller should fill with predicted PV later
    house_frame[PV_COL] = pd.NA
    house_frame["house_consumption_lag24"] = lag24
    house_frame["house_consumption_lag48"] = lag48
    house_frame["house_consumption_lag72"] = lag72
    house_feats = [
        "relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","cloud_cover","weather_code","pressure_msl",
        "wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
        *TIME_FEATURES,
        PV_COL,
        "house_consumption_lag24","house_consumption_lag48","house_consumption_lag72",
        "HDD","CDD","temp_sq",
    ]
    house_X = house_frame[house_feats]

    return {"pv_X": pv_X, "house_X": house_X, "ha_norm": ha}
