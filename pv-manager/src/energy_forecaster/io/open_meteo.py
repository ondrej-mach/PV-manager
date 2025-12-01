
from typing import List
import math
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


def _resample_to_interval(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    interval_minutes = max(1, int(interval_minutes))
    freq = f"{interval_minutes}min"
    df = df.sort_index()
    resampled = df.resample(freq).mean()
    if len(df.index) >= 2:
        src_minutes = int((df.index[1] - df.index[0]).total_seconds() / 60)
    else:
        src_minutes = interval_minutes
    if interval_minutes < src_minutes:
        resampled = resampled.interpolate(method="time")
    return resampled.ffill().bfill()

def fetch_openmeteo_archive(
    start: pd.Timestamp,
    end: pd.Timestamp,
    lat: float,
    lon: float,
    tz: str = "Europe/Prague",
    interval_minutes: int = 60,
) -> pd.DataFrame:
    cache_sess = requests_cache.CachedSession(".cache", expire_after=-1)
    session = retry(cache_sess, retries=4, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)

    start_date = start.date().isoformat()
    end_date = end.date().isoformat()

    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars: List[str] = [
        "temperature_2m","relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","weather_code","pressure_msl","cloud_cover","wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": tz,
    }

    responses = client.weather_api(url, params=params)
    resp = responses[0]
    hourly = resp.Hourly()

    idx = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    # keep timestamps tz-aware in UTC
    data = {"timestamp": idx}
    for i, name in enumerate(hourly_vars):
        values = hourly.Variables(i).ValuesAsNumpy()
        if values is None:
            values = np.full(len(idx), np.nan)
        data[name] = values

    wx = pd.DataFrame(data).set_index("timestamp").sort_index()
    for var in hourly_vars:
        if var not in wx.columns:
            wx[var] = 0.0
    wx = wx.astype(float).fillna(0.0)
    wx = _resample_to_interval(wx, interval_minutes)
    return wx

def fetch_openmeteo_forecast(
    lat: float,
    lon: float,
    tz: str = "Europe/Prague",
    hours: int = 24,
    use_mock: bool = False,
    interval_minutes: int = 60,
) -> pd.DataFrame:
    """Fetch hourly weather forecast for the next `hours` hours.

    Returns a DataFrame indexed by UTC timestamps with the same columns as the archive fetch
    (including temperature_2m required for engineered features).
    
    Args:
        use_mock: If True, generate mock forecast data (for testing when API unavailable)
    """
    hourly_vars: List[str] = [
        "temperature_2m","relative_humidity_2m","dew_point_2m","rain","precipitation","snowfall",
        "snow_depth","weather_code","pressure_msl","cloud_cover","wind_speed_10m","wind_speed_100m",
        "vapour_pressure_deficit","et0_fao_evapotranspiration",
        "shortwave_radiation","direct_radiation","diffuse_radiation","direct_normal_irradiance",
    ]
    
    if use_mock:
        # Generate mock forecast data for testing
        now = pd.Timestamp.utcnow().tz_localize("UTC").ceil("h")
        periods = max(1, math.ceil((hours * 60) / max(1, interval_minutes)))
        timeline = pd.date_range(start=now, periods=periods, freq=f"{interval_minutes}min")
        time_hours = np.arange(periods) * (interval_minutes / 60.0)

        data = {
            "timestamp": timeline,
            "temperature_2m": 10 + 5 * np.sin(time_hours * 2 * np.pi / 24),
            "relative_humidity_2m": np.full(periods, 70.0),
            "dew_point_2m": 5 + 3 * np.sin(time_hours * 2 * np.pi / 24),
            "rain": np.zeros(periods),
            "precipitation": np.zeros(periods),
            "snowfall": np.zeros(periods),
            "snow_depth": np.zeros(periods),
            "weather_code": np.full(periods, 1.0),
            "pressure_msl": np.full(periods, 1013.0),
            "cloud_cover": np.full(periods, 50.0),
            "wind_speed_10m": np.full(periods, 3.0),
            "wind_speed_100m": np.full(periods, 8.0),
            "vapour_pressure_deficit": np.full(periods, 0.5),
            "et0_fao_evapotranspiration": np.full(periods, 0.1),
            "shortwave_radiation": 200 + 150 * np.clip(np.sin(time_hours * 2 * np.pi / 24), 0, None),
            "direct_radiation": 150 + 120 * np.clip(np.sin(time_hours * 2 * np.pi / 24), 0, None),
            "diffuse_radiation": 50 + 40 * np.clip(np.sin(time_hours * 2 * np.pi / 24), 0, None),
            "direct_normal_irradiance": 250 + 180 * np.clip(np.sin(time_hours * 2 * np.pi / 24), 0, None),
        }
        return pd.DataFrame(data).set_index("timestamp")

    cache_sess = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(cache_sess, retries=4, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)

    url = "https://api.open-meteo.com/v1/forecast"
    now_utc = pd.Timestamp.utcnow()
    hours_since_midnight = (now_utc - now_utc.floor("D")).total_seconds() / 3600.0
    hours_needed = hours + math.ceil(hours_since_midnight)
    forecast_days = max(1, math.ceil(hours_needed / 24))
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    responses = client.weather_api(url, params=params)
    resp = responses[0]
    hourly = resp.Hourly()

    idx = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": idx}
    for i, name in enumerate(hourly_vars):
        values = hourly.Variables(i).ValuesAsNumpy()
        if values is None:
            values = np.full(len(idx), np.nan)
        data[name] = values

    wx = pd.DataFrame(data).set_index("timestamp").sort_index()
    for var in hourly_vars:
        if var not in wx.columns:
            wx[var] = 0.0
    wx = wx.astype(float).fillna(0.0)
    wx = _resample_to_interval(wx, interval_minutes)

    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    # Only keep forecast rows after "now"; if the API doesn't reach far enough into the future,
    # extend the last known value but note the shortfall via logging.
    wx_future_raw = wx[wx.index >= now]
    periods = max(1, math.ceil((hours * 60) / max(1, interval_minutes)))
    if len(wx_future_raw) == 0:
        future_start = now.ceil(f"{interval_minutes}min")
    else:
        future_start = wx_future_raw.index[0]

    future_index = pd.date_range(start=future_start, periods=periods, freq=f"{interval_minutes}min", tz="UTC")
    wx_future = wx_future_raw.reindex(future_index).interpolate(method="time").ffill().bfill()

    if len(wx_future) < periods:
        pad_index = pd.date_range(start=wx_future.index[-1] + pd.Timedelta(minutes=interval_minutes),
                                  periods=periods - len(wx_future), freq=f"{interval_minutes}min")
        pad_df = pd.DataFrame(index=pad_index, columns=wx_future.columns)
        if len(wx_future) > 0:
            pad_df = pad_df.fillna(wx_future.iloc[-1])
        wx_future = pd.concat([wx_future, pad_df])
        wx_future = wx_future.ffill().bfill()

    return wx_future
