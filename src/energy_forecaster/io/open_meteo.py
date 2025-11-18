
from typing import List
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

def fetch_openmeteo_archive(
    start: pd.Timestamp,
    end: pd.Timestamp,
    lat: float,
    lon: float,
    tz: str = "Europe/Prague",
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
    wx = wx.astype(float).fillna(0.0).resample("h").mean()
    return wx

def fetch_openmeteo_forecast(
    lat: float,
    lon: float,
    tz: str = "Europe/Prague",
    hours: int = 24,
    use_mock: bool = False,
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
        timestamps = pd.date_range(start=now, periods=hours, freq="h")
        
        # Realistic-ish values
        data = {
            "timestamp": timestamps,
            "temperature_2m": 10 + 5 * np.sin(np.arange(hours) * 2 * np.pi / 24),
            "relative_humidity_2m": np.full(hours, 70.0),
            "dew_point_2m": 5 + 3 * np.sin(np.arange(hours) * 2 * np.pi / 24),
            "rain": np.zeros(hours),
            "precipitation": np.zeros(hours),
            "snowfall": np.zeros(hours),
            "snow_depth": np.zeros(hours),
            "weather_code": np.full(hours, 1.0),
            "pressure_msl": np.full(hours, 1013.0),
            "cloud_cover": np.full(hours, 50.0),
            "wind_speed_10m": np.full(hours, 3.0),
            "wind_speed_100m": np.full(hours, 8.0),
            "vapour_pressure_deficit": np.full(hours, 0.5),
            "et0_fao_evapotranspiration": np.full(hours, 0.1),
            "shortwave_radiation": 200 + 150 * np.clip(np.sin(np.arange(hours) * 2 * np.pi / 24), 0, None),
            "direct_radiation": 150 + 120 * np.clip(np.sin(np.arange(hours) * 2 * np.pi / 24), 0, None),
            "diffuse_radiation": 50 + 40 * np.clip(np.sin(np.arange(hours) * 2 * np.pi / 24), 0, None),
            "direct_normal_irradiance": 250 + 180 * np.clip(np.sin(np.arange(hours) * 2 * np.pi / 24), 0, None),
        }
        return pd.DataFrame(data).set_index("timestamp")

    cache_sess = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(cache_sess, retries=4, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)

    url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = max(1, (hours + 23) // 24)
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

    now = pd.Timestamp.utcnow()
    wx_future = wx[wx.index >= now].iloc[:hours]
    return wx_future
