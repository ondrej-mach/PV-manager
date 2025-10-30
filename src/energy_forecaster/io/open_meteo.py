
from typing import List
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
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    wx = pd.DataFrame(data).set_index("timestamp").sort_index()
    wx = wx.astype(float).resample("h").mean()
    return wx
