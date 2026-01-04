from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class EntsoeClientConfig:
    # We might keep this class for future config, but token_path is deprecated/unused for now 
    # based on user request to rely on env var. 
    # However, to avoid breaking signature if other code uses it, verify usage.
    # It is used in signature below.
    pass


def guess_country_code_from_tz(tz: str) -> str:
    tz = (tz or "").lower()
    mapping = {
        "europe/prague": "CZ",
        "europe/vienna": "AT",
        "europe/berlin": "DE",
        "europe/warsaw": "PL",
        "europe/bratislava": "SK",
        "europe/budapest": "HU",
        "europe/paris": "FR",
        "europe/madrid": "ES",
        "europe/rome": "IT",
        "europe/amsterdam": "NL",
        "europe/copenhagen": "DK",
        "europe/oslo": "NO",
        "europe/stockholm": "SE",
        "europe/lisbon": "PT",
        "europe/dublin": "IE",
        "europe/athens": "GR",
        "europe/sofia": "BG",
        "europe/bucharest": "RO",
        "europe/ljubljana": "SI",
        "europe/zagreb": "HR",
    }
    return os.getenv("ENTSOE_COUNTRY_CODE") or mapping.get(tz, "CZ")


def fetch_day_ahead_prices_country(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz: str = "Europe/Prague",
    cfg: Optional[EntsoeClientConfig] = None,
) -> pd.DataFrame:
    """Fetch day-ahead prices for a country using entsoe-py.

    Returns hourly UTC-indexed DataFrame with 'price_eur_per_kwh'.
    """
    from entsoe import EntsoePandasClient  # lazy import to avoid hard dependency at module import

    # cfg is currently unused but kept for signature compatibility
    cfg = cfg or EntsoeClientConfig()
    
    import requests
    from entsoe import EntsoePandasClient

    token = os.getenv("ENTSOE_TOKEN")
    if not token:
             raise ValueError("No ENTSO-E token found. Set ENTSOE_TOKEN env var.")

    with requests.Session() as session:
        client = EntsoePandasClient(api_key=token, session=session)
        # entsoe-py expects tz-aware timestamps in local bidding zone tz
    s = pd.Timestamp(start).tz_convert(tz) if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz=tz)
    e = pd.Timestamp(end).tz_convert(tz) if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz=tz)

    series = client.query_day_ahead_prices(country_code, start=s, end=e)
    # series is EUR/MWh; convert to €/kWh and use UTC index
    df = series.to_frame("eur_per_mwh")
    df.index = df.index.tz_convert("UTC")
    df["price_eur_per_kwh"] = df["eur_per_mwh"] / 1000.0
    return df[["price_eur_per_kwh"]]
