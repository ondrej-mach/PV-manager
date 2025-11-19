import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Environment variable for token
HASS_TOKEN = os.getenv("HASS_TOKEN")
if not HASS_TOKEN:
    raise ValueError("HASS_TOKEN environment variable not set")

# Home Assistant URL
BASE_URL = "http://homeassistant:8123/api/history/period"

# Calculate 72 hours ago
end_time = datetime.now()
start_time = end_time - timedelta(hours=48)
end_time_iso = end_time.isoformat()
start_time_iso = start_time.isoformat()

# Entities to retrieve
entities = ["sensor.pv_power", "sensor.house_consumption"]
entity_filter = ",".join(entities)

# Prepare headers
headers = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json"
}

# Make the request
params = {
    "filter_entity_id": entity_filter,
    "minimal_response": "true",
    "end_time": end_time_iso
}

response = requests.get(f"{BASE_URL}/{start_time_iso}", headers=headers, params=params)
response.raise_for_status()

# Load JSON data
data = response.json()


rows = []
for ent_list in data:
    if not ent_list:
        continue
    base = ent_list[0]
    entity_id = base.get("entity_id") or base.get("entity")
    for measurement in ent_list:
        rows.append({
            "entity_id": entity_id,
            "state": measurement.get("state"),
            "timestamp": measurement.get("last_changed")
        })

df = pd.DataFrame(rows)

# ---------- normalize types ----------
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)   # keep tz-aware
df["state"] = pd.to_numeric(df["state"], errors="coerce")    # numeric states, others -> NaN

# ---------- pivot so each entity is a column ----------
df_pivot = (
    df.pivot_table(index="timestamp", columns="entity_id", values="state", aggfunc="mean")
      .sort_index()
)

# ---------- create 5-minute grid (natural 5-min marks) ----------
start = df_pivot.index.min().floor("5min")
end   = df_pivot.index.max().ceil("5min")
bins = pd.date_range(start, end, freq="5min", tz=df_pivot.index.tz)   # e.g. 12:00,12:05,...

left = pd.DataFrame({"timestamp": bins})

# ---------- for each entity pick latest reading <= bin timestamp ----------
df_5min_latest = pd.DataFrame(index=bins, columns=df_pivot.columns, dtype=float)

for col in df_pivot.columns:
    ser = df_pivot[col].dropna().reset_index()
    if ser.empty:
        continue
    # ensure reset_index columns are ['timestamp','value']
    ser.columns = ["timestamp", "value"]
    ser = ser.sort_values("timestamp")
    merged = pd.merge_asof(left, ser, on="timestamp", direction="backward")
    df_5min_latest[col] = merged["value"].values

# df_5min_latest now holds, for every 5-min mark, the latest-known reading at or before that mark.
# If you want to carry the very first known value backwards to earlier bins, do:
# df_5min_latest = df_5min_latest.ffill(axis=0)

# Save to CSV
csv_filename = "home_assistant_history.csv"
df_5min_latest.to_csv(csv_filename)
logger.info("Saved %s records to %s", len(df_5min_latest), csv_filename)