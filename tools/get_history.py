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
entities = ["sensor.pv_power"]
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

# Convert to pandas DataFrame
all_records = []
for entity_id, entity_list in zip(entities, data):
    for item in entity_list:
        all_records.append({
            "entity_id": entity_id,
            "last_changed": item.get("last_changed"),
            "state": item.get("state")
        })

df = pd.DataFrame(all_records)

# Save to CSV
csv_filename = "home_assistant_history.csv"
df.to_csv(csv_filename, index=False)
logger.info("Saved %s records to %s", len(df), csv_filename)
