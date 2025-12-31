
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Add app directory to path
# Add app directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from pv_manager.state import AppContext
from pv_manager.settings import AppSettings, EntitySelection

def test_cycle():
    print("Initializing AppContext...")
    ctx = AppContext()
    
    # Mock Home Assistant
    mock_ha = MagicMock()
    ctx._ha = mock_ha
    ctx._lat = 50.0
    ctx._lon = 14.0
    ctx._tz = "Europe/Prague"
    
    # Mock fetch_history_sync
    def mock_fetch_history(entities, hours, period):
        print(f"Mock fetch_history called for {hours} hours")
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=hours*4, freq="15min")
        df = pd.DataFrame(index=dates)
        for entity_id, _ in entities:
            df[entity_id] = np.random.rand(len(dates)) * 1000 # Random power
        return df
    
    mock_ha.fetch_history_sync.side_effect = mock_fetch_history
    
    # Mock get_entity_state_sync
    def mock_get_state(entity_id):
        return {"state": "50", "attributes": {"unit_of_measurement": "%"}}
    
    mock_ha.get_entity_state_sync.side_effect = mock_get_state
    
    # Mock run_prediction_pipeline
    # We want to test _run_cycle_sync logic, but run_prediction_pipeline is complex and external.
    # However, to test _run_cycle_sync fully, we should let it call run_prediction_pipeline if possible,
    # OR mock it if it's too hard to setup.
    # Given the user's error was in _run_cycle_sync AFTER prediction, let's mock run_prediction_pipeline to return dummy data
    # so we can focus on the optimization part where the error occurred.
    
    with patch("pv_manager.state.run_prediction_pipeline") as mock_pred:
        # Setup dummy prediction result
        dates = pd.date_range(start=datetime.now(timezone.utc), periods=24*4, freq="15min")
        pv_pred = pd.DataFrame({"pv_power_kw": np.random.rand(len(dates))}, index=dates)
        house_pred = pd.DataFrame({"power_consumption_kw": np.random.rand(len(dates))}, index=dates)
        ha_recent = pd.DataFrame()
        
        mock_pred.return_value = {
            "pv_pred": pv_pred,
            "house_pred": house_pred,
            "ha_recent": ha_recent
        }
        
        print("Running _run_cycle_sync...")
        stat_ids = [("sensor.house", "mean"), ("sensor.pv", "mean")]
        rename_map = {"sensor.house": "House_consumption", "sensor.pv": "PV_power"}
        scales = {"House_consumption": 0.001, "PV_power": 0.001}
        
        try:
            snapshot = ctx._run_cycle_sync(stat_ids, rename_map, scales)
            print("Cycle finished successfully!")
            print("Snapshot summary:", snapshot.summary)
        except Exception as e:
            print(f"Cycle failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_cycle()
