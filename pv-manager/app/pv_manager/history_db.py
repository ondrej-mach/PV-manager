import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

_LOGGER = logging.getLogger(__name__)

class HistoryDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # sqlite3 context manager only handles transaction (commit/rollback), NOT closing!
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS deferrable_logs (
                            timestamp TEXT NOT NULL,
                            entity_id TEXT NOT NULL,
                            val REAL,
                            PRIMARY KEY (timestamp, entity_id)
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON deferrable_logs(timestamp)")
            finally:
                conn.close()
        except Exception as exc:
            _LOGGER.error("Failed to initialize history database at %s: %s", self.db_path, exc)

    def log_states(self, timestamp: datetime, states: Dict[str, float]) -> None:
        """Log a batch of entity states for a given timestamp."""
        if not states:
            return

        ts_str = timestamp.isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO deferrable_logs (timestamp, entity_id, val) VALUES (?, ?, ?)",
                        [(ts_str, eid, val) for eid, val in states.items()]
                    )
            finally:
                conn.close()
        except Exception as exc:
            _LOGGER.error("Failed to log states to history DB: %s", exc)

    def get_history(self, start: datetime, end: datetime, entity_ids: List[str]) -> pd.DataFrame:
        """Retrieve history for specific entities as a DataFrame (pivoted)."""
        if not entity_ids:
            return pd.DataFrame()

        start_str = start.isoformat()
        end_str = end.isoformat()
        
        placeholders = ",".join("?" for _ in entity_ids)
        query = f"""
            SELECT timestamp, entity_id, val 
            FROM deferrable_logs 
            WHERE timestamp >= ? AND timestamp <= ? AND entity_id IN ({placeholders})
        """
        params = [start_str, end_str] + entity_ids

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                # read_sql_query uses the connection. We don't need transaction for SELECT, 
                # but closing is mandatory.
                df = pd.read_sql_query(query, conn, params=params)
            finally:
                conn.close()
            
            if df.empty:
                return pd.DataFrame()

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Pivot: rows=timestamp, cols=entity_id, values=val
            pivot_df = df.pivot(index="timestamp", columns="entity_id", values="val")
            # Ensure index is sorted
            pivot_df.sort_index(inplace=True)
            return pivot_df

        except Exception as exc:
            _LOGGER.error("Failed to retrieve history from DB: %s", exc)
            return pd.DataFrame()
