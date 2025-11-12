
import os
import json
import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import websockets
from websockets.client import WebSocketClientProtocol


class HomeAssistant:
    """Class for managing Home Assistant WebSocket API connections and data retrieval."""

    def __init__(self, token: Optional[str] = None, url: Optional[str] = None):
        """Initialize Home Assistant connection manager.

        Args:
            token: Authentication token. Defaults to SUPERVISOR_TOKEN env var.
            url: WebSocket URL. Defaults to supervisor WebSocket URL.
        """
        self.token = token or os.getenv("SUPERVISOR_TOKEN")
        self.url = url or os.getenv("HASS_WS_URL") or "ws://supervisor/core/websocket"
        self._ws: Optional[WebSocketClientProtocol] = None
        self._msg_id = 1
        # Background loop and thread for sync façade (persistent connection)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._cached_config: Optional[Dict[str, Any]] = None
        self._cached_stat_metadata: Optional[List[Dict[str, Any]]] = None
        
    async def _ensure_connected(self) -> WebSocketClientProtocol:
        """Ensure WebSocket connection is established and authenticated.
        
        Reuses existing connection if still open, otherwise creates a new one.
        """
        # Check if we have an existing connection that's still open
        if self._ws:
            # Check if connection is still alive
            try:
                # The websockets library's connection has an 'open' property
                if hasattr(self._ws, 'open') and self._ws.open:
                    return self._ws
                # Fallback: check if 'closed' attribute exists and is False
                if hasattr(self._ws, 'closed') and not self._ws.closed:
                    return self._ws
            except Exception:
                # If checking the state fails, assume connection is dead
                self._ws = None
            
        if not self.token:
            raise RuntimeError("No authentication token available")
            
        # Establish new connection
        self._ws = await websockets.connect(self.url)
        await self._ws.recv()  # auth required message
        await self._ws.send(json.dumps({"type": "auth", "access_token": self.token}))
        auth_resp = json.loads(await self._ws.recv())
        
        if auth_resp["type"] != "auth_ok":
            await self._ws.close()
            raise RuntimeError("Authentication failed")
            
        return self._ws
        
    async def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and receive response via WebSocket."""
        ws = await self._ensure_connected()
        if "id" not in message:
            message["id"] = self._msg_id
            self._msg_id += 1
            
        await ws.send(json.dumps(message))
        return json.loads(await ws.recv())
        
    async def fetch_config_async(self) -> Optional[Dict]:
        """Fetch Home Assistant core configuration including location and timezone."""
        try:
            resp = await self._send_message({"type": "get_config"})
            if resp.get("success"):
                config = resp.get("result", {}) or {}
                self._cached_config = config
                return config
            return None
        except Exception:
            # Connection error - reset connection for next attempt
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            raise

    def _get_core_config_sync(self) -> Dict[str, Any]:
        """Retrieve and cache Home Assistant core config synchronously."""
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.fetch_config_async(), self._loop)
            try:
                config = fut.result()
            except FutureTimeoutError:
                config = None
        else:
            config = asyncio.run(self.fetch_config_async())
        if config is None:
            config = {}
        self._cached_config = config
        return config

    def get_location(self, 
                   fallback_lat: float = 49.6069, 
                   fallback_lon: float = 15.5808, 
                   fallback_tz: str = "Europe/Prague") -> Tuple[float, float, str]:
        """Get coordinates and timezone from Home Assistant if available, otherwise use fallbacks."""
        config = self._cached_config or self._get_core_config_sync()
        return (
            config.get("latitude", fallback_lat) if config else fallback_lat,
            config.get("longitude", fallback_lon) if config else fallback_lon,
            config.get("time_zone", fallback_tz) if config else fallback_tz
        )

    def get_locale(self, fallback_locale: str = "en-US") -> str:
        """Return preferred locale string derived from HA config."""
        config = self._cached_config or self._get_core_config_sync()
        language = (config.get("language") if config else None) or ""
        country = (config.get("country") if config else None) or ""
        if language:
            lang_clean = language.replace("_", "-")
            if country:
                country_clean = country.replace("_", "-")
                if len(country_clean) == 2:
                    country_clean = country_clean.upper()
                return f"{lang_clean}-{country_clean}"
            return lang_clean
        return fallback_locale

    async def fetch_statistics_async(
        self,
        entities_with_types: List[Tuple[str, str]],
        days: int,
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        """Fetch hourly statistics for multiple entities from Home Assistant."""
        days = min(days, 730)  # Limit to 2 years of data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        entities = [e for e, _t in entities_with_types]
        type_for_entity = {e: t for e, t in entities_with_types}
        requested_types = sorted(set(type_for_entity.values()))

        # Split into time chunks to avoid huge requests
        ranges = []
        cursor = start_time
        while cursor < end_time:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_time)
            ranges.append((cursor, chunk_end))
            cursor = chunk_end

        results: Dict[str, list] = {e: [] for e in entities}

        try:
            for chunk_start, chunk_end in ranges:
                req = {
                    "type": "call_service",
                    "domain": "recorder",
                    "service": "get_statistics",
                    "service_data": {
                        "statistic_ids": entities,
                        "period": "hour",
                        "start_time": chunk_start.isoformat(),
                        "end_time": chunk_end.isoformat(),
                        "types": requested_types,
                    },
                    "return_response": True,
                }
                
                resp = await self._send_message(req)
                stats = resp.get("result", {}).get("response", {}).get("statistics", {})

                for entity in entities:
                    if entity in stats:
                        results[entity].extend(stats[entity])
        except Exception:
            # On error, close the connection so it gets re-established on next call
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            raise

        # Process results into DataFrame
        frames = []
        for entity, rows in results.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if "start" not in df.columns:
                continue
            df["start"] = pd.to_datetime(df["start"], utc=True)
            df = df.set_index("start").sort_index()

            col = type_for_entity[entity]
            if col in df.columns:
                frames.append(df[[col]].rename(columns={col: entity}))

        if frames:
            out = pd.concat(frames, axis=1).sort_index()
            return out
        return pd.DataFrame()

    # --- Short-term helpers ---
    async def fetch_statistics_window_async(
        self,
        entities_with_types: List[Tuple[str, str]],
        start_time: datetime,
        end_time: datetime,
        period: str = "hour",
    ) -> pd.DataFrame:
        """Fetch statistics for a custom time window.

        Args:
            entities_with_types: [(entity_id, type)] pairs (e.g., ("sensor.pv_power","mean"))
            start_time: window start (timezone-aware, UTC preferred)
            end_time: window end (timezone-aware, UTC preferred)
            period: "5minute", "hour", "day", or "month"
        Returns:
            DataFrame indexed by timestamp with columns per entity id.
        """
        entities = [e for e, _t in entities_with_types]
        type_for_entity = {e: t for e, t in entities_with_types}
        requested_types = sorted(set(type_for_entity.values()))

        req = {
            "type": "call_service",
            "domain": "recorder",
            "service": "get_statistics",
            "service_data": {
                "statistic_ids": entities,
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "types": requested_types,
            },
            "return_response": True,
        }

        resp = await self._send_message(req)
        stats = resp.get("result", {}).get("response", {}).get("statistics", {})

        frames = []
        for entity in entities:
            rows = stats.get(entity, [])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if "start" not in df.columns:
                continue
            df["start"] = pd.to_datetime(df["start"], utc=True)
            df = df.set_index("start").sort_index()
            col = type_for_entity[entity]
            if col in df.columns:
                frames.append(df[[col]].rename(columns={col: entity}))

        if frames:
            return pd.concat(frames, axis=1).sort_index()
        return pd.DataFrame()

    async def list_statistic_ids_async(self, include_units: bool = True) -> List[Dict[str, Any]]:
        """Return statistic metadata (id, names, units) from Home Assistant recorder."""
        service_data: Dict[str, Any] = {}
        req = {
            "type": "call_service",
            "domain": "recorder",
            "service": "list_statistic_ids",
            "service_data": service_data,
            "return_response": True,
        }

        resp = await self._send_message(req)
        payload: Any = resp.get("result", {})
        if isinstance(payload, dict):
            payload = payload.get("response", payload)
        if isinstance(payload, dict):
            payload = payload.get("statistic_ids", payload.get("statistics", payload))
        if not isinstance(payload, list):
            payload = []

        normalized: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            stat_id = item.get("statistic_id") or item.get("name")
            if not stat_id:
                continue
            unit = (
                item.get("unit_of_measurement")
                or item.get("statistics_unit_of_measurement")
                or item.get("unit")
            )
            normalized.append(
                {
                    "statistic_id": stat_id,
                    "display_name": item.get("display_name") or item.get("name") or stat_id,
                    "unit": unit,
                    "has_mean": bool(item.get("has_mean")),
                    "has_sum": bool(item.get("has_sum")),
                    "unit_class": item.get("unit_class"),
                    "device_class": item.get("device_class"),
                    "supported_statistics": item.get("supported_statistics") or item.get("statistics_types") or [],
                }
            )
        self._cached_stat_metadata = normalized
        return normalized

    def list_statistic_ids_sync(self, include_units: bool = True) -> List[Dict[str, Any]]:
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.list_statistic_ids_async(include_units=include_units), self._loop)
            return fut.result()
        return asyncio.run(self.list_statistic_ids_async(include_units=include_units))

    async def list_entities_async(self) -> List[Dict[str, Any]]:
        """Return raw entity state metadata from Home Assistant."""
        result = await self._send_message({"type": "get_states"})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            payload = result.get("result")
            if isinstance(payload, list):
                return payload
        return []

    def list_entities_sync(self) -> List[Dict[str, Any]]:
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.list_entities_async(), self._loop)
            return fut.result()
        return asyncio.run(self.list_entities_async())

    def fetch_last_hours_sync(
        self,
        entities_with_types: List[Tuple[str, str]],
        hours: int,
        period: str = "hour",
    ) -> pd.DataFrame:
        """Fetch last N hours of statistics.
        
        Args:
            period: "5minute" for high-res recent data, "hour" for longer lookback
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_statistics_window_async(entities_with_types, start_time, end_time, period),
                self._loop,
            )
            return fut.result()
        return asyncio.run(self.fetch_statistics_window_async(entities_with_types, start_time, end_time, period))

    def fetch_statistics_sync(
        self,
        entities_with_types: List[Tuple[str, str]],
        days: int,
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        """Synchronous wrapper for fetch_statistics_async."""
        # Preferred: run the async method on the background loop that is kept
        # alive for the lifetime of this HomeAssistant instance. If no loop is
        # active, fall back to running it with asyncio.run (short-lived loop).
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_statistics_async(entities_with_types, days, chunk_days), self._loop
            )
            return fut.result()
        return asyncio.run(self.fetch_statistics_async(entities_with_types, days, chunk_days))

    # --- Background loop management and sync helpers ---
    def _start_background_loop(self) -> None:
        if self._loop and self._loop.is_running():
            return

        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=_run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def connect_sync(self, timeout: Optional[float] = None) -> None:
        """Start background loop and connect (authenticate) synchronously."""
        self._start_background_loop()
        fut = asyncio.run_coroutine_threadsafe(self._ensure_connected(), self._loop)
        return fut.result(timeout)

    def close_sync(self, timeout: Optional[float] = None) -> None:
        """Close connection and stop background loop synchronously."""
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.close(), self._loop)
            try:
                fut.result(timeout)
            except Exception:
                pass

            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1.0)
            self._loop = None
            self._thread = None

    async def close(self):
        """Close the WebSocket connection if it's open.
        
        Call this when shutting down the addon or when you're done with all operations.
        """
        if not self._ws:
            return

        try:
            await self._ws.close()
        except Exception:
            # Ignore errors during close
            pass
        finally:
            self._ws = None
