
import os
import json
import asyncio
import threading
import logging
import requests
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
from urllib.parse import urljoin

import pandas as pd
import websockets

_LOGGER = logging.getLogger(__name__)


class HomeAssistant:
    """Class for managing Home Assistant WebSocket API connections and data retrieval."""

    def __init__(
        self,
        token: Optional[str] = None,
        instance_url: Optional[str] = None,
    ):
        """Initialize Home Assistant connection manager.

        Args:
            token: Authentication token. Defaults to SUPERVISOR_TOKEN env var.
            url: WebSocket URL. Defaults to supervisor WebSocket URL.
        """
        self.token = token or os.getenv("SUPERVISOR_TOKEN")

        self.ws_url = urljoin(instance_url.replace("http", "ws", 1), "/api/websocket") if instance_url else "ws://supervisor/core/websocket"
        self.rest_url = urljoin(instance_url, "/api") if instance_url else "http://supervisor/core/api"
        self._ws: Any = None
        self._msg_id = 1
        # Background loop and thread for sync facade (persistent connection)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._cached_config: Optional[Dict[str, Any]] = None
        self._cached_stat_metadata: Optional[List[Dict[str, Any]]] = None
        self._ws_lock: Optional[asyncio.Lock] = None
        self._ws_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection_lock: Optional[asyncio.Lock] = None
        self._connection_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
        self._history_stats_supported: Optional[bool] = None

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    @staticmethod
    def _period_to_freq(period: str) -> str:
        normalized = (period or "hour").strip().lower()
        mapping = {
            "minute": "1min",
            "1minute": "1min",
            "5minute": "5min",
            "10minute": "10min",
            "quarter": "15min",
            "15minute": "15min",
            "hour": "1h",
            "1hour": "1h",
            "day": "1d",
        }
        return mapping.get(normalized, period or "1h")

    @staticmethod
    def _normalize_entities_with_types(entities: Sequence[Union[str, Tuple[str, str]]]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for item in entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], str):
                entity_id = item[0]
                stat_type = str(item[1] or "mean").lower()
            elif isinstance(item, str):
                entity_id = item
                stat_type = "mean"
            else:
                continue
            if entity_id:
                normalized.append((entity_id, stat_type))
        return normalized

    @staticmethod
    def _normalize_history_response(resp: Any) -> Dict[str, List[Dict[str, Any]]]:
        payload = resp
        if isinstance(payload, dict):
            if payload.get("success") is False:
                return {}
            payload = payload.get("result", payload.get("response", payload))

        by_entity: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(payload, list):
            for bucket in payload:
                if not isinstance(bucket, list) or not bucket:
                    continue
                entity_id = None
                for row in bucket:
                    if isinstance(row, dict) and row.get("entity_id"):
                        entity_id = row["entity_id"]
                        break
                if entity_id:
                    by_entity[entity_id] = [row for row in bucket if isinstance(row, dict)]
        return by_entity

    async def _fetch_history_statistics_window_async(
        self,
        entities_with_types: List[Tuple[str, str]],
        start_time: datetime,
        end_time: datetime,
        *,
        period: str,
        chunk_hours: int,
    ) -> pd.DataFrame:
        if not entities_with_types:
            return pd.DataFrame()

        entities = [entity for entity, _t in entities_with_types]
        type_for_entity = {entity: stat_type for entity, stat_type in entities_with_types}
        requested_types = sorted(set(type_for_entity.values()))

        chunk_hours = max(1, int(chunk_hours))
        frames_by_entity: Dict[str, List[Dict[str, Any]]] = {entity: [] for entity in entities}
        cursor = start_time
        while cursor < end_time:
            chunk_end = min(cursor + timedelta(hours=chunk_hours), end_time)
            req = {
                "type": "history/history_statistics_during_period",
                "start_time": cursor.isoformat(),
                "end_time": chunk_end.isoformat(),
                "statistic_ids": entities,
                "period": period,
                "types": requested_types,
            }

            resp = await self._send_message(req)
            stats = self._normalize_statistics_response(resp)
            if not isinstance(stats, dict):
                stats = {}
            for entity in entities:
                rows = stats.get(entity)
                if isinstance(rows, list):
                    frames_by_entity[entity].extend(rows)
            cursor = chunk_end

        frames: List[pd.DataFrame] = []
        for entity, rows in frames_by_entity.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if "start" not in df.columns:
                continue
            df["start"] = pd.to_datetime(df["start"], utc=True)
            df = df.set_index("start").sort_index()
            col = type_for_entity.get(entity, "mean")
            if col in df.columns:
                frames.append(df[[col]].rename(columns={col: entity}))

        if frames:
            return pd.concat(frames, axis=1).sort_index()
        return pd.DataFrame()

    def _history_rows_to_series(
        self,
        rows: List[Dict[str, Any]],
        timeline: pd.DatetimeIndex,
        freq: str,
        entity_id: str,
        entity_info: Optional[Dict[str, Any]],
    ) -> pd.Series:
        if not rows:
            return pd.Series(index=timeline, dtype=float)
        df = pd.DataFrame(rows)
        ts_col = None
        for candidate in ("last_updated", "last_changed", "time"):
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None or "state" not in df.columns:
            return pd.Series(index=timeline, dtype=float)
        df["__ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["__ts"])
        if df.empty:
            return pd.Series(index=timeline, dtype=float)
        values = pd.to_numeric(df["state"], errors="coerce")
        series = pd.Series(values.values, index=pd.DatetimeIndex(df["__ts"]))
        series = series[~series.index.duplicated(keep="last")].sort_index()
        freq = freq or getattr(timeline, "freqstr", None) or "5min"
        freq_delta = pd.to_timedelta(freq, errors="coerce")
        if pd.isna(freq_delta) or freq_delta.total_seconds() <= 0:
            freq_delta = pd.Timedelta(minutes=5)
        hours_per_bucket = freq_delta.total_seconds() / 3600.0 or 1.0

        is_cumulative = self._is_cumulative_entity(entity_id, entity_info, series)
        try:
            if is_cumulative:
                resampled = series.resample(freq).ffill()
                diff = resampled.diff().clip(lower=0.0)
                resampled = diff / hours_per_bucket  # type: ignore[operator]
            else:
                resampled = series.resample(freq).mean()
        except ValueError:
            if is_cumulative:
                resampled = series.ffill().diff().clip(lower=0.0) / hours_per_bucket  # type: ignore[operator]
            else:
                resampled = series

        aligned = resampled.reindex(timeline)
        if aligned.notna().sum() == 0:
            aligned = aligned.ffill().fillna(0.0)
        else:
            aligned = aligned.ffill().fillna(0.0)
        return aligned

    @staticmethod
    def _is_cumulative_entity(
        entity_id: str,
        entity_info: Optional[Dict[str, Any]],
        series: Optional[pd.Series] = None,
    ) -> bool:
        attrs: Dict[str, Any] = {}
        if isinstance(entity_info, dict):
            attrs = entity_info.get("attributes") or {}
        state_class = str(attrs.get("state_class") or "").lower()
        if state_class in {"total_increasing", "total"}:
            return True
        device_class = str(attrs.get("device_class") or "").lower()
        if device_class in {"energy", "gas", "water", "monetary"}:
            return True
        unit = str(attrs.get("unit_of_measurement") or "").lower()
        if unit.endswith("wh") or unit.endswith("kwh") or unit.endswith("mwh"):
            return True
        if unit in {"kwh", "wh", "mwh"}:
            return True
        if series is not None and len(series) > 4:
            diffs = series.diff().dropna()
            if len(diffs) > 0:
                non_decreasing = (diffs >= -1e-6).sum() / len(diffs)  # type: ignore[operator]
                if non_decreasing > 0.9:
                    median_step = diffs.abs().median()
                    total_range = series.max() - series.min()
                    if median_step > 0 and total_range > median_step * 5:
                        return True
        return False

    def _get_lock_for_loop(self, attr: str, loop_attr: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock: Optional[asyncio.Lock] = getattr(self, attr)
        lock_loop: Optional[asyncio.AbstractEventLoop] = getattr(self, loop_attr)
        if lock is None or lock_loop is not loop:
            lock = asyncio.Lock()
            setattr(self, attr, lock)
            setattr(self, loop_attr, loop)
        return lock

    async def _get_entity_info_map(self, entity_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        missing = [entity for entity in entity_ids if entity not in self._entity_cache]
        if missing:
            try:
                states = await self.list_entities_async()
            except Exception:
                states = []
            for entry in states:
                if isinstance(entry, dict):
                    eid = entry.get("entity_id")
                    if isinstance(eid, str):
                        self._entity_cache[eid] = entry
        return {entity: self._entity_cache.get(entity) for entity in entity_ids}
        
    async def _ensure_connected(self) -> Any:
        """Ensure WebSocket connection is established and authenticated.
        
        Reuses existing connection if still open, otherwise creates a new one.
        """
        lock = self._get_lock_for_loop("_connection_lock", "_connection_lock_loop")
        async with lock:
            # Re-check connection inside the lock in case another coroutine already reconnected
            if self._ws:
                try:
                    # Check for modern websockets (v11+) state
                    if hasattr(self._ws, "state") and getattr(self._ws.state, "name", "") == "OPEN":
                        return self._ws
                    # Check for legacy websockets open/closed
                    if hasattr(self._ws, "open") and self._ws.open:
                        return self._ws
                    if hasattr(self._ws, "closed") and not self._ws.closed:
                        return self._ws
                except Exception:
                    self._ws = None

            if not self.token:
                raise RuntimeError("No authentication token available")

            # Ensure we connect using the current event loop
            # This is critical for asyncio safety when running in different contexts (e.g. uvicorn vs background thread)
            ws = await websockets.connect(self.ws_url)
            await ws.recv()  # auth required message
            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth_resp = json.loads(await ws.recv())

            if auth_resp.get("type") != "auth_ok":
                await ws.close()
                raise RuntimeError("Authentication failed")

            self._ws = ws
            # Reset message counter on new connection so ids stay consistent
            self._msg_id = 1
            return self._ws
        
    async def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and receive response via WebSocket."""
        # Ensure we are connected on the current loop
        ws = await self._ensure_connected()
        
        # Verify the connection belongs to the current loop to avoid "Future attached to a different loop"
        # If the connection was created on a different loop, we must reconnect.
        try:
            # This is a heuristic check. websockets usually bind to the loop they were created on.
            # If we can't check, we proceed and let it fail if it must.
            if hasattr(ws, "loop") and ws.loop is not asyncio.get_running_loop():
                _LOGGER.warning("WebSocket connection belongs to a different loop. Reconnecting...")
                self._ws = None
                ws = await self._ensure_connected()
        except Exception:
            pass

        lock = self._get_lock_for_loop("_ws_lock", "_ws_lock_loop")
        async with lock:
            if "id" not in message:
                message["id"] = self._msg_id
                self._msg_id += 1

            await ws.send(json.dumps(message))
            return json.loads(await ws.recv())

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a Home Assistant service."""
        msg = {
            "type": "call_service",
            "domain": domain,
            "service": service,
            "service_data": service_data or {},
        }
        if target:
            msg["target"] = target
            
        return await self._send_message(msg)
        
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
        period: str = "hour",
    ) -> pd.DataFrame:
        """Fetch recorder statistics for multiple entities from Home Assistant."""
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
                        "period": period,
                        "start_time": chunk_start.isoformat(),
                        "end_time": chunk_end.isoformat(),
                        "types": requested_types,
                    },
                    "return_response": True,
                }
                
                resp = await self._send_message(req)
                stats = self._normalize_statistics_response(resp)

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
        stats = self._normalize_statistics_response(resp)

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

    @staticmethod
    def _normalize_statistics_response(resp: Any) -> Dict[str, Any]:
        payload = resp
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            return {}
        # payload may already be the statistics dict
        result = payload.get("result", payload)
        if isinstance(result, list):
            result = result[0] if result else {}
        if isinstance(result, dict) and "response" in result:
            result = result["response"]
        if isinstance(result, dict):
            stats = result.get("statistics")
            if isinstance(stats, dict):
                return stats
        return {}

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

    async def get_state_async(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Fetch current state record for a specific entity."""
        if not entity_id:
            return None
        states = await self.list_entities_async()
        for entry in states:
            if isinstance(entry, dict) and entry.get("entity_id") == entity_id:
                return entry
        return None

    def get_entity_state_sync(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.get_state_async(entity_id), self._loop)
            return fut.result()
        return asyncio.run(self.get_state_async(entity_id))

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

    def fetch_history_sync(
        self,
        entities_with_types: List[Tuple[str, str]],
        hours: int = 24,
        period: str = "5minute",
    ) -> pd.DataFrame:
        """Fetch recent entity history via the REST API and resample to the requested period."""
        normalized = self._normalize_entities_with_types(entities_with_types)
        entity_ids = [entity for entity, _typ in normalized]
        if not entity_ids:
            return pd.DataFrame()

        if not self.token:
            raise RuntimeError("No authentication token available for history fetch")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=max(1, int(hours or 24)))
        freq = self._period_to_freq(period)
        timeline = pd.date_range(start=start_time, end=end_time, freq=freq, tz=timezone.utc)
        history_url = f"{self.rest_url.rstrip('/')}/history/period/{start_time.isoformat()}"
        params = {
            "filter_entity_id": ",".join(entity_ids),
            "minimal_response": "true",
            "end_time": end_time.isoformat(),
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            with requests.Session() as session:
                with session.get(history_url, headers=headers, params=params, timeout=30) as resp:
                    resp.raise_for_status()
                    payload = resp.json()
        except Exception as exc:
            _LOGGER.error("Failed to download Home Assistant history: %s", exc)
            raise

        history_map = self._normalize_history_response(payload)

        # Build unified rows list (entity_id, state, timestamp)
        rows = []
        for ent in entity_ids:
            ent_rows = history_map.get(ent, [])
            for item in ent_rows:
                ts = item.get("last_changed") or item.get("last_updated") or item.get("time")
                rows.append({"entity_id": ent, "state": item.get("state"), "timestamp": ts})

        if not rows:
            return pd.DataFrame(index=timeline)

        df = pd.DataFrame(rows)
        # normalize types similar to tools/get_history.py example
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])  # drop rows without valid timestamp
        if df.empty:
            return pd.DataFrame(index=timeline)

        # Prefer numeric states where possible; non-numeric become NaN
        df["state"] = pd.to_numeric(df["state"], errors="coerce")

        # Pivot so each entity is a column with timestamp index
        df_pivot = (
            df.pivot_table(index="timestamp", columns="entity_id", values="state", aggfunc="mean")
            .sort_index()
        )

        if df_pivot.empty:
            return pd.DataFrame(index=timeline)

        # Use the requested timeline as target bins (already tz-aware)
        bins = timeline
        left = pd.DataFrame({"timestamp": bins})

        # For each entity, pick the latest reading <= bin timestamp using merge_asof
        out_df = pd.DataFrame(index=bins, columns=df_pivot.columns, dtype=float)
        for col in df_pivot.columns:
            ser = df_pivot[col].dropna().reset_index()
            if ser.empty:
                continue
            ser.columns = ["timestamp", "value"]
            ser = ser.sort_values("timestamp")
            merged = pd.merge_asof(left, ser, on="timestamp", direction="backward")
            out_df[col] = merged["value"].values

        # If no values available at all for a column, it'll remain NaN; caller can ffill if desired
        return out_df

    def fetch_statistics_sync(
        self,
        entities_with_types: List[Tuple[str, str]],
        days: int,
        chunk_days: int = 30,
        period: str = "hour",
    ) -> pd.DataFrame:
        """Synchronous wrapper for fetch_statistics_async."""
        # Preferred: run the async method on the background loop that is kept
        # alive for the lifetime of this HomeAssistant instance. If no loop is
        # active, fall back to running it with asyncio.run (short-lived loop).
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_statistics_async(entities_with_types, days, chunk_days, period), self._loop
            )
            return fut.result()
        return asyncio.run(self.fetch_statistics_async(entities_with_types, days, chunk_days, period))



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
        if self._loop is None:
            raise RuntimeError("Failed to start background loop")
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
        try:
            await self._ws.close()
        except Exception:
            # Ignore errors during close
            pass
        finally:
            self._ws = None
