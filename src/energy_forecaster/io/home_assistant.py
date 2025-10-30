
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
        
    async def _ensure_connected(self) -> WebSocketClientProtocol:
        """Ensure WebSocket connection is established and authenticated."""
        if self._ws:
            # Some websocket client implementations may not expose a `closed`
            # attribute. Be defensive: if the attribute exists and is False,
            # reuse the connection; if it doesn't exist, assume the connection
            # object is still valid and reuse it.
            try:
                if not getattr(self._ws, "closed"):
                    return self._ws
            except Exception:
                return self._ws
            
        if not self.token:
            raise RuntimeError("No authentication token available")
            
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
                result = resp.get("result", {})
                # Close the connection after fetching config to avoid
                # leaving a WebSocket object attached to a now-finished
                # asyncio loop (this prevents cross-loop Future errors
                # when callers use `asyncio.run` multiple times).
                try:
                    if self._ws:
                        close_coro = getattr(self._ws, "close", None)
                        if callable(close_coro):
                            res = close_coro()
                            if asyncio.iscoroutine(res):
                                await res
                except Exception:
                    # Ignore close errors here; we'll reset the attribute
                    pass
                finally:
                    self._ws = None

                return result
            return None
        except Exception:
            # Connection error - reset connection for next attempt
            if self._ws:
                await self._ws.close()
                self._ws = None
            raise

    def get_location(self, 
                   fallback_lat: float = 49.6069, 
                   fallback_lon: float = 15.5808, 
                   fallback_tz: str = "Europe/Prague") -> Tuple[float, float, str]:
        """Get coordinates and timezone from Home Assistant if available, otherwise use fallbacks."""
        # Synchronous façade: run fetch_config_async on the background loop if available,
        # otherwise run it in a fresh loop (rare - better to call connect_sync first).
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self.fetch_config_async(), self._loop)
            try:
                config = fut.result()
            except FutureTimeoutError:
                config = None
        else:
            config = asyncio.run(self.fetch_config_async())
        return (
            config.get("latitude", fallback_lat) if config else fallback_lat,
            config.get("longitude", fallback_lon) if config else fallback_lon,
            config.get("time_zone", fallback_tz) if config else fallback_tz
        )

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
        finally:
            # Always close connection after bulk operations
            if self._ws:
                await self._ws.close()
                self._ws = None

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
        """Close the WebSocket connection if it's open."""
        if not self._ws:
            return

        # Some connection objects expose a coroutine `close()` but not a
        # `closed` attribute. Be defensive: if a `close` callable exists,
        # call it and await if it returns a coroutine. Finally unset
        # the stored connection.
        close_attr = getattr(self._ws, "close", None)
        if callable(close_attr):
            try:
                result = close_attr()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Ignore errors during close; connection is being discarded.
                pass

        self._ws = None
