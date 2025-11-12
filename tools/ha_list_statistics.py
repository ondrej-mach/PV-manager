import asyncio
import json
import os
import sys
from urllib.parse import urlparse

import websockets


def _build_ws_url(http_url: str) -> str:
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc}/api/websocket"


async def main() -> None:
    http_url = os.getenv("HA_HTTP_URL", "http://homeassistant.lan:8123")
    token = os.getenv("HASS_TOKEN")
    if not token:
        print("HASS_TOKEN environment variable is not set", file=sys.stderr)
        raise SystemExit(1)

    ws_url = _build_ws_url(http_url)
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth = json.loads(await ws.recv())
        if auth.get("type") != "auth_ok":
            print(f"Authentication failed: {auth}", file=sys.stderr)
            raise SystemExit(1)

        request = {"id": 1, "type": "recorder/list_statistic_ids"}
        await ws.send(json.dumps(request))
        response = json.loads(await ws.recv())
        print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
