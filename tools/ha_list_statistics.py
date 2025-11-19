import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import websockets

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _build_ws_url(http_url: str) -> str:
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc}/api/websocket"


async def main() -> None:
    http_url = os.getenv("HA_HTTP_URL", "http://homeassistant.lan:8123")
    token = os.getenv("HASS_TOKEN")
    if not token:
        logger.error("HASS_TOKEN environment variable is not set")
        raise SystemExit(1)

    ws_url = _build_ws_url(http_url)
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth = json.loads(await ws.recv())
        if auth.get("type") != "auth_ok":
            logger.error("Authentication failed: %s", auth)
            raise SystemExit(1)

        request = {"id": 1, "type": "recorder/list_statistic_ids"}
        await ws.send(json.dumps(request))
        response = json.loads(await ws.recv())
    logger.info(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
