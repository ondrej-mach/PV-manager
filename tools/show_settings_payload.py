import asyncio
import logging
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from app.pv_manager.state import AppContext


def main() -> None:
    token = os.getenv("HASS_TOKEN")
    if token and not os.getenv("SUPERVISOR_TOKEN"):
        os.environ["SUPERVISOR_TOKEN"] = token
    os.environ.setdefault("HASS_WS_URL", "ws://homeassistant.lan:8123/api/websocket")

    ctx = AppContext()

    async def run() -> None:
        await ctx.start()
        payload = await ctx.get_settings_payload()
        logger.info("statistics count: %s", len(payload.get('statistics', [])))
        for entry in payload.get("statistics", [])[:5]:
            logger.info("statistic: %s", entry.get("statistic_id"))
        logger.info(
            "house entry: %s",
            next((item for item in payload.get("statistics", []) if item.get("statistic_id") == "sensor.house_consumption"), None),
        )
        logger.info(
            "pv entry: %s",
            next((item for item in payload.get("statistics", []) if item.get("statistic_id") == "sensor.pv_power"), None),
        )
        await ctx.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
