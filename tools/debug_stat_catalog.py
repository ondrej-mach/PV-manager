import asyncio
import logging
import os
import sys
from pathlib import Path

from app.pv_manager.state import AppContext

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    token = os.getenv("HASS_TOKEN")
    if token and not os.getenv("SUPERVISOR_TOKEN"):
        os.environ["SUPERVISOR_TOKEN"] = token
    os.environ.setdefault("HASS_WS_URL", "ws://homeassistant.lan:8123/api/websocket")
    ctx = AppContext()

    async def run():
        await ctx._ensure_home_assistant()  # pylint: disable=protected-access
        catalog = await ctx.refresh_statistics_catalog()
        logger.info("statistics fetched: %s", len(catalog))
        ids = {item.get("statistic_id") for item in catalog if item.get("statistic_id")}
        for stat_id in ["sensor.house_consumption", "sensor.pv_power"]:
            logger.info("%s present: %s", stat_id, stat_id in ids)

    asyncio.run(run())


if __name__ == "__main__":
    main()
