import asyncio
import os

from app.pv_manager.state import AppContext


def main() -> None:
    token = os.getenv("HASS_TOKEN")
    if token and not os.getenv("SUPERVISOR_TOKEN"):
        os.environ["SUPERVISOR_TOKEN"] = token
    os.environ.setdefault("HASS_WS_URL", "ws://homeassistant.lan:8123/api/websocket")
    ctx = AppContext()

    async def run():
        await ctx._ensure_home_assistant()  # pylint: disable=protected-access
        catalog = await ctx.refresh_statistics_catalog()
        print(f"statistics fetched: {len(catalog)}")
        ids = {item.get("statistic_id") for item in catalog if item.get("statistic_id")}
        for stat_id in ["sensor.house_consumption", "sensor.pv_power"]:
            print(f"{stat_id}:", stat_id in ids)

    asyncio.run(run())


if __name__ == "__main__":
    main()
