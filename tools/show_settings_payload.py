import asyncio
import os

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
        print(f"statistics count: {len(payload.get('statistics', []))}")
        for entry in payload.get("statistics", [])[:5]:
            print(entry.get("statistic_id"))
        print("house entry:", next((item for item in payload.get("statistics", []) if item.get("statistic_id") == "sensor.house_consumption"), None))
        print("pv entry:", next((item for item in payload.get("statistics", []) if item.get("statistic_id") == "sensor.pv_power"), None))
        await ctx.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
