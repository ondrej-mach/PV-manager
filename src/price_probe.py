import logging
import os
import sys
import pandas as pd

src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz
from energy_forecaster.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def main():
    # Choose a window covering today+2 days in local tz
    tz = os.getenv("TZ", "Europe/Prague")
    country = os.getenv("ENTSOE_COUNTRY_CODE") or guess_country_code_from_tz(tz)
    now_local = pd.Timestamp.now(tz=tz)
    start = now_local.floor("D")
    end = start + pd.Timedelta(days=3)

    try:
        prices = fetch_day_ahead_prices_country(country_code=country, start=start, end=end, tz=tz)
    except Exception as e:
        # Avoid printing the full URL or token; provide high-level diagnostics only
        msg = str(e)
        logger.error("[ENTSOE] Price fetch failed: %s", msg)
        logger.info(
            "Hint: Ensure ENTSOE token exists at ENTSOE_TOKEN_PATH and the country code/tz are correct"
        )
        return

    logger.info("[ENTSOE] Retrieved prices:")
    logger.info("%s", prices.head())
    logger.info("…")
    logger.info("%s", prices.tail())
    logger.info(
        "Count: %s, range: %.4f–%.4f €/kWh",
        len(prices),
        prices['price_eur_per_kwh'].min(),
        prices['price_eur_per_kwh'].max(),
    )


if __name__ == "__main__":
    main()
