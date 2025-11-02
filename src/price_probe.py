import os
import sys
import pandas as pd

src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz


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
        print("[ENTSOE] Price fetch failed:", msg)
        print("Hint: Ensure the token file exists at ENTSOE_TOKEN_PATH (default /home/ondra/Documents/VUT/DIP/data/ENTSO-E_token.txt) and the country code/tz are correct.")
        return

    print("[ENTSOE] Retrieved prices:")
    print(prices.head())
    print("…")
    print(prices.tail())
    print(f"Count: {len(prices)}, range: {prices['price_eur_per_kwh'].min():.4f}–{prices['price_eur_per_kwh'].max():.4f} €/kWh")


if __name__ == "__main__":
    main()
