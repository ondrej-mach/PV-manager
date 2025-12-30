#!/bin/bash
set -e

docker stop pv-manager-dev 2>/dev/null || true
docker rm pv-manager-dev 2>/dev/null || true

docker run -it \
    --name pv-manager-dev \
    --entrypoint python3 \
    -p "8099:8099" \
    -e SUPERVISOR_TOKEN="$HASS_TOKEN" \
    -e HASS_SERVER="http://homeassistant.lan:8123/" \
    -e DEBUG_DIR="/data/output/debug" \
    -e LOG_LEVEL="DEBUG" \
    -e ENTSOE_TOKEN="$(cat /home/ondra/Documents/VUT/DIP/ENTSO-E_token.txt)" \
    pv-manager-local \
    /app/app/main.py
