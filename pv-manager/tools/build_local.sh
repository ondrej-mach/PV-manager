#!/bin/bash
# Stop on error
set -e

# Change to project root (one level up from tools)
cd "$(dirname "$0")/.."

echo "Building PV Manager Docker image locally..."
# Build with local tag
docker build -t pv-manager-local .

echo "Build complete! Image tagged as 'pv-manager-local'"
