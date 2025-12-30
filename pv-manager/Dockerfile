# Hardcoded Debian base to ignore HA build args (which force Alpine)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# build-essential: for compiling any missing wheels
# git: for potential git dependencies
# libgomp1: required for scikit-learn/xgboost OpenMP support
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Clean up build dependencies if minimizing size is desired,
# but keeping them ensures stability if new pip packages are added.
# For now, we clean aptitude cache only (done above).

# Copy application and library code
COPY app/ /app/app/
COPY src/ /app/src/

# Create necessary directories for runtime data
RUN mkdir -p /data/trained_models /data/output /data/output/debug

ENV PYTHONPATH="/app/src"

CMD [ "python3", "/app/app/main.py" ]