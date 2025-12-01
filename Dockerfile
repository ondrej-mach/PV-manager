ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-alpine3.19
FROM $BUILD_FROM

WORKDIR /app

# Install Python dependencies and build tools
RUN \
    apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    g++ \
    gfortran \
    lapack-dev \
    openblas-dev \
    cmake \
    make \
    git

# Install Python packages
# Set CFLAGS to suppress warnings that cause build failures in CVXcanon/Eigen
COPY requirements.txt ./
RUN CFLAGS="-Wno-int-in-bool-context -Wno-deprecated-declarations" \
    CXXFLAGS="-Wno-int-in-bool-context -Wno-deprecated-declarations" \
    pip install --no-cache-dir -r requirements.txt

# Clean up build dependencies to reduce image size (keep runtime libraries)
RUN apk del gcc musl-dev python3-dev libffi-dev openssl-dev g++ gfortran cmake make git lapack-dev openblas-dev && \
    apk add --no-cache openblas lapack

# Copy application and library code
COPY app/ /app/app/
COPY src/ /app/src/

# Create necessary directories for runtime data
RUN mkdir -p /data/trained_models /data/output /data/output/debug

ENV PYTHONPATH="/app/src"

CMD [ "python3", "/app/app/main.py" ]