# PV Manager Add-on Testing Guide

This project now ships as a Home Assistant add-on. You can exercise the add-on image without
pushing it to the Supervisor by building and running the container locally.

## 1. Build the add-on image

```bash
# From the repository root
docker build \
  --build-arg BUILD_FROM="ghcr.io/home-assistant/amd64-base:3.19" \
  -t pv-manager-addon .
```

Pick the `BUILD_FROM` value that matches your Home Assistant architecture
(`amd64`, `aarch64`, `armv7`, ...). When building inside Home Assistant, the
Supervisor sets this automatically.

## 2. Run the add-on locally

Expose the HTTP ingress port (`8099` by default) and mount a volume for cached
benchmark data so runs persist between tests.

```bash
docker run --rm -it \
  -p 8099:8099 \
  -v "$(pwd)/benchmark_data:/app/benchmark_data" \
  -v "$(pwd)/output:/app/output" \
  pv-manager-addon
```

Navigate to [http://localhost:8099](http://localhost:8099) to open the add-on UI.
The status panel calls `/api/status` to verify cached benchmark data and the
latest optimization summary. After the first forecast cycle completes, the page
renders charts for the predicted load/PV profiles and the battery dispatch
plan.

## 3. Test inside Home Assistant

1. Copy the repository into Home Assistant's `addons` folder (either via SSH,
   Samba, or the VS Code add-on).
2. In Home Assistant, open *Settings → Add-ons → Add-on Store → ⋮ → Repositories*
   and add the path to your cloned repository.
3. Install the *PV Manager* add-on and start it. Ingress should open the same
   frontend as the local container.

## 4. Functional smoke tests

* Ensure `/api/status` returns `{"ok": true}` and reports the benchmark cache
  window after running any of the analysis scripts in `tools/`.
* Confirm the frontend updates when cached data becomes available.
* Run the helper scripts from the host machine (for example
  `python tools/benchmark.py`) to verify the production library is unaffected by
  the add-on packaging.

These steps provide a repeatable workflow on Linux with Docker. When the add-on
is ready for Home Assistant, copy `config.yaml`, `Dockerfile`, and the `app/`
folder into `addons/pv-manager/` and rebuild through the Supervisor UI.
