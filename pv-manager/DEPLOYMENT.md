# PV Manager - Deployment Checklist

## ✅ Configuration Files

### Core Add-on Files
- [x] `config.yaml` - Add-on configuration with ingress, arch support, and metadata
- [x] `build.yaml` - Multi-architecture build configuration  
- [x] `Dockerfile` - Container build instructions with all dependencies
- [x] `requirements.txt` - Python dependencies
- [x] `.dockerignore` - Excludes unnecessary files from build

### Documentation
- [x] `README.md` - Add-on overview and installation
- [x] `DOCS.md` - Comprehensive user documentation
- [x] `CHANGELOG.md` - Version history

## ✅ Code Configuration

### Home Assistant Integration
- [x] **SUPERVISOR_TOKEN** - Automatically provided by HA, used in `home_assistant.py`
- [x] **WebSocket URL** - Defaults to `ws://supervisor/core/websocket`
- [x] **REST API URL** - Defaults to `http://supervisor/core/api`
- [x] **Ingress** - Enabled on port 8099, configured in `config.yaml` and `main.py`

### Data Persistence
- [x] **/data directory** - Automatically detected in `state.py`
- [x] **trained_models/** - Stored in `/data/trained_models`
- [x] **State files** - `control_state.json`, `inverter_driver.json` in `/data`
- [x] **Settings** - `settings.json` in app directory (survives restarts)

### Directory Creation
- [x] Dockerfile creates `/data/trained_models`, `/data/output`, `/data/output/debug`
- [x] `AppContext.__init__` creates models_dir if missing
- [x] `settings.py` creates parent directory for settings.json

## ✅ Dependencies

### Python Packages (requirements.txt)
- [x] pandas - Data manipulation
- [x] numpy - Numerical computing
- [x] scikit-learn - ML framework
- [x] xgboost - Gradient boosting
- [x] joblib - Model serialization
- [x] cvxpy - Linear programming optimizer
- [x] entsoe-py - Spot price API
- [x] fastapi - Web framework
- [x] uvicorn - ASGI server
- [x] websockets - HA WebSocket client
- [x] openmeteo-requests - Weather API
- [x] requests, requests-cache, retry-requests - HTTP utilities
- [x] jinja2 - Template rendering
- [x] matplotlib - Plotting (used for benchmarks)

### System Dependencies (Dockerfile)
- [x] python3, py3-pip - Python runtime
- [x] gcc, musl-dev, python3-dev - Build tools
- [x] libffi-dev, openssl-dev - Crypto libraries
- [x] g++, gfortran - C++ and Fortran compilers (for numpy/scipy)
- [x] lapack-dev - Linear algebra (for cvxpy)

## ✅ Runtime Configuration

### Environment Variables (Auto-provided by HA)
- [x] `SUPERVISOR_TOKEN` - HA authentication
- [x] `DATA_DIR` - Optional, defaults to `/data`

### Optional Configuration (Environment Variables)
- [ ] `UVICORN_HOST` - Default: `0.0.0.0`
- [ ] `UVICORN_PORT` - Default: `8099`
- [ ] `UVICORN_LOG_LEVEL` - Default: `info`
- [ ] `DEBUG_DIR` - Enable debug CSV output
- [ ] `BATTERY_CAP_KWH` - Override battery capacity
- [ ] `BATTERY_P_MAX_KW` - Override power limit
- [ ] `RESERVE_SOC_HARD` - Hard minimum SoC
- [ ] `RESERVE_SOC_SOFT` - Soft minimum SoC
- [ ] `RESERVE_SOFT_PENALTY_EUR_PER_KWH` - Soft reserve penalty

## 🔧 Deployment Steps

### 1. Local Testing (Optional)
```bash
# Build the Docker image locally
docker build -t pv-manager-test .

# Test run (requires HA instance)
docker run --rm \
  -e SUPERVISOR_TOKEN=your_token \
  -v /path/to/data:/data \
  -p 8099:8099 \
  pv-manager-test
```

### 2. Add Repository to Home Assistant
1. Go to **Settings → Add-ons → Add-on Store**
2. Click **⋮** (top right) → **Repositories**
3. Add your repository URL
4. Click **Add**

### 3. Install Add-on
1. Find "PV Manager" in the add-on store
2. Click **Install**
3. Wait for build to complete (may take 10-20 minutes)

### 4. Start Add-on
1. Go to add-on configuration page
2. Click **Start**
3. Enable **Start on boot** if desired
4. Enable **Watchdog** for automatic restart on crash

### 5. Access Web Interface
1. Click **Open Web UI** (ingress)
2. Or navigate to sidebar → PV Manager

### 6. Initial Configuration
1. **Settings → HA Entities**
   - Select inverter driver
   - Configure PV power sensor
   - Configure consumption sensor
   - Configure battery SoC sensor
   - Click **Save Configuration**

2. **Settings → Battery**
   - Set usable capacity (kWh)
   - Set charge/discharge limit (kW)
   - Configure reserve and max SoC
   - Set wear cost

3. **Settings → Grid**
   - Configure import tariff
   - Enable/disable export
   - Configure export tariff (if enabled)

4. **Settings → Training**
   - Enable automatic nightly training

5. **Main Dashboard**
   - Click **Trigger Training** (wait for completion)
   - Enable **Automatic control**

## ⚠️ Common Issues

### Build Failures
- **Out of memory** - Increase Docker memory limit (needs ~2GB for build)
- **Missing dependencies** - Check Dockerfile has all required system packages
- **Architecture not supported** - Remove unsupported arch from config.yaml

### Runtime Issues
- **Can't connect to HA** - Check SUPERVISOR_TOKEN is available
- **No data persistence** - Verify /data directory is mounted
- **Training fails** - Check at least 2-3 weeks of historical data exists
- **Control doesn't work** - Verify inverter driver is configured

### Permission Issues
- Add-on runs as user `root` in container (normal for HA add-ons)
- /data directory is automatically managed by Supervisor
- No manual permission configuration needed

## 📊 Post-Deployment Verification

### Check Logs
```
Settings → Add-ons → PV Manager → Log
```
Look for:
- `INFO: Started server process` - Server started
- `INFO: Waiting for application startup` - Initialization
- `Training in progress` - Models training
- `[FORECAST] ...` - Cycle running

### Check Data Directory
In container shell or from add-on logs:
```bash
ls -la /data/
ls -la /data/trained_models/
```
Should see:
- `trained_models/house_consumption.joblib`
- `trained_models/pv_power.joblib`
- `control_state.json`
- `inverter_driver.json`
- `settings.json` (in app directory)

### Test API
From HA Developer Tools → Services:
```yaml
service: rest_command.pv_status
```
Or use curl:
```bash
curl http://localhost:8099/api/status
```

## 🚀 Ready for Production

Once verified:
1. ✅ Add-on starts without errors
2. ✅ Web interface accessible
3. ✅ Can connect to HA (no "home_assistant_error")
4. ✅ Training completes successfully
5. ✅ Forecast shows data
6. ✅ Optimization plan displays
7. ✅ Automatic control toggle works
8. ✅ Data persists after restart

Your add-on is ready for use! 🎉
