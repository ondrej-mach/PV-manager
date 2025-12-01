# PV Manager Documentation

## Overview

PV Manager is a Home Assistant add-on that optimizes battery storage operations for residential photovoltaic systems. It uses machine learning to forecast energy production and consumption, then applies linear programming to determine optimal battery charge/discharge schedules based on electricity prices.

## Initial Setup

### 1. Configure Home Assistant Entities

Navigate to **Settings → HA Entities** and configure:

- **Inverter Driver**: Select your inverter type from the dropdown
- **PV Power**: Entity measuring photovoltaic generation (W or kW)
- **House Consumption**: Entity measuring household energy consumption (W or kW)
- **Battery SoC**: Entity measuring battery state of charge (%)

Click "Save Configuration" after entering your entities.

### 2. Battery Settings

Configure your battery parameters in **Settings → Battery**:

- **Usable capacity (kWh)**: Total usable energy storage
- **Charge/discharge limit (kW)**: Maximum power rating
- **Reserve floor (% SoC)**: Minimum charge to maintain for backup
- **Maximum usable SoC (%)**: Upper limit to reduce wear
- **Battery wear cost (€/kWh)**: Degradation cost per throughput cycle

### 3. Grid Settings

Configure your electricity tariff in **Settings → Grid**:

#### Import Tariff
- **Constant price**: Fixed rate (€/kWh)
- **Spot price with offset**: Dynamic pricing + fixed markup
- **Dual-rate schedule**: Different rates for peak/off-peak hours

#### Export Settings
- Enable/disable export to grid
- Set export power limit if applicable
- Configure export tariff structure

### 4. Training

Go to **Settings → Training**:

- Enable **automatic nightly training** to keep models up-to-date
- Manual training can be triggered from the main dashboard

## Operation

### Automatic Control

1. Ensure entities are configured correctly
2. Run initial training (button on main dashboard)
3. Toggle "Automatic control" switch on
4. The system will update every 15 minutes with new optimization plans

### Understanding the Dashboard

**System Overview**
- Import/Export energy totals
- Net cost for current window
- PV generation and consumption

**Forecast vs Load**
- Predicted PV generation (orange)
- Predicted house consumption (blue)

**Optimization Plan**
- Battery state of charge trajectory
- Electricity price timeline
- Grid import/export schedule

### Intervention Banner

Shows the current battery control action:
- **Charge**: Actively charging the battery
- **Discharge**: Discharging to supply load or export
- **Idle**: No active control

## Advanced Configuration

### Battery Wear Cost

Set the degradation cost to balance arbitrage profit against battery lifetime:
- Higher values → less cycling, longer battery life
- Lower values → more arbitrage, potentially faster degradation
- Typical range: €0.05-0.15/kWh

### Reserve Settings

- **Hard reserve**: Absolute minimum SoC (for backup)
- **Soft reserve**: Encouraged minimum with penalty (configurable via environment variables)

### Export Limiting

Enable "PV power is limited by the inverter" if your system throttles generation when:
- Batteries are full
- Export is disabled/limited
- This improves forecast accuracy

## Troubleshooting

### "Training unavailable until Home Assistant connects"
- Check that the add-on can reach the Home Assistant API
- Verify SUPERVISOR_TOKEN is available (automatic in add-on mode)

### "No inverter driver configured"
- Select a driver in Settings → HA Entities
- Automatic control requires a configured driver

### Poor Forecast Accuracy
- Ensure you have at least 2-3 weeks of historical data
- Run training after significant system changes
- Check that entities are reporting correct units (W or kW)

### Battery Not Charging from Surplus
- Check battery reserve settings aren't too high
- Verify export is disabled if you want to prioritize storage
- Review battery wear cost (high values discourage charging)

## API Endpoints

The add-on exposes a REST API for automation:

- `GET /api/status` - Current system status
- `GET /api/forecast` - Latest forecast and optimization plan
- `POST /api/training` - Trigger model training
- `POST /api/cycle` - Recompute optimization
- `POST /api/control` - Enable/disable automatic control
- `GET /api/settings` - Get current settings
- `PATCH /api/settings` - Update settings

## Environment Variables

Optional environment variables for advanced tuning:

- `BATTERY_CAP_KWH` - Override battery capacity
- `BATTERY_P_MAX_KW` - Override power limit
- `RESERVE_SOC_HARD` - Hard minimum SoC (fraction)
- `RESERVE_SOC_SOFT` - Soft minimum SoC (fraction)
- `RESERVE_SOFT_PENALTY_EUR_PER_KWH` - Penalty for soft reserve violations
- `DEBUG_DIR` - Enable debug CSV output to directory

## Notes

- The optimizer runs every 15 minutes by default
- Models retrain at 02:00 UTC when automatic training is enabled
- All times in the UI are in your Home Assistant's configured timezone
- The add-on requires continuous operation for effective control
