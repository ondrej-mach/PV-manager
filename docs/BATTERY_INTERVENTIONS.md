# Battery Intervention Types

This document defines the four types of battery interventions that the PV Manager system can command, and explains how to determine which intervention is needed based on the optimization plan's battery flow data.

## 1. Intervention Types

### 1.1 Cover Load from Battery
**Description**: Battery discharges to help cover house consumption. PV may also contribute. Grid import is minimized.

**Characteristics**:
- Battery provides power to household loads
- PV may also provide power to loads (pv_to_load_kw)
- Grid may provide power to loads (grid_to_load_kw)
- Battery is NOT being charged from grid
- Battery is NOT exporting to grid
- SoC decreases
- Total load balance: `pv_to_load_kw + batt_to_load_kw + grid_to_load_kw = load_kw`

**Mathematical Check**:
```
batt_to_load_kw > MIN_POWER_KW  # Battery is discharging to load
batt_to_grid_kw ≈ 0              # Not exporting to grid
grid_to_batt_kw ≈ 0              # Not charging from grid
SoC decreases
```

### 1.2 Charge from Grid
**Description**: All load is covered from grid, battery charges with specific power from grid.

**Characteristics**:
- Grid provides power to both household and battery
- Battery energy in > battery energy out
- SoC rises
- Grid import is high

**Mathematical Check**:
```
grid_to_batt_kw > 0
batt_to_load_kw ≈ 0
batt_to_grid_kw ≈ 0
SoC increases
```

### 1.3 Discharge to Grid
**Description**: All load is covered from battery, and battery discharges additional power to grid (export).

**Characteristics**:
- Battery provides power to both household loads AND grid export
- Battery energy out exceeds house consumption
- SoC decreases significantly
- Grid export is high

**Mathematical Check**:
```
batt_to_grid_kw > 0
batt_to_load_kw > 0
(batt_to_load_kw + batt_to_grid_kw) > load_kw
SoC decreases
```

### 1.4 Disable Battery
**Description**: All load is covered from grid, battery neither charges nor discharges.

**Characteristics**:
- Battery flows in and out are approximately equal (accounting for self-discharge/efficiency)
- SoC remains stable
- Grid provides all power to household

**Mathematical Check**:
```
grid_to_batt_kw ≈ 0
batt_to_load_kw ≈ 0
batt_to_grid_kw ≈ 0
SoC stable (~ constant)
```

## 2. Determining Intervention from Optimization Plan

The optimization plan DataFrame from `solve_lp_optimal_control` contains the following relevant columns:

- `pv_kw`: PV generation
- `load_kw`: House consumption
- `pv_to_load_kw`: PV directly to load
- `pv_to_batt_kw`: PV directly to battery
- `pv_to_grid_kw`: PV directly to grid export
- `grid_to_load_kw`: Grid to load
- `grid_to_batt_kw`: Grid to battery (grid charging)
- `batt_to_load_kw`: Battery to load
- `batt_to_grid_kw`: Battery to grid export
- `grid_import_kw`: Total grid import
- `grid_export_kw`: Total grid export
- `soc`: State of charge (0-1)

### 2.1 Decision Logic

```python
# Energy flow analysis (for a single timestep)
battery_charge_kw = grid_to_batt_kw + pv_to_batt_kw
battery_discharge_kw = batt_to_load_kw + batt_to_grid_kw

# Thresholds (accounting for efficiencies)
EFFICIENCY_TOLERANCE = 0.05  # 5%
MIN_POWER_KW = 0.05  # 50W minimum threshold

if battery_charge_kw < MIN_POWER_KW and battery_discharge_kw < MIN_POWER_KW:
    intervention = "DISABLE_BATTERY"
    
elif grid_to_batt_kw > MIN_POWER_KW:
    # Battery is being charged from grid
    intervention = "CHARGE_FROM_GRID"
    target_power_kw = grid_to_batt_kw
    
elif batt_to_grid_kw > MIN_POWER_KW:
    # Battery is discharging to grid
    intervention = "DISCHARGE_TO_GRID"
    target_power_kw = batt_to_grid_kw
    
elif batt_to_load_kw > MIN_POWER_KW:
    # Battery is covering load
    intervention = "COVER_LOAD"
    target_power_kw = batt_to_load_kw
    
else:
    intervention = "DISABLE_BATTERY"
```

## 3. Mapping to GoodWe Inverter Modes

Based on the available GoodWe modes:
- **General mode** - Normal self-consumption
- Off grid mode
- Backup mode
- Eco mode
- Peak shaving mode
- **Eco charge mode** - Used for "Charge from Grid"
- **Eco discharge mode** - Used for "Discharge to Grid"

### Mode Mapping:

| Intervention | GoodWe Mode | Eco Power % | Eco SoC % | Notes |
|--------------|-------------|-------------|-----------|-------|
| Disable Battery | General mode | N/A | N/A | Normal self-consumption |
| Cover Load | General mode | N/A | N/A | Let inverter manage naturally |
| Charge from Grid | Eco charge mode | (target_kw/max_kw)*100 | 100 | Charge until full |
| Discharge to Grid | Eco discharge mode | (target_kw/max_kw)*100 | 0 | Discharge until empty |

### Notes:
- **General mode**: Normal self-consumption mode where the inverter automatically uses PV and battery to minimize grid import
- **Eco charge mode**: Forces grid charging with specified power limit
- **Eco discharge mode**: Forces grid export with specified power limit
- Power percentage is calculated as: `(target_power_kw / max_inverter_power_kw) * 100`
- SoC limit of 100% means "charge until full", 0% means "discharge until empty"

## 4. Implementation Reference

See `app/pv_manager/drivers.py` - `GoodWeDriver.apply_control()` for the actual implementation of this logic.
