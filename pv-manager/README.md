# PV Manager

> [!CAUTION]
> **Active Development Warning**
> This add-on is currently in active development. Features may break, configuration formats may change, and strict backward compatibility is not guaranteed.
> **Do not use in production environments** unless you are actively developing or testing the system.

Smart energy manager for home photovoltaic systems with battery storage.

## Features

- **Automatic battery control** - Optimizes battery charging/discharging based on electricity prices and forecasts
- **ML-based forecasting** - Predicts PV generation and household consumption
- **Price-aware optimization** - Works with spot pricing and various tariff structures
- **Home Assistant integration** - Native add-on with full HA integration
- **Flexible configuration** - Supports multiple inverter drivers and customizable settings

## Installation

1. Add this repository to your Home Assistant add-on store
2. Install the PV Manager add-on
3. Configure your Home Assistant entities in the add-on UI
4. Select your inverter driver
5. Enable automatic control

## Configuration

The add-on uses ingress for its web interface - no manual port configuration needed.

All configuration is done through the web UI:
- **HA Entities**: Configure sensors for PV power, consumption, and battery SoC
- **Battery**: Set capacity, power limits, and wear costs
- **Grid**: Configure import/export tariffs (constant, spot-based, or dual-rate)
- **Training**: Enable automatic nightly model retraining

## Requirements

- Home Assistant with Supervisor
- Battery storage system with supported inverter
- PV power sensor
- House consumption sensor
- Battery SoC sensor

## Support

For issues and feature requests, visit the [GitHub repository](https://github.com/ondrej-mach/PV-manager).
