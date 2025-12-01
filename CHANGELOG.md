# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2025-11-30

### Added
- Complete Home Assistant add-on configuration
- Automatic /data directory detection for persistent storage
- Build configuration for multi-architecture support
- Comprehensive documentation (README.md and DOCS.md)
- .dockerignore for faster builds

### Fixed
- Terminal value calculation now properly accounts for discharge efficiency
- Type safety improvements in optimization code
- Proper null checks for cvxpy variable values

### Changed
- Moved automatic training toggle to Settings → Training section
- Updated CSS with better dark mode support and organization
- Models and state files now stored in /data directory when running as add-on

## [0.1.2] - 2025-11-26

### Fixed
- Optimization now allows battery charging from PV surplus
- Terminal SoC constraint relaxed from equality to inequality
- Dark mode fixes for HA Entities table

### Added
- CSS refactoring with semantic color variables
- Improved terminal value calculation in optimization

## [0.1.1] - 2025-11-25

### Initial Release
- ML-based PV and consumption forecasting
- Linear programming optimization for battery control
- Web-based configuration interface
- Support for multiple inverter drivers
- Spot pricing and various tariff structures
