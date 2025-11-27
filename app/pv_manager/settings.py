from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from energy_forecaster.features.data_prep import PV_COL, TARGET_COL

_CONFIG_PATH = Path(__file__).resolve().parent / "settings.json"

TARIFF_MODES = {"constant", "spot_plus_constant", "dual_rate"}


def _sanitize_float(value: Any, *, allow_none: bool = True) -> Optional[float]:
    if value is None:
        return None if allow_none else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None if allow_none else None
    if not math.isfinite(numeric):
        return None if allow_none else None
    return numeric


def _sanitize_positive_float(value: Any, *, fallback: float, allow_zero: bool = True, minimum: float = 0.0) -> float:
    numeric = _sanitize_float(value, allow_none=False)
    if numeric is None:
        return fallback
    if allow_zero:
        if numeric < max(minimum, 0.0):
            return fallback
    else:
        if numeric <= max(minimum, 0.0):
            return fallback
    return float(numeric)


def _sanitize_time_literal(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    candidate = value.strip()
    if not candidate:
        return fallback
    if re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", candidate):
        return candidate
    return fallback


def _default_import_tariff() -> "TariffSettings":
    return TariffSettings(spot_offset_eur_per_kwh=0.14)


def _default_export_tariff() -> "TariffSettings":
    return TariffSettings()


@dataclass
class TariffSettings:
    mode: str = "spot_plus_constant"
    constant_eur_per_kwh: Optional[float] = None
    spot_offset_eur_per_kwh: float = 0.0
    dual_peak_eur_per_kwh: Optional[float] = None
    dual_offpeak_eur_per_kwh: Optional[float] = None
    dual_peak_start_local: str = "07:00"
    dual_peak_end_local: str = "21:00"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "constant_eur_per_kwh": None if self.constant_eur_per_kwh is None else float(self.constant_eur_per_kwh),
            "spot_offset_eur_per_kwh": float(self.spot_offset_eur_per_kwh),
            "dual_peak_eur_per_kwh": None if self.dual_peak_eur_per_kwh is None else float(self.dual_peak_eur_per_kwh),
            "dual_offpeak_eur_per_kwh": None if self.dual_offpeak_eur_per_kwh is None else float(self.dual_offpeak_eur_per_kwh),
            "dual_peak_start_local": self.dual_peak_start_local,
            "dual_peak_end_local": self.dual_peak_end_local,
        }


@dataclass
class PricingSettings:
    import_tariff: TariffSettings = field(default_factory=_default_import_tariff)
    export_tariff: TariffSettings = field(default_factory=_default_export_tariff)
    export_enabled: bool = True
    export_limit_kw: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "import": self.import_tariff.to_dict(),
            "export": self.export_tariff.to_dict(),
            "export_enabled": self.export_enabled,
            "export_limit_kw": self.export_limit_kw,
        }


@dataclass
class EntitySelection:
    entity_id: str
    unit: Optional[str] = None

    def scale_to_kw(self) -> float:
        return unit_to_kw_factor(self.unit)


@dataclass
class HAEntitiesSettings:
    house_consumption: EntitySelection
    pv_power: EntitySelection
    soc_sensor: Optional[EntitySelection] = None
    export_power_limited: bool = False
    driver_id: Optional[str] = None
    driver_config: Dict[str, Any] = field(default_factory=dict)
    entity_map: Dict[str, str] = field(default_factory=dict)

    def rename_map(self) -> Dict[str, str]:
        m = {}
        if self.house_consumption.entity_id:
            m[self.house_consumption.entity_id] = TARGET_COL
        if self.pv_power.entity_id:
            m[self.pv_power.entity_id] = PV_COL
        return m

    def stat_ids(self) -> list[tuple[str, str]]:
        ids = []
        if self.house_consumption.entity_id:
            ids.append((self.house_consumption.entity_id, "mean"))
        if self.pv_power.entity_id:
            ids.append((self.pv_power.entity_id, "mean"))
        return ids

    def scales(self) -> Dict[str, float]:
        return {
            TARGET_COL: self.house_consumption.scale_to_kw(),
            PV_COL: self.pv_power.scale_to_kw(),
        }


@dataclass
class BatterySettings:
    wear_cost_eur_per_kwh: float = 0.10
    capacity_kwh: float = 10.0
    power_limit_kw: float = 3.0
    soc_min: float = 0.1
    soc_max: float = 0.9


@dataclass
class AppSettings:
    ha_entities: HAEntitiesSettings
    battery: BatterySettings = field(default_factory=BatterySettings)
    pricing: PricingSettings = field(default_factory=PricingSettings)
    auto_training_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        ha_payload = {
            "house_consumption": asdict(self.ha_entities.house_consumption),
            "pv_power": asdict(self.ha_entities.pv_power),
            "export_power_limited": self.ha_entities.export_power_limited,
            "driver_id": self.ha_entities.driver_id,
            "driver_config": self.ha_entities.driver_config,
            "entity_map": self.ha_entities.entity_map,
        }
        if self.ha_entities.soc_sensor:
            ha_payload["soc_sensor"] = asdict(self.ha_entities.soc_sensor)
        else:
            ha_payload["soc_sensor"] = None

        payload: Dict[str, Any] = {
            "ha_entities": ha_payload,
            "battery": {
                "wear_cost_eur_per_kwh": float(self.battery.wear_cost_eur_per_kwh),
                "capacity_kwh": float(self.battery.capacity_kwh),
                "power_limit_kw": float(self.battery.power_limit_kw),
                "soc_min": float(self.battery.soc_min),
                "soc_max": float(self.battery.soc_max),
            },
            "pricing": self.pricing.to_dict(),
            "auto_training_enabled": self.auto_training_enabled,
        }
        return payload


_DEFAULT_SETTINGS = AppSettings(
    ha_entities=HAEntitiesSettings(
        house_consumption=EntitySelection("sensor.house_consumption", "W"),
        pv_power=EntitySelection("sensor.pv_power", "W"),
        soc_sensor=None,
        export_power_limited=False,
    ),
    battery=BatterySettings(),
    pricing=PricingSettings(),
    auto_training_enabled=False,
)


def unit_to_kw_factor(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    cleaned = unit.strip().lower()
    if cleaned in {"w", "watt", "watts", "wh"}:
        return 0.001
    if cleaned in {"kw", "kilowatt", "kilowatts", "kwh"}:
        return 1.0
    if cleaned in {"mw", "megawatt", "megawatts", "mwh"}:
        return 1000.0
    return 1.0


def load_settings() -> AppSettings:
    if not _CONFIG_PATH.exists():
        return _DEFAULT_SETTINGS
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _DEFAULT_SETTINGS

    # Migration logic:
    # If "ha_entities" exists, use it.
    # Else, try to build it from "inverter" and "battery" (old structure).
    
    ha_data = data.get("ha_entities", {})
    inverter_data = data.get("inverter", {})
    battery_data = data.get("battery", {})
    
    # House Consumption
    if "house_consumption" in ha_data:
        house_block = ha_data["house_consumption"]
    else:
        house_block = inverter_data.get("house_consumption", {})
        
    house_sel = EntitySelection(
        entity_id=house_block.get("entity_id", _DEFAULT_SETTINGS.ha_entities.house_consumption.entity_id),
        unit=house_block.get("unit"),
    )

    # PV Power
    if "pv_power" in ha_data:
        pv_block = ha_data["pv_power"]
    else:
        pv_block = inverter_data.get("pv_power", {})
        
    pv_sel = EntitySelection(
        entity_id=pv_block.get("entity_id", _DEFAULT_SETTINGS.ha_entities.pv_power.entity_id),
        unit=pv_block.get("unit"),
    )

    # SoC Sensor
    if "soc_sensor" in ha_data:
        soc_block = ha_data["soc_sensor"]
    else:
        soc_block = battery_data.get("soc_sensor")
        
    soc_id = soc_block.get("entity_id") if isinstance(soc_block, dict) else None
    if soc_id:
        soc_sel = EntitySelection(
            entity_id=str(soc_id),
            unit=soc_block.get("unit") if isinstance(soc_block, dict) else None,
        )
    else:
        soc_sel = None

    # Export Limit
    if "export_power_limited" in ha_data:
        export_limit = ha_data["export_power_limited"]
    else:
        export_limit = inverter_data.get("export_power_limited", _DEFAULT_SETTINGS.ha_entities.export_power_limited)
        
    if isinstance(export_limit, str):
        export_limit = export_limit.strip().lower() in {"1", "true", "yes", "on"}
    elif not isinstance(export_limit, bool):
        export_limit = bool(export_limit)

    # Driver Config
    driver_id = ha_data.get("driver_id")
    driver_config = ha_data.get("driver_config", {})
    entity_map = ha_data.get("entity_map", {})

    ha_entities = HAEntitiesSettings(
        house_consumption=house_sel,
        pv_power=pv_sel,
        soc_sensor=soc_sel,
        export_power_limited=export_limit,
        driver_id=driver_id,
        driver_config=driver_config,
        entity_map=entity_map,
    )

    # Battery Settings (minus soc_sensor)
    wear_cost_raw = battery_data.get("wear_cost_eur_per_kwh", _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh)
    try:
        wear_cost_val = float(wear_cost_raw)
    except (TypeError, ValueError):
        wear_cost_val = _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh
    if wear_cost_val < 0 or wear_cost_val != wear_cost_val:  # also guard NaN
        wear_cost_val = _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh

    capacity_val = _sanitize_positive_float(
        battery_data.get("capacity_kwh"),
        fallback=_DEFAULT_SETTINGS.battery.capacity_kwh,
        allow_zero=False,
        minimum=0.1,
    )
    power_limit_val = _sanitize_positive_float(
        battery_data.get("power_limit_kw"),
        fallback=_DEFAULT_SETTINGS.battery.power_limit_kw,
        allow_zero=False,
        minimum=0.1,
    )

    soc_min_val = _sanitize_float(battery_data.get("soc_min"), allow_none=False)
    if soc_min_val is None or not (0 <= soc_min_val < 1):
        soc_min_val = _DEFAULT_SETTINGS.battery.soc_min
    soc_max_val = _sanitize_float(battery_data.get("soc_max"), allow_none=False)
    if soc_max_val is None or not (0 < soc_max_val <= 1):
        soc_max_val = _DEFAULT_SETTINGS.battery.soc_max
    if soc_max_val <= soc_min_val:
        soc_max_val = max(soc_min_val + 0.05, _DEFAULT_SETTINGS.battery.soc_max)
        soc_max_val = min(soc_max_val, 1.0)

    pricing_block = data.get("pricing")
    pricing_settings = coerce_pricing_payload(pricing_block, _DEFAULT_SETTINGS.pricing)

    auto_training = bool(data.get("auto_training_enabled", False))

    return AppSettings(
        ha_entities=ha_entities,
        battery=BatterySettings(
            wear_cost_eur_per_kwh=wear_cost_val,
            capacity_kwh=capacity_val,
            power_limit_kw=power_limit_val,
            soc_min=float(soc_min_val),
            soc_max=float(soc_max_val),
        ),
        pricing=pricing_settings,
        auto_training_enabled=auto_training,
    )


def save_settings(settings: AppSettings) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def coerce_tariff_payload(payload: Any, base: Optional[TariffSettings], scope: str) -> TariffSettings:
    defaults = _default_import_tariff() if scope == "import" else _default_export_tariff()
    current = replace(base) if base is not None else replace(defaults)
    if not isinstance(payload, dict):
        return current
    mode_raw = payload.get("mode")
    if isinstance(mode_raw, str):
        mode_val = mode_raw.strip()
        if mode_val in TARIFF_MODES:
            current.mode = mode_val
    if "constant_eur_per_kwh" in payload:
        constant_val = _sanitize_float(payload.get("constant_eur_per_kwh"))
        current.constant_eur_per_kwh = constant_val
    if "spot_offset_eur_per_kwh" in payload:
        offset_val = _sanitize_float(payload.get("spot_offset_eur_per_kwh"), allow_none=False)
        if offset_val is None:
            offset_val = defaults.spot_offset_eur_per_kwh
        current.spot_offset_eur_per_kwh = float(offset_val)
    if "dual_peak_eur_per_kwh" in payload:
        peak_val = _sanitize_float(payload.get("dual_peak_eur_per_kwh"))
        current.dual_peak_eur_per_kwh = peak_val
    if "dual_offpeak_eur_per_kwh" in payload:
        offpeak_val = _sanitize_float(payload.get("dual_offpeak_eur_per_kwh"))
        current.dual_offpeak_eur_per_kwh = offpeak_val
    if "dual_peak_start_local" in payload:
        current.dual_peak_start_local = _sanitize_time_literal(payload.get("dual_peak_start_local"), current.dual_peak_start_local or defaults.dual_peak_start_local)
    if "dual_peak_end_local" in payload:
        current.dual_peak_end_local = _sanitize_time_literal(payload.get("dual_peak_end_local"), current.dual_peak_end_local or defaults.dual_peak_end_local)
    return current


def coerce_pricing_payload(payload: Any, base: Optional[PricingSettings]) -> PricingSettings:
    base_settings = base if base is not None else PricingSettings()
    if not isinstance(payload, dict):
        return base_settings
    import_block = payload.get("import")
    export_block = payload.get("export")

    export_enabled = payload.get("export_enabled")
    if not isinstance(export_enabled, bool):
        export_enabled = base_settings.export_enabled

    export_limit_kw = _sanitize_float(payload.get("export_limit_kw"))
    if export_limit_kw is None and "export_limit_kw" not in payload:
        export_limit_kw = base_settings.export_limit_kw

    return PricingSettings(
        import_tariff=coerce_tariff_payload(import_block, base_settings.import_tariff, "import"),
        export_tariff=coerce_tariff_payload(export_block, base_settings.export_tariff, "export"),
        export_enabled=export_enabled,
        export_limit_kw=export_limit_kw,
    )