from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from energy_forecaster.features.data_prep import PV_COL, TARGET_COL

_CONFIG_PATH = Path(__file__).resolve().parent / "settings.json"


@dataclass
class EntitySelection:
    entity_id: str
    unit: Optional[str] = None

    def scale_to_kw(self) -> float:
        return unit_to_kw_factor(self.unit)


@dataclass
class InverterSettings:
    house_consumption: EntitySelection
    pv_power: EntitySelection
    export_power_limited: bool = False

    def rename_map(self) -> Dict[str, str]:
        return {
            self.house_consumption.entity_id: TARGET_COL,
            self.pv_power.entity_id: PV_COL,
        }

    def stat_ids(self) -> list[tuple[str, str]]:
        return [
            (self.house_consumption.entity_id, "mean"),
            (self.pv_power.entity_id, "mean"),
        ]

    def scales(self) -> Dict[str, float]:
        return {
            TARGET_COL: self.house_consumption.scale_to_kw(),
            PV_COL: self.pv_power.scale_to_kw(),
        }


@dataclass
class BatterySettings:
    soc_sensor: Optional[EntitySelection] = None
    wear_cost_eur_per_kwh: float = 0.10


@dataclass
class AppSettings:
    inverter: InverterSettings
    battery: BatterySettings = field(default_factory=BatterySettings)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "inverter": {
                "house_consumption": asdict(self.inverter.house_consumption),
                "pv_power": asdict(self.inverter.pv_power),
                "export_power_limited": self.inverter.export_power_limited,
            }
        }
        if self.battery.soc_sensor:
            payload["battery"] = {
                "soc_sensor": asdict(self.battery.soc_sensor),
            }
        else:
            payload["battery"] = {"soc_sensor": None}
        payload["battery"]["wear_cost_eur_per_kwh"] = float(self.battery.wear_cost_eur_per_kwh)
        return payload


_DEFAULT_SETTINGS = AppSettings(
    inverter=InverterSettings(
        house_consumption=EntitySelection("sensor.house_consumption", "W"),
        pv_power=EntitySelection("sensor.pv_power", "W"),
        export_power_limited=False,
    ),
    battery=BatterySettings(soc_sensor=None),
)


def unit_to_kw_factor(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    cleaned = unit.strip().lower()
    if cleaned in {"w", "watt", "watts"}:
        return 0.001
    if cleaned in {"kw", "kilowatt", "kilowatts"}:
        return 1.0
    if cleaned in {"mw", "megawatt", "megawatts"}:
        return 1000.0
    return 1.0


def load_settings() -> AppSettings:
    if not _CONFIG_PATH.exists():
        return _DEFAULT_SETTINGS
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _DEFAULT_SETTINGS

    inverter = data.get("inverter", {})
    house = inverter.get("house_consumption", {})
    pv = inverter.get("pv_power", {})
    export_limit = inverter.get("export_power_limited", _DEFAULT_SETTINGS.inverter.export_power_limited)
    if isinstance(export_limit, str):
        export_limit = export_limit.strip().lower() in {"1", "true", "yes", "on"}
    elif not isinstance(export_limit, bool):
        export_limit = bool(export_limit)

    battery_data = data.get("battery", {})
    soc_block = battery_data.get("soc_sensor") or {}
    soc_entity = soc_block.get("entity_id") if isinstance(soc_block, dict) else None
    if soc_entity:
        soc_selection: Optional[EntitySelection] = EntitySelection(
            entity_id=soc_entity,
            unit=soc_block.get("unit") if isinstance(soc_block, dict) else None,
        )
    else:
        soc_selection = None

    wear_cost_raw = battery_data.get("wear_cost_eur_per_kwh", _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh)
    try:
        wear_cost_val = float(wear_cost_raw)
    except (TypeError, ValueError):
        wear_cost_val = _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh
    if wear_cost_val < 0 or wear_cost_val != wear_cost_val:  # also guard NaN
        wear_cost_val = _DEFAULT_SETTINGS.battery.wear_cost_eur_per_kwh

    return AppSettings(
        inverter=InverterSettings(
            house_consumption=EntitySelection(
                entity_id=house.get("entity_id", _DEFAULT_SETTINGS.inverter.house_consumption.entity_id),
                unit=house.get("unit"),
            ),
            pv_power=EntitySelection(
                entity_id=pv.get("entity_id", _DEFAULT_SETTINGS.inverter.pv_power.entity_id),
                unit=pv.get("unit"),
            ),
            export_power_limited=export_limit,
        ),
        battery=BatterySettings(soc_sensor=soc_selection, wear_cost_eur_per_kwh=wear_cost_val),
    )


def save_settings(settings: AppSettings) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")