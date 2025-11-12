from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
class AppSettings:
    inverter: InverterSettings

    def to_dict(self) -> Dict[str, Any]:
        return {"inverter": {
            "house_consumption": asdict(self.inverter.house_consumption),
            "pv_power": asdict(self.inverter.pv_power),
        }}


_DEFAULT_SETTINGS = AppSettings(
    inverter=InverterSettings(
        house_consumption=EntitySelection("sensor.house_consumption", "W"),
        pv_power=EntitySelection("sensor.pv_power", "W"),
    )
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
        )
    )


def save_settings(settings: AppSettings) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")