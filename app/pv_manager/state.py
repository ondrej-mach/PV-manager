from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from energy_forecaster.features.data_prep import PV_COL, TARGET_COL
from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz
from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.services.baseline_control import BatteryConfig
from energy_forecaster.services.optimal_control import solve_lp_optimal_control
from energy_forecaster.services.prediction_service import run_prediction_pipeline
from energy_forecaster.services.training_service import run_training_pipeline
from .settings import AppSettings, load_settings, save_settings

_LOGGER = logging.getLogger(__name__)

INTERVAL_MINUTES_DEFAULT = 15
HORIZON_HOURS_DEFAULT = 24
MODELS_DIR_DEFAULT = Path("trained_models")
LOOKBACK_DAYS_DEFAULT = 730
@dataclass
class TrainingStatus:
    running: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }


@dataclass
class ForecastSnapshot:
    timestamp: datetime
    interval_minutes: int
    horizon_hours: int
    forecast_index: pd.DatetimeIndex
    pv_forecast: pd.Series
    load_forecast: pd.Series
    price_forecast: pd.Series
    price_imputed: pd.Series
    plan: pd.DataFrame
    ha_recent: pd.DataFrame
    summary: dict[str, float]
    intervention: dict[str, Any]
    timezone: str
    locale: Optional[str] = None

    def as_payload(self) -> dict[str, Any]:
        idx_iso = [ts.isoformat() for ts in self.forecast_index]
        plan_grid_import = self.plan["grid_import_kw"].reindex(self.forecast_index, fill_value=0.0)
        plan_grid_export = self.plan["grid_export_kw"].reindex(self.forecast_index, fill_value=0.0)
        plan_soc = self.plan["soc"].reindex(self.forecast_index).ffill().bfill()
        plan_batt_to_grid = self.plan.get("batt_to_grid_kw", pd.Series(dtype=float)).reindex(self.forecast_index, fill_value=0.0)
        plan_batt_to_load = self.plan.get("batt_to_load_kw", pd.Series(dtype=float)).reindex(self.forecast_index, fill_value=0.0)
        plan_grid_to_batt = self.plan.get("grid_to_batt_kw", pd.Series(dtype=float)).reindex(self.forecast_index, fill_value=0.0)
        plan_pv_to_batt = self.plan.get("pv_to_batt_kw", pd.Series(dtype=float)).reindex(self.forecast_index, fill_value=0.0)
        battery_discharge = plan_batt_to_grid.add(plan_batt_to_load)
        battery_charge = plan_grid_to_batt.add(plan_pv_to_batt)
        battery_net = battery_discharge.sub(battery_charge)
        history_tail = self.ha_recent.iloc[-96:] if not self.ha_recent.empty else self.ha_recent
        history_idx = [ts.isoformat() for ts in history_tail.index]
        payload: dict[str, Any] = {
            "timestamp": self.timestamp.isoformat(),
            "interval_minutes": self.interval_minutes,
            "horizon_hours": self.horizon_hours,
            "series": {
                "timestamps": idx_iso,
                "pv_kw": self.pv_forecast.reindex(self.forecast_index, fill_value=0.0).tolist(),
                "load_kw": self.load_forecast.reindex(self.forecast_index, method="ffill").tolist(),
                "price_eur_per_kwh": self.price_forecast.reindex(self.forecast_index, method="ffill").tolist(),
                "price_imputed": self.price_imputed.reindex(self.forecast_index, fill_value=False).astype(bool).tolist(),
                "grid_import_kw": plan_grid_import.tolist(),
                "grid_export_kw": plan_grid_export.tolist(),
                "soc": plan_soc.tolist(),
                "battery_net_kw": battery_net.tolist(),
                "battery_discharge_kw": battery_discharge.tolist(),
                "battery_charge_kw": battery_charge.tolist(),
                "battery_to_grid_kw": plan_batt_to_grid.tolist(),
                "battery_to_load_kw": plan_batt_to_load.tolist(),
            },
            "history": {
                "timestamps": history_idx,
                "load_kw": history_tail[TARGET_COL].ffill().tolist() if TARGET_COL in history_tail.columns else [],
                "pv_kw": history_tail[PV_COL].ffill().tolist() if PV_COL in history_tail.columns else [],
            },
            "summary": self.summary,
            "summary_window": {
                "start": idx_iso[0] if idx_iso else None,
                "end": idx_iso[-1] if idx_iso else None,
            },
            "intervention": self.intervention,
        }
        payload["timezone"] = self.timezone
        if self.locale:
            payload["locale"] = self.locale
        return payload


class AppContext:
    def __init__(
        self,
        models_dir: Path | str = MODELS_DIR_DEFAULT,
        interval_minutes: int = INTERVAL_MINUTES_DEFAULT,
        horizon_hours: int = HORIZON_HOURS_DEFAULT,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.interval_minutes = interval_minutes
        self.horizon_hours = horizon_hours
        self._lock = asyncio.Lock()
        self._cycle_lock = asyncio.Lock()
        self._snapshot: Optional[ForecastSnapshot] = None
        self._ha: Optional[HomeAssistant] = None
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._tz: Optional[str] = None
        self._locale: Optional[str] = None
        self._battery_cfg = BatteryConfig()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._training_task: Optional[asyncio.Task] = None
        self._manual_cycle_task: Optional[asyncio.Task] = None
        self._training_status = TrainingStatus()
        self._ha_error: Optional[str] = None
        self._settings: AppSettings = load_settings()
        self._settings_lock = asyncio.Lock()
        self._stat_catalog: list[dict[str, Any]] = []
        self._entity_catalog: list[dict[str, Any]] = []

    async def start(self) -> None:
        try:
            await self._ensure_home_assistant()
        except Exception as exc:  # pragma: no cover - best-effort bootstrap
            _LOGGER.warning("Initial Home Assistant connection failed: %s", exc)
        if not self._scheduler_task:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        await self.refresh_statistics_catalog()
        await self.refresh_entity_catalog()

    async def stop(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        if self._training_task:
            self._training_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._training_task
            self._training_task = None
        if self._manual_cycle_task:
            self._manual_cycle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._manual_cycle_task
            self._manual_cycle_task = None
        if self._ha:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._ha.close_sync)
            self._ha = None

    async def _ensure_home_assistant(self) -> None:
        if self._ha is not None:
            return

        def _init() -> tuple[HomeAssistant, float, float, str, Optional[str]]:
            ha = HomeAssistant()
            ha.connect_sync()
            lat, lon, tz = ha.get_location()
            try:
                locale = ha.get_locale()
            except Exception:  # pragma: no cover - locale is best-effort
                locale = None
            return ha, lat, lon, tz, locale

        loop = asyncio.get_running_loop()
        try:
            ha, lat, lon, tz, locale = await loop.run_in_executor(None, _init)
        except Exception as exc:
            self._ha_error = f"Home Assistant connection failed: {exc}"
            raise
        self._ha_error = None
        self._ha = ha
        self._lat = lat
        self._lon = lon
        self._tz = tz
        self._locale = locale
        _LOGGER.info(
            "Home Assistant connection established (lat=%.4f lon=%.4f tz=%s locale=%s)",
            lat,
            lon,
            tz,
            locale or "n/a",
        )

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Forecast cycle failed: %s", exc)
            await asyncio.sleep(self.interval_minutes * 60)

    async def run_cycle(self) -> None:
        async with self._cycle_lock:
            stat_ids, rename_map, scales = await self._get_inverter_config()
            await self._run_cycle(stat_ids, rename_map, scales)

    async def _run_cycle(
        self,
        stat_ids: List[Tuple[str, str]],
        rename_map: Dict[str, str],
        scales: Dict[str, float],
    ) -> None:
        await self._ensure_home_assistant()
        if self._ha is None or self._lat is None or self._lon is None or self._tz is None:
            raise RuntimeError("Home Assistant not initialized")

        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(
            None,
            self._compute_snapshot,
            self._ha,
            self._lat,
            self._lon,
            self._tz,
            stat_ids,
            rename_map,
            scales,
        )

        async with self._lock:
            self._snapshot = snapshot
        _LOGGER.info("Forecast cycle complete at %s", snapshot.timestamp.isoformat())

    def _compute_snapshot(
        self,
        ha: HomeAssistant,
        lat: float,
        lon: float,
        tz: str,
        stat_ids: List[Tuple[str, str]],
        rename_map: Dict[str, str],
        scales: Dict[str, float],
    ) -> ForecastSnapshot:
        now = datetime.now(timezone.utc)
        preds = run_prediction_pipeline(
            ha=ha,
            lat=lat,
            lon=lon,
            tz=tz,
            models_dir=str(self.models_dir),
            horizon_hours=self.horizon_hours,
            interval_minutes=self.interval_minutes,
            entities=stat_ids,
            rename_map=rename_map,
            scales=scales,
        )

        pv_forecast = preds["pv_pred"][PV_COL].astype(float)
        if TARGET_COL in preds["house_pred"].columns:
            load_series = preds["house_pred"][TARGET_COL].reindex(pv_forecast.index)
        else:
            load_series = pd.Series(index=pv_forecast.index, dtype=float)
        load_series = load_series.fillna(method="ffill").fillna(method="bfill").clip(lower=0.0)
        pv_series = pv_forecast.clip(lower=0.0)

        price_series, price_imputed = self._build_price_series(tz, pv_series.index)

        dt_hours = self.interval_minutes / 60.0
        S = pd.DataFrame(
            {
                "pv_kw": pv_series.values,
                "load_kw": load_series.values,
                "dt_h": np.full(len(pv_series), dt_hours),
                "price_eur_per_kwh": price_series.reindex(pv_series.index, method="ffill").values,
            },
            index=pv_series.index,
        )

        plan = solve_lp_optimal_control(S, self._battery_cfg, soc0=None, allow_grid_charging=True)
        plan.index = pv_series.index

        summary = self._summarize_plan(plan)
        intervention = self._determine_intervention(plan, price_series, tz)

        ha_recent = preds.get("ha_recent", pd.DataFrame(index=pd.Index([], name="time")))
        if not ha_recent.empty:
            ha_recent = ha_recent.sort_index()

        return ForecastSnapshot(
            timestamp=now,
            interval_minutes=self.interval_minutes,
            horizon_hours=self.horizon_hours,
            forecast_index=pv_series.index,
            pv_forecast=pv_series,
            load_forecast=load_series,
            price_forecast=price_series,
            price_imputed=price_imputed,
            plan=plan,
            ha_recent=ha_recent,
            summary=summary,
            intervention=intervention,
            timezone=tz,
            locale=self._locale,
        )

    def _build_price_series(self, tz: str, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
        start_local = index[0].tz_convert(tz) if index.tz is not None else index[0].tz_localize("UTC").tz_convert(tz)
        end_local = index[-1].tz_convert(tz) if index.tz is not None else index[-1].tz_localize("UTC").tz_convert(tz)
        start = start_local - timedelta(days=2)
        end = end_local + timedelta(hours=2)
        # add buffer to ensure complete coverage
        country = guess_country_code_from_tz(tz)
        prices_df = fetch_day_ahead_prices_country(country, start=start, end=end, tz=tz)
        prices = prices_df["price_eur_per_kwh"].tz_convert("UTC")
        aligned = prices.reindex(index)
        imputed_mask = pd.Series(False, index=index)
        missing_idx = aligned.index[aligned.isna()]
        if len(missing_idx) > 0:
            for ts in missing_idx:
                fallback_ts = ts - timedelta(days=1)
                if fallback_ts in prices.index:
                    fallback_val = prices.loc[fallback_ts]
                    if pd.notna(fallback_val):
                        aligned.at[ts] = fallback_val
                        imputed_mask.at[ts] = True
        still_missing = aligned.isna()
        if still_missing.any():
            aligned = aligned.ffill().bfill()
            imputed_mask = imputed_mask | still_missing
        return aligned, imputed_mask

    def _determine_intervention(self, plan: pd.DataFrame, price_series: pd.Series, tz: str) -> dict[str, Any]:
        if plan.empty:
            return {
                "mode": "none",
                "power_kw": 0.0,
                "price_eur_per_kwh": None,
                "soc": None,
                "reason": "Plan not available",
                "timestamp": None,
            }

        row = plan.iloc[0]
        ts = plan.index[0]

        batt_to_grid = float(row.get("batt_to_grid_kw", 0.0) or 0.0)
        batt_to_load = float(row.get("batt_to_load_kw", 0.0) or 0.0)
        pv_to_batt = float(row.get("pv_to_batt_kw", 0.0) or 0.0)
        grid_to_batt = float(row.get("grid_to_batt_kw", 0.0) or 0.0)
        grid_export = float(row.get("grid_export_kw", 0.0) or 0.0)
        grid_import = float(row.get("grid_import_kw", 0.0) or 0.0)
        soc = float(row.get("soc", float("nan")))
        net_batt = batt_to_grid + batt_to_load - pv_to_batt - grid_to_batt

        try:
            price = float(price_series.loc[ts])
        except KeyError:
            price = float(price_series.reindex([ts], method="nearest", tolerance=pd.Timedelta(minutes=self.interval_minutes)).iloc[0]) if len(price_series) else float("nan")
        except Exception:
            price = float("nan")

        power_tol = 0.05
        near_full = not math.isnan(soc) and soc >= (self._battery_cfg.soc_max - 0.02)
        price_negative = price < 0 if not math.isnan(price) else False

        reason = None
        mode = "none"
        power = 0.0

        if net_batt > power_tol:
            mode = "discharge"
            power = net_batt
            if batt_to_load >= batt_to_grid:
                reason = "Covering load from battery"
            else:
                reason = "Exporting surplus from battery"
        elif net_batt < -power_tol:
            mode = "charge"
            power = abs(net_batt)
            if pv_to_batt >= grid_to_batt:
                reason = "Charging from PV surplus"
            else:
                reason = "Charging from grid"
        elif near_full and price_negative and grid_export > power_tol:
            mode = "limit_export"
            power = grid_export
            reason = "Negative price while battery near full"
        else:
            mode = "none"
            reason = None

        return {
            "mode": mode,
            "power_kw": round(power, 4),
            "price_eur_per_kwh": price if not math.isnan(price) else None,
            "soc": soc if not math.isnan(soc) else None,
            "reason": reason,
            "timestamp": ts.isoformat(),
            "grid_export_kw": round(grid_export, 4),
            "grid_import_kw": round(grid_import, 4),
            "battery_net_kw": round(net_batt, 4),
        }

    async def get_settings_payload(self) -> dict[str, Any]:
        if not self._stat_catalog:
            await self.refresh_statistics_catalog()
        if not self._entity_catalog:
            await self.refresh_entity_catalog()
        async with self._settings_lock:
            serialized = self._serialize_settings(self._settings)
        serialized = await self._attach_recorder_status(serialized)
        return {
            "settings": serialized,
            "statistics": list(self._stat_catalog),
            "entities": list(self._entity_catalog),
        }

    async def update_inverter_settings(self, data: Dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Payload must be an object")
        catalog = await self.refresh_statistics_catalog()
        catalog_map = {item.get("statistic_id"): item for item in catalog if item.get("statistic_id")}

        def _supports_mean(meta: Optional[Dict[str, Any]]) -> bool:
            if not meta:
                return False
            if meta.get("has_mean"):
                return True
            supported = meta.get("supported_statistics")
            if isinstance(supported, list) and any(str(s).lower() == "mean" for s in supported):
                return True
            return False

        changed = False
        async with self._settings_lock:
            settings = self._settings
            inverter = settings.inverter

            def _apply(key: str, selection) -> None:
                nonlocal changed
                block = data.get(key)
                if not isinstance(block, dict):
                    return
                entity_id = block.get("entity_id") or selection.entity_id
                meta = catalog_map.get(entity_id)
                if not meta:
                    _LOGGER.warning("Recorder metadata missing for statistic '%s'", entity_id)
                elif not _supports_mean(meta):
                    _LOGGER.warning("Statistic '%s' does not declare mean values; proceeding with best effort", entity_id)
                if selection.entity_id != entity_id:
                    selection.entity_id = entity_id
                    changed = True
                unit = block.get("unit") or (meta.get("unit") if meta else selection.unit)
                if unit and selection.unit != unit:
                    selection.unit = unit
                    changed = True

            _apply("house_consumption", inverter.house_consumption)
            _apply("pv_power", inverter.pv_power)

            if changed:
                save_settings(settings)
            serialized = self._serialize_settings(settings)

        serialized = await self._attach_recorder_status(serialized)

        if changed:
            await self.refresh_statistics_catalog()
            await self.refresh_entity_catalog()
        elif not self._entity_catalog:
            await self.refresh_entity_catalog()
        return {
            "settings": serialized,
            "statistics": list(self._stat_catalog),
            "entities": list(self._entity_catalog),
        }

    async def refresh_statistics_catalog(self) -> List[Dict[str, Any]]:
        try:
            await self._ensure_home_assistant()
        except Exception:
            return self._stat_catalog
        if self._ha is None:
            return self._stat_catalog

        loop = asyncio.get_running_loop()
        try:
            catalog = await loop.run_in_executor(None, self._ha.list_statistic_ids_sync)
        except Exception as exc:  # pragma: no cover - best-effort logging
            _LOGGER.debug("Failed to fetch statistic metadata: %s", exc)
            return self._stat_catalog

        await self._sync_units_from_catalog(catalog)
        self._stat_catalog = catalog
        return catalog

    async def refresh_entity_catalog(self) -> List[Dict[str, Any]]:
        try:
            await self._ensure_home_assistant()
        except Exception:
            return self._entity_catalog
        if self._ha is None:
            return self._entity_catalog

        loop = asyncio.get_running_loop()
        try:
            entities = await loop.run_in_executor(None, self._ha.list_entities_sync)
        except Exception as exc:  # pragma: no cover - best-effort logging
            _LOGGER.debug("Failed to fetch entity metadata: %s", exc)
            return self._entity_catalog

        allowed_units = {"w", "kw", "mw", "wh", "kwh", "mwh"}
        filtered: list[dict[str, Any]] = []
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            entity_id = entry.get("entity_id")
            if not entity_id:
                continue
            domain = entity_id.split(".", 1)[0]
            if domain != "sensor":
                continue
            attrs = entry.get("attributes") or {}
            device_class = (attrs.get("device_class") or "").lower()
            unit = attrs.get("unit_of_measurement")
            unit_norm = unit.lower() if isinstance(unit, str) else None
            if device_class not in {"power", "energy"} and (unit_norm not in allowed_units):
                continue
            filtered.append(
                {
                    "entity_id": entity_id,
                    "name": attrs.get("friendly_name") or entity_id,
                    "device_class": attrs.get("device_class"),
                    "unit": unit,
                    "state": entry.get("state"),
                    "domain": domain,
                }
            )

        filtered.sort(key=lambda item: item["name"].lower())
        self._entity_catalog = filtered
        return filtered

    async def _sync_units_from_catalog(self, catalog: List[Dict[str, Any]]) -> None:
        unit_map: Dict[str, str] = {}
        for item in catalog:
            stat_id = item.get("statistic_id")
            unit = item.get("unit")
            if stat_id and unit:
                unit_map[stat_id] = unit

        async with self._settings_lock:
            settings = self._settings
            inverter = settings.inverter
            changed = False
            for selection in (inverter.house_consumption, inverter.pv_power):
                unit = unit_map.get(selection.entity_id)
                if unit and unit != selection.unit:
                    selection.unit = unit
                    changed = True
            if changed:
                save_settings(settings)

    @staticmethod
    def _meta_supports_mean(meta: Optional[Dict[str, Any]]) -> bool:
        if not meta:
            return False
        if meta.get("has_mean"):
            return True
        supported = meta.get("supported_statistics")
        if isinstance(supported, list) and any(str(item).lower() == "mean" for item in supported):
            return True
        return False

    async def _ensure_statistic_metadata(self, stat_id: Optional[str], unit: Optional[str]) -> Optional[Dict[str, Any]]:
        if not stat_id:
            return None
        for item in self._stat_catalog:
            if item.get("statistic_id") == stat_id:
                return item
        probe = await self._probe_statistic_metadata(stat_id, unit)
        if probe and probe.get("statistic_id") == stat_id:
            self._stat_catalog.append(probe)
            return probe
        return None

    async def _attach_recorder_status(self, serialized: dict[str, Any]) -> dict[str, Any]:
        inverter = serialized.get("inverter")
        if not isinstance(inverter, dict):
            return serialized

        async def _apply(key: str) -> None:
            block = inverter.get(key)
            if not isinstance(block, dict):
                return
            stat_id = block.get("entity_id")
            unit = block.get("unit")
            meta = await self._ensure_statistic_metadata(stat_id, unit)
            if meta is None:
                block["recorder_status"] = "missing"
            elif self._meta_supports_mean(meta):
                block["recorder_status"] = "ok"
            else:
                block["recorder_status"] = "no_mean"

        await _apply("house_consumption")
        await _apply("pv_power")
        return serialized

    async def _probe_statistic_metadata(self, stat_id: str, unit: Optional[str]) -> Optional[dict[str, Any]]:
        if not stat_id or self._ha is None:
            return None
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(
                None,
                lambda: self._ha.fetch_last_hours_sync([(stat_id, "mean")], hours=48),
            )
        except Exception as exc:  # pragma: no cover - diagnostic aid
            _LOGGER.debug("Probe for statistic '%s' failed: %s", stat_id, exc)
            return None
        if df.empty or stat_id not in df.columns:
            return None
        return {
            "statistic_id": stat_id,
            "display_name": stat_id,
            "unit": unit,
            "has_mean": True,
            "has_sum": False,
            "unit_class": None,
            "device_class": None,
            "supported_statistics": ["mean"],
        }

    async def _get_inverter_config(self) -> tuple[List[Tuple[str, str]], Dict[str, str], Dict[str, float]]:
        async with self._settings_lock:
            inverter = self._settings.inverter
            stat_ids = list(inverter.stat_ids())
            rename_map = dict(inverter.rename_map())
            scales = dict(inverter.scales())
        return stat_ids, rename_map, scales

    def _serialize_settings(self, settings: AppSettings) -> dict[str, Any]:
        inverter = settings.inverter
        return {
            "inverter": {
                "house_consumption": {
                    "entity_id": inverter.house_consumption.entity_id,
                    "unit": inverter.house_consumption.unit,
                    "scale_to_kw": inverter.house_consumption.scale_to_kw(),
                },
                "pv_power": {
                    "entity_id": inverter.pv_power.entity_id,
                    "unit": inverter.pv_power.unit,
                    "scale_to_kw": inverter.pv_power.scale_to_kw(),
                },
            }
        }

    @staticmethod
    def _summarize_plan(plan: pd.DataFrame) -> dict[str, float]:
        import_kwh = float(plan.get("import_kwh", pd.Series(dtype=float)).sum())
        export_kwh = float(plan.get("export_kwh", pd.Series(dtype=float)).sum())
        net_cost = float((plan.get("cost_import_eur", pd.Series(dtype=float)) - plan.get("rev_export_eur", pd.Series(dtype=float))).sum())
        peak_import = float(plan.get("grid_import_kw", pd.Series(dtype=float)).max()) if not plan.empty else 0.0
        peak_export = float(plan.get("grid_export_kw", pd.Series(dtype=float)).max()) if not plan.empty else 0.0
        return {
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "net_cost_eur": net_cost,
            "peak_import_kw": peak_import,
            "peak_export_kw": peak_export,
        }

    async def get_snapshot(self) -> Optional[ForecastSnapshot]:
        async with self._lock:
            return self._snapshot

    def is_cycle_running(self) -> bool:
        return self._cycle_lock.locked()

    async def trigger_cycle(self) -> None:
        if self._cycle_lock.locked():
            raise RuntimeError("Forecast cycle already running")

        async def _runner() -> None:
            try:
                await self.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive logging
                _LOGGER.exception("Manual forecast cycle failed: %s", exc)

        loop = asyncio.get_running_loop()
        self._manual_cycle_task = loop.create_task(_runner())
        self._manual_cycle_task.add_done_callback(lambda _task: setattr(self, "_manual_cycle_task", None))

    def get_training_status(self) -> TrainingStatus:
        return self._training_status

    def get_ha_error(self) -> Optional[str]:
        return self._ha_error

    async def trigger_training(self) -> None:
        if self._training_task and not self._training_task.done():
            raise RuntimeError("Training already in progress")

        if self._ha_error:
            raise RuntimeError(self._ha_error)

        await self._ensure_home_assistant()
        if self._ha is None or self._lat is None or self._lon is None or self._tz is None:
            raise RuntimeError("Home Assistant not initialized")

        stat_ids, rename_map, scales = await self._get_inverter_config()
        loop = asyncio.get_running_loop()
        self._training_status = TrainingStatus(running=True, started_at=datetime.now(timezone.utc), finished_at=None, error=None)
        self._training_task = loop.create_task(self._run_training(stat_ids, rename_map, scales))

    async def _run_training(
        self,
        stat_ids: List[Tuple[str, str]],
        rename_map: Dict[str, str],
        scales: Dict[str, float],
    ) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                self._run_training_sync,
                stat_ids,
                rename_map,
                scales,
            )
            self._training_status.finished_at = datetime.now(timezone.utc)
            self._training_status.running = False
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.exception("Training failed: %s", exc)
            self._training_status.error = str(exc)
            self._training_status.running = False
            self._training_status.finished_at = datetime.now(timezone.utc)

    def _run_training_sync(
        self,
        stat_ids: List[Tuple[str, str]],
        rename_map: Dict[str, str],
        scales: Dict[str, float],
    ) -> None:
        if self._ha is None or self._lat is None or self._lon is None or self._tz is None:
            raise RuntimeError("Home Assistant not initialized")

        _LOGGER.info("Starting training pipeline")
        run_training_pipeline(
            stat_ids=stat_ids,
            lookback_days=LOOKBACK_DAYS_DEFAULT,
            lat=self._lat,
            lon=self._lon,
            tz=self._tz,
            save_dir=str(self.models_dir),
            ha=self._ha,
            rename_map=rename_map,
            scales=scales,
        )
        _LOGGER.info("Training pipeline finished")
