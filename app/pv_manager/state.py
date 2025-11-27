from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from energy_forecaster.features.data_prep import PV_COL, TARGET_COL
from energy_forecaster.io.entsoe import fetch_day_ahead_prices_country, guess_country_code_from_tz
from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.services.optimal_control import solve_lp_optimal_control, BatteryConfig
from energy_forecaster.services.prediction_service import run_prediction_pipeline
from energy_forecaster.services.training_service import run_training_pipeline
from .settings import (
    AppSettings,
    EntitySelection,
    PricingSettings,
    TariffSettings,
    coerce_pricing_payload,
    load_settings,
    save_settings,
)
from .drivers import InverterManager

_LOGGER = logging.getLogger(__name__)

INTERVAL_MINUTES_DEFAULT = 15
HORIZON_HOURS_DEFAULT = 24
MODELS_DIR_DEFAULT = Path("trained_models")
LOOKBACK_DAYS_DEFAULT = 730
T = TypeVar("T")
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
    export_price_forecast: pd.Series
    spot_price_forecast: pd.Series
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
                "price_import_eur_per_kwh": self.price_forecast.reindex(self.forecast_index, method="ffill").tolist(),
                "price_export_eur_per_kwh": self.export_price_forecast.reindex(self.forecast_index, method="ffill").tolist(),
                "price_spot_eur_per_kwh": self.spot_price_forecast.reindex(self.forecast_index, method="ffill").tolist(),
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
        self._last_cycle_error: Optional[str] = None
        self._ha: Optional[HomeAssistant] = None
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._tz: Optional[str] = None
        self._locale: Optional[str] = None
        self._settings: AppSettings = load_settings()
        self._settings_lock = asyncio.Lock()
        self._battery_cfg = BatteryConfig()
        self._battery_wear_cost = max(0.0, float(self._settings.battery.wear_cost_eur_per_kwh))
        self._apply_battery_settings_to_config()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._training_task: Optional[asyncio.Task] = None
        self._manual_cycle_task: Optional[asyncio.Task] = None
        self._training_status = TrainingStatus()
        self._ha_error: Optional[str] = None
        self._stat_catalog: list[dict[str, Any]] = []
        self._entity_catalog: list[dict[str, Any]] = []
        self._control_active: bool = False
        self._saved_inverter_mode: Optional[Any] = None
        debug_dir_env = os.getenv("DEBUG_DIR")
        self._debug_dir = Path(debug_dir_env).expanduser() if debug_dir_env else None
        self._inverter_manager = InverterManager()
        self._load_inverter_driver_config()
        self._load_control_state()

    async def _call_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        ctx = contextvars.copy_context()

        def _runner() -> None:
            try:
                result = ctx.run(func, *args, **kwargs)
            except Exception as exc:  # pragma: no cover - thread bridge
                if future.cancelled():
                    return
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                if future.cancelled():
                    return
                loop.call_soon_threadsafe(future.set_result, result)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise

    async def start(self) -> None:
        try:
            await self._ensure_home_assistant()
        except Exception as exc:  # pragma: no cover - best-effort bootstrap
            _LOGGER.warning("Initial Home Assistant connection failed: %s", exc)
        if not self._scheduler_task:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self._load_inverter_driver_config()
        
        await self.refresh_statistics_catalog()
        await self.refresh_entity_catalog()

    def is_control_active(self) -> bool:
        return self._control_active
        
    def get_last_cycle_error(self) -> Optional[str]:
        return self._last_cycle_error

    async def set_control_active(self, active: bool) -> None:
        if self._control_active == active:
            return
        
        self._control_active = active
        # Save state
        await self._save_control_state(active)
        
        if active:
            _LOGGER.info("Automatic control ENABLED.")
            # Apply control immediately based on current forecast plan
            async with self._lock:
                snapshot = self._snapshot
            
            if snapshot and snapshot.plan is not None and not snapshot.plan.empty:
                # Extract the first step from the plan with all battery flows
                first_row = snapshot.plan.iloc[0]
                
                battery_flows = {
                    "grid_to_batt_kw": float(first_row.get("grid_to_batt_kw", 0.0)),
                    "pv_to_batt_kw": float(first_row.get("pv_to_batt_kw", 0.0)),
                    "batt_to_load_kw": float(first_row.get("batt_to_load_kw", 0.0)),
                    "batt_to_grid_kw": float(first_row.get("batt_to_grid_kw", 0.0)),
                    "load_kw": float(first_row.get("load_kw", 0.0)),
                    "grid_import_kw": float(first_row.get("grid_import_kw", 0.0)),
                    "grid_export_kw": float(first_row.get("grid_export_kw", 0.0)),
                }
                
                _LOGGER.info("Applying immediate control with battery flows: %s", battery_flows)
                if self._ha:
                    await self._inverter_manager.apply_control(self._ha, battery_flows)
            else:
                _LOGGER.warning("No forecast plan available for immediate control application")
        else:
            _LOGGER.info("Automatic control DISABLED.")
            # When disabling, reset the inverter to idle state
            if self._ha:
                await self._inverter_manager.apply_control(self._ha, None)

    async def _save_control_state(self, active: bool) -> None:
        state_path = self.models_dir.parent / "control_state.json"
        try:
            import json
            data = {"control_active": active}
            state_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            _LOGGER.error("Failed to save control state: %s", exc)

    def _load_control_state(self) -> None:
        state_path = self.models_dir.parent / "control_state.json"
        if state_path.exists():
            try:
                import json
                data = json.loads(state_path.read_text())
                self._control_active = bool(data.get("control_active", False))
                _LOGGER.info("Loaded control state: active=%s", self._control_active)
            except Exception as exc:
                _LOGGER.error("Failed to load control state: %s", exc)

    def get_inverter_manager(self) -> InverterManager:
        return self._inverter_manager

    def _load_inverter_driver_config(self) -> None:
        # Load from settings
        ha_entities = self._settings.ha_entities
        if ha_entities.driver_id:
            self._inverter_manager.set_active_driver(
                ha_entities.driver_id,
                ha_entities.entity_map,
                ha_entities.driver_config
            )
        else:
            # Migration: Try to load from old inverter_driver.json
            config_path = self.models_dir.parent / "inverter_driver.json"
            if config_path.exists():
                try:
                    import json
                    data = json.loads(config_path.read_text())
                    driver_id = data.get("driver_id")
                    if driver_id:
                        entity_map = data.get("entity_map", {})
                        config = data.get("config", {})
                        self._inverter_manager.set_active_driver(driver_id, entity_map, config)
                        
                        # Save to new settings structure
                        # We can't await here easily as this is sync, but we can update settings object
                        # and save it. Since this runs at startup, we can just write it.
                        self._settings.ha_entities.driver_id = driver_id
                        self._settings.ha_entities.entity_map = entity_map
                        self._settings.ha_entities.driver_config = config
                        save_settings(self._settings)
                        _LOGGER.info("Migrated inverter driver config to settings")
                except Exception as exc:
                    _LOGGER.error("Failed to migrate inverter driver config: %s", exc)

    async def save_inverter_driver_config(self, driver_id: str, entity_map: dict, config: dict) -> None:
        self._inverter_manager.set_active_driver(driver_id, entity_map, config)
        async with self._settings_lock:
            self._settings.ha_entities.driver_id = driver_id
            self._settings.ha_entities.entity_map = entity_map
            self._settings.ha_entities.driver_config = config
            save_settings(self._settings)


    async def stop(self) -> None:
        # Disable control on shutdown to reset inverter to safe state
        if self._control_active:
            _LOGGER.info("Shutting down: Disabling automatic control and resetting inverter...")
            # We don't save state here because we want to remember it was active on next boot
            # But we do want to physically reset the inverter now.
            if self._ha:
                try:
                    await self._inverter_manager.apply_control(self._ha, None)
                except Exception as exc:
                    _LOGGER.error("Failed to reset inverter on shutdown: %s", exc)

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
            await self._call_in_thread(self._ha.close_sync)
            self._ha = None

    async def _ensure_home_assistant(self) -> None:
        if self._ha is not None:
            return

        def _init() -> tuple[HomeAssistant, float, float, str, Optional[str]]:
            hass_server = os.getenv("HASS_SERVER")
            ha = HomeAssistant(instance_url=hass_server)
            ha.connect_sync()
            lat, lon, tz = ha.get_location()
            try:
                locale = ha.get_locale()
            except Exception:  # pragma: no cover - locale is best-effort
                locale = None
            return ha, lat, lon, tz, locale

        try:
            ha, lat, lon, tz, locale = await self._call_in_thread(_init)
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
                
                # Check for auto-training
                if self._settings.auto_training_enabled:
                    now = datetime.now(timezone.utc)
                    # Run training if it's between 02:00 and 03:00 UTC and not running
                    if now.hour == 2 and not self._training_status.running:
                        # Check if we already trained today
                        last_run = self._training_status.finished_at
                        if not last_run or last_run.date() < now.date():
                            _LOGGER.info("Triggering scheduled auto-training")
                            try:
                                await self.trigger_training()
                            except Exception as exc:
                                _LOGGER.error("Auto-training failed to start: %s", exc)

            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Forecast cycle failed: %s", exc)
                self._last_cycle_error = str(exc)
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
        if self._ha is None:
            _LOGGER.warning("Home Assistant not connected, skipping cycle")
            self._last_cycle_error = "Home Assistant not connected"
            return

        # ... (existing forecast logic) ...
        # We need to capture the plan result to apply control
        
        # For now, let's assume we have a plan. 
        # The existing code runs prediction and optimization but doesn't seem to return the plan structure easily 
        # without modifying run_prediction_pipeline or solve_lp_optimal_control return values or side effects.
        # However, looking at the code, `solve_lp_optimal_control` returns a schedule.
        
        # Let's peek at where `solve_lp_optimal_control` is called.
        # It is called inside `run_prediction_pipeline`? No, `run_prediction_pipeline` returns a snapshot.
        # Wait, `run_cycle` calls `_run_cycle`.
        
        # I need to see the rest of `_run_cycle` to insert the control logic.
        # Run pipeline in thread
        snapshot = await self._call_in_thread(
            self._run_cycle_sync,
            stat_ids,
            rename_map,
            scales,
        )

        async with self._lock:
            self._snapshot = snapshot
            self._last_cycle_error = None
            
        if snapshot:
            _LOGGER.info("Forecast cycle complete at %s", snapshot.timestamp.isoformat())
            
        # Apply control if active
        if self._control_active and snapshot and snapshot.plan is not None and not snapshot.plan.empty:
            try:
                # Get the first step of the plan (current interval)
                first_step = snapshot.plan.iloc[0]
                # Extract battery flows from the first step
                battery_flows = {
                    "grid_to_batt_kw": float(first_step.get("grid_to_batt_kw", 0.0)),
                    "pv_to_batt_kw": float(first_step.get("pv_to_batt_kw", 0.0)),
                    "batt_to_load_kw": float(first_step.get("batt_to_load_kw", 0.0)),
                    "batt_to_grid_kw": float(first_step.get("batt_to_grid_kw", 0.0)),
                    "load_kw": float(first_step.get("load_kw", 0.0)),
                    "grid_import_kw": float(first_step.get("grid_import_kw", 0.0)),
                    "grid_export_kw": float(first_step.get("grid_export_kw", 0.0)),
                }
                
                await self._inverter_manager.apply_control(self._ha, battery_flows)
            except Exception as exc:
                _LOGGER.error("Failed to apply inverter control: %s", exc)

    def _build_price_series(
        self,
        tz: str,
        index: pd.DatetimeIndex,
        pricing_cfg: Optional[PricingSettings],
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        start_local = index[0].tz_convert(tz) if index.tz is not None else index[0].tz_localize("UTC").tz_convert(tz)
        end_local = index[-1].tz_convert(tz) if index.tz is not None else index[-1].tz_localize("UTC").tz_convert(tz)
        start = start_local - timedelta(days=2)
        end = end_local + timedelta(hours=2)
        # add buffer to ensure complete coverage
        country = guess_country_code_from_tz(tz)
        prices_df = fetch_day_ahead_prices_country(country, start=start, end=end, tz=tz)
        prices = prices_df["price_eur_per_kwh"].tz_convert("UTC")
        aligned = prices.reindex(index)
        
        # Check if we have enough data for dynamic pricing
        if aligned.isna().all():
            # If we are in a dynamic mode, we cannot proceed without spot prices
            is_dynamic = False
            if pricing_cfg:
                if pricing_cfg.import_tariff.mode == "spot_plus_constant" or pricing_cfg.export_tariff.mode == "spot_plus_constant":
                    is_dynamic = True
            
            if is_dynamic:
                raise ValueError(f"No spot price data available for {country} (ENTSO-E). Cannot run optimization in spot_plus_constant mode.")
        
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
        import_cfg = pricing_cfg.import_tariff if pricing_cfg else None
        export_cfg = pricing_cfg.export_tariff if pricing_cfg else None
        import_series = self._apply_tariff_series(aligned, index, tz, import_cfg, "import")
        export_series = self._apply_tariff_series(aligned, index, tz, export_cfg, "export")
        return aligned, import_series, export_series, imputed_mask

    def _apply_tariff_series(
        self,
        base_series: pd.Series,
        index: pd.DatetimeIndex,
        tz: str,
        tariff: Optional[TariffSettings],
        scope: str,
    ) -> pd.Series:
        series = base_series.reindex(index, method="ffill")
        cfg = tariff or (PricingSettings().import_tariff if scope == "import" else PricingSettings().export_tariff)
        mode = cfg.mode if cfg.mode in {"constant", "spot_plus_constant", "dual_rate"} else "spot_plus_constant"
        if mode == "constant" and cfg.constant_eur_per_kwh is not None:
            return pd.Series(float(cfg.constant_eur_per_kwh), index=index)
        if mode == "spot_plus_constant":
            offset = float(cfg.spot_offset_eur_per_kwh)
            return series.add(offset, fill_value=0.0)
        if mode == "dual_rate":
            peak_val = cfg.dual_peak_eur_per_kwh
            offpeak_val = cfg.dual_offpeak_eur_per_kwh
            if peak_val is None or offpeak_val is None:
                if cfg.constant_eur_per_kwh is not None:
                    return pd.Series(float(cfg.constant_eur_per_kwh), index=index)
                return series.copy()
            mask = self._build_dual_rate_mask(index, tz, cfg.dual_peak_start_local, cfg.dual_peak_end_local)
            values = pd.Series(float(offpeak_val), index=index)
            values[mask] = float(peak_val)
            return values
        if cfg.constant_eur_per_kwh is not None:
            return pd.Series(float(cfg.constant_eur_per_kwh), index=index)
        return series.copy()

    def _build_dual_rate_mask(
        self,
        index: pd.DatetimeIndex,
        tz: str,
        start_literal: str,
        end_literal: str,
    ) -> pd.Series:
        try:
            start_hour = int(start_literal[:2])
            start_min = int(start_literal[3:5])
            end_hour = int(end_literal[:2])
            end_min = int(end_literal[3:5])
        except (ValueError, TypeError):
            start_hour = 7
            start_min = 0
            end_hour = 21
            end_min = 0
        local_index = index.tz_convert(tz) if index.tz is not None else index.tz_localize("UTC").tz_convert(tz)
        local_minutes = local_index.hour * 60 + local_index.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        if start_minutes == end_minutes:
            mask = pd.Series(True, index=index)
        elif start_minutes < end_minutes:
            mask = pd.Series((local_minutes >= start_minutes) & (local_minutes < end_minutes), index=index)
        else:
            mask = pd.Series((local_minutes >= start_minutes) | (local_minutes < end_minutes), index=index)
        return mask

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
            if batt_to_grid > power_tol:
                mode = "discharge"
                power = net_batt
                reason = "Exporting surplus from battery"
            else:
                mode = "none"
                power = 0.0
                reason = "Covering load from battery"
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

    async def _update_entity_selection(
        self,
        current: Optional[EntitySelection],
        block: Dict[str, Any],
        catalog_map: Dict[str, Any],
    ) -> Tuple[Optional[EntitySelection], bool]:
        """Updates EntitySelection. Returns (updated_selection, changed)."""
        entity_id = block.get("entity_id", current.entity_id if current else None)
        
        if entity_id and entity_id not in catalog_map:
            _LOGGER.warning("Recorder metadata missing for statistic '%s'", entity_id)

        changed = False
        if not current and entity_id:
            current = EntitySelection(entity_id=entity_id)
            changed = True
        elif current and current.entity_id != entity_id:
            if entity_id is None: return None, True
            current.entity_id = entity_id
            changed = True
            
        if not current: return None, changed

        new_unit = block.get("unit")
        if not new_unit and current.entity_id:
            meta = catalog_map.get(current.entity_id)
            new_unit = meta.get("unit") if meta else None
            if not new_unit and self._ha:
                try:
                    state = await self._call_in_thread(self._ha.get_entity_state_sync, current.entity_id)
                    if state: new_unit = state.get("attributes", {}).get("unit_of_measurement")
                except Exception: pass
        
        if new_unit and current.unit != new_unit:
            current.unit = new_unit
            changed = True
            
        return current, changed

    async def get_settings_payload(self) -> dict[str, Any]:
        async with self._settings_lock:
            serialized = self._serialize_settings(self._settings)
        return await self._build_settings_response(serialized)

    async def update_ha_entities_settings(self, data: Dict[str, Any]) -> dict[str, Any]:
        payload = self._require_mapping(data)
        catalog = await self.refresh_statistics_catalog()
        catalog_map: Dict[str, Any] = {str(item.get("statistic_id")): item for item in catalog if item.get("statistic_id")}
        
        entities = await self.refresh_entity_catalog()
        entity_map_catalog: Dict[str, Any] = {
            str(item.get("entity_id")): item
            for item in entities
            if isinstance(item, dict) and item.get("entity_id")
        }

        changed = False
        catalog_changed = False
        async with self._settings_lock:
            settings = self._settings
            ha_entities = settings.ha_entities

            async def _apply(key: str, current: Optional[EntitySelection]) -> Tuple[Optional[EntitySelection], bool]:
                block = payload.get(key)
                if not isinstance(block, dict):
                    return current, False
                return await self._update_entity_selection(current, block, catalog_map)

            new_house, changed_house = await _apply("house_consumption", ha_entities.house_consumption)
            if new_house is not None:
                ha_entities.house_consumption = new_house
                if changed_house:
                    changed = True
                    catalog_changed = True

            new_pv, changed_pv = await _apply("pv_power", ha_entities.pv_power)
            if new_pv is not None:
                ha_entities.pv_power = new_pv
                if changed_pv:
                    changed = True
                    catalog_changed = True

            if "soc_sensor" in payload:
                block = payload.get("soc_sensor")
                if block is None:
                    if ha_entities.soc_sensor is not None:
                        ha_entities.soc_sensor = None
                        changed = True
                elif isinstance(block, dict):
                    # Use entity_map_catalog for soc_sensor as it might not be a statistic
                    ha_entities.soc_sensor, sensor_changed = await self._update_entity_selection(
                        ha_entities.soc_sensor, block, entity_map_catalog
                    )
                    if sensor_changed:
                        changed = True
                else:
                    raise ValueError("soc_sensor must be null or an object with entity_id")

            if "export_power_limited" in payload:
                flag = payload.get("export_power_limited")
                if isinstance(flag, str):
                    normalized = flag.strip().lower()
                    if normalized in {"true", "1", "yes", "on"}:
                        flag_bool = True
                    elif normalized in {"false", "0", "no", "off"}:
                        flag_bool = False
                    else:
                        raise ValueError("export_power_limited must be a boolean")
                elif isinstance(flag, bool):
                    flag_bool = flag
                elif flag is None:
                    flag_bool = False
                else:
                    raise ValueError("export_power_limited must be a boolean")
                if ha_entities.export_power_limited != flag_bool:
                    ha_entities.export_power_limited = flag_bool
                    changed = True

            if changed:
                save_settings(settings)
            serialized = self._serialize_settings(settings)

        return await self._build_settings_response(
            serialized,
            refresh_stats=catalog_changed,
            refresh_entities=catalog_changed,
        )

    async def update_battery_settings(self, data: Dict[str, Any]) -> dict[str, Any]:
        payload = self._require_mapping(data)

        changed = False
        async with self._settings_lock:
            battery = self._settings.battery

            def _parse_positive(name: str, raw_value: Any) -> float:
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} must be a positive number") from None
                if not math.isfinite(numeric) or numeric <= 0:
                    raise ValueError(f"{name} must be a positive number")
                return numeric

            def _parse_soc_fraction(name: str, raw_value: Any, *, minimum: float, maximum: float) -> float:
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} must be between {minimum} and {maximum}") from None
                if not math.isfinite(numeric) or not (minimum <= numeric <= maximum):
                    raise ValueError(f"{name} must be between {minimum} and {maximum}")
                return numeric

            if "wear_cost_eur_per_kwh" in payload:
                raw_cost = payload.get("wear_cost_eur_per_kwh")
                if raw_cost is None:
                     raise ValueError("wear_cost_eur_per_kwh cannot be null")
                try:
                    cost_value = float(raw_cost)
                except (TypeError, ValueError):
                    raise ValueError("wear_cost_eur_per_kwh must be a number") from None
                if not math.isfinite(cost_value) or cost_value < 0:
                    raise ValueError("wear_cost_eur_per_kwh must be a non-negative finite number")
                if not math.isclose(battery.wear_cost_eur_per_kwh, cost_value, rel_tol=1e-9, abs_tol=1e-9):
                    battery.wear_cost_eur_per_kwh = cost_value
                    changed = True

            if "capacity_kwh" in payload:
                cap_value = _parse_positive("capacity_kwh", payload.get("capacity_kwh"))
                if not math.isclose(battery.capacity_kwh, cap_value, rel_tol=1e-9, abs_tol=1e-9):
                    battery.capacity_kwh = cap_value
                    changed = True

            if "power_limit_kw" in payload:
                power_value = _parse_positive("power_limit_kw", payload.get("power_limit_kw"))
                if not math.isclose(battery.power_limit_kw, power_value, rel_tol=1e-9, abs_tol=1e-9):
                    battery.power_limit_kw = power_value
                    changed = True

            soc_min_value = battery.soc_min
            soc_max_value = battery.soc_max
            if "soc_min" in payload:
                soc_min_value = _parse_soc_fraction("soc_min", payload.get("soc_min"), minimum=0.0, maximum=0.98)
            if "soc_max" in payload:
                soc_max_value = _parse_soc_fraction("soc_max", payload.get("soc_max"), minimum=0.02, maximum=1.0)
            if soc_max_value <= soc_min_value:
                raise ValueError("soc_max must be greater than soc_min")
            if not math.isclose(battery.soc_min, soc_min_value, rel_tol=1e-9, abs_tol=1e-9):
                battery.soc_min = soc_min_value
                changed = True
            if not math.isclose(battery.soc_max, soc_max_value, rel_tol=1e-9, abs_tol=1e-9):
                battery.soc_max = soc_max_value
                changed = True

            self._battery_wear_cost = battery.wear_cost_eur_per_kwh
            self._apply_battery_settings_to_config()

            if changed:
                save_settings(self._settings)
            serialized = self._serialize_settings(self._settings)

        return await self._build_settings_response(serialized)

    async def update_pricing_settings(self, data: Dict[str, Any]) -> dict[str, Any]:
        payload = self._require_mapping(data)

        changed = False
        async with self._settings_lock:
            current_pricing = self._settings.pricing
            new_pricing = coerce_pricing_payload(payload, current_pricing)
            if current_pricing != new_pricing:
                self._settings.pricing = new_pricing
                changed = True
            if changed:
                save_settings(self._settings)
            serialized = self._serialize_settings(self._settings)

        return await self._build_settings_response(serialized)

    async def update_settings(self, data: Dict[str, Any]) -> dict[str, Any]:
        """Update one or more settings sections with a partial payload."""
        payload = self._require_mapping(data)

        # Track if catalogs need refresh (only inverter updates trigger this)
        refresh_stats = False
        refresh_entities = False

        # Update each section if present in payload
        if "ha_entities" in payload:
            # HA entities updates may require catalog refresh
            result = await self.update_ha_entities_settings(payload["ha_entities"])
            # Extract refresh flags from result by checking if catalogs changed
            # The update already saved settings and refreshed catalogs if needed
            # We'll skip the final refresh since it was already done
            return result  # Already includes full response
        
        # Backward compatibility for "inverter" key
        if "inverter" in payload:
             result = await self.update_ha_entities_settings(payload["inverter"])
             return result

        if "battery" in payload:
            await self.update_battery_settings(payload["battery"])

        if "pricing" in payload:
            await self.update_pricing_settings(payload["pricing"])

        # Return unified response
        async with self._settings_lock:
            serialized = self._serialize_settings(self._settings)

        return await self._build_settings_response(
            serialized,
            refresh_stats=refresh_stats,
            refresh_entities=refresh_entities,
        )


    async def refresh_statistics_catalog(self) -> List[Dict[str, Any]]:
        try:
            await self._ensure_home_assistant()
        except Exception:
            return self._stat_catalog
        if self._ha is None:
            return self._stat_catalog

        try:
            catalog = await self._call_in_thread(self._ha.list_statistic_ids_sync)
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

        try:
            entities = await self._call_in_thread(self._ha.list_entities_sync)
        except Exception as exc:  # pragma: no cover - best-effort logging
            _LOGGER.debug("Failed to fetch entity metadata: %s", exc)
            return self._entity_catalog

        allowed_power_units = {"w", "kw", "mw", "wh", "kwh", "mwh"}
        percent_units = {"%", "percent", "percentage"}
        filtered: list[dict[str, Any]] = []
        allowed_domains = {"sensor", "number", "input_number", "select", "input_select"}
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            entity_id = entry.get("entity_id")
            if not entity_id:
                continue
            domain = entity_id.split(".", 1)[0]
            if domain not in allowed_domains:
                continue
            attrs = entry.get("attributes") or {}
            device_class = (attrs.get("device_class") or "").lower()
            unit = attrs.get("unit_of_measurement")
            unit_norm = unit.lower() if isinstance(unit, str) else None
            category = "other"
            if device_class in {"power", "energy"} or (unit_norm in allowed_power_units):
                category = "power"
            elif device_class in {"battery"} or (unit_norm in percent_units):
                category = "battery"
            
            # For driver config, we want to include everything in allowed domains
            filtered.append({
                "entity_id": entity_id,
                "name": attrs.get("friendly_name") or entity_id,
                "unit": unit,
                "device_class": device_class,
                "category": category,
            })
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
            ha_entities = settings.ha_entities
            changed = False
            for selection in (ha_entities.house_consumption, ha_entities.pv_power):
                unit = unit_map.get(selection.entity_id)
                if unit and unit != selection.unit:
                    selection.unit = unit
                    changed = True
            if changed:
                save_settings(settings)

    def _apply_battery_settings_to_config(self) -> None:
        battery = self._settings.battery
        cap = float(battery.capacity_kwh) if battery.capacity_kwh is not None else 0.0
        power = float(battery.power_limit_kw) if battery.power_limit_kw is not None else 0.0
        self._battery_cfg.cap_kwh = max(0.1, cap)
        self._battery_cfg.p_max_kw = max(0.1, power)
        try:
            soc_min = float(battery.soc_min)
        except (TypeError, ValueError):
            soc_min = 0.1
        try:
            soc_max = float(battery.soc_max)
        except (TypeError, ValueError):
            soc_max = 0.9
        soc_min = float(np.clip(soc_min, 0.0, 0.95))
        soc_max = float(np.clip(soc_max, soc_min + 0.01, 1.0))
        if soc_max <= soc_min:
            soc_max = min(1.0, soc_min + 0.05)
        self._battery_cfg.soc_min = soc_min
        self._battery_cfg.soc_max = soc_max

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

    @staticmethod
    def _require_mapping(value: Any, label: str = "Payload") -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        raise ValueError(f"{label} must be an object")

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

    async def _build_settings_response(
        self,
        serialized: dict[str, Any],
        *,
        refresh_stats: bool = False,
        refresh_entities: bool = False,
    ) -> dict[str, Any]:
        if refresh_stats or not self._stat_catalog:
            await self.refresh_statistics_catalog()
        if refresh_entities or not self._entity_catalog:
            await self.refresh_entity_catalog()
        serialized = await self._attach_recorder_status(serialized)
        return {
            "settings": serialized,
            "statistics": list(self._stat_catalog),
            "entities": list(self._entity_catalog),
        }

    async def _attach_recorder_status(self, serialized: dict[str, Any]) -> dict[str, Any]:
        ha_entities = serialized.get("ha_entities")
        if not isinstance(ha_entities, dict):
            return serialized

        async def _apply(key: str) -> None:
            block = ha_entities.get(key)
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
        try:
            df = await self._call_in_thread(self._ha.fetch_last_hours_sync, [(stat_id, "mean")], 48)
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
            ha_entities = self._settings.ha_entities
            stat_ids = list(ha_entities.stat_ids())
            rename_map = dict(ha_entities.rename_map())
            scales = dict(ha_entities.scales())
        return stat_ids, rename_map, scales

    async def _get_pricing_config(self) -> PricingSettings:
        async with self._settings_lock:
            return copy.deepcopy(self._settings.pricing)

    def _serialize_settings(self, settings: AppSettings) -> dict[str, Any]:
        return settings.to_dict()

    def _run_cycle_sync(
        self,
        stat_ids: List[Tuple[str, str]],
        rename_map: Dict[str, str],
        scales: Dict[str, float],
    ) -> Optional[ForecastSnapshot]:
        if self._ha is None or self._lat is None or self._lon is None or self._tz is None:
            raise RuntimeError("Home Assistant not initialized")

        # 1. Run Prediction Pipeline
        _LOGGER.info("Starting forecast cycle")
        pred_result = run_prediction_pipeline(
            ha=self._ha,
            lat=self._lat,
            lon=self._lon,
            tz=self._tz,
            models_dir=str(self.models_dir),
            horizon_hours=HORIZON_HOURS_DEFAULT,
            interval_minutes=INTERVAL_MINUTES_DEFAULT,
            entities=stat_ids,
            rename_map=rename_map,
            scales=scales,
            return_features=True,  # We might need features for debug or advanced logic
        )
        
        pv_pred = pred_result["pv_pred"]
        house_pred = pred_result["house_pred"]
        ha_recent = pred_result["ha_recent"]
        
        # 2. Prepare Optimization Inputs
        pricing_settings = self._settings.pricing
        battery_settings = self._settings.battery
        
        # Fetch prices
        now = datetime.now(timezone.utc)
        
        try:
            spot_prices, import_price, export_price, price_imputed = self._build_price_series(
                self._tz, pv_pred.index, pricing_settings
            )
        except ValueError as exc:
            _LOGGER.error("Price data error: %s", exc)
            # Re-raise to stop the cycle and show error in UI
            raise
        
        # 3. Run Optimization
        # We need battery config
        batt_cfg = BatteryConfig(
            cap_kwh=battery_settings.capacity_kwh,
            p_max_kw=battery_settings.power_limit_kw,
            soc_min=battery_settings.soc_min,
            soc_max=battery_settings.soc_max,
        )
        
        # Fetch initial SoC
        initial_soc = 0.5
        soc_sensor = self._settings.ha_entities.soc_sensor
        if soc_sensor and soc_sensor.entity_id:
            try:
                state = self._ha.get_entity_state_sync(soc_sensor.entity_id)
                if state:
                    raw_state = state.get("state")
                    if raw_state is not None:
                        val = float(raw_state)
                        # Normalize
                        initial_soc = self._normalize_soc(val, soc_sensor.unit) or 0.5
            except Exception as exc:
                _LOGGER.warning("Failed to fetch initial SoC: %s", exc)

        # Construct S DataFrame for optimization
        # Columns: 'pv_kw', 'load_kw', 'dt_h', 'price_eur_per_kwh' (base), 'import_price_eur_per_kwh', 'export_price_eur_per_kwh'
        dt_h = INTERVAL_MINUTES_DEFAULT / 60.0
        S = pd.DataFrame(index=pv_pred.index)
        S["pv_kw"] = pv_pred[PV_COL]
        S["load_kw"] = house_pred[TARGET_COL]
        S["dt_h"] = dt_h
        S["import_price_eur_per_kwh"] = import_price
        S["export_price_eur_per_kwh"] = export_price
        # Base price for reference, can be avg
        S["price_eur_per_kwh"] = (import_price + export_price) / 2.0

        plan = solve_lp_optimal_control(
            S=S,
            cfg=batt_cfg,
            deg_eur_per_kwh=battery_settings.wear_cost_eur_per_kwh,
            soc0=initial_soc,
            export_enabled=self._settings.pricing.export_enabled,
            export_limit_kw=self._settings.pricing.export_limit_kw,
        )
        
        # 4. Detect intervention type from first step
        intervention = self._detect_intervention(plan)
        
        # 5. Create Snapshot
        summary = self._summarize_plan(plan)
        
        snapshot = ForecastSnapshot(
            timestamp=now,
            interval_minutes=INTERVAL_MINUTES_DEFAULT,
            horizon_hours=HORIZON_HOURS_DEFAULT,
            forecast_index=pv_pred.index,
            pv_forecast=pv_pred[PV_COL],
            load_forecast=house_pred[TARGET_COL],
            price_forecast=import_price, # simplified
            export_price_forecast=export_price,
            spot_price_forecast=spot_prices,
            price_imputed=price_imputed,
            plan=plan,
            ha_recent=ha_recent,
            summary=summary,
            intervention=intervention,
            timezone=self._tz,
            locale=self._ha.get_locale()
        )
        
        return snapshot
    
    def _detect_intervention(self, plan: pd.DataFrame) -> dict:
        """Detect intervention type from optimization plan."""
        if plan.empty:
            return {"type": "Unknown", "description": "No plan available"}
        
        # Get first step
        first_row = plan.iloc[0]
        grid_to_batt = float(first_row.get("grid_to_batt_kw", 0.0))
        batt_to_load = float(first_row.get("batt_to_load_kw", 0.0))
        batt_to_grid = float(first_row.get("batt_to_grid_kw", 0.0))
        pv_to_batt = float(first_row.get("pv_to_batt_kw", 0.0))
        
        MIN_POWER_KW = 0.05  # 50W threshold
        
        battery_charge_kw = grid_to_batt + pv_to_batt
        battery_discharge_kw = batt_to_load + batt_to_grid
        
        # Determine intervention type
        if battery_charge_kw < MIN_POWER_KW and battery_discharge_kw < MIN_POWER_KW:
            return {
                "type": "Disable Battery",
                "description": "Battery idle, all load covered from grid"
            }
        elif grid_to_batt > MIN_POWER_KW:
            return {
                "type": "Charge from Grid",
                "description": f"Charging battery from grid at {grid_to_batt:.2f} kW",
                "target_power_kw": grid_to_batt
            }
        elif batt_to_grid > MIN_POWER_KW:
            return {
                "type": "Discharge to Grid",
                "description": f"Discharging battery to grid at {batt_to_grid:.2f} kW",
                "target_power_kw": batt_to_grid
            }
        elif batt_to_load > MIN_POWER_KW:
            return {
                "type": "Cover Load from Battery",
                "description": f"Battery covering {batt_to_load:.2f} kW of load",
                "target_power_kw": batt_to_load
            }
        else:
            return {
                "type": "Disable Battery",
                "description": "Battery idle"
            }

    async def _get_initial_soc(self) -> Optional[float]:
        async with self._settings_lock:
            selection = self._settings.ha_entities.soc_sensor
            entity_id = selection.entity_id if selection else None
            stored_unit = selection.unit if selection else None
        if not entity_id:
            return None

        await self._ensure_home_assistant()
        if self._ha is None:
            return None

        try:
            state = await self._call_in_thread(self._ha.get_entity_state_sync, entity_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.debug("Failed to fetch battery SoC from '%s': %s", entity_id, exc)
            return None

        if not state:
            return None

        raw_value = state.get("state")
        if raw_value is None:
            return None
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            _LOGGER.debug("Battery SoC sensor '%s' returned non-numeric state '%s'", entity_id, raw_value)
            return None

        attributes = state.get("attributes") or {}
        state_unit = attributes.get("unit_of_measurement") if isinstance(attributes, dict) else None
        unit_for_norm = state_unit or stored_unit

        if state_unit and state_unit != stored_unit:
            async with self._settings_lock:
                current = self._settings.ha_entities.soc_sensor
                if current and current.entity_id == entity_id and current.unit != state_unit:
                    current.unit = state_unit
                    save_settings(self._settings)
            unit_for_norm = state_unit

        fraction = self._normalize_soc(numeric, unit_for_norm)
        if fraction is None:
            return None
        return fraction

    @staticmethod
    def _normalize_soc(value: float, unit: Optional[str]) -> Optional[float]:
        if not math.isfinite(value):
            return None
        unit_clean = unit.strip().lower() if isinstance(unit, str) else None
        if unit_clean in {"%", "percent", "percentage"}:
            value = value / 100.0
        elif unit_clean in {"fraction", "ratio"}:
            pass
        elif value > 1.5 and not unit_clean:
            # Heuristic: treat large values without units as percentage inputs.
            value = value / 100.0
        fraction = float(np.clip(value, 0.0, 1.0))
        return fraction

    @staticmethod
    def _summarize_plan(plan: pd.DataFrame) -> dict[str, float]:
        import_kwh = float(plan.get("import_kwh", pd.Series(dtype=float)).sum())
        export_kwh = float(plan.get("export_kwh", pd.Series(dtype=float)).sum())
        net_cost = float((plan.get("cost_import_eur", pd.Series(dtype=float)) - plan.get("rev_export_eur", pd.Series(dtype=float))).sum())
        peak_import = float(plan.get("grid_import_kw", pd.Series(dtype=float)).max()) if not plan.empty else 0.0
        peak_export = float(plan.get("grid_export_kw", pd.Series(dtype=float)).max()) if not plan.empty else 0.0
        pv_energy = float(
            plan.get("pv_kw", pd.Series(dtype=float)).mul(plan.get("dt_h", pd.Series(dtype=float)), fill_value=0).sum()
        )
        consumption = float(
            plan.get("load_kw", pd.Series(dtype=float)).mul(plan.get("dt_h", pd.Series(dtype=float)), fill_value=0).sum()
        )
        return {
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "net_cost_eur": net_cost,
            "peak_import_kw": peak_import,
            "peak_export_kw": peak_export,
            "pv_energy_kwh": pv_energy,
            "consumption_kwh": consumption,
        }

    async def get_snapshot(self) -> Optional[ForecastSnapshot]:
        async with self._lock:
            if self._snapshot is None:
                _LOGGER.debug("get_snapshot returning None")
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
            await self._call_in_thread(
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
