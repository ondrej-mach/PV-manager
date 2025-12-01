"""Benchmark dumb eco mode vs optimal control with realistic rolling predictions.

This benchmark simulates real-world operation:
- Each day at 13:00, run predictions for next 24 hours
- Optimize battery control based on predictions (not actual values)
- Execute hourly, then repeat next day
- Compare actual costs vs baseline
"""
import logging
import os
import sys
from pathlib import Path

# Aggressive warning suppression - must be before ALL imports
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="unclosed.*ssl.SSLSocket")
warnings.filterwarnings("ignore", message="unclosed transport")
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.io.home_assistant import HomeAssistant
from energy_forecaster.io.benchmark_data import download_benchmark_data, load_benchmark_data
from energy_forecaster.services.baseline_control import simulate_dumb_eco_mode, BatteryConfig
from energy_forecaster.services.prediction_service import load_models
from energy_forecaster.features.data_prep import (
    PV_COL,
    TARGET_COL,
    assemble_forecast_features,
)
from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
def get_optimal_control():
    try:
        from energy_forecaster.services.optimal_control import solve_lp_optimal_control
        return solve_lp_optimal_control
    except ModuleNotFoundError:
        return None


MODELS_DIR = os.getenv("MODELS_DIR", "trained_models")
FALLBACK_LAT = 49.6069
FALLBACK_LON = 15.5808
FALLBACK_TZ = "Europe/Prague"
HORIZON_HOURS = 24  # Prediction/optimization horizon
REOPTIMIZE_INTERVAL_HOURS = float(os.getenv("REOPTIMIZE_INTERVAL_HOURS", "1.0"))  # How often to re-optimize (1h default, can be 0.25 for 15min)

# TODO: expose as configuration; legacy models currently emit power in watts.
MODEL_OUTPUT_DIVISOR = 1000.0


def _to_month_period(index: pd.DatetimeIndex, target_tz: str | None = None) -> pd.PeriodIndex:
    """Convert a datetime index to monthly periods without timezone warnings."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex for monthly conversion")
    if index.tz is not None:
        tz_name = target_tz or "UTC"
        index = index.tz_convert(tz_name).tz_localize(None)
    return index.to_period("M")


def _get_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _plan_to_result(plan: pd.DataFrame) -> pd.DataFrame:
    if plan is None or plan.empty:
        return plan.iloc[0:0]

    plan = plan.copy()
    dt_h = _get_series(plan, "dt_h", 1.0)
    pv_kw = _get_series(plan, "pv_kw")
    load_kw = _get_series(plan, "load_kw")
    soc = _get_series(plan, "soc")

    pv_to_batt = _get_series(plan, "pv_to_batt_kw")
    grid_to_batt = _get_series(plan, "grid_to_batt_kw")
    batt_to_load = _get_series(plan, "batt_to_load_kw")
    batt_to_grid = _get_series(plan, "batt_to_grid_kw")
    grid_import = _get_series(plan, "grid_import_kw")
    grid_export = _get_series(plan, "grid_export_kw")

    buy_price = _get_series(plan, "buy_price_eur_per_kwh")
    sell_price = _get_series(plan, "sell_price_eur_per_kwh")
    if sell_price.eq(0).all() and "price_eur_per_kwh" in plan.columns:
        sell_price = _get_series(plan, "price_eur_per_kwh")
    if buy_price.eq(0).all() and "price_eur_per_kwh" in plan.columns:
        buy_price = _get_series(plan, "price_eur_per_kwh")

    import_kwh = grid_import * dt_h
    export_kwh = grid_export * dt_h

    cost_import = _get_series(plan, "cost_import_eur")
    rev_export = _get_series(plan, "rev_export_eur")
    if cost_import.eq(0).all() and buy_price.notna().any():
        cost_import = import_kwh * buy_price
    if rev_export.eq(0).all() and sell_price.notna().any():
        rev_export = export_kwh * sell_price

    return pd.DataFrame(index=plan.index, data={
        "pv_kw": pv_kw,
        "load_kw": load_kw,
        "dt_h": dt_h,
        "soc": soc,
        "batt_charge_kw": pv_to_batt + grid_to_batt,
        "batt_discharge_kw": batt_to_load + batt_to_grid,
        "grid_import_kw": grid_import,
        "grid_export_kw": grid_export,
        "import_kwh": import_kwh,
        "export_kwh": export_kwh,
        "cost_import_eur": cost_import,
        "rev_export_eur": rev_export,
        "price_eur_per_kwh": sell_price,
    })


def _solve_true_optimal(S: pd.DataFrame, cfg: BatteryConfig) -> pd.DataFrame | None:
    solve_lp = get_optimal_control()
    if solve_lp is None:
        return None

    if S.empty:
        return S.copy()

    chunk_sizes = [len(S), 7 * 24, 3 * 24, 24]
    soc0 = 0.5
    last_exc: Exception | None = None

    for chunk_size in chunk_sizes:
        chunk_size = max(24, min(chunk_size, len(S)))
        try:
            results: list[pd.DataFrame] = []
            start = 0
            current_soc = soc0
            while start < len(S):
                end = min(start + chunk_size, len(S))
                segment = S.iloc[start:end]
                plan = solve_lp(
                    segment,
                    cfg,
                    soc0=current_soc,
                    force_terminal_soc=False,
                )
                segment_result = _plan_to_result(plan)
                if segment_result.empty:
                    break
                results.append(segment_result)
                current_soc = float(segment_result["soc"].iloc[-1])
                start = end
            if results:
                return pd.concat(results)
            return pd.DataFrame(index=S.index, columns=[])
        except Exception as exc:  # pragma: no cover - solver issues fallback
            last_exc = exc
            continue

    if last_exc is not None:
        logger.warning("True optimal solve failed: %s", last_exc)
    return None

# Benchmark window selection controls
# Option A: Choose a specific month like "2024-06"
BENCHMARK_MONTH = os.getenv("BENCHMARK_MONTH", "").strip()
# Option B: Explicit date range (YYYY-MM-DD). If set, overrides cached data range when downloading
BENCHMARK_START = os.getenv("BENCHMARK_START", "").strip()
BENCHMARK_END = os.getenv("BENCHMARK_END", "").strip()
# Option C: Segment analysis (e.g., 14-day non-overlapping windows across available data)
SEGMENT_DAYS = int(os.getenv("SEGMENT_DAYS", "0").strip() or 0)


def simulate_rolling_optimal(actual_data, weather, prices, cfg, reoptimize_interval_h=1.0,
                             tz: str = "Europe/Prague", models_dir: str = MODELS_DIR):
    """Simulate optimal control with Model Predictive Control (MPC) style rolling optimization.
    
    NOTE: Uses trained forecasting models to generate PV/load predictions at each
    re-optimization step. If models or features are unavailable the routine
    falls back to perfect foresight using actual values.
    
    MPC Loop:
    - Every `reoptimize_interval_h` hours:
      1. Get predictions for next 24 hours (currently: use actual data)
      2. Run optimization for next 24 hours
      3. Execute only the next `reoptimize_interval_h` hours from the plan
      4. Update battery SoC
      5. Repeat
    
    Args:
        actual_data: DataFrame with actual pv_kw, load_kw (hourly UTC)
        prices: DataFrame with prices
        cfg: BatteryConfig
        reoptimize_interval_h: How often to re-optimize (1.0 = hourly, 0.25 = every 15min)
    
    Returns:
        DataFrame with actual execution results (soc, flows, costs)
    """
    solve_lp = get_optimal_control()
    if solve_lp is None:
        raise RuntimeError("cvxpy not available")

    try:
        house_model, pv_model = load_models(models_dir)
    except Exception as exc:
        house_model = None
        pv_model = None
        logger.warning("Failed to load prediction models (%s); using actuals as forecasts", exc)
    
    # Initialize battery state
    soc = 0.5  # Start at 50%

    pv_clip_upper = float(actual_data["pv_kw"].max()) if not actual_data["pv_kw"].empty else 0.0
    load_clip_upper = float(actual_data["load_kw"].max()) if not actual_data["load_kw"].empty else 0.0
    pv_clip_upper = pv_clip_upper * 1.5 if pv_clip_upper > 0 else None
    load_clip_upper = load_clip_upper * 1.5 if load_clip_upper > 0 else None
    
    # Results storage
    results = []

    load_forecast_fallback_warned = False

    # Convert reoptimize interval to number of timesteps
    dt_h = 1.0  # We have hourly data
    steps_per_reopt = int(reoptimize_interval_h / dt_h)
    if steps_per_reopt < 1:
        steps_per_reopt = 1
        logger.warning("reoptimize_interval_h=%s < data resolution (1h), using 1 step", reoptimize_interval_h)
    
    horizon_steps = int(HORIZON_HOURS / dt_h)  # 24 steps for 24h horizon
    
    # Simulate hour by hour with periodic re-optimization
    total_steps = len(actual_data)
    current_step = 0
    
    logger.info("MPC simulation: re-optimize every %sh, horizon=%sh", reoptimize_interval_h, HORIZON_HOURS)
    logger.info("Total timesteps: %s, re-optimization points: %s", total_steps, total_steps // steps_per_reopt)
    
    reopt_count = 0
    while current_step < total_steps:
        # Time to re-optimize?
        if current_step % steps_per_reopt == 0:
            reopt_count += 1
            if reopt_count % 50 == 0:
                current_time = actual_data.index[current_step]
                logger.info(
                    "Re-optimization %s at step %s/%s: %s",
                    reopt_count,
                    current_step,
                    total_steps,
                    current_time,
                )
            
            # Get next horizon_steps (or remaining) for optimization
            end_step = min(current_step + horizon_steps, total_steps)
            horizon_data = actual_data.iloc[current_step:end_step]
            
            if len(horizon_data) == 0:
                break
            
            # Get prices for this horizon
            horizon_prices = prices.reindex(horizon_data.index).ffill().bfill()["price_eur_per_kwh"]
            
            # Build optimization input S with actual data (perfect foresight)
            S_actual = pd.DataFrame(index=horizon_data.index, data={
                "pv_kw": horizon_data["pv_kw"].values,
                "load_kw": horizon_data["load_kw"].values,
                "dt_h": dt_h,
                "price_eur_per_kwh": horizon_prices.values,
            })

            use_forecast = house_model is not None and pv_model is not None
            if use_forecast:
                try:
                    history_hours = max(96, int(5 * 24 / dt_h))
                    history_start_idx = max(0, current_step - history_hours)
                    history_slice = actual_data.iloc[history_start_idx:current_step]
                    if history_slice.empty:
                        raise ValueError("insufficient history for forecast features")

                    ha_recent_raw = pd.DataFrame(index=history_slice.index, data={
                        "sensor.house_consumption": history_slice["load_kw"] * 1000.0,
                        "sensor.pv_power": history_slice["pv_kw"] * 1000.0,
                    })

                    wx_future = weather.reindex(S_actual.index).ffill().bfill()
                    feats = assemble_forecast_features(ha_recent_raw, wx_future, tz=tz)

                    pv_X = feats["pv_X"]
                    pv_pred_vals = pv_model.predict(pv_X.values)
                    pv_pred_raw = pd.Series(pv_pred_vals, index=pv_X.index, dtype=float)
                    pv_pred = (pv_pred_raw / MODEL_OUTPUT_DIVISOR).clip(lower=0.0)
                    pv_pred = pv_pred.fillna(0.0)

                    house_X = feats["house_X"].copy()
                    house_X[PV_COL] = pv_pred
                    house_X_clean = house_X.dropna()

                    load_pred = None
                    if len(house_X_clean) > 0:
                        load_pred_vals = house_model.predict(house_X_clean.values)
                        load_pred_raw = pd.Series(load_pred_vals, index=house_X_clean.index, dtype=float)
                        load_pred_series = (load_pred_raw / MODEL_OUTPUT_DIVISOR).clip(lower=0.0)
                        load_pred = load_pred_series.reindex(pv_pred.index)
                    else:
                        load_pred = pd.Series(index=pv_pred.index, dtype=float)

                    load_pred = load_pred.ffill().bfill()
                    if load_pred.isna().any():
                        if not history_slice.empty:
                            fallback_val = float(history_slice["load_kw"].iloc[-1])
                        else:
                            fallback_val = 0.0
                        load_pred = load_pred.fillna(fallback_val)
                        if load_pred.isna().any():
                            raise ValueError("forecast load contains NaNs after fallback fill")
                        if not load_forecast_fallback_warned:
                            logger.warning("Load forecast missing entries; using persistence fallback")
                            load_forecast_fallback_warned = True

                    pv_forecast = pv_pred.reindex(S_actual.index).ffill().bfill().clip(lower=0.0)
                    load_forecast = load_pred.reindex(S_actual.index).ffill().bfill().clip(lower=0.0)

                    if pv_clip_upper is not None:
                        pv_forecast = pv_forecast.clip(upper=pv_clip_upper)
                    if load_clip_upper is not None:
                        load_forecast = load_forecast.clip(upper=load_clip_upper)

                    S_horizon = pd.DataFrame(index=S_actual.index, data={
                        "pv_kw": pv_forecast.values,
                        "load_kw": load_forecast.values,
                        "dt_h": dt_h,
                        "price_eur_per_kwh": horizon_prices.values,
                    })
                except Exception as forecast_err:
                    logger.warning("Forecast generation failed, using actuals: %s", forecast_err)
                    S_horizon = S_actual
            else:
                S_horizon = S_actual
            
            # Run optimization for this horizon starting from the current SoC
            try:
                opt_plan = solve_lp(S_horizon, cfg, soc0=soc)
            except Exception as e:
                logger.warning("Optimization failed at step %s: %s", current_step, e)
                # Fall back to baseline for this horizon
                opt_plan = simulate_dumb_eco_mode(S_horizon, cfg, soc0=soc)
        
        # Execute the next step from the current plan
        step_in_plan = current_step % steps_per_reopt if current_step % steps_per_reopt < len(opt_plan) else 0
        
        # If we're past the plan length, use the last available step
        if step_in_plan >= len(opt_plan):
            step_in_plan = len(opt_plan) - 1
        
        hour_result = opt_plan.iloc[step_in_plan]
        
        # Map column names from optimal_control output
        # optimal_control outputs: pv_to_batt_kw, grid_to_batt_kw (charging)
        #                         batt_to_load_kw, batt_to_grid_kw (discharging)
        batt_charge = hour_result.get("pv_to_batt_kw", 0) + hour_result.get("grid_to_batt_kw", 0)
        batt_discharge = hour_result.get("batt_to_load_kw", 0) + hour_result.get("batt_to_grid_kw", 0)
        
        # Update SoC based on actual execution
        # (In real system, the SoC would be measured)
        # For now, use the SoC from the plan but this is a simplification
        soc = hour_result["soc"]
        
        results.append({
            "time": actual_data.index[current_step],
            "pv_kw": hour_result["pv_kw"],
            "load_kw": hour_result["load_kw"],
            "soc": soc,
            "batt_charge_kw": batt_charge,
            "batt_discharge_kw": batt_discharge,
            "grid_import_kw": hour_result["grid_import_kw"],
            "grid_export_kw": hour_result["grid_export_kw"],
            "import_kwh": hour_result["import_kwh"],
            "export_kwh": hour_result["export_kwh"],
            "cost_import_eur": hour_result["cost_import_eur"],
            "rev_export_eur": hour_result["rev_export_eur"],
            "price_eur_per_kwh": hour_result["sell_price_eur_per_kwh"] if "sell_price_eur_per_kwh" in hour_result else hour_result.get("price_eur_per_kwh", 0),
        })
        
        current_step += 1
    
    return pd.DataFrame(results).set_index("time")


def _subset_by_month(df: pd.DataFrame, month_str: str) -> pd.DataFrame:
    """Return only rows in the given YYYY-MM month; empty if none."""
    try:
        month = pd.Period(month_str, freq="M")
    except Exception:
        return df.iloc[0:0]
    mask = (df.index.to_period("M") == month)
    return df.loc[mask]


def _run_window_benchmark(S: pd.DataFrame,
                          weather: pd.DataFrame,
                          prices: pd.DataFrame,
                          cfg: BatteryConfig,
                          label: str,
                          reoptimize_interval_h: float,
                          tz: str,
                          models_dir: str = MODELS_DIR) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Run baseline, MPC, and true optimal on a window; return (dumb, mpc, true, summary)."""
    logger.info("Running window: %s | %s to %s (%s hours)", label, S.index[0], S.index[-1], len(S))

    # Dumb eco baseline
    dumb_result = simulate_dumb_eco_mode(S, cfg)
    dumb_cost_import = float(dumb_result["cost_import_eur"].sum())
    dumb_rev_export = float(dumb_result["rev_export_eur"].sum())
    dumb_net_cost = dumb_cost_import - dumb_rev_export

    # Optimal (MPC)
    logger.info("MPC re-opt=%sh, horizon=%sh", reoptimize_interval_h, HORIZON_HOURS)
    opt_result = simulate_rolling_optimal(
        S,
        weather,
        prices,
        cfg,
        reoptimize_interval_h=reoptimize_interval_h,
        tz=tz,
        models_dir=models_dir,
    )
    opt_cost_import = float(opt_result["cost_import_eur"].sum())
    opt_rev_export = float(opt_result["rev_export_eur"].sum())
    opt_net_cost = opt_cost_import - opt_rev_export

    true_result = _solve_true_optimal(S, cfg)
    if true_result is not None and not true_result.empty:
        true_cost_import = float(true_result["cost_import_eur"].sum())
        true_rev_export = float(true_result["rev_export_eur"].sum())
        true_net_cost = true_cost_import - true_rev_export
    else:
        true_result = None
        true_cost_import = float("nan")
        true_rev_export = float("nan")
        true_net_cost = float("nan")

    savings = dumb_net_cost - opt_net_cost
    savings_pct = (savings / abs(dumb_net_cost) * 100.0) if dumb_net_cost else 0.0

    true_savings = dumb_net_cost - true_net_cost if not np.isnan(true_net_cost) else float("nan")
    true_savings_pct = (true_savings / abs(dumb_net_cost) * 100.0) if dumb_net_cost and not np.isnan(true_savings) else float("nan")
    opt_gap_eur = opt_net_cost - true_net_cost if not np.isnan(true_net_cost) else float("nan")
    opt_gap_pct = (opt_gap_eur / abs(true_net_cost) * 100.0) if not np.isnan(true_net_cost) and true_net_cost else float("nan")

    summary = {
        "label": label,
        "period_start": str(S.index[0]),
        "period_end": str(S.index[-1]),
        "hours": len(S),
        "dumb_import_kwh": float(dumb_result["import_kwh"].sum()),
        "dumb_export_kwh": float(dumb_result["export_kwh"].sum()),
        "dumb_cost_eur": dumb_net_cost,
        "opt_import_kwh": float(opt_result["import_kwh"].sum()),
        "opt_export_kwh": float(opt_result["export_kwh"].sum()),
        "opt_cost_eur": opt_net_cost,
        "savings_eur": savings,
        "savings_pct": savings_pct,
        "true_opt_import_kwh": float(true_result["import_kwh"].sum()) if true_result is not None else float("nan"),
        "true_opt_export_kwh": float(true_result["export_kwh"].sum()) if true_result is not None else float("nan"),
        "true_opt_cost_eur": true_net_cost,
        "true_savings_eur": true_savings,
        "true_savings_pct": true_savings_pct,
        "opt_vs_true_gap_eur": opt_gap_eur,
        "opt_vs_true_gap_pct": opt_gap_pct,
    }

    logger.info("Result: MPC savings %.2f € (%.1f%%)", savings, savings_pct)
    if not np.isnan(true_net_cost):
        logger.info("True optimal net cost %.2f € (gap vs MPC %.2f €)", true_net_cost, opt_gap_eur)
    else:
        logger.info("True optimal solution unavailable (see logs)")

    return dumb_result, opt_result, true_result, summary


def main():
    # Setup HA connection
    if not os.getenv("SUPERVISOR_TOKEN"):
        token = os.getenv("HASS_TOKEN")
        url = os.getenv("HASS_WS_URL") or "ws://192.168.1.11:8123/api/websocket"
        if not token:
            logger.error("Please set HASS_TOKEN environment variable")
            return
    else:
        token = None
        url = None

    ha = HomeAssistant(token=token, ws_url=url)

    try:
        lat, lon, tz = ha.get_location(FALLBACK_LAT, FALLBACK_LON, FALLBACK_TZ)

        # Download/load benchmark data
        force_refresh = os.getenv("BENCHMARK_REFRESH", "0").lower() in ("true", "1", "yes")
        
        # Allow overriding the cached window via env
        if BENCHMARK_START and BENCHMARK_END:
            logger.info("Downloading benchmark data for %s to %s...", BENCHMARK_START, BENCHMARK_END)
            download_benchmark_data(ha, lat, lon, tz,
                                    start=BENCHMARK_START, end=BENCHMARK_END,
                                    force_refresh=True)
            ha_stats, weather, prices = load_benchmark_data()
        else:
            if force_refresh:
                logger.info("BENCHMARK_REFRESH requested, re-downloading default range...")
                download_benchmark_data(ha, lat, lon, tz, force_refresh=True)
                ha_stats, weather, prices = load_benchmark_data()
            else:
                try:
                    ha_stats, weather, prices = load_benchmark_data()
                    logger.info("Loaded benchmark data from cache")
                except FileNotFoundError:
                    logger.info("Benchmark data not found, downloading default range...")
                    download_benchmark_data(ha, lat, lon, tz, force_refresh=True)
                    ha_stats, weather, prices = load_benchmark_data()

        # Align all data to hourly UTC
        # Normalize HA stats column names (raw sensor names -> friendly names)
        # Expected columns: sensor.pv_power, sensor.house_consumption
        # HA returns values in Watts, convert to kW
        col_map = {}
        for col in ha_stats.columns:
            if "pv_power" in col.lower():
                col_map[col] = "pv_power_kw"
            elif "house_consumption" in col.lower() or "consumption" in col.lower():
                col_map[col] = "power_consumption_kw"
        
        ha_stats = ha_stats.rename(columns=col_map)
        
        # Convert W to kW
        for col in ["pv_power_kw", "power_consumption_kw"]:
            if col in ha_stats.columns:
                ha_stats.loc[:, col] = ha_stats[col] / 1000.0
        
        # Find common hourly index
        idx = ha_stats.index.intersection(weather.index).intersection(prices.index)
        if len(idx) == 0:
            logger.error("No overlapping data between HA stats, weather, and prices")
            return

        logger.info("Full available period: %s to %s (%s hours, %.0f days)", idx[0], idx[-1], len(idx), len(idx) / 24)

        # Build input DataFrame with actual data
        dt_h = 1.0  # hourly
        S = pd.DataFrame(index=idx, data={
            "pv_kw": ha_stats.loc[idx, "pv_power_kw"].fillna(0).clip(lower=0),
            "load_kw": ha_stats.loc[idx, "power_consumption_kw"].fillna(0).clip(lower=0),
            "dt_h": dt_h,
            "price_eur_per_kwh": prices.loc[idx, "price_eur_per_kwh"].ffill().bfill(),
        })
        weather_aligned = weather.reindex(idx).ffill().bfill()

        # Battery config
        cfg = BatteryConfig()

        dumb_result: pd.DataFrame | None = None
        opt_result: pd.DataFrame | None = None
        true_result: pd.DataFrame | None = None
        summary: dict[str, object] | None = None

        # Optionally narrow to a specific month first
        if BENCHMARK_MONTH:
            S_month = _subset_by_month(S, BENCHMARK_MONTH)
            if S_month.empty:
                logger.warning(
                    "No data found for month %s. Available range: %s .. %s",
                    BENCHMARK_MONTH,
                    S.index[0],
                    S.index[-1],
                )
                return
            prices_month = prices.loc[S_month.index]
            weather_month = weather_aligned.loc[S_month.index]
            dumb_result, opt_result, true_result, summary = _run_window_benchmark(
                S_month,
                weather_month,
                prices_month,
                cfg,
                label=f"month-{BENCHMARK_MONTH}",
                reoptimize_interval_h=REOPTIMIZE_INTERVAL_HOURS,
                tz=tz,
                models_dir=MODELS_DIR,
            )
        elif SEGMENT_DAYS and SEGMENT_DAYS > 0:
            logger.info("Running non-overlapping %s-day segments across available data…", SEGMENT_DAYS)
            segment_hours = SEGMENT_DAYS * 24
            summaries = []
            all_dumb: list[pd.DataFrame] = []
            all_opt: list[pd.DataFrame] = []
            all_true: list[pd.DataFrame] = []
            start_i = 0
            seg_idx = 1
            while start_i < len(S):
                end_i = min(start_i + segment_hours, len(S))
                S_seg = S.iloc[start_i:end_i]
                if len(S_seg) < 24:  # require at least 1 day
                    break
                prices_seg = prices.loc[S_seg.index]
                weather_seg = weather_aligned.loc[S_seg.index]
                label = f"segment-{seg_idx}"
                dumb_res, opt_res, true_res, summ = _run_window_benchmark(
                    S_seg,
                    weather_seg,
                    prices_seg,
                    cfg,
                    label=label,
                    reoptimize_interval_h=REOPTIMIZE_INTERVAL_HOURS,
                    tz=tz,
                    models_dir=MODELS_DIR,
                )
                summaries.append(summ)
                dumb_res = dumb_res.copy(); dumb_res["segment"] = label
                all_dumb.append(dumb_res)
                if opt_res is not None:
                    opt_res = opt_res.copy(); opt_res["segment"] = label
                    all_opt.append(opt_res)
                if true_res is not None:
                    true_res = true_res.copy(); true_res["segment"] = label
                    all_true.append(true_res)
                start_i = end_i
                seg_idx += 1

            # Save aggregated results
            os.makedirs("output", exist_ok=True)
            if all_dumb:
                pd.concat(all_dumb).to_csv("output/benchmark_dumb_eco.csv")
            if all_opt:
                pd.concat(all_opt).to_csv("output/benchmark_optimal.csv")
            if all_true:
                pd.concat(all_true).to_csv("output/benchmark_true_optimal.csv")
            pd.DataFrame(summaries).to_csv("output/benchmark_segments_summary.csv", index=False)
            logger.info("Saved: output/benchmark_segments_summary.csv")

            dumb_result = pd.concat(all_dumb).sort_index() if all_dumb else simulate_dumb_eco_mode(S, cfg)
            opt_result = pd.concat(all_opt).sort_index() if all_opt else simulate_rolling_optimal(
                S,
                weather_aligned,
                prices,
                cfg,
                reoptimize_interval_h=REOPTIMIZE_INTERVAL_HOURS,
                tz=tz,
                models_dir=MODELS_DIR,
            )
            true_result = pd.concat(all_true).sort_index() if all_true else None

            dumb_net_cost = float(dumb_result["cost_import_eur"].sum() - dumb_result["rev_export_eur"].sum())
            opt_net_cost = float(opt_result["cost_import_eur"].sum() - opt_result["rev_export_eur"].sum()) if opt_result is not None else float("nan")
            true_net_cost = float(true_result["cost_import_eur"].sum() - true_result["rev_export_eur"].sum()) if true_result is not None else float("nan")
            savings = dumb_net_cost - opt_net_cost
            savings_pct = (savings / abs(dumb_net_cost) * 100.0) if dumb_net_cost else 0.0
            true_savings = dumb_net_cost - true_net_cost if not np.isnan(true_net_cost) else float("nan")
            true_savings_pct = (true_savings / abs(dumb_net_cost) * 100.0) if dumb_net_cost and not np.isnan(true_savings) else float("nan")
            opt_gap_eur = opt_net_cost - true_net_cost if not np.isnan(true_net_cost) else float("nan")
            opt_gap_pct = (opt_gap_eur / abs(true_net_cost) * 100.0) if not np.isnan(true_net_cost) and true_net_cost else float("nan")
            summary = {
                "label": "segments_total",
                "period_start": str(S.index[0]),
                "period_end": str(S.index[-1]),
                "hours": len(S),
                "dumb_cost_eur": dumb_net_cost,
                "opt_cost_eur": opt_net_cost,
                "savings_eur": savings,
                "savings_pct": savings_pct,
                "true_opt_cost_eur": true_net_cost,
                "true_savings_eur": true_savings,
                "true_savings_pct": true_savings_pct,
                "opt_vs_true_gap_eur": opt_gap_eur,
                "opt_vs_true_gap_pct": opt_gap_pct,
            }
        else:
            # Default: run on the full available window
            prices_full = prices.loc[idx, :]
            weather_full = weather_aligned
            dumb_result, opt_result, true_result, summary = _run_window_benchmark(
                S,
                weather_full,
                prices_full,
                cfg,
                label="full-period",
                reoptimize_interval_h=REOPTIMIZE_INTERVAL_HOURS,
                tz=tz,
                models_dir=MODELS_DIR,
            )

        # Save results
        os.makedirs("output", exist_ok=True)
        # Save CSVs
        dumb_result.to_csv("output/benchmark_dumb_eco.csv")
        logger.info("Saved: output/benchmark_dumb_eco.csv")
        if opt_result is not None:
            opt_result.to_csv("output/benchmark_optimal.csv")
            logger.info("Saved: output/benchmark_optimal.csv")
        if true_result is not None:
            true_result.to_csv("output/benchmark_true_optimal.csv")
            logger.info("Saved: output/benchmark_true_optimal.csv")

        # Create comparison plots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: SoC comparison
        axes[0].plot(dumb_result.index, dumb_result["soc"], label="Dumb eco", color="#1f77b4", alpha=0.7)
        if opt_result is not None:
            axes[0].plot(opt_result.index, opt_result["soc"], label="MPC optimal", color="#ff7f0e", alpha=0.7)
        if true_result is not None:
            axes[0].plot(true_result.index, true_result["soc"], label="True optimal", color="#2ca02c", alpha=0.7)
        axes[0].set_ylabel("SoC")
        axes[0].set_title("Battery State of Charge")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Grid import/export comparison
        axes[1].plot(dumb_result.index, dumb_result["grid_import_kw"], label="Dumb import", color="#2ca02c", alpha=0.6)
        axes[1].plot(dumb_result.index, -dumb_result["grid_export_kw"], label="Dumb export", color="#d62728", alpha=0.6)
        if opt_result is not None:
            axes[1].plot(opt_result.index, opt_result["grid_import_kw"], label="MPC import", color="#9467bd", alpha=0.6, linestyle="--")
            axes[1].plot(opt_result.index, -opt_result["grid_export_kw"], label="MPC export", color="#8c564b", alpha=0.6, linestyle="--")
        if true_result is not None:
            axes[1].plot(true_result.index, true_result["grid_import_kw"], label="True import", color="#17becf", alpha=0.6, linestyle=":")
            axes[1].plot(true_result.index, -true_result["grid_export_kw"], label="True export", color="#bcbd22", alpha=0.6, linestyle=":")
        axes[1].set_ylabel("Power (kW)")
        axes[1].set_title("Grid Import/Export (negative = export)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Cumulative cost comparison
        dumb_cumcost = (dumb_result["cost_import_eur"] - dumb_result["rev_export_eur"]).cumsum()
        axes[2].plot(dumb_result.index, dumb_cumcost, label="Dumb eco", color="#1f77b4")
        if opt_result is not None:
            opt_cumcost = (opt_result["cost_import_eur"] - opt_result["rev_export_eur"]).cumsum()
            axes[2].plot(opt_result.index, opt_cumcost, label="MPC optimal", color="#ff7f0e")
        if true_result is not None:
            true_cumcost = (true_result["cost_import_eur"] - true_result["rev_export_eur"]).cumsum()
            axes[2].plot(true_result.index, true_cumcost, label="True optimal", color="#2ca02c")
        axes[2].set_ylabel("Cumulative net cost (€)")
        axes[2].set_xlabel("Time")
        axes[2].set_title("Cumulative Cost Over Time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig("output/benchmark_comparison.png", dpi=150)
        logger.info("Saved: output/benchmark_comparison.png")

        # Save summary metrics
        # Persist summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv("output/benchmark_summary.csv", index=False)
        logger.info("Saved: output/benchmark_summary.csv")

        # Monthly breakdown analysis
        if opt_result is not None and len(idx) > 24*30:  # Only if we have at least a month of data
            logger.info("Generating monthly breakdown...")
            
            # Add month column to both results
            dumb_monthly = dumb_result.copy()
            dumb_monthly['month'] = _to_month_period(dumb_monthly.index, tz)
            opt_monthly = opt_result.copy()
            opt_monthly['month'] = _to_month_period(opt_monthly.index, tz)
            true_monthly = None
            if true_result is not None:
                true_monthly = true_result.copy()
                true_monthly['month'] = _to_month_period(true_monthly.index, tz)
            
            # Calculate monthly costs
            monthly_stats = []
            for month in sorted(dumb_monthly['month'].unique()):
                dumb_month = dumb_monthly[dumb_monthly['month'] == month]
                opt_month = opt_monthly[opt_monthly['month'] == month]
                true_month = true_monthly[true_monthly['month'] == month] if true_monthly is not None else None
                
                dumb_cost = (dumb_month["cost_import_eur"] - dumb_month["rev_export_eur"]).sum()
                opt_cost = (opt_month["cost_import_eur"] - opt_month["rev_export_eur"]).sum()
                true_cost = (true_month["cost_import_eur"] - true_month["rev_export_eur"]).sum() if true_month is not None else float("nan")
                pv_sum = dumb_month["pv_kw"].sum()
                load_sum = dumb_month["load_kw"].sum()
                
                mpc_savings = dumb_cost - opt_cost
                mpc_savings_pct = (mpc_savings / abs(dumb_cost)) * 100 if dumb_cost != 0 else 0
                true_savings = dumb_cost - true_cost if not np.isnan(true_cost) else float("nan")
                true_savings_pct = (true_savings / abs(dumb_cost)) * 100 if dumb_cost != 0 and not np.isnan(true_savings) else float("nan")
                opt_cost_pct = (opt_cost / dumb_cost) * 100 if dumb_cost != 0 else float("nan")
                true_cost_pct = (true_cost / dumb_cost) * 100 if dumb_cost != 0 and not np.isnan(true_cost) else float("nan")
                gap_pct = ((opt_cost - true_cost) / abs(true_cost) * 100) if not np.isnan(true_cost) and true_cost else float("nan")
                
                monthly_stats.append({
                    'month': str(month),
                    'pv_kwh': pv_sum,
                    'load_kwh': load_sum,
                    'dumb_cost_eur': dumb_cost,
                    'opt_cost_eur': opt_cost,
                    'true_cost_eur': true_cost,
                    'opt_cost_pct_of_dumb': opt_cost_pct,
                    'true_cost_pct_of_dumb': true_cost_pct,
                    'mpc_savings_eur': mpc_savings,
                    'mpc_savings_pct': mpc_savings_pct,
                    'true_savings_eur': true_savings,
                    'true_savings_pct': true_savings_pct,
                    'mpc_vs_true_gap_pct': gap_pct,
                })
            
            monthly_df = pd.DataFrame(monthly_stats)
            monthly_df.to_csv("output/benchmark_monthly.csv", index=False)
            logger.info("Saved: output/benchmark_monthly.csv")
            
            # Plot monthly savings
            fig_monthly, ax_monthly = plt.subplots(2, 1, figsize=(12, 8))
            
            months = range(len(monthly_df))
            
            # Plot 1: Costs comparison
            ax_monthly[0].bar([m - 0.25 for m in months], monthly_df['dumb_cost_eur'], 
                            width=0.25, label='Dumb eco', color='#1f77b4', alpha=0.7)
            ax_monthly[0].bar(months, monthly_df['opt_cost_eur'], 
                            width=0.25, label='MPC optimal', color='#ff7f0e', alpha=0.7)
            if 'true_cost_eur' in monthly_df and not monthly_df['true_cost_eur'].isna().all():
                ax_monthly[0].bar([m + 0.25 for m in months], monthly_df['true_cost_eur'], 
                                width=0.25, label='True optimal', color='#2ca02c', alpha=0.7)
            ax_monthly[0].set_ylabel('Net cost (€)')
            ax_monthly[0].set_title('Monthly Cost Comparison')
            ax_monthly[0].set_xticks(months)
            ax_monthly[0].set_xticklabels(monthly_df['month'], rotation=45, ha='right')
            ax_monthly[0].legend()
            ax_monthly[0].grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Savings percentage
            ax_monthly[1].bar([m - 0.15 for m in months], monthly_df['mpc_savings_pct'], width=0.3,
                               color='#2ca02c', alpha=0.7, label='MPC vs baseline')
            if 'true_savings_pct' in monthly_df and not monthly_df['true_savings_pct'].isna().all():
                ax_monthly[1].bar([m + 0.15 for m in months], monthly_df['true_savings_pct'], width=0.3,
                                   color='#17becf', alpha=0.7, label='True vs baseline')
            ax_monthly[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_monthly[1].set_ylabel('Savings (%)')
            ax_monthly[1].set_xlabel('Month')
            ax_monthly[1].set_title('Monthly Cost Savings Percentage')
            ax_monthly[1].set_xticks(months)
            ax_monthly[1].set_xticklabels(monthly_df['month'], rotation=45, ha='right')
            ax_monthly[1].legend()
            ax_monthly[1].grid(True, alpha=0.3, axis='y')
            
            fig_monthly.tight_layout()
            fig_monthly.savefig("output/benchmark_monthly_savings.png", dpi=150)
            logger.info("Saved: output/benchmark_monthly_savings.png")
            
            # Print summary
            logger.info("Monthly breakdown:")
            for _, row in monthly_df.iterrows():
                dumb_cost = row.get('dumb_cost_eur', float('nan'))
                opt_cost = row.get('opt_cost_eur', float('nan'))
                true_cost = row.get('true_cost_eur', float('nan'))
                opt_pct_of_dumb = row.get('opt_cost_pct_of_dumb', float('nan'))
                true_pct_of_dumb = row.get('true_cost_pct_of_dumb', float('nan'))

                opt_display = "n/a"
                if not np.isnan(opt_cost):
                    pct_str = f"{opt_pct_of_dumb:>5.1f}%" if not np.isnan(opt_pct_of_dumb) else "  n/a"
                    opt_display = f"{opt_cost:>7.2f}€ ({pct_str} of dumb)"

                true_display = "n/a"
                if not np.isnan(true_cost):
                    pct_str = f"{true_pct_of_dumb:>5.1f}%" if not np.isnan(true_pct_of_dumb) else "  n/a"
                    true_display = f"{true_cost:>7.2f}€ ({pct_str} of dumb)"

                logger.info(
                    "  %s: dumb %7.2f€, mpc %s, true %s, PV %7.0f kWh, load %7.0f kWh",
                    row['month'],
                    dumb_cost,
                    opt_display,
                    true_display,
                    row['pv_kwh'],
                    row['load_kwh'],
                )

    finally:
        import asyncio
        import time
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ha.close())
            loop.run_until_complete(asyncio.sleep(0.1))  # Allow cleanup
            loop.close()
            time.sleep(0.05)  # Final cleanup
        except Exception:
            pass


if __name__ == "__main__":
    main()
