"""Baseline 'dumb eco mode' battery control simulation."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BatteryConfig:
    """Battery configuration (same as in optimal_control)."""
    cap_kwh: float = float(os.getenv("BATTERY_CAP_KWH", "10.0"))
    p_max_kw: float = float(os.getenv("BATTERY_P_MAX_KW", "3.0"))
    soc_min: float = 0.1
    soc_max: float = 0.9
    ec: float = 0.95  # AC->DC charge efficiency
    ed: float = 0.95  # DC->AC discharge efficiency
    eb_rt: float = 0.96  # internal round-trip


def simulate_dumb_eco_mode(S: pd.DataFrame, cfg: BatteryConfig, soc0: float = 0.5) -> pd.DataFrame:
    """Simulate 'dumb eco mode' battery control.

    Rules:
    - Always charge from PV surplus (after covering load)
    - Never charge from grid
    - Never discharge to grid
    - Cover self-consumption from battery if available
    - Respect SoC limits and power limits

    Args:
        S: DataFrame with columns 'pv_kw', 'load_kw', 'dt_h', 'price_eur_per_kwh'
        cfg: BatteryConfig
        soc0: initial SoC (fraction)

    Returns:
        DataFrame with dispatch decisions, SoC, and costs
    """
    # sanitize inputs
    S = S.copy()
    for c in ("pv_kw", "load_kw"):
        S.loc[:, c] = pd.to_numeric(S[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    S.loc[:, "dt_h"] = pd.to_numeric(S["dt_h"], errors="coerce").fillna(0.0).clip(lower=0.0)
    S.loc[:, "price_eur_per_kwh"] = pd.to_numeric(S["price_eur_per_kwh"], errors="coerce").ffill().bfill()

    pv = S["pv_kw"].to_numpy(float)
    load = S["load_kw"].to_numpy(float)
    dt = S["dt_h"].to_numpy(float)
    prices = S["price_eur_per_kwh"].to_numpy(float)
    N = len(S)

    # Battery parameters
    cap_kwh = cfg.cap_kwh
    soc_min = cfg.soc_min
    soc_max = cfg.soc_max
    p_max = cfg.p_max_kw
    eta_c = cfg.ec
    eta_d = cfg.ed
    eta_b_c = float(np.sqrt(cfg.eb_rt))
    eta_b_d = float(np.sqrt(cfg.eb_rt))

    # Initialize tracking arrays
    e = np.zeros(N + 1)  # energy in battery [kWh]
    soc = np.zeros(N)
    pv_to_load = np.zeros(N)
    pv_to_batt = np.zeros(N)
    pv_to_grid = np.zeros(N)
    batt_to_load = np.zeros(N)
    grid_to_load = np.zeros(N)

    e[0] = cap_kwh * np.clip(soc0, soc_min, soc_max)

    for t in range(N):
        # Step 1: Cover load from PV first
        pv_used_for_load = min(pv[t], load[t])
        pv_to_load[t] = pv_used_for_load
        remaining_load = load[t] - pv_used_for_load
        remaining_pv = pv[t] - pv_used_for_load

        # Step 2: Try to cover remaining load from battery
        if remaining_load > 0:
            # Max discharge power (AC side)
            max_dis_ac = min(p_max, remaining_load)
            # Max discharge energy available (DC side)
            max_dis_dc_kwh = e[t] - cap_kwh * soc_min
            max_dis_ac_kwh = max_dis_dc_kwh * (eta_d * eta_b_d)  # DC -> AC
            # Actual discharge
            dis_ac_kwh = min(max_dis_ac * dt[t], max_dis_ac_kwh)
            dis_ac_kw = dis_ac_kwh / dt[t] if dt[t] > 0 else 0
            batt_to_load[t] = min(dis_ac_kw, remaining_load)
            remaining_load -= batt_to_load[t]

        # Step 3: Import from grid if still needed
        grid_to_load[t] = remaining_load

        # Step 4: Charge battery from remaining PV
        if remaining_pv > 0:
            # Max charge power (AC side)
            max_ch_ac = min(p_max, remaining_pv)
            # Max charge energy capacity (DC side)
            max_ch_dc_kwh = cap_kwh * soc_max - e[t]
            max_ch_ac_kwh = max_ch_dc_kwh / (eta_c * eta_b_c)  # AC -> DC
            # Actual charge
            ch_ac_kwh = min(max_ch_ac * dt[t], max_ch_ac_kwh)
            ch_ac_kw = ch_ac_kwh / dt[t] if dt[t] > 0 else 0
            pv_to_batt[t] = min(ch_ac_kw, remaining_pv)
            remaining_pv -= pv_to_batt[t]

        # Step 5: Export remaining PV to grid
        pv_to_grid[t] = remaining_pv

        # Update battery state
        ch_dc = (eta_c * eta_b_c) * pv_to_batt[t] * dt[t]
        dis_dc = (1.0 / (eta_d * eta_b_d)) * batt_to_load[t] * dt[t]
        e[t + 1] = np.clip(e[t] + ch_dc - dis_dc, cap_kwh * soc_min, cap_kwh * soc_max)
        soc[t] = e[t + 1] / cap_kwh

    # Build result DataFrame
    result = pd.DataFrame(index=S.index, data={
        "pv_kw": pv,
        "load_kw": load,
        "dt_h": dt,
        "pv_to_load_kw": pv_to_load,
        "pv_to_batt_kw": pv_to_batt,
        "pv_to_grid_kw": pv_to_grid,
        "batt_to_load_kw": batt_to_load,
        "grid_to_load_kw": grid_to_load,
        "grid_import_kw": grid_to_load,  # no grid charging
        "grid_export_kw": pv_to_grid,
        "buy_price_eur_per_kwh": prices + 0.14,  # add fixed components
        "sell_price_eur_per_kwh": prices,
        "soc": soc,
    })

    result.loc[:, "import_kwh"] = result["grid_import_kw"] * result["dt_h"]
    result.loc[:, "export_kwh"] = result["grid_export_kw"] * result["dt_h"]
    result.loc[:, "cost_import_eur"] = result["import_kwh"] * result["buy_price_eur_per_kwh"]
    result.loc[:, "rev_export_eur"] = result["export_kwh"] * result["sell_price_eur_per_kwh"]

    return result
