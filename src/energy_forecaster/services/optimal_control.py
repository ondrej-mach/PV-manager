from __future__ import annotations

import numpy as np
import os
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class BatteryConfig:
    cap_kwh: float = float(os.getenv("BATTERY_CAP_KWH", "10.0"))
    p_max_kw: float = float(os.getenv("BATTERY_P_MAX_KW", "3.0"))
    soc_min: float = 0.1
    soc_max: float = 0.9
    ec: float = 0.95  # AC->DC charge efficiency
    ed: float = 0.95  # DC->AC discharge efficiency
    eb_rt: float = 0.96  # internal round-trip (battery cell)


def solve_lp_optimal_control(S: pd.DataFrame, cfg: BatteryConfig, deg_eur_per_kwh: float = 0.10,
                             throughput_mode: str = "avg") -> pd.DataFrame:
    """Solve linear program for optimal battery dispatch.

    Input S must contain columns: 'pv_kw', 'load_kw', 'dt_h', 'price_eur_per_kwh'.
    Index should be a DateTimeIndex.
    Returns DataFrame with dispatch decisions and SoC.
    """
    # sanitize inputs
    S = S.copy()
    for c in ("pv_kw", "load_kw"):
        S[c] = pd.to_numeric(S[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    S["dt_h"] = pd.to_numeric(S["dt_h"], errors="coerce").fillna(0.0).clip(lower=0.0)
    S["price_eur_per_kwh"] = pd.to_numeric(S["price_eur_per_kwh"], errors="coerce").ffill().bfill()

    cap_kwh, soc_min, soc_max = cfg.cap_kwh, cfg.soc_min, cfg.soc_max
    p_batt_max = cfg.p_max_kw
    eta_c, eta_d = cfg.ec, cfg.ed
    eta_b_c = float(np.sqrt(cfg.eb_rt))
    eta_b_d = float(np.sqrt(cfg.eb_rt))

    if "soc_meas" in S.columns:
        s = pd.to_numeric(S["soc_meas"], errors="coerce").ffill().bfill()
        soc0 = float(np.clip(s.iloc[0] if len(s) else 0.5, soc_min, soc_max))
    else:
        soc0 = float(np.clip(0.5, soc_min, soc_max))
    e0 = cap_kwh * soc0

    pv = S["pv_kw"].to_numpy(float)
    load = S["load_kw"].to_numpy(float)
    dt = S["dt_h"].to_numpy(float)
    prices = S["price_eur_per_kwh"].to_numpy(float)
    N = len(S)

    # decision variables
    pv_to_load = cp.Variable(N, nonneg=True)
    pv_to_batt = cp.Variable(N, nonneg=True)
    pv_to_grid = cp.Variable(N, nonneg=True)
    pv_curt = cp.Variable(N, nonneg=True)
    grid_to_load = cp.Variable(N, nonneg=True)
    grid_to_batt = cp.Variable(N, nonneg=True)
    batt_to_load = cp.Variable(N, nonneg=True)
    batt_to_grid = cp.Variable(N, nonneg=True)
    e = cp.Variable(N + 1)  # kWh

    grid_import = grid_to_load + grid_to_batt
    grid_export = pv_to_grid + batt_to_grid
    ch_ac = pv_to_batt + grid_to_batt
    dis_ac = batt_to_load + batt_to_grid

    cons = [
        e[0] == e0,
        e[-1] == e0,
        e >= cap_kwh * soc_min,
        e <= cap_kwh * soc_max,
    ]
    for t in range(N):
        cons += [
            pv[t] == pv_to_load[t] + pv_to_batt[t] + pv_to_grid[t] + pv_curt[t],
            load[t] == pv_to_load[t] + batt_to_load[t] + grid_to_load[t],
            ch_ac[t] <= p_batt_max,
            dis_ac[t] <= p_batt_max,
        ]
        e_next = e[t] + (eta_c * eta_b_c) * ch_ac[t] * dt[t] - (1.0 / (eta_d * eta_b_d)) * dis_ac[t] * dt[t]
        cons += [e[t + 1] == e_next]

    BUY = cp.Constant(prices + 0.14)  # add fixed components to import price if needed
    SELL = cp.Constant(prices)
    DT = cp.Constant(dt)

    trade_step = cp.multiply(BUY, grid_import) - cp.multiply(SELL, grid_export)
    trade_cost = cp.sum(cp.multiply(trade_step, DT))

    dc_in_kw = (eta_c * eta_b_c) * ch_ac
    dc_out_kw = (1.0 / (eta_d * eta_b_d)) * dis_ac
    if throughput_mode == "discharge":
        throughput_kwh = cp.multiply(dc_out_kw, DT)
    else:
        throughput_kwh = 0.5 * cp.multiply(dc_in_kw + dc_out_kw, DT)
    deg_cost = deg_eur_per_kwh * cp.sum(throughput_kwh)

    obj = cp.Minimize(trade_cost + deg_cost)

    prob = cp.Problem(obj, cons)
    used = None
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=2000, tol_gap_rel=1e-8, tol_feas=1e-8, tol_kkt=1e-8)
        used = "CLARABEL"
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=100000, eps=1e-5)
        used = "SCS"

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")

    opt = pd.DataFrame(index=S.index, data={
        "pv_kw": pv, "load_kw": load, "dt_h": dt,
        "pv_to_load_kw": pv_to_load.value,
        "pv_to_batt_kw": pv_to_batt.value,
        "pv_to_grid_kw": pv_to_grid.value,
        "pv_curt_kw": pv_curt.value,
        "grid_to_load_kw": grid_to_load.value,
        "grid_to_batt_kw": grid_to_batt.value,
        "batt_to_load_kw": batt_to_load.value,
        "batt_to_grid_kw": batt_to_grid.value,
        "grid_import_kw": grid_import.value,
        "grid_export_kw": grid_export.value,
        "buy_price_eur_per_kwh": BUY.value,
        "sell_price_eur_per_kwh": SELL.value,
    })
    opt["soc"] = (np.array(e.value[1:]) / cap_kwh).clip(0, 1)
    opt["import_kwh"] = opt["grid_import_kw"] * opt["dt_h"]
    opt["export_kwh"] = opt["grid_export_kw"] * opt["dt_h"]
    opt["cost_import_eur"] = opt["import_kwh"] * opt["buy_price_eur_per_kwh"]
    opt["rev_export_eur"] = opt["export_kwh"] * opt["sell_price_eur_per_kwh"]

    return opt
