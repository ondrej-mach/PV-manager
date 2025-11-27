from __future__ import annotations

import numpy as np
import os
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional


@dataclass
class BatteryConfig:
    cap_kwh: float = float(os.getenv("BATTERY_CAP_KWH", "10.0"))
    p_max_kw: float = float(os.getenv("BATTERY_P_MAX_KW", "3.0"))
    soc_min: float = 0.1
    soc_max: float = 0.9
    ec: float = 0.95  # AC->DC charge efficiency
    ed: float = 0.95  # DC->AC discharge efficiency
    eb_rt: float = 0.96  # internal round-trip (battery cell)


def solve_lp_optimal_control(
    S: pd.DataFrame,
    cfg: BatteryConfig,
    deg_eur_per_kwh: float = 0.10,
    throughput_mode: str = "avg",
    soc0: Optional[float] = None,
    reserve_soc_hard: Optional[float] = None,
    reserve_soc_soft: Optional[float] = None,
    reserve_soft_penalty_eur_per_kwh: Optional[float] = None,
    force_terminal_soc: bool = True,
    terminal_soc: Optional[float] = None,
    allow_grid_charging: bool = True,
    export_enabled: bool = True,
    export_limit_kw: Optional[float] = None,
) -> pd.DataFrame:
    """Solve linear program for optimal battery dispatch.

    Input S must contain columns: 'pv_kw', 'load_kw', 'dt_h', 'price_eur_per_kwh'.
    Index should be a DateTimeIndex.
    Returns DataFrame with dispatch decisions and SoC.

    Optional reserves:
    - reserve_soc_hard enforces an absolute minimum SoC (fraction of capacity).
    - reserve_soc_soft encourages staying above a higher SoC via linear penalty.
    - reserve_soft_penalty_eur_per_kwh overrides the penalty weight (€/kWh) for soft reserve violations.
    """
    # sanitize inputs
    S = S.copy()
    for c in ("pv_kw", "load_kw"):
        S.loc[:, c] = pd.to_numeric(S[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    S.loc[:, "dt_h"] = pd.to_numeric(S["dt_h"], errors="coerce").fillna(0.0).clip(lower=0.0)

    base_price_series = None
    if "price_eur_per_kwh" in S.columns:
        base_price_series = pd.to_numeric(S["price_eur_per_kwh"], errors="coerce").ffill().bfill()
    import_price_series = None
    if "import_price_eur_per_kwh" in S.columns:
        import_price_series = pd.to_numeric(S["import_price_eur_per_kwh"], errors="coerce").ffill().bfill()
    if import_price_series is None:
        if base_price_series is None:
            base_price_series = pd.Series(0.0, index=S.index)
        import_price_series = base_price_series + 0.14
    export_price_series = None
    if "export_price_eur_per_kwh" in S.columns:
        export_price_series = pd.to_numeric(S["export_price_eur_per_kwh"], errors="coerce").ffill().bfill()
    if export_price_series is None:
        if base_price_series is None:
            base_price_series = pd.Series(0.0, index=S.index)
        export_price_series = base_price_series.copy()
    if base_price_series is None:
        base_price_series = (import_price_series + export_price_series) / 2.0

    cap_kwh, soc_min, soc_max = cfg.cap_kwh, cfg.soc_min, cfg.soc_max
    p_batt_max = cfg.p_max_kw
    eta_c, eta_d = cfg.ec, cfg.ed
    eta_b_c = float(np.sqrt(cfg.eb_rt))
    eta_b_d = float(np.sqrt(cfg.eb_rt))

    def _parse_env_float(name: str) -> Optional[float]:
        raw = os.getenv(name, "")
        raw = raw.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    if reserve_soc_hard is None:
        reserve_soc_hard = _parse_env_float("RESERVE_SOC_HARD")
    if reserve_soc_soft is None:
        reserve_soc_soft = _parse_env_float("RESERVE_SOC_SOFT")
    if reserve_soft_penalty_eur_per_kwh is None:
        reserve_soft_penalty_eur_per_kwh = _parse_env_float("RESERVE_SOFT_PENALTY_EUR_PER_KWH")

    def _clip_soc(value: float) -> float:
        return float(min(max(value, 0.0), soc_max))

    hard_soc_floor = soc_min
    if reserve_soc_hard is not None:
        hard_soc_floor = max(hard_soc_floor, _clip_soc(reserve_soc_hard))

    soft_soc_target: Optional[float] = None
    if reserve_soc_soft is not None:
        soft_soc_target = _clip_soc(reserve_soc_soft)
        if soft_soc_target < hard_soc_floor:
            soft_soc_target = hard_soc_floor

    if soc0 is not None:
        soc0 = float(np.clip(soc0, hard_soc_floor, soc_max))
    elif "soc_meas" in S.columns:
        s = pd.to_numeric(S["soc_meas"], errors="coerce").ffill().bfill()
        soc0 = float(np.clip(s.iloc[0] if len(s) else 0.5, hard_soc_floor, soc_max))
    else:
        soc0 = float(np.clip(0.5, hard_soc_floor, soc_max))
    e0 = cap_kwh * soc0

    pv = S["pv_kw"].to_numpy(float)
    load = S["load_kw"].to_numpy(float)
    dt = S["dt_h"].to_numpy(float)
    prices_import = import_price_series.to_numpy(float)
    prices_export = export_price_series.to_numpy(float)
    prices_base = base_price_series.to_numpy(float)
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
        e >= cap_kwh * hard_soc_floor,
        e <= cap_kwh * soc_max,
    ]

    terminal_energy: Optional[float] = None
    if force_terminal_soc:
        terminal_energy = e0
    elif terminal_soc is not None:
        terminal_target = float(np.clip(terminal_soc, hard_soc_floor, soc_max))
        terminal_energy = cap_kwh * terminal_target

    if terminal_energy is not None:
        # Relax equality to inequality to allow charging from surplus
        cons.append(e[-1] >= terminal_energy)

    reserve_shortfall = None
    if soft_soc_target is not None and cap_kwh > 0:
        reserve_shortfall = cp.Variable(N, nonneg=True)

    for t in range(N):
        cons += [
            pv[t] == pv_to_load[t] + pv_to_batt[t] + pv_to_grid[t] + pv_curt[t],
            load[t] == pv_to_load[t] + batt_to_load[t] + grid_to_load[t],
            ch_ac[t] <= p_batt_max,
            dis_ac[t] <= p_batt_max,
        ]
        if not allow_grid_charging:
            cons += [grid_to_batt[t] == 0]
        
        if not export_enabled:
            cons += [grid_export[t] == 0]
        elif export_limit_kw is not None and export_limit_kw >= 0:
            cons += [grid_export[t] <= export_limit_kw]

        e_next = e[t] + (eta_c * eta_b_c) * ch_ac[t] * dt[t] - (1.0 / (eta_d * eta_b_d)) * dis_ac[t] * dt[t]
        cons += [e[t + 1] == e_next]
        if reserve_shortfall is not None and soft_soc_target is not None:
            cons += [reserve_shortfall[t] >= cap_kwh * soft_soc_target - e[t + 1]]

    BUY = cp.Constant(prices_import)
    SELL = cp.Constant(prices_export)
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

    reserve_penalty = 0.0
    avg_buy_price = float(np.mean(prices_import)) if len(prices_import) else 0.0

    if reserve_shortfall is not None:
        # Default to average import price unless an explicit penalty is provided
        penalty_weight = reserve_soft_penalty_eur_per_kwh
        if penalty_weight is None:
            penalty_weight = max(avg_buy_price, 0.0)
        else:
            penalty_weight = max(penalty_weight, 0.0)
        reserve_penalty = penalty_weight * cp.sum(reserve_shortfall)

    # Value the stored energy at the end of horizon to incentivize charging from surplus
    # We must account for discharge efficiency (battery -> load), otherwise the optimizer 
    # might buy grid power thinking it can store it at 100% efficiency.
    # Value = Energy * Discharge_Efficiency * Avg_Price
    total_discharge_eff = eta_d * eta_b_d
    terminal_value = e[-1] * total_discharge_eff * avg_buy_price

    obj = cp.Minimize(trade_cost + deg_cost + reserve_penalty - terminal_value)

    prob = cp.Problem(obj, cons)
    used = None
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=2000)
        used = "CLARABEL"
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=100000, eps=1e-5)
        used = "SCS"

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")

    if e.value is None:
        raise RuntimeError("Optimization solved but variable values are missing")

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
        "buy_price_eur_per_kwh": prices_import,
        "sell_price_eur_per_kwh": prices_export,
        "spot_price_eur_per_kwh": prices_base,
    })
    if cap_kwh > 0:
        opt.loc[:, "soc"] = (np.array(e.value[1:]) / cap_kwh).clip(0, 1)
    else:
        opt.loc[:, "soc"] = 0.0
    opt.loc[:, "import_kwh"] = opt["grid_import_kw"] * opt["dt_h"]
    opt.loc[:, "export_kwh"] = opt["grid_export_kw"] * opt["dt_h"]
    opt.loc[:, "cost_import_eur"] = opt["import_kwh"] * opt["buy_price_eur_per_kwh"]
    opt.loc[:, "rev_export_eur"] = opt["export_kwh"] * opt["sell_price_eur_per_kwh"]
    if reserve_shortfall is not None:
        opt.loc[:, "reserve_shortfall_kwh"] = reserve_shortfall.value

    return opt
