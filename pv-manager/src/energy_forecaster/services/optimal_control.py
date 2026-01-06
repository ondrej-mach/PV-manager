from __future__ import annotations

import numpy as np
import cvxpy as cp
import os
import pandas as pd
import pandas as pd
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Optional, List
import logging

_LOGGER = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    cap_kwh: float = float(os.getenv("BATTERY_CAP_KWH", "10.0"))
    p_max_kw: float = float(os.getenv("BATTERY_P_MAX_KW", "3.0"))
    soc_min: float = 0.1
    soc_max: float = 0.9
    ec: float = 0.95  # AC->DC charge efficiency
    ed: float = 0.95  # DC->AC discharge efficiency
    eb_rt: float = 0.96  # internal round-trip (battery cell)


@dataclass
class DeferrableLoadConfig:
    name: str  # Unique identifier for column matching
    nominal_power_kw: float
    opt_mode: str = "smart_dump"  # "smart_dump" or "custom_curve"
    opt_value_start_eur: float = 0.05
    opt_saturation_kwh: float = 10.0
    opt_prevent_overshoot: bool = False
    is_integer: bool = False
    opt_max_energy_kwh: Optional[float] = None

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
    deferrable_loads: Optional[List[DeferrableLoadConfig]] = None,
) -> pd.DataFrame:
    """
    Solves the optimal control problem using Linear Programming (or MILP if integers present).
    """
    # cvxpy is imported globally now to ensure any signal handler manipulation
    # happens BEFORE Uvicorn takes control.
    
    # sanitize inputs
    input_S = S
    S = S.copy()

    # Check for integer requirements and solver availability
    target_mip = False
    selected_solver = None
    if deferrable_loads:
        for d in deferrable_loads:
            if d.is_integer:
                target_mip = True
                break

    if target_mip:
        available = cp.installed_solvers()
        # Common MIP solvers supported by cvxpy
        mip_solvers = {"CBC", "GLPK_MI", "CPLEX", "GUROBI", "SCIP", "XPRESS", "MOSEK", "COPT"}
        usable_mip = [s for s in available if s in mip_solvers]

        if usable_mip:
            # CBC is often preferred if available, but take first found
            if "CBC" in usable_mip:
                selected_solver = "CBC"
            elif "GLPK_MI" in usable_mip:
                selected_solver = "GLPK_MI"
            else:
                selected_solver = usable_mip[0]
            _LOGGER.info("Integer control requested. Using MIP solver: %s", selected_solver)
        else:
            _LOGGER.warning("Integer control requested but no MIP solver found (installed: %s). Falling back to continuous control.", available)
            # Downgrade all loads to continuous to prevent SolverError
            for d in deferrable_loads:
                if d.is_integer:
                    d.is_integer = False
    
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

    # Deferrable Loads Variables
    def_load_vars = []  # List of (config, cp.Variable(N))
    if deferrable_loads:
        for dload in deferrable_loads:
            # Variable represents the power consumed by this specific load [kW]
            if dload.is_integer:
                # Integer Mode: Use binary variable scaled by nominal power
                # Requires integer solver (e.g., CBC, GLPK_MI)
                _LOGGER.debug("Creating binary variable for load '%s'", dload.name)
                bin_var = cp.Variable(N, boolean=True)
                var = bin_var * dload.nominal_power_kw
            else:
                # Continuous Mode: Variable between 0 and Nominal
                var = cp.Variable(N, nonneg=True)
            
            def_load_vars.append((dload, var))

    grid_import = grid_to_load + grid_to_batt
    grid_export = pv_to_grid + batt_to_grid
    ch_ac = pv_to_batt + grid_to_batt
    dis_ac = batt_to_load + batt_to_grid
    
    # Total consumption = Base Load + Deferrable Loads
    # Note: 'load' array in input S is the "Base Load" (inflexible house consumption)
    total_load_t = grid_to_load + batt_to_load + pv_to_load
    
    # Base load satisfaction constraint
    # base_load[t] == pv_to_load[t] + batt_to_load[t] + grid_to_load[t]
    # BUT we need to account for where the deferrable load power comes from.
    # Actually, the standard convention is:
    # Sources: PV(t) + GridImport(t) + BatteryDischarge(t)
    # Sinks: BaseLoad(t) + BatteryCharge(t) + GridExport(t) + Curtailed(t) + Sum(DeferrableLoads(t))
    #
    # We need to restructure the energy balance constraints.
    # Let total_supply = pv[t] + grid_import[t] + dis_ac[t] (AC side)
    # Let total_demand = load[t] + ch_ac[t] + grid_export[t] + pv_curt[t] + sum(def_load[t])
    #
    # However, to maintain the specific flows (grid->batt, pv->batt etc.) logic:
    # We can assume deferrable loads draw from the common "AC bus".
    # Since we don't track "PV -> Deferrable" explicitly to avoid huge variable explosion,
    # we can simplify:
    # The existing variables (pv_to_load, etc.) cover the BASE load.
    # We need NEW variables source -> def_load?
    # Or simpler:
    # Relax equality: Supply >= Demand? No, must be balanced.
    # 
    # Alternative:
    # Treat Deferrable Loads as an addition to the 'load' array?
    # No, because we optimize them.
    #
    # Let's add explicit source flows for aggregate deferrable demand.
    # source_pv_to_def = cp.Variable(N)
    # source_grid_to_def = cp.Variable(N)
    # source_batt_to_def = cp.Variable(N)
    # 
    # This adds 3*M variables.
    # Simpler: Just ensure Global Power Balance.
    # pv[t] + grid_import[t] + dis_ac[t] == load[t] + ch_ac[t] + grid_export[t] + pv_curt[t] + sum(def_load[t])
    # 
    # AND keep the specific breakdown for Base Load to track "Self Consumption"?
    # The original code uses `pv_to_load`, etc. to track components.
    # `load[t] == pv_to_load[t] + batt_to_load[t] + grid_to_load[t]`
    # 
    # We can add:
    # `sum(def_load[t]) == pv_to_def[t] + batt_to_def[t] + grid_to_def[t]`
    # 
    # Let's define these aggregate flows to flexible loads.
    pv_to_def = cp.Variable(N, nonneg=True)
    grid_to_def = cp.Variable(N, nonneg=True)
    batt_to_def = cp.Variable(N, nonneg=True)
    
    def_load_total = np.zeros(N) # shape placeholder
    if def_load_vars:
        def_load_total = sum(v for _, v in def_load_vars)

    grid_import = grid_to_load + grid_to_batt + grid_to_def
    grid_export = pv_to_grid + batt_to_grid
    ch_ac = pv_to_batt + grid_to_batt
    dis_ac = batt_to_load + batt_to_grid + batt_to_def

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
        # Strict equality as requested by user
        cons.append(e[-1] == terminal_energy)

    reserve_shortfall = None
    if soft_soc_target is not None and cap_kwh > 0:
        reserve_shortfall = cp.Variable(N, nonneg=True)

    for t in range(N):
        # Power Balance
        # 1. PV Balance
        cons += [pv[t] == pv_to_load[t] + pv_to_batt[t] + pv_to_grid[t] + pv_curt[t] + pv_to_def[t]]
        
        # 2. Base Load Satisfaction
        cons += [load[t] == pv_to_load[t] + batt_to_load[t] + grid_to_load[t]]
        
        # 3. Deferrable Load Satisfaction (Aggregate)
        if def_load_vars:
             cons += [def_load_total[t] == pv_to_def[t] + batt_to_def[t] + grid_to_def[t]]
        else:
             cons += [pv_to_def[t] == 0, batt_to_def[t] == 0, grid_to_def[t] == 0]

        cons += [
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
    
    # Deferrable Constraints
    for dload, var in def_load_vars:
        # Power Limit
        cons += [var <= dload.nominal_power_kw]
        
        # Overshoot Limit (Optional)
        if dload.opt_prevent_overshoot and dload.opt_saturation_kwh > 0:
            total_energy = cp.sum(cp.multiply(var, dt))
            cons += [total_energy <= dload.opt_saturation_kwh]
            _LOGGER.info("Constraint added for '%s': MaxEnergy=%.2f kWh", dload.name, dload.opt_saturation_kwh)
        else:
            _LOGGER.info("No energy constraint for '%s' (PreventOvershoot=%s, Saturation=%.2f)", dload.name, dload.opt_prevent_overshoot, dload.opt_saturation_kwh)

        # Max Energy Hard Limit
        if dload.opt_max_energy_kwh is not None and dload.opt_max_energy_kwh > 0:
            total_e = cp.sum(cp.multiply(var, dt))
            cons += [total_e <= dload.opt_max_energy_kwh]
            _LOGGER.info("Constraint added for '%s': HardLimit=%.2f kWh", dload.name, dload.opt_max_energy_kwh)

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
    
    _LOGGER.info("Solving LP: Avg Buy Price=%.3f, Avg Sell Price=%.3f, Deferrables=%d. ThroughputMode=%s", avg_buy_price, float(np.mean(prices_export)) if len(prices_export) else 0.0, len(def_load_vars), throughput_mode)

    # Deferrable Loads Utility (Negative Cost)
    def_load_utility = 0.0
    for dload, var in def_load_vars:
        # Base Dump Value (always active)
        # Using a small value e.g. 0.01 or derived from start value if simple mode
        
        # Calculate Utility integral: U = Integral(u(e) * de)
        # For "smart_dump": Utility = Value * Energy (Linear)
        # For "custom_curve": Utility = V_start * E - (V_start / 2*E_sat) * E^2
        
        # Total Energy consumed by this load
        # Since 'var' is power profile, E_total = sum(var * dt)
        # BUT this quadratic logic applies to the AGGREGATE energy, which connects variables across time.
        # Quadratic Objective in CVXPY: cp.sum_squares(x) is convex.
        # We need Concave Utility (maximize).
        # Minimize (-Utility) -> Minimize ( Quadratic Term - Linear Term )
        # Convexity: -(-(a*x^2)) = a*x^2. If a > 0, it is convex.
        # Our Utility is V*E - k*E^2.
        # We minimize -(V*E - k*E^2) = k*E^2 - V*E.
        # This is Convex if k > 0.
        # k = V_start / (2 * E_sat). V_start > 0, E_sat > 0 => k > 0.
        # So we can minimize (k * E_total^2 - V_start * E_total).
        
        E_total = cp.sum(cp.multiply(var, DT))
        
        if dload.opt_mode == "custom_curve" and dload.opt_saturation_kwh > 0:
            v_start = max(dload.opt_value_start_eur, 0.0)
            e_sat = dload.opt_saturation_kwh
            v_dump = 0.01 # Base dump value for tail
            
            # Boost component
            # U_boost(E) approx V_start * E - (V_start / 2 E_sat) * E^2
            # We want to transition to v_dump after saturation?
            # User requirement: "Value" decreases to "Value when cost is zero" (dump).
            # The simplified math in plan:
            # U_boost = (V_start - V_dump) * E - ...
            
            val_diff = max(v_start - v_dump, 0.0)
            k = val_diff / (2.0 * e_sat)
            if target_mip:
                # CBC does not support quadratic costs (MIQP). Linearize by setting k=0.
                k = 0.0
            
            # Since we minimize:
            # Objective += k * E^2 - (V_start) * E
            # Wait, V_total = V_boost + V_dump
            # V_dump * E is linear.
            # V_boost is the quadratic part.
            
            # Linear part: (V_start) * E is applied?
            # No, if we use the boost formula:
            # U_boost = val_diff * E - k * E^2
            # U_dump = v_dump * E
            # Total U = (val_diff + v_dump) * E - k * E^2 = V_start * E - k * E^2
            # Min(-U) = k * E^2 - V_start * E
            
            if k > 0:
                def_load_utility += (k * cp.square(E_total)) - (v_start * E_total)
            else:
                def_load_utility += -1.0 * (v_start * E_total)
            
        else:
            # Smart Dump or fallback
            # Simple constant value per kWh
            # Default to slightly above typical export (e.g. dynamic or fixed)
            # If user didn't specify, we use a heuristic or the start value
            val = 0.01 # Default dump value
            if dload.opt_mode == "custom_curve":
                val = dload.opt_value_start_eur 
            else:
                 # Smart Dump: User defines the value
                 val = max(dload.opt_value_start_eur, 0.01)

            
            def_load_utility += -1.0 * (val * E_total)

    # Value the stored energy at the end of horizon to incentivize charging from surplus
    # We must account for discharge efficiency (battery -> load), otherwise the optimizer 
    # might buy grid power thinking it can store it at 100% efficiency.
    # Value = Energy * Discharge_Efficiency * Avg_Price
    total_discharge_eff = eta_d * eta_b_d
    terminal_value = e[-1] * total_discharge_eff * avg_buy_price

    obj = cp.Minimize(trade_cost + deg_cost + reserve_penalty + def_load_utility - terminal_value)

    prob = cp.Problem(obj, cons)
    used = None
    # Solve
    try:
        if selected_solver:
            prob.solve(solver=selected_solver, verbose=False)
            used = selected_solver
        else:
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=2000)
            used = "CLARABEL"
    except cp.error.SolverError as exc:
        if selected_solver:
            _LOGGER.error("Solver %s failed: %s", selected_solver, exc)
            raise
        # Fallback for standard LP
        prob.solve(solver=cp.SCS, verbose=False, max_iters=100000, eps=1e-5)
        used = "SCS"
    except Exception as exc:
        if selected_solver:
            _LOGGER.error("Solver %s failed: %s", selected_solver, exc)
            raise
        # Fallback for standard LP
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

    if deferrable_loads:
        for idx, (dload, var) in enumerate(def_load_vars):
            col_name = f"deferrable_{idx}_kw"
            if dload.name:
                # Sanitize name for column
                safe_name = "".join(c if c.isalnum() else "_" for c in dload.name)
                col_name = f"load_{safe_name}_kw"
            
            if var.value is not None:
                 raw_vals = np.array(var.value)
                 # Clip solver noise (tiny values < 0.1W) to 0
                 raw_vals[raw_vals < 1e-4] = 0.0
                 
                 total_planned = np.sum(raw_vals * dt) # Fix: multiply by dt for energy
                 _LOGGER.info("Optimization Result '%s': TotalPlanned=%.2f kWh, Value=%.3f, Max=%.2f kW", dload.name, total_planned, dload.opt_value_start_eur, dload.nominal_power_kw)
                 opt.loc[:, col_name] = raw_vals
            else:
                 _LOGGER.warning("Optimization Result '%s': Variable value is None! (Solver %s)", dload.name, used)
                 opt.loc[:, col_name] = 0.0
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
