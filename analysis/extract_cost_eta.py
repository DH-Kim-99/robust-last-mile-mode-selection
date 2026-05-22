"""
Extract cost decomposition and eta (uncertainty direction) data.
Re-runs a small subset of Exp1 instances (10 seeds × 4 DI × Optimal = 40 runs)
and extracts detailed cost components + worst-case scenario eta values.

Expected runtime: ~2-3 minutes (median solve time ~2s per instance).
"""

import os
import sys
import numpy as np
import pandas as pd

# Add codes directory to path
CODES_DIR = os.path.join(os.path.dirname(__file__), '..', 'codes')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
sys.path.insert(0, os.path.abspath(CODES_DIR))

from config import ProblemConfig
from data_gen import SupplyChainData
from algo import CCGAlgorithm

import gurobipy as gp

OUT_DIR = os.path.dirname(__file__)

SEEDS = range(1, 11)  # 10 seeds
DI_SCENARIOS = ['HD', 'MD', 'LD', 'Mixed']
INSTANCE = 'full'
GAMMA = 10


def _compute_scenario_profit(master, data, scenario_idx, eta_plus, eta_minus):
    """
    Compute the second-stage profit for a specific scenario from solved Gurobi variables.
    Returns (profit, cost_dict) or (None, None) on failure.
    """
    K, I, J, R, M = data.K, data.I, data.J, data.R, data.M
    l = scenario_idx

    # Realized demand for this scenario
    d_realized = {}
    for r in range(R):
        for k in range(K):
            d_nom = data.mu[(r, k)]
            d_hat = data.mu_hat[(r, k)]
            ep = eta_plus.get((r, k), 0)
            em = eta_minus.get((r, k), 0)
            d_realized[(r, k)] = d_nom + d_hat * ep - d_hat * em

    try:
        revenue = sum(
            data.S * (d_realized[(r, k)] - master.u[(r, k, l)].X)
            for r in range(R) for k in range(K)
        )
        HC = sum(
            (data.h[j] / 2) * master.A_ij[(k, i, j, l)].X
            for k in range(K) for i in range(I) for j in range(J)
        )
        TC1 = sum(
            data.D1[(k, i, j)] * data.t * master.A_ij[(k, i, j, l)].X
            for k in range(K) for i in range(I) for j in range(J)
        )
        TC2 = sum(
            data.D2[(j, r)] * data.TC[m] * master.X[(j, r, m, k, l)].X
            for k in range(K) for j in range(J) for r in range(R) for m in range(M)
        )
        PC = sum(
            data.F[(k, i)] * master.A_ij[(k, i, j, l)].X
            for k in range(K) for i in range(I) for j in range(J)
        )
        SC = sum(
            data.SC * master.u[(r, k, l)].X
            for r in range(R) for k in range(K)
        )
        profit = revenue - HC - TC1 - TC2 - PC - SC
        return profit, {
            'Revenue': revenue, 'Holding_Cost': HC,
            'TC1_PlantDC': TC1, 'TC2_LastMile': TC2,
            'Production_Cost': PC, 'Shortage_Cost': SC,
        }
    except (AttributeError, KeyError) as e:
        return None, None


def extract_cost_components(master, data, solution, critical_scenarios):
    """
    Extract individual cost components from a solved master problem.
    Finds the binding (worst-case) scenario by evaluating all scenarios
    and selecting the one with minimum second-stage profit.
    """
    K, I, J, R, M = data.K, data.I, data.J, data.R, data.M

    # === First-stage costs (from solution dict) ===
    OC = sum(data.O[j] * solution['y'][j] for j in range(J))
    plant_cost = sum(
        data.C_plant[(k, i)] * solution['x'][(k, i)]
        for k in range(K) for i in range(I)
    )
    dc_cost = sum(data.C_dc[j] * solution['y'][j] for j in range(J))
    route1_cost = sum(
        data.L1[(k, i, j)] * solution['z'][(k, i, j)]
        for k in range(K) for i in range(I) for j in range(J)
    )
    route2_cost = sum(
        data.L2[(j, r)] * solution['w'][(j, r)]
        for j in range(J) for r in range(R)
    )
    first_stage_cost = OC + plant_cost + dc_cost + route1_cost + route2_cost

    # === Find binding scenario (minimum second-stage profit) ===
    binding_l = None
    binding_profit = float('inf')
    binding_costs = None

    for l, (_, eta_plus, eta_minus) in enumerate(critical_scenarios):
        profit, costs = _compute_scenario_profit(master, data, l, eta_plus, eta_minus)
        if profit is not None and profit < binding_profit:
            binding_profit = profit
            binding_costs = costs
            binding_l = l

    if binding_costs is None:
        print(f"  Warning: Could not extract costs from any scenario")
        return {
            'OC': OC, 'Plant_Cost': plant_cost, 'DC_Cost': dc_cost,
            'Route1_Cost': route1_cost, 'Route2_Cost': route2_cost,
            'First_Stage_Cost': first_stage_cost,
            'Revenue': np.nan, 'Holding_Cost': np.nan,
            'TC1_PlantDC': np.nan, 'TC2_LastMile': np.nan,
            'Production_Cost': np.nan, 'Shortage_Cost': np.nan,
            'Second_Stage_Profit': np.nan, 'Total_Profit': np.nan,
            'Binding_Scenario': -1,
        }

    return {
        'OC': OC,
        'Plant_Cost': plant_cost,
        'DC_Cost': dc_cost,
        'Route1_Cost': route1_cost,
        'Route2_Cost': route2_cost,
        'First_Stage_Cost': first_stage_cost,
        **binding_costs,
        'Second_Stage_Profit': binding_profit,
        'Total_Profit': -first_stage_cost + binding_profit,
        'Binding_Scenario': binding_l,
    }


def extract_eta_summary(critical_scenarios, data):
    """
    Extract eta direction summary from critical scenarios.
    For the last (binding) scenario, count how many customers have
    eta_plus=1 (demand increase) vs eta_minus=1 (demand decrease).
    """
    K, R = data.K, data.R
    results = []

    for l, (sc_id, eta_plus, eta_minus) in enumerate(critical_scenarios):
        n_plus = sum(1 for r in range(R) for k in range(K) if eta_plus.get((r, k), 0) > 0.5)
        n_minus = sum(1 for r in range(R) for k in range(K) if eta_minus.get((r, k), 0) > 0.5)
        n_neutral = R * K - n_plus - n_minus

        results.append({
            'Scenario_ID': l,
            'N_Plus': n_plus,
            'N_Minus': n_minus,
            'N_Neutral': n_neutral,
            'Pct_Plus': n_plus / (R * K) * 100,
            'Pct_Minus': n_minus / (R * K) * 100,
        })

    return results


def extract_eta_by_mode(critical_scenarios, solution, data, binding_l):
    """
    For the binding scenario, cross-tabulate eta direction with assigned delivery mode.
    binding_l: index of the binding (worst-case) scenario in critical_scenarios.
    """
    K, R, J, M = data.K, data.R, data.J, data.M

    # Get mode assignment for each customer from alpha
    customer_mode = {}
    for r in range(R):
        for m in range(M):
            for j in range(J):
                if solution['alpha'].get((j, r, m), 0) > 0.5:
                    customer_mode[r] = m
                    break
            if r in customer_mode:
                break

    if not critical_scenarios or binding_l < 0:
        return {}

    _, eta_plus, eta_minus = critical_scenarios[binding_l]

    mode_eta = {m: {'plus': 0, 'minus': 0, 'neutral': 0, 'total': 0} for m in range(M)}

    for r in range(R):
        m_assigned = customer_mode.get(r, -1)
        if m_assigned < 0:
            continue
        for k in range(K):
            mode_eta[m_assigned]['total'] += 1
            if eta_plus.get((r, k), 0) > 0.5:
                mode_eta[m_assigned]['plus'] += 1
            elif eta_minus.get((r, k), 0) > 0.5:
                mode_eta[m_assigned]['minus'] += 1
            else:
                mode_eta[m_assigned]['neutral'] += 1

    return mode_eta


# ============================================================
# MAIN EXECUTION
# ============================================================
print("=" * 60)
print("COST DECOMPOSITION & ETA EXTRACTION")
print(f"Running {len(list(SEEDS))} seeds × {len(DI_SCENARIOS)} DI = {len(list(SEEDS)) * len(DI_SCENARIOS)} instances")
print("=" * 60)

cost_rows = []
eta_rows = []
eta_mode_rows = []

for seed in SEEDS:
    # Load data once per seed (DI is applied after loading)
    config = ProblemConfig(INSTANCE)
    config.set_gamma(GAMMA)
    data_file = os.path.join(DATA_DIR, f"data_{INSTANCE}_seed{seed}.pkl")
    if not os.path.exists(data_file):
        print(f"  Data file not found: {data_file}, skipping seed {seed}")
        continue
    base_data = SupplyChainData.load(data_file)

    for di in DI_SCENARIOS:
        print(f"\n--- Seed {seed}, DI={di} ---")

        # Deep copy data and apply DI scenario
        import copy
        data = copy.deepcopy(base_data)

        if hasattr(data, 'DI_scenarios') and di in data.DI_scenarios:
            DI_matrix = data.DI_scenarios[di]
            for m in range(config.M):
                for k in range(config.K):
                    data.DI[(m, k)] = DI_matrix[k][m]
        else:
            print(f"  Warning: DI scenario '{di}' not found, using defaults")

        # Run C&CG
        ccg = CCGAlgorithm(data, config)
        results = ccg.run()

        if not results['converged'] or results['optimal_solution'] is None:
            print(f"  SKIPPED (not converged)")
            continue

        solution = results['optimal_solution']
        critical_scenarios = results['critical_scenarios']

        # Extract cost components (finds binding scenario internally)
        costs = extract_cost_components(ccg.master, data, solution, critical_scenarios)
        binding_l = costs.get('Binding_Scenario', -1)
        costs['Seed'] = seed
        costs['DI_Scenario'] = di
        costs['Optimal_Value'] = results['optimal_value']
        cost_rows.append(costs)
        print(f"  Binding scenario: {binding_l} of {len(critical_scenarios)}")

        # Extract eta summary (all scenarios)
        eta_summaries = extract_eta_summary(critical_scenarios, data)
        for es in eta_summaries:
            es['Seed'] = seed
            es['DI_Scenario'] = di
            es['Is_Binding'] = (es['Scenario_ID'] == binding_l)
            eta_rows.append(es)

        # Extract eta × mode cross-tabulation (binding scenario only)
        mode_eta = extract_eta_by_mode(critical_scenarios, solution, data, binding_l)
        for m, counts in mode_eta.items():
            eta_mode_rows.append({
                'Seed': seed,
                'DI_Scenario': di,
                'Mode': m,
                'N_Plus': counts['plus'],
                'N_Minus': counts['minus'],
                'N_Neutral': counts['neutral'],
                'N_Total': counts['total'],
            })

# Save results
df_cost = pd.DataFrame(cost_rows)
df_cost.to_csv(os.path.join(OUT_DIR, 'cost_decomposition.csv'), index=False)

df_eta = pd.DataFrame(eta_rows)
df_eta.to_csv(os.path.join(OUT_DIR, 'eta_summary.csv'), index=False)

df_eta_mode = pd.DataFrame(eta_mode_rows)
df_eta_mode.to_csv(os.path.join(OUT_DIR, 'eta_by_mode.csv'), index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COST DECOMPOSITION SUMMARY")
print("=" * 60)

if not df_cost.empty:
    # Overall averages
    print("\n--- Average Cost Components (across all instances) ---")
    cost_cols = ['First_Stage_Cost', 'Revenue', 'Holding_Cost', 'TC1_PlantDC',
                 'TC2_LastMile', 'Production_Cost', 'Shortage_Cost', 'Total_Profit']
    for col in cost_cols:
        print(f"  {col:25s}: {df_cost[col].mean():>12,.0f}")

    # Last-mile proportion
    total_transport = df_cost['TC1_PlantDC'].mean() + df_cost['TC2_LastMile'].mean()
    total_operating_cost = (df_cost['Holding_Cost'].mean() + total_transport +
                            df_cost['Production_Cost'].mean() + df_cost['Shortage_Cost'].mean())
    total_cost_incl_first = df_cost['First_Stage_Cost'].mean() + total_operating_cost

    print(f"\n--- Last-Mile Cost Proportion ---")
    print(f"  TC2 (last-mile):     {df_cost['TC2_LastMile'].mean():>12,.0f}")
    print(f"  TC1 (plant-DC):      {df_cost['TC1_PlantDC'].mean():>12,.0f}")
    print(f"  Total transport:     {total_transport:>12,.0f}")
    print(f"  TC2 / Total transp:  {df_cost['TC2_LastMile'].mean() / total_transport * 100:>11.1f}%")
    print(f"  TC2 / Total cost:    {df_cost['TC2_LastMile'].mean() / total_cost_incl_first * 100:>11.1f}%")

    # By DI scenario
    print(f"\n--- Last-Mile Share by DI Scenario ---")
    for di in DI_SCENARIOS:
        sub = df_cost[df_cost['DI_Scenario'] == di]
        tc2_mean = sub['TC2_LastMile'].mean()
        tc1_mean = sub['TC1_PlantDC'].mean()
        tt = tc1_mean + tc2_mean
        rev_mean = sub['Revenue'].mean()
        print(f"  {di:6s}: TC2={tc2_mean:>8,.0f}, TC2/Transport={tc2_mean/tt*100:.1f}%, TC2/Revenue={tc2_mean/rev_mean*100:.1f}%")

print("\n" + "=" * 60)
print("ETA (UNCERTAINTY DIRECTION) SUMMARY")
print("=" * 60)

if not df_eta.empty:
    # Only binding scenario per instance
    binding = df_eta[df_eta['Is_Binding'] == True].copy()

    print("\n--- Binding Scenario: Demand Direction (% of customer-product pairs) ---")
    for di in DI_SCENARIOS:
        sub = binding[binding['DI_Scenario'] == di]
        print(f"  {di:6s}: Up={sub['Pct_Plus'].mean():.1f}%, Down={sub['Pct_Minus'].mean():.1f}%, "
              f"Neutral={100-sub['Pct_Plus'].mean()-sub['Pct_Minus'].mean():.1f}%")

    overall = binding.agg({'Pct_Plus': 'mean', 'Pct_Minus': 'mean'})
    print(f"  {'Overall':6s}: Up={overall['Pct_Plus']:.1f}%, Down={overall['Pct_Minus']:.1f}%, "
          f"Neutral={100-overall['Pct_Plus']-overall['Pct_Minus']:.1f}%")

if not df_eta_mode.empty:
    print("\n--- Eta Direction × Delivery Mode (binding scenario, averages) ---")
    mode_names = {0: 'Slow (m=0)', 1: 'Medium (m=1)', 2: 'Fast (m=2)'}
    for m in range(3):
        sub = df_eta_mode[df_eta_mode['Mode'] == m]
        if sub['N_Total'].sum() == 0:
            continue
        n_total = sub['N_Total'].mean()
        pct_plus = sub['N_Plus'].sum() / sub['N_Total'].sum() * 100
        pct_minus = sub['N_Minus'].sum() / sub['N_Total'].sum() * 100
        print(f"  {mode_names[m]:15s}: N={n_total:.1f}, Up={pct_plus:.1f}%, Down={pct_minus:.1f}%")

print("\n=== DONE ===")
