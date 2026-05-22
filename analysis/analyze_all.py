"""
Comprehensive analysis of all experiment results.
Reads 6,100 CSV files, computes statistics, and generates LaTeX table snippets.
"""

import os
import glob
import re
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "result"
OUT_TABLES = HERE / "generated_tables"
OUT_FIGURES = HERE / "generated_figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

DI_ORDER = ["HD", "MD", "LD", "Mixed"]
POLICY_ORDER = ["Optimal", "FM0", "FM1", "FM2"]
GAMMA_ORDER = [20, 40, 60, 80, 100]
TC2_ORDER = [0.20, 0.40, 0.60, 0.80, 1.00]
COV_ORDER = ["No limit", "Relaxed", "Moderate", "Tight"]

# ============================================================
# Helper: read all CSVs from a folder into a single DataFrame
# ============================================================
def read_experiment(folder, parse_filename=True):
    """Read all CSVs from folder. Parse filename for metadata."""
    rows = []
    for fpath in sorted(glob.glob(str(BASE / folder / "*.csv"))):
        fname = os.path.basename(fpath)
        try:
            df_row = pd.read_csv(fpath)
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
            continue
        if df_row.empty:
            continue

        # Parse policy from filename
        if fname.startswith("optimal_"):
            df_row["Policy"] = "Optimal"
        elif fname.startswith("fm0_"):
            df_row["Policy"] = "FM0"
        elif fname.startswith("fm1_"):
            df_row["Policy"] = "FM1"
        elif fname.startswith("fm2_"):
            df_row["Policy"] = "FM2"

        # Parse optional tags from filename
        if "_difunc_linear" in fname:
            df_row["DI_Function"] = "Linear"
        elif "_difunc_" in fname:
            df_row["DI_Function"] = "Exponential"
        else:
            df_row["DI_Function"] = "Exponential"

        m = re.search(r"_tc2_([\d]+\.[\d]+)", fname)
        if m:
            df_row["TC2"] = float(m.group(1))

        m = re.search(r"_cov_(\w+)", fname)
        if m:
            df_row["Coverage"] = m.group(1).capitalize()
        else:
            df_row["Coverage"] = "No limit"

        rows.append(df_row)

    df = pd.concat(rows, ignore_index=True)
    print(f"  {folder}: {len(df)} rows loaded")
    return df

# ============================================================
# Statistical tests with Holm correction
# ============================================================
def holm_correction(pvalues):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(pvalues[idx] * (n - rank), 1.0)
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i-1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted

# ============================================================
# Format helpers
# ============================================================
def fmt_num(x, precision=0):
    """Format number with thousands separator."""
    if pd.isna(x):
        return "--"
    if precision == 0:
        return f"\\num{{{int(round(x))}}}"
    return f"\\num{{{x:.{precision}f}}}"

def fmt_pct(x, precision=1):
    if pd.isna(x):
        return "--"
    return f"{x:.{precision}f}\\%"

def fmt_pval(p):
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"

def fmt_time(x):
    """Format time in seconds."""
    if pd.isna(x):
        return "--"
    if x < 1:
        return f"{x:.2f}"
    elif x < 60:
        return f"{x:.1f}"
    else:
        return f"{x:.0f}"

# ============================================================
# LOAD ALL DATA
# ============================================================
print("Loading experiment data...")
df_exp1 = read_experiment("exp1")
df_exp2 = read_experiment("exp2")
df_break = read_experiment("breakeven")
df_linear = read_experiment("linear_di")
df_cov = read_experiment("coverage")

# ============================================================
# EXP1 ANALYSIS
# ============================================================
print("\n=== EXP1 Analysis ===")

# T1: Base Results — Policy × DI Mean Profit
exp1_summary = df_exp1.groupby(["Policy", "DI_Scenario"]).agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
    Median_Time=("Total_Time", "median"),
).reset_index()

# Also compute overall (across all DI)
exp1_overall = df_exp1.groupby("Policy").agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
    Median_Time=("Total_Time", "median"),
).reset_index()
exp1_overall["DI_Scenario"] = "Overall"
exp1_summary = pd.concat([exp1_summary, exp1_overall], ignore_index=True)

# T2: VoF Analysis (paired by seed × DI)
vof_rows = []
for di in DI_ORDER:
    opt = df_exp1[(df_exp1["Policy"] == "Optimal") & (df_exp1["DI_Scenario"] == di)].set_index("Seed")["Optimal_Value"]
    for fm_name in ["FM0", "FM1", "FM2"]:
        fm = df_exp1[(df_exp1["Policy"] == fm_name) & (df_exp1["DI_Scenario"] == di)].set_index("Seed")["Optimal_Value"]
        common = opt.index.intersection(fm.index)
        abs_vof = opt.loc[common] - fm.loc[common]
        pct_vof = (abs_vof / fm.loc[common].abs()) * 100
        vof_rows.append({
            "DI_Scenario": di,
            "Baseline": fm_name,
            "Mean_AbsVoF": abs_vof.mean(),
            "Std_AbsVoF": abs_vof.std(),
            "Mean_PctVoF": pct_vof.mean(),
            "Std_PctVoF": pct_vof.std(),
            "Min_PctVoF": pct_vof.min(),
            "Max_PctVoF": pct_vof.max(),
        })

# Overall VoF
for fm_name in ["FM0", "FM1", "FM2"]:
    abs_vofs_all = []
    pct_vofs_all = []
    for di in DI_ORDER:
        opt = df_exp1[(df_exp1["Policy"] == "Optimal") & (df_exp1["DI_Scenario"] == di)].set_index("Seed")["Optimal_Value"]
        fm = df_exp1[(df_exp1["Policy"] == fm_name) & (df_exp1["DI_Scenario"] == di)].set_index("Seed")["Optimal_Value"]
        common = opt.index.intersection(fm.index)
        abs_vofs_all.extend((opt.loc[common] - fm.loc[common]).tolist())
        pct_vofs_all.extend(((opt.loc[common] - fm.loc[common]) / fm.loc[common].abs() * 100).tolist())
    vof_rows.append({
        "DI_Scenario": "Overall",
        "Baseline": fm_name,
        "Mean_AbsVoF": np.mean(abs_vofs_all),
        "Std_AbsVoF": np.std(abs_vofs_all),
        "Mean_PctVoF": np.mean(pct_vofs_all),
        "Std_PctVoF": np.std(pct_vofs_all),
        "Min_PctVoF": np.min(pct_vofs_all),
        "Max_PctVoF": np.max(pct_vofs_all),
    })

df_vof = pd.DataFrame(vof_rows)

# T3: Statistical Tests (paired t-test + Wilcoxon, Holm corrected)
test_rows = []
for di in DI_ORDER + ["Overall"]:
    if di == "Overall":
        opt_vals = df_exp1[df_exp1["Policy"] == "Optimal"].sort_values(["DI_Scenario", "Seed"])["Optimal_Value"].values
    else:
        opt_vals = df_exp1[(df_exp1["Policy"] == "Optimal") & (df_exp1["DI_Scenario"] == di)].sort_values("Seed")["Optimal_Value"].values

    for fm_name in ["FM0", "FM1", "FM2"]:
        if di == "Overall":
            fm_vals = df_exp1[df_exp1["Policy"] == fm_name].sort_values(["DI_Scenario", "Seed"])["Optimal_Value"].values
        else:
            fm_vals = df_exp1[(df_exp1["Policy"] == fm_name) & (df_exp1["DI_Scenario"] == di)].sort_values("Seed")["Optimal_Value"].values

        n = min(len(opt_vals), len(fm_vals))
        opt_v = opt_vals[:n]
        fm_v = fm_vals[:n]

        t_stat, t_pval = stats.ttest_rel(opt_v, fm_v)
        w_stat, w_pval = stats.wilcoxon(opt_v - fm_v, alternative='two-sided')

        test_rows.append({
            "DI_Scenario": di,
            "Comparison": f"Optimal vs {fm_name}",
            "Baseline": fm_name,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "w_stat": w_stat,
            "w_pval": w_pval,
            "n": n,
        })

df_tests = pd.DataFrame(test_rows)
# Apply Holm correction across all tests
df_tests["t_pval_holm"] = holm_correction(df_tests["t_pval"].values)
df_tests["w_pval_holm"] = holm_correction(df_tests["w_pval"].values)

# ============================================================
# EXP2 ANALYSIS (Gamma Sensitivity)
# ============================================================
print("\n=== EXP2 Analysis ===")

# Filter converged only
df_exp2_conv = df_exp2[df_exp2["Converged"] == True].copy()

exp2_summary = df_exp2_conv.groupby(["Gamma", "Policy"]).agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
).reset_index()

# VoF by Gamma (paired)
gamma_vof_rows = []
for gamma in GAMMA_ORDER:
    for di in DI_ORDER + ["Overall"]:
        if di == "Overall":
            opt = df_exp2_conv[(df_exp2_conv["Policy"] == "Optimal") & (df_exp2_conv["Gamma"] == gamma)]
            opt = opt.sort_values(["DI_Scenario", "Seed"]).reset_index(drop=True)
        else:
            opt = df_exp2_conv[(df_exp2_conv["Policy"] == "Optimal") & (df_exp2_conv["Gamma"] == gamma) & (df_exp2_conv["DI_Scenario"] == di)]
            opt = opt.sort_values("Seed").reset_index(drop=True)

        for fm_name in ["FM0", "FM2"]:
            if di == "Overall":
                fm = df_exp2_conv[(df_exp2_conv["Policy"] == fm_name) & (df_exp2_conv["Gamma"] == gamma)]
                fm = fm.sort_values(["DI_Scenario", "Seed"]).reset_index(drop=True)
            else:
                fm = df_exp2_conv[(df_exp2_conv["Policy"] == fm_name) & (df_exp2_conv["Gamma"] == gamma) & (df_exp2_conv["DI_Scenario"] == di)]
                fm = fm.sort_values("Seed").reset_index(drop=True)

            n = min(len(opt), len(fm))
            if n == 0:
                continue
            abs_vof = opt["Optimal_Value"].values[:n] - fm["Optimal_Value"].values[:n]
            pct_vof = abs_vof / np.abs(fm["Optimal_Value"].values[:n]) * 100

            gamma_vof_rows.append({
                "Gamma": gamma,
                "DI_Scenario": di,
                "Baseline": fm_name,
                "Mean_AbsVoF": np.mean(abs_vof),
                "Mean_PctVoF": np.mean(pct_vof),
                "Std_PctVoF": np.std(pct_vof),
            })

df_gamma_vof = pd.DataFrame(gamma_vof_rows)

# ============================================================
# BREAKEVEN ANALYSIS
# ============================================================
print("\n=== Breakeven Analysis ===")

df_break_conv = df_break[df_break["Converged"] == True].copy()

break_summary = df_break_conv.groupby(["TC2", "Policy"]).agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
).reset_index()

# Paired VoF by TC2
break_vof_rows = []
for tc2 in TC2_ORDER:
    opt = df_break_conv[(df_break_conv["Policy"] == "Optimal") & (df_break_conv["TC2"] == tc2)].sort_values("Seed").reset_index(drop=True)
    fm2 = df_break_conv[(df_break_conv["Policy"] == "FM2") & (df_break_conv["TC2"] == tc2)].sort_values("Seed").reset_index(drop=True)
    n = min(len(opt), len(fm2))
    if n == 0:
        continue
    abs_vof = opt["Optimal_Value"].values[:n] - fm2["Optimal_Value"].values[:n]
    pct_vof = abs_vof / np.abs(fm2["Optimal_Value"].values[:n]) * 100

    break_vof_rows.append({
        "TC2": tc2,
        "Opt_Mean": opt["Optimal_Value"].mean(),
        "FM2_Mean": fm2["Optimal_Value"].mean(),
        "Mean_AbsVoF": np.mean(abs_vof),
        "Mean_PctVoF": np.mean(pct_vof),
        "Std_PctVoF": np.std(pct_vof),
        "Pct_OptBetter": np.mean(abs_vof > 0) * 100,
        "N": n,
    })

df_break_vof = pd.DataFrame(break_vof_rows)

# ============================================================
# LINEAR DI ANALYSIS
# ============================================================
print("\n=== Linear DI Analysis ===")

linear_summary = df_linear.groupby(["DI_Function", "Policy", "DI_Scenario"]).agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
).reset_index()

# VoF comparison: exponential vs linear
linear_vof_rows = []
for di_func in ["Exponential", "Linear"]:
    for di in DI_ORDER + ["Overall"]:
        if di == "Overall":
            opt = df_linear[(df_linear["DI_Function"] == di_func) & (df_linear["Policy"] == "Optimal")]
            opt = opt.sort_values(["DI_Scenario", "Seed"]).reset_index(drop=True)
        else:
            opt = df_linear[(df_linear["DI_Function"] == di_func) & (df_linear["Policy"] == "Optimal") & (df_linear["DI_Scenario"] == di)]
            opt = opt.sort_values("Seed").reset_index(drop=True)

        for fm_name in ["FM0", "FM2"]:
            if di == "Overall":
                fm = df_linear[(df_linear["DI_Function"] == di_func) & (df_linear["Policy"] == fm_name)]
                fm = fm.sort_values(["DI_Scenario", "Seed"]).reset_index(drop=True)
            else:
                fm = df_linear[(df_linear["DI_Function"] == di_func) & (df_linear["Policy"] == fm_name) & (df_linear["DI_Scenario"] == di)]
                fm = fm.sort_values("Seed").reset_index(drop=True)

            n = min(len(opt), len(fm))
            if n == 0:
                continue
            abs_vof = opt["Optimal_Value"].values[:n] - fm["Optimal_Value"].values[:n]
            pct_vof = abs_vof / np.abs(fm["Optimal_Value"].values[:n]) * 100

            linear_vof_rows.append({
                "DI_Function": di_func,
                "DI_Scenario": di,
                "Baseline": fm_name,
                "Mean_AbsVoF": np.mean(abs_vof),
                "Mean_PctVoF": np.mean(pct_vof),
            })

df_linear_vof = pd.DataFrame(linear_vof_rows)

# ============================================================
# COVERAGE ANALYSIS
# ============================================================
print("\n=== Coverage Analysis ===")

# Feasibility summary
cov_feasibility = df_cov.groupby(["Coverage", "Policy"]).agg(
    Total=("Converged", "count"),
    Converged=("Converged", "sum"),
).reset_index()
cov_feasibility["Feasible_Pct"] = cov_feasibility["Converged"] / cov_feasibility["Total"] * 100

# Profit for converged only
df_cov_conv = df_cov[df_cov["Converged"] == True].copy()

cov_profit = df_cov_conv.groupby(["Coverage", "Policy"]).agg(
    Mean_Profit=("Optimal_Value", "mean"),
    Std_Profit=("Optimal_Value", "std"),
).reset_index()

# VoF by coverage level (paired, feasible seeds only)
cov_vof_rows = []
for cov in ["Relaxed", "Moderate", "Tight"]:
    for di in DI_ORDER + ["Overall"]:
        if di == "Overall":
            opt = df_cov_conv[(df_cov_conv["Coverage"] == cov) & (df_cov_conv["Policy"] == "Optimal")]
        else:
            opt = df_cov_conv[(df_cov_conv["Coverage"] == cov) & (df_cov_conv["Policy"] == "Optimal") & (df_cov_conv["DI_Scenario"] == di)]
        opt_seeds = set(opt["Seed"].values)

        for fm_name in ["FM0", "FM2"]:
            if di == "Overall":
                fm = df_cov_conv[(df_cov_conv["Coverage"] == cov) & (df_cov_conv["Policy"] == fm_name)]
            else:
                fm = df_cov_conv[(df_cov_conv["Coverage"] == cov) & (df_cov_conv["Policy"] == fm_name) & (df_cov_conv["DI_Scenario"] == di)]
            fm_seeds = set(fm["Seed"].values)

            common_seeds = sorted(opt_seeds & fm_seeds)
            if len(common_seeds) == 0:
                cov_vof_rows.append({
                    "Coverage": cov, "DI_Scenario": di, "Baseline": fm_name,
                    "Mean_PctVoF": np.nan, "Mean_AbsVoF": np.nan,
                    "N_Feasible": 0,
                })
                continue

            if di == "Overall":
                opt_v = opt[opt["Seed"].isin(common_seeds)].sort_values(["DI_Scenario", "Seed"])["Optimal_Value"].values
                fm_v = fm[fm["Seed"].isin(common_seeds)].sort_values(["DI_Scenario", "Seed"])["Optimal_Value"].values
            else:
                opt_v = opt[opt["Seed"].isin(common_seeds)].sort_values("Seed")["Optimal_Value"].values
                fm_v = fm[fm["Seed"].isin(common_seeds)].sort_values("Seed")["Optimal_Value"].values

            n = min(len(opt_v), len(fm_v))
            abs_vof = opt_v[:n] - fm_v[:n]
            pct_vof = abs_vof / np.abs(fm_v[:n]) * 100

            cov_vof_rows.append({
                "Coverage": cov,
                "DI_Scenario": di,
                "Baseline": fm_name,
                "Mean_PctVoF": np.mean(pct_vof),
                "Mean_AbsVoF": np.mean(abs_vof),
                "N_Feasible": len(common_seeds),
            })

df_cov_vof = pd.DataFrame(cov_vof_rows)

# Add exp1 as "No limit" baseline for comparison
exp1_vof_as_nolimit = []
for di in DI_ORDER + ["Overall"]:
    for fm_name in ["FM0", "FM2"]:
        row = df_vof[(df_vof["DI_Scenario"] == di) & (df_vof["Baseline"] == fm_name)]
        if len(row) > 0:
            exp1_vof_as_nolimit.append({
                "Coverage": "No limit",
                "DI_Scenario": di,
                "Baseline": fm_name,
                "Mean_PctVoF": row.iloc[0]["Mean_PctVoF"],
                "Mean_AbsVoF": row.iloc[0]["Mean_AbsVoF"],
                "N_Feasible": 50 if di != "Overall" else 200,
            })
df_cov_vof_full = pd.concat([pd.DataFrame(exp1_vof_as_nolimit), df_cov_vof], ignore_index=True)

# ============================================================
# COMPUTATIONAL PERFORMANCE (all experiments)
# ============================================================
print("\n=== Computational Performance ===")

perf_rows = []
for name, df_exp in [("Exp1 (R=50)", df_exp1), ("Exp2 (R=200)", df_exp2)]:
    conv = df_exp[df_exp["Converged"] == True]
    perf_rows.append({
        "Experiment": name,
        "Total_Runs": len(df_exp),
        "Converged": len(conv),
        "Conv_Rate": len(conv) / len(df_exp) * 100,
        "Mean_Time": conv["Total_Time"].mean(),
        "Median_Time": conv["Total_Time"].median(),
        "Q75_Time": conv["Total_Time"].quantile(0.75),
        "Max_Time": conv["Total_Time"].max(),
        "Mean_Iter": conv["Iterations"].mean(),
        "Median_Iter": conv["Iterations"].median(),
        "Pct_Under60s": (conv["Total_Time"] < 60).mean() * 100,
    })

df_perf = pd.DataFrame(perf_rows)

# Network structure (exp1)
net_summary = df_exp1.groupby(["Policy", "DI_Scenario"]).agg(
    Mean_Plants=("Num_Plants_Opened", "mean"),
    Mean_DCs=("Num_DCs_Opened", "mean"),
).reset_index()

# ============================================================
# GENERATE LATEX TABLE SNIPPETS
# ============================================================
print("\n=== Generating LaTeX Tables ===")

# --- T1: Base Results ---
lines = []
lines.append("% Table: Base Results (Exp1)")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Mean robust profit by policy and demand sensitivity scenario ($\\Gamma=10$, $|R|=50$, 50~instances).}")
lines.append("\\label{tab:base_results}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=0,table-format=6.0] S[round-precision=0,table-format=5.0] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{DI Scenario} & {Policy} & {Mean Profit (\\$)} & {Std Dev} & {Median Time (s)} \\\\")
lines.append("\\midrule")

for di in DI_ORDER + ["Overall"]:
    first = True
    for pol in POLICY_ORDER:
        row = exp1_summary[(exp1_summary["DI_Scenario"] == di) & (exp1_summary["Policy"] == pol)]
        if row.empty:
            continue
        r = row.iloc[0]
        di_label = f"\\multirow{{4}}{{*}}{{{di}}}" if first else ""
        lines.append(f"{di_label} & {pol} & {r['Mean_Profit']:.0f} & {r['Std_Profit']:.0f} & {r['Median_Time']:.1f} \\\\")
        first = False
    if di != "Overall":
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T1_base_results.tex", "w") as f:
    f.write("\n".join(lines))

# --- T2: VoF Analysis ---
lines = []
lines.append("% Table: Value of Fast Delivery / Value of Flexibility")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Value of fast delivery and value of flexibility by demand sensitivity scenario. VoF is computed as paired differences over 50~instances.}")
lines.append("\\label{tab:vof_analysis}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=0,table-format=6.0] S[round-precision=0,table-format=5.0] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{DI Scenario} & {Baseline} & {Mean VoF (\\$)} & {Std VoF} & {Mean VoF\\%} & {Std VoF\\%} \\\\")
lines.append("\\midrule")

for di in DI_ORDER + ["Overall"]:
    first = True
    for bl in ["FM0", "FM1", "FM2"]:
        row = df_vof[(df_vof["DI_Scenario"] == di) & (df_vof["Baseline"] == bl)]
        if row.empty:
            continue
        r = row.iloc[0]
        di_label = f"\\multirow{{3}}{{*}}{{{di}}}" if first else ""
        lines.append(f"{di_label} & {bl} & {r['Mean_AbsVoF']:.0f} & {r['Std_AbsVoF']:.0f} & {r['Mean_PctVoF']:.1f} & {r['Std_PctVoF']:.1f} \\\\")
        first = False
    if di != "Overall":
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item VoF = $Z^*_{\\text{Optimal}} - Z^*_{\\text{Baseline}}$; VoF\\% = VoF $/ |Z^*_{\\text{Baseline}}| \\times 100$.")
lines.append("\\item Optimal vs FM0 captures the \\emph{value of fast delivery}; Optimal vs FM2 captures the \\emph{value of flexibility}.")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T2_vof_analysis.tex", "w") as f:
    f.write("\n".join(lines))

# --- T3: Statistical Tests ---
lines = []
lines.append("% Table: Statistical Tests")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Statistical significance of profit differences (paired $t$-test and Wilcoxon signed-rank test, Holm-corrected $p$-values).}")
lines.append("\\label{tab:stat_tests}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll rr rr}")
lines.append("\\toprule")
lines.append("{DI Scenario} & {Comparison} & {$t$-stat} & {$p_{\\text{Holm}}$} & {$W$-stat} & {$p_{\\text{Holm}}$} \\\\")
lines.append("\\midrule")

for di in DI_ORDER + ["Overall"]:
    first = True
    for _, r in df_tests[df_tests["DI_Scenario"] == di].iterrows():
        di_label = f"\\multirow{{3}}{{*}}{{{di}}}" if first else ""
        lines.append(f"{di_label} & {r['Comparison']} & {r['t_stat']:.2f} & {fmt_pval(r['t_pval_holm'])} & {fmt_num(r['w_stat'])} & {fmt_pval(r['w_pval_holm'])} \\\\")
        first = False
    if di != "Overall":
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item Holm--Bonferroni correction applied across all 15 comparisons ($\\alpha=0.05$).")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T3_stat_tests.tex", "w") as f:
    f.write("\n".join(lines))

# --- T4: Gamma Sensitivity ---
lines = []
lines.append("% Table: Gamma Sensitivity")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Effect of uncertainty budget $\\Gamma$ on mean robust profit and value of fast delivery ($|R|=200$, 50~instances per cell).}")
lines.append("\\label{tab:gamma_sensitivity}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{r S[round-precision=0,table-format=7.0] S[round-precision=0,table-format=7.0] S[round-precision=0,table-format=6.0] S[round-precision=1,table-format=3.1] S[round-precision=0,table-format=6.0] S[round-precision=1,table-format=2.1]}")
lines.append("\\toprule")
lines.append("{$\\Gamma$} & {Optimal} & {FM0} & {VoFD (\\$)}\\tnote{a} & {VoFD (\\%)} & {VoFF (\\$)}\\tnote{b} & {VoFF (\\%)} \\\\")
lines.append("\\midrule")

for gamma in GAMMA_ORDER:
    # Get overall values
    opt_row = exp2_summary[(exp2_summary["Gamma"] == gamma) & (exp2_summary["Policy"] == "Optimal")]
    fm0_row = exp2_summary[(exp2_summary["Gamma"] == gamma) & (exp2_summary["Policy"] == "FM0")]
    vofd_row = df_gamma_vof[(df_gamma_vof["Gamma"] == gamma) & (df_gamma_vof["DI_Scenario"] == "Overall") & (df_gamma_vof["Baseline"] == "FM0")]
    voff_row = df_gamma_vof[(df_gamma_vof["Gamma"] == gamma) & (df_gamma_vof["DI_Scenario"] == "Overall") & (df_gamma_vof["Baseline"] == "FM2")]

    opt_val = opt_row.iloc[0]["Mean_Profit"] if len(opt_row) > 0 else np.nan
    fm0_val = fm0_row.iloc[0]["Mean_Profit"] if len(fm0_row) > 0 else np.nan
    vofd_abs = vofd_row.iloc[0]["Mean_AbsVoF"] if len(vofd_row) > 0 else np.nan
    vofd_pct = vofd_row.iloc[0]["Mean_PctVoF"] if len(vofd_row) > 0 else np.nan
    voff_abs = voff_row.iloc[0]["Mean_AbsVoF"] if len(voff_row) > 0 else np.nan
    voff_pct = voff_row.iloc[0]["Mean_PctVoF"] if len(voff_row) > 0 else np.nan

    lines.append(f"{gamma} & {opt_val:.0f} & {fm0_val:.0f} & {vofd_abs:.0f} & {vofd_pct:.1f} & {voff_abs:.0f} & {voff_pct:.1f} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item[a] VoFD = Value of Fast Delivery (Optimal $-$ FM0).")
lines.append("\\item[b] VoFF = Value of Flexibility (Optimal $-$ FM2).")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T4_gamma_sensitivity.tex", "w") as f:
    f.write("\n".join(lines))

# --- T5: Breakeven Analysis ---
lines = []
lines.append("% Table: Breakeven Analysis")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Transportation cost breakeven analysis: Optimal vs.\\ FM2 under increasing fast-delivery cost $TC_2$ (LD scenario, $\\Gamma=10$, 50~instances).}")
lines.append("\\label{tab:breakeven}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{r r S[round-precision=0,table-format=6.0] S[round-precision=0,table-format=6.0] S[round-precision=0,table-format=5.0] S[round-precision=1,table-format=3.1] S[round-precision=0,table-format=3.0]}")
lines.append("\\toprule")
lines.append("{$TC_2$} & {$TC_2/TC_0$} & {Optimal (\\$)} & {FM2 (\\$)} & {VoFF (\\$)} & {VoFF (\\%)} & {\\% Opt $>$ FM2} \\\\")
lines.append("\\midrule")

for _, r in df_break_vof.iterrows():
    ratio = r["TC2"] / 0.05  # TC_0 = 0.05
    lines.append(f"{r['TC2']:.2f} & {ratio:.0f}$\\times$ & {r['Opt_Mean']:.0f} & {r['FM2_Mean']:.0f} & {r['Mean_AbsVoF']:.0f} & {r['Mean_PctVoF']:.1f} & {r['Pct_OptBetter']:.0f} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item $TC_0 = 0.05$. VoFF = Optimal profit $-$ FM2 profit. \\% Opt $>$ FM2 = percentage of instances where Optimal strictly outperforms FM2.")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T5_breakeven.tex", "w") as f:
    f.write("\n".join(lines))

# --- T6: Coverage SA ---
lines = []
lines.append("% Table: Coverage Sensitivity Analysis")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Effect of service coverage constraints on feasibility and value of flexibility (10~instances per cell).}")
lines.append("\\label{tab:coverage}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=0,table-format=3.0] S[round-precision=0,table-format=6.0] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{Coverage} & {Policy} & {Feasible (\\%)} & {Mean Profit (\\$)} & {VoFD (\\%)}\\tnote{a} & {VoFF (\\%)}\\tnote{b} \\\\")
lines.append("\\midrule")

for cov in COV_ORDER:
    first = True
    for pol in POLICY_ORDER:
        # Feasibility
        if cov == "No limit":
            feas_pct = 100.0
            # Get profit from exp1
            profit_row = exp1_summary[(exp1_summary["DI_Scenario"] == "Overall") & (exp1_summary["Policy"] == pol)]
            mean_profit = profit_row.iloc[0]["Mean_Profit"] if len(profit_row) > 0 else np.nan
        else:
            feas_row = cov_feasibility[(cov_feasibility["Coverage"] == cov) & (cov_feasibility["Policy"] == pol)]
            feas_pct = feas_row.iloc[0]["Feasible_Pct"] if len(feas_row) > 0 else np.nan
            profit_row = cov_profit[(cov_profit["Coverage"] == cov) & (cov_profit["Policy"] == pol)]
            mean_profit = profit_row.iloc[0]["Mean_Profit"] if len(profit_row) > 0 else np.nan

        # VoF for this coverage level (Overall DI)
        vofd_row = df_cov_vof_full[(df_cov_vof_full["Coverage"] == cov) & (df_cov_vof_full["DI_Scenario"] == "Overall") & (df_cov_vof_full["Baseline"] == "FM0")]
        voff_row = df_cov_vof_full[(df_cov_vof_full["Coverage"] == cov) & (df_cov_vof_full["DI_Scenario"] == "Overall") & (df_cov_vof_full["Baseline"] == "FM2")]

        vofd_pct_val = vofd_row.iloc[0]["Mean_PctVoF"] if len(vofd_row) > 0 and pol == "Optimal" else np.nan
        voff_pct_val = voff_row.iloc[0]["Mean_PctVoF"] if len(voff_row) > 0 and pol == "Optimal" else np.nan

        cov_label = f"\\multirow{{4}}{{*}}{{{cov}}}" if first else ""

        profit_str = f"{mean_profit:.0f}" if not pd.isna(mean_profit) else "--"
        feas_str = f"{feas_pct:.0f}" if not pd.isna(feas_pct) else "--"
        vofd_str = f"{vofd_pct_val:.1f}" if not pd.isna(vofd_pct_val) else ""
        voff_str = f"{voff_pct_val:.1f}" if not pd.isna(voff_pct_val) else ""

        lines.append(f"{cov_label} & {pol} & {feas_str} & {profit_str} & {vofd_str} & {voff_str} \\\\")
        first = False
    if cov != "Tight":
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item[a] VoFD = Value of Fast Delivery (Optimal $-$ FM0).")
lines.append("\\item[b] VoFF = Value of Flexibility (Optimal $-$ FM2).")
lines.append("\\item No limit results from Exp.\\ 1 (50~instances); coverage results from 10~instances.")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T6_coverage.tex", "w") as f:
    f.write("\n".join(lines))

# --- T7: Linear vs Exponential ---
lines = []
lines.append("% Table: Robustness Check — DI Function Form")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Robustness check: exponential vs.\\ linear DI function. Both anchored at $DI=1.60$ for $m=2, \\kappa=1$ (10~instances per cell).}")
lines.append("\\label{tab:linear_robustness}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=2.1] S[round-precision=1,table-format=2.1]}")
lines.append("\\toprule")
lines.append("& & \\multicolumn{2}{c}{VoFD (\\%)} & \\multicolumn{2}{c}{VoFF (\\%)} \\\\")
lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
lines.append("{DI Scenario} & {} & {Exponential} & {Linear} & {Exponential} & {Linear} \\\\")
lines.append("\\midrule")

for di in DI_ORDER + ["Overall"]:
    exp_vofd = df_linear_vof[(df_linear_vof["DI_Function"] == "Exponential") & (df_linear_vof["DI_Scenario"] == di) & (df_linear_vof["Baseline"] == "FM0")]
    lin_vofd = df_linear_vof[(df_linear_vof["DI_Function"] == "Linear") & (df_linear_vof["DI_Scenario"] == di) & (df_linear_vof["Baseline"] == "FM0")]
    exp_voff = df_linear_vof[(df_linear_vof["DI_Function"] == "Exponential") & (df_linear_vof["DI_Scenario"] == di) & (df_linear_vof["Baseline"] == "FM2")]
    lin_voff = df_linear_vof[(df_linear_vof["DI_Function"] == "Linear") & (df_linear_vof["DI_Scenario"] == di) & (df_linear_vof["Baseline"] == "FM2")]

    ev1 = exp_vofd.iloc[0]["Mean_PctVoF"] if len(exp_vofd) > 0 else np.nan
    lv1 = lin_vofd.iloc[0]["Mean_PctVoF"] if len(lin_vofd) > 0 else np.nan
    ev2 = exp_voff.iloc[0]["Mean_PctVoF"] if len(exp_voff) > 0 else np.nan
    lv2 = lin_voff.iloc[0]["Mean_PctVoF"] if len(lin_voff) > 0 else np.nan

    lines.append(f"{di} & & {ev1:.1f} & {lv1:.1f} & {ev2:.1f} & {lv2:.1f} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item Exponential: $DI_m^k = (\\sqrt{1.6})^{\\kappa \\cdot m}$. Linear: $DI_m^k = 1 + 0.3\\kappa m$. Both yield $DI=1.60$ at $m=2, \\kappa=1$.")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T7_linear_robustness.tex", "w") as f:
    f.write("\n".join(lines))

# --- T8: Computational Performance ---
lines = []
lines.append("% Table: Computational Performance")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Computational performance of the C\\&CG algorithm.}")
lines.append("\\label{tab:comp_performance}")
lines.append("\\begin{threeparttable}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{l S[round-precision=0,table-format=4.0] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=4.1] S[round-precision=1,table-format=4.1] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{Experiment} & {Runs} & {Conv.\\%} & {Mean (s)} & {Median (s)} & {Q75 (s)} & {Mean Iter.} & {\\% $<$60s} \\\\")
lines.append("\\midrule")

for _, r in df_perf.iterrows():
    lines.append(f"{r['Experiment']} & {r['Total_Runs']:.0f} & {r['Conv_Rate']:.1f} & {r['Mean_Time']:.1f} & {r['Median_Time']:.1f} & {r['Q75_Time']:.1f} & {r['Mean_Iter']:.1f} & {r['Pct_Under60s']:.1f} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\footnotesize")
lines.append("\\item Convergence tolerance $\\epsilon = 100$. Statistics computed over converged instances only.")
lines.append("\\end{tablenotes}")
lines.append("\\end{threeparttable}")
lines.append("\\end{table}")

with open(OUT_TABLES / "T8_comp_performance.tex", "w") as f:
    f.write("\n".join(lines))

# --- APPENDIX: A1 — DI-Scenario Detailed VoF ---
lines = []
lines.append("% Appendix Table: Detailed VoF by DI Scenario (Exp1)")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Detailed value of fast delivery and flexibility by DI scenario and baseline policy (Exp.\\ 1, 50~instances).}")
lines.append("\\label{tab:app_vof_detail}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=0,table-format=6.0] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=4.1] S[round-precision=1,table-format=3.1] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{DI} & {Baseline} & {Mean VoF (\\$)} & {Mean VoF\\%} & {Std VoF\\%} & {Min VoF\\%} & {Max VoF\\%} \\\\")
lines.append("\\midrule")

for di in DI_ORDER:
    first = True
    for bl in ["FM0", "FM1", "FM2"]:
        row = df_vof[(df_vof["DI_Scenario"] == di) & (df_vof["Baseline"] == bl)]
        if row.empty:
            continue
        r = row.iloc[0]
        di_label = f"\\multirow{{3}}{{*}}{{{di}}}" if first else ""
        lines.append(f"{di_label} & {bl} & {r['Mean_AbsVoF']:.0f} & {r['Mean_PctVoF']:.1f} & {r['Std_PctVoF']:.1f} & {r['Min_PctVoF']:.1f} & {r['Max_PctVoF']:.1f} \\\\")
        first = False
    lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open(OUT_TABLES / "A1_vof_detail.tex", "w") as f:
    f.write("\n".join(lines))

# --- APPENDIX: A2 — Gamma × DI Detailed ---
lines = []
lines.append("% Appendix Table: Gamma × DI Scenario VoF")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Value of fast delivery by $\\Gamma$ and DI scenario ($|R|=200$, Exp.\\ 2).}")
lines.append("\\label{tab:app_gamma_di}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{r l S[round-precision=0,table-format=6.0] S[round-precision=1,table-format=3.1]}")
lines.append("\\toprule")
lines.append("{$\\Gamma$} & {DI} & {VoFD (\\$)} & {VoFD (\\%)} \\\\")
lines.append("\\midrule")

for gamma in GAMMA_ORDER:
    first = True
    for di in DI_ORDER:
        row = df_gamma_vof[(df_gamma_vof["Gamma"] == gamma) & (df_gamma_vof["DI_Scenario"] == di) & (df_gamma_vof["Baseline"] == "FM0")]
        if row.empty:
            continue
        r = row.iloc[0]
        g_label = f"\\multirow{{4}}{{*}}{{{gamma}}}" if first else ""
        lines.append(f"{g_label} & {di} & {r['Mean_AbsVoF']:.0f} & {r['Mean_PctVoF']:.1f} \\\\")
        first = False
    if gamma != GAMMA_ORDER[-1]:
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open(OUT_TABLES / "A2_gamma_di_detail.tex", "w") as f:
    f.write("\n".join(lines))

# --- APPENDIX: A3 — Network Structure ---
lines = []
lines.append("% Appendix Table: Network Structure")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Average number of opened facilities by policy and DI scenario (Exp.\\ 1).}")
lines.append("\\label{tab:app_network}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll S[round-precision=2,table-format=1.2] S[round-precision=2,table-format=1.2]}")
lines.append("\\toprule")
lines.append("{DI} & {Policy} & {Mean Plants} & {Mean DCs} \\\\")
lines.append("\\midrule")

for di in DI_ORDER:
    first = True
    for pol in POLICY_ORDER:
        row = net_summary[(net_summary["DI_Scenario"] == di) & (net_summary["Policy"] == pol)]
        if row.empty:
            continue
        r = row.iloc[0]
        di_label = f"\\multirow{{4}}{{*}}{{{di}}}" if first else ""
        lines.append(f"{di_label} & {pol} & {r['Mean_Plants']:.2f} & {r['Mean_DCs']:.2f} \\\\")
        first = False
    if di != DI_ORDER[-1]:
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open(OUT_TABLES / "A3_network_structure.tex", "w") as f:
    f.write("\n".join(lines))

# --- APPENDIX: A4 — Coverage Infeasibility Detail ---
lines = []
lines.append("% Appendix Table: Coverage Infeasibility Detail")
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Feasibility of fixed-mode policies under service coverage constraints (10~instances per cell).}")
lines.append("\\label{tab:app_coverage_infeasibility}")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.3}")
lines.append("\\begin{tabular}{ll cccc}")
lines.append("\\toprule")
lines.append("{Coverage} & {Policy} & {HD} & {MD} & {LD} & {Mixed} \\\\")
lines.append("\\midrule")

for cov in ["Relaxed", "Moderate", "Tight"]:
    first = True
    for pol in POLICY_ORDER:
        cov_label = f"\\multirow{{4}}{{*}}{{{cov}}}" if first else ""
        cells = []
        for di in DI_ORDER:
            sub = df_cov[(df_cov["Coverage"] == cov) & (df_cov["Policy"] == pol) & (df_cov["DI_Scenario"] == di)]
            n_total = len(sub)
            n_conv = sub["Converged"].sum()
            cells.append(f"{int(n_conv)}/{int(n_total)}")
        lines.append(f"{cov_label} & {pol} & {' & '.join(cells)} \\\\")
        first = False
    if cov != "Tight":
        lines.append("\\midrule")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open(OUT_TABLES / "A4_coverage_infeasibility.tex", "w") as f:
    f.write("\n".join(lines))

# ============================================================
# GENERATE FIGURE DATA (CSV for pgfplots)
# ============================================================
print("\n=== Generating Figure Data ===")

# F1: Breakeven plot data
df_break_vof.to_csv(OUT_FIGURES / "F1_breakeven.csv", index=False)

# F2: Gamma sensitivity — VoFD% by DI and Gamma (vs FM0)
gamma_fig = df_gamma_vof[(df_gamma_vof["Baseline"] == "FM0") & (df_gamma_vof["DI_Scenario"].isin(DI_ORDER))].copy()
gamma_fig_pivot = gamma_fig.pivot_table(index="Gamma", columns="DI_Scenario", values="Mean_PctVoF")
gamma_fig_pivot = gamma_fig_pivot[DI_ORDER]
gamma_fig_pivot.to_csv(OUT_FIGURES / "F2_gamma_vofd.csv")

# Also VoFF by Gamma
gamma_fig_ff = df_gamma_vof[(df_gamma_vof["Baseline"] == "FM2") & (df_gamma_vof["DI_Scenario"].isin(DI_ORDER))].copy()
gamma_fig_ff_pivot = gamma_fig_ff.pivot_table(index="Gamma", columns="DI_Scenario", values="Mean_PctVoF")
gamma_fig_ff_pivot = gamma_fig_ff_pivot[DI_ORDER]
gamma_fig_ff_pivot.to_csv(OUT_FIGURES / "F2_gamma_voff.csv")

# F3: Coverage VoFF by coverage level (Overall)
cov_fig = df_cov_vof_full[(df_cov_vof_full["DI_Scenario"] == "Overall") & (df_cov_vof_full["Baseline"] == "FM2")].copy()
cov_fig = cov_fig[["Coverage", "Mean_PctVoF", "N_Feasible"]]
cov_fig.to_csv(OUT_FIGURES / "F3_coverage_voff.csv", index=False)

# ============================================================
# PRINT KEY SUMMARY NUMBERS
# ============================================================
print("\n" + "="*60)
print("KEY RESULTS SUMMARY")
print("="*60)

overall_vof = df_vof[df_vof["DI_Scenario"] == "Overall"]
for _, r in overall_vof.iterrows():
    print(f"  Optimal vs {r['Baseline']}: VoF = ${r['Mean_AbsVoF']:,.0f} ({r['Mean_PctVoF']:.1f}%)")

print(f"\nGamma sensitivity (VoFD% range): {df_gamma_vof[(df_gamma_vof['Baseline']=='FM0') & (df_gamma_vof['DI_Scenario']=='Overall')]['Mean_PctVoF'].min():.1f}% — {df_gamma_vof[(df_gamma_vof['Baseline']=='FM0') & (df_gamma_vof['DI_Scenario']=='Overall')]['Mean_PctVoF'].max():.1f}%")

print(f"\nBreakeven analysis:")
for _, r in df_break_vof.iterrows():
    marker = " <<<" if r["Mean_PctVoF"] > 1 else ""
    print(f"  TC2={r['TC2']:.2f} ({r['TC2']/0.05:.0f}x): VoFF = {r['Mean_PctVoF']:.1f}%, Opt>FM2 in {r['Pct_OptBetter']:.0f}% cases{marker}")

print(f"\nCoverage VoFF (Overall, Opt vs FM2):")
for _, r in cov_fig.iterrows():
    val = f"{r['Mean_PctVoF']:.1f}%" if not pd.isna(r['Mean_PctVoF']) else "N/A"
    print(f"  {r['Coverage']}: VoFF = {val} (N={int(r['N_Feasible'])})")

print(f"\nLinear vs Exponential (Overall VoFD%):")
for func in ["Exponential", "Linear"]:
    row = df_linear_vof[(df_linear_vof["DI_Function"] == func) & (df_linear_vof["DI_Scenario"] == "Overall") & (df_linear_vof["Baseline"] == "FM0")]
    if len(row) > 0:
        print(f"  {func}: {row.iloc[0]['Mean_PctVoF']:.1f}%")

print(f"\nComputational performance:")
for _, r in df_perf.iterrows():
    print(f"  {r['Experiment']}: {r['Conv_Rate']:.1f}% conv, median {r['Median_Time']:.1f}s, {r['Pct_Under60s']:.1f}% <60s")

print("\n=== DONE ===")
