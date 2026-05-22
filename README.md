# Robust last-mile mode selection in supply chain design under speed-dependent demand

Replication code and results for the paper *"Robust last-mile mode selection in supply chain design under speed-dependent demand"* by Donghwan Kim and Biswajit Sarkar (Yonsei University), submitted to *European Journal of Operational Research*.

The repository contains a Python/Gurobi implementation of a two-stage budgeted-robust supply chain network design model with mode-dependent demand, solved by a tailored Column-and-Constraint Generation (C&CG) algorithm with McCormick linearization, together with the 6,100 raw CSV result files and the post-processing scripts used to produce every table and figure in the paper.

## Repository structure

```
.
├── codes/             Python source code (model, algorithm, experiment drivers)
├── data/              100 pre-generated instances (.pkl), full (R=50) and full200 (R=200), seeds 1-50
├── result/            6,100 raw CSV result files
│   ├── exp1/          Experiment 1: base case (R=50), 800 runs
│   ├── exp2/          Experiment 2: Gamma sensitivity (R=200), 4,000 runs
│   ├── breakeven/     Experiment 3: tau_2 breakeven, 500 runs
│   ├── coverage/      Experiment 4: service coverage, 480 runs
│   └── linear_di/     Experiment 5: linear DI functional form, 320 runs
├── analysis/          Post-processing: produces tables and figures from result/
│   ├── analyze_all.py         Generates all main and supplementary tables/figures
│   ├── extract_cost_eta.py    Generates cost decomposition and eta-direction statistics
│   ├── generated_tables/      LaTeX snippets for paper tables (T1-T8, A1-A4)
│   └── generated_figures/     CSV data underlying paper figures (F1-F3)
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.8
- Gurobi Optimizer >= 10.0 with a valid license (free academic licenses available at <https://www.gurobi.com/academia/>)
- Python packages: `gurobipy`, `numpy`, `pandas`, `matplotlib`

Install Python dependencies with

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

All commands below assume the working directory is `codes/`.

```bash
cd codes
```

### Single-run sanity check

```bash
# Optimal policy
python main.py full 10 --seed 1 --di HD

# A fixed-mode benchmark
python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2
```

### Full experiments

The five experiment drivers reproduce every CSV in `result/`. The 100 pre-generated instances in `data/` are picked up automatically.

```bash
# Experiment 1: base case (R=50), 800 runs
python run_exp1.py

# Experiment 2: Gamma sensitivity (R=200), 4,000 runs
python run_exp2.py

# Experiment 3: tau_2 breakeven, 500 runs
python run_breakeven.py

# Experiment 4: service coverage, 480 runs
python run_coverage.py

# Experiment 5: linear DI functional form, 320 runs
python run_linear_di.py
```

### Regenerating the instance data

The instances shipped in `data/` can be regenerated deterministically from seeds 1--50:

```bash
cd codes
python generate_50_seeds.py full
python generate_50_seeds.py full200
```

## Reproducing tables and figures

After the experiments have finished (or using the shipped CSVs in `result/`):

```bash
cd analysis
python analyze_all.py        # Tables T1-T8, A1-A4 and figure-CSVs F1-F3
python extract_cost_eta.py   # Cost decomposition and eta-direction statistics
```

Outputs are written into `analysis/generated_tables/` and `analysis/generated_figures/`.

## Command-line options

The following options are accepted by `main.py` and `main_fixed_mode.py`:

| Option | Description |
|---|---|
| `--tc2 0.60` | Overrides the fast-mode transportation cost tau_2 (used by the breakeven experiment) |
| `--di-func linear` | Switches to the linear demand-increase function (robustness check) |
| `--coverage moderate` | Applies a distance-based service-coverage restriction (`tight`, `moderate`, or `relaxed`) |

## Citation

If you use this code, please cite:

> Kim, D., and Sarkar, B. Robust last-mile mode selection in supply chain design under speed-dependent demand. 
