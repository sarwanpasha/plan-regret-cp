# Plan-Regret Conformal Prediction

Code for the paper *Plan-Regret Conformal Prediction: A Decision-Aware Framework for Robust Cardinality Estimation* (anonymous submission).

This repository contains the full experimental pipeline that produced the results reported in the paper, plus sample outputs for direct inspection.

## What this code does

The paper studies how a query optimizer should consume cardinality-estimate uncertainty when selecting an execution plan. Concretely, it implements and compares six methods:

| Method | What it calibrates | Plan picker |
|---|---|---|
| Point-est | nothing (baseline) | argmin of point-estimate costs |
| T1 Split-CP | global cardinality residual quantile | argmin of upper-bound costs (provably == point-est, see paper Prop. 1) |
| T1+ MUB-by-Size *(ours)* | cardinality residuals stratified by intermediate size | argmin of stratified upper-bound costs |
| T2 LW-S-CP | locally-weighted cardinality residual (xgboost) | argmin of per-intermediate adaptive upper-bound costs |
| T3 CQR | conformalized quantile regression on log-cardinality | argmin of CQR upper-bound costs |
| M1 PR-CP *(ours)* | log-ratio of chosen-plan cost to oracle-plan cost | point-estimate picker with certified plan-regret bound |

All six are evaluated on TPC-H (scale factor 1) and a TPC-DS subset (scale factor 1) using a shared MSCN-style neural cardinality estimator. The headline metric is multiplicative plan-regret (chosen plan's true cost ÷ oracle plan's true cost).

## Quick start

```bash
# Dependencies
pip install duckdb numpy pandas torch scikit-learn xgboost

# Generate the data, train the estimator, calibrate everything,
# and produce the paper's headline tables. Total time: ~30-60 min on a modern laptop.
python scripts/run_all.py
```

The orchestration script runs each stage in order and reuses cached outputs from earlier stages, so partial reruns are fast.

## Repository layout

```
plan-regret-cp/
├── README.md                       # this file
├── requirements.txt                # pinned-ish dependencies
├── scripts/
│   ├── run_all.py                  # one-button driver -- runs everything end-to-end
│   ├── 01_generate_data.py         # generate TPC-H + TPC-DS workloads, true cardinalities, plans
│   ├── 02_train_estimator.py       # train MSCN-style neural CE
│   ├── 03_calibrate_and_evaluate.py # all 6 methods, both benchmarks, all 4 alphas
│   └── 04_summarize.py             # produce paper-shaped tables
├── src/
│   ├── data_gen.py                 # query workload generation, ground-truth cardinalities
│   ├── plan_enumeration.py         # left-deep plan enumeration, C_out cost computation
│   ├── featurize.py                # query / subquery feature vectors
│   ├── estimator.py                # MSCN-style neural network
│   ├── conformal.py                # all conformal methods (T1, T1+, T2, T3, M1)
│   └── pickers.py                  # plan picker implementations
├── sample_outputs/                 # the actual numbers from the paper
│   ├── comparison_tpch.csv         # per-query plan-regret for all methods on TPC-H
│   ├── comparison_tpcds.csv        # same on TPC-DS
│   └── round5_summary.json         # all calibration constants and aggregated statistics
└── docs/
    └── REPRODUCTION.md             # which paper number lives where, exact reproduction recipe
```

## Reproducing the paper's tables

After `python scripts/run_all.py` completes, the paper's three result tables are available as:

- **Table I (cardinality coverage):** `results/card_coverage.csv`
- **Table II (TPC-H plan-regret):** `results/plan_regret_tpch.csv`
- **Table III (TPC-DS plan-regret):** `results/plan_regret_tpcds.csv`
- **Table IV (PR-CP certified bounds):** `results/pr_cp_bounds.csv`

For convenience, `sample_outputs/` contains the exact numbers that appear in the paper, in case a reviewer wants to inspect results without re-running anything.

## Compute requirements

The pipeline is designed to run on a single machine with no special hardware. CPU-only is sufficient; no GPU is needed for the small MSCN estimator we use.

- Disk: ~3 GB for TPC-H SF=1 + TPC-DS SF=1 + intermediate caches
- RAM: 8 GB sufficient, 16 GB comfortable (for plan-enumeration on TPC-DS)
- Time: ~30-60 minutes end-to-end on a modern laptop, dominated by ground-truth cardinality computation

If you want a faster smoke test, set `SCALE_FACTOR = 0.1` in `scripts/run_all.py` (≈ 5 min total).

## Design choices worth mentioning

**DuckDB for benchmark data.** TPC-H and TPC-DS are generated programmatically inside DuckDB via its built-in `tpch` and `tpcds` extensions (`CALL dbgen(sf=1)`, `CALL dsdgen(sf=1)`). This means no external data download is required — the benchmarks are reproduced exactly.

**C_out as the cost metric.** Following Leis et al. (VLDB 2015) and the broader learned-CE literature, we use $C_{\mathrm{out}}$ (the sum of true intermediate cardinalities) as the cost metric. This isolates CE quality from physical-operator effects.

**Left-deep plans only.** To make plan enumeration tractable, we restrict to left-deep join orderings, capped at 6-8 valid orderings per query.

**Shared base estimator across methods.** All six methods use the same trained MSCN-style network as the underlying point estimator. The differences between methods are entirely in the calibration scheme and the plan-picker integration, isolating the wrapper effect from estimator-quality differences.

## Numbers from the paper, at a glance

From `sample_outputs/` (TPC-H, $\alpha=0.10$, $n_{\mathrm{test}}=500$ queries):

| Method | P50 regret | P90 regret | P99 regret | Max regret |
|---|---|---|---|---|
| Point-est | 1.00× | 1.33× | 196× | 48,225× |
| T1 Split-CP | 1.00× | 1.33× | 196× | 48,225× |
| **T1+ MUB-by-Size** | 1.00× | 1.43× | **116×** | **10,031×** |
| T2 LW-S-CP | 1.00× | 8.85× | 9,517× | 810,001× |
| T3 CQR | 1.00× | 3.34× | 4,503× | 48,225× |

PR-CP certifies plan-regret ≤ **1.53× of oracle** with 91.4% empirical coverage at α=0.10 on TPC-H, and **1.99×** with 95.0% coverage on TPC-DS.

