# Reproducing the paper

This document maps each numerical claim in the paper to the file and computation that produces it.

## End-to-end recipe

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

When the script finishes, `results/` contains the CSVs and JSON below.

## Where each paper number lives

### Proposition 1 (Section IV.A): T1 split-CP picker == point-estimate picker

**Claim in paper:** "On every test query of TPC-H ($n=500$) and TPC-DS ($n=300$), at every $\alpha \in \{0.01, 0.05, 0.10, 0.20\}$, the maximum absolute difference between the T1 split-CP picker's regret and the point-estimate picker's regret is exactly 0."

**Where in the code:** `results/comparison_tpch.csv` and `results/comparison_tpcds.csv`. For any row, `t1_regret_<alpha> == pt_regret_lin` to floating-point exactness.

**Reproduce the check directly:**
```python
import pandas as pd
df = pd.read_csv("results/comparison_tpch.csv")
for a in [0.01, 0.05, 0.10, 0.20]:
    diff = (df[f"t1_regret_{a}"] - df["pt_regret_lin"]).abs().max()
    print(f"alpha={a}: max diff = {diff}")     # all should print 0.0
```

### Table I — cardinality coverage

**File:** `results/card_coverage_tpch.csv`, `results/card_coverage_tpcds.csv`

Columns: `alpha`, `t1_cov`, `t1_w` (T1 split CP), `t1s_cov`, `t1s_w` (MUB-by-Size), `t2_cov`, `t2_w` (LW-S-CP), `t3_cov`, `t3_w` (CQR). `_w` is median log-space interval width.

### Tables II and III — plan regret distributions

**Files:** `results/comparison_tpch.csv`, `results/comparison_tpcds.csv`

The CSV has one row per test query, with columns:
- `pt_regret_lin`: point-estimate plan regret (matches T1 by Prop. 1)
- `t1s_regret_<alpha>`: MUB-by-Size regret at each alpha
- `t2_regret_<alpha>`: LW-S-CP regret at each alpha
- `t3_regret_<alpha>`: CQR regret at each alpha

To produce the paper's table format (P50/P90/P99/mean/max), aggregate per column:
```python
import numpy as np, pandas as pd
df = pd.read_csv("results/comparison_tpch.csv")
def summarize(s):
    v = s.dropna().values
    return {"P50": np.percentile(v,50), "P90": np.percentile(v,90),
            "P99": np.percentile(v,99), "mean": v.mean(), "max": v.max()}
for a in [0.01, 0.05, 0.10, 0.20]:
    for code in ["pt_regret_lin", f"t1s_regret_{a}", f"t2_regret_{a}", f"t3_regret_{a}"]:
        print(a, code, summarize(df[code]))
```

### Table IV — PR-CP certified bounds

**File:** `results/summary.json`, under keys `tpch.plan_regret_cp_*` and `tpcds.plan_regret_cp_*`.

Specifically:
- `plan_regret_cp_tau[<alpha>]`: calibrated log-regret quantile $\hat\tau_\alpha$
- `plan_regret_cp_exp[<alpha>]`: certified multiplicative bound $\exp(\hat\tau_\alpha)$
- `plan_regret_cp_coverage[<alpha>]`: empirical coverage on test queries

### Hard / easy query analysis (Section VII.E)

A query is "easy" iff `is_trivial_oracle == 1` in the CSV (i.e., `pt_regret_lin < 1.001`). The paper reports the 81.4% (TPC-H) and 79.7% (TPC-DS) trivial-oracle rates from this column.

### Plan-change counts (Section IV.C)

The paper reports: "T2 changes the chosen plan in 146/500 test queries; 111 of those changes are worse than point-estimate, and only 35 are better."

To reproduce:
```python
import numpy as np
df = pd.read_csv("results/comparison_tpch.csv")
mask = (df["t2_regret_0.1"] - df["pt_regret_lin"]).abs() > 0.01
worse = ((df["t2_regret_0.1"] > df["pt_regret_lin"]) & mask).sum()
better = ((df["t2_regret_0.1"] < df["pt_regret_lin"]) & mask).sum()
print(f"changed={mask.sum()}, worse={worse}, better={better}")
```

## Cached numbers shipped in `sample_outputs/`

The CSVs and JSON under `sample_outputs/` are the exact outputs that produced the numbers in the paper, included so a reviewer can inspect results without re-running. The naming matches what `run_all.py` produces under `results/`.

## Determinism notes

- All random operations use `seed=7` (configurable in `scripts/run_all.py`).
- DuckDB-side query results are deterministic for a fixed scale factor.
- Neural network training is deterministic up to PyTorch ops that don't have deterministic CUDA kernels; on CPU (default) it is fully deterministic.
- Expect identical numbers on a re-run on the same machine; numbers across different machines (different BLAS, etc.) may differ in the 4th decimal place of the certified bound but not in the qualitative ordering of methods.

## Computational cost

On a single laptop with no GPU, at SF=1:
- TPC-H pipeline: 15-25 min
- TPC-DS pipeline: 25-40 min
- Dominated by ground-truth intermediate-cardinality computation (one COUNT(*) per (plan, intermediate-prefix))

For a quick smoke test, set `SCALE_FACTOR = 0.1` in `scripts/run_all.py` — the full pipeline finishes in ~5 minutes and the qualitative findings persist.
