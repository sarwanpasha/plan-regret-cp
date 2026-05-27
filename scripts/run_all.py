"""
End-to-end driver: generate data, train estimator, calibrate, evaluate, summarize.

For both TPC-H and TPC-DS, this script:
  1. Generates the database via DuckDB's built-in benchmark extensions.
  2. Samples a multi-join workload and computes ground-truth cardinalities.
  3. Enumerates left-deep plans and their intermediate cardinalities.
  4. Trains an MSCN-style neural cardinality estimator.
  5. Calibrates all six methods (T1, T1+, T2, T3, M1, baseline point-est).
  6. Evaluates plan-regret on the test split.
  7. Writes paper-shaped result CSVs to results/.

All intermediate artifacts are cached on disk under cache/, so reruns are cheap.

Run:
    python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the `src` package importable from this script's location
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from src.benchmarks import get_schema
from src.conformal import (
    calibrate_split_cp, calibrate_split_cp_by_size,
    fit_difficulty_xgb, calibrate_lw_split_cp,
    fit_cqr, calibrate_cqr,
    calibrate_plan_regret,
)
from src.data_gen import generate_workload
from src.estimator import (
    CardinalityEstimator, train_estimator, compute_plan_estimates,
)
from src.intermediate_table import build_intermediate_table
from src.pickers import (
    picker_point_estimate, picker_split_cp, picker_split_cp_by_size,
    picker_lw_split_cp, picker_cqr, plan_regret,
)
from src.plan_enumeration import enumerate_plans


# =============================================================
# Config — change these to scale the experiment up or down
# =============================================================
SCALE_FACTOR = 1            # TPC-H/TPC-DS scale factor; 1 ~ 1 GB / 600 MB
ALPHAS = [0.01, 0.05, 0.10, 0.20]
SEED = 7

TPCH_N = (1500, 500, 500)   # train, calib, test
TPCDS_N = (900, 300, 300)

MAX_TABLES = 5
MAX_PLAN_ORDERS = 8

CACHE_DIR = Path("./cache")
RESULTS_DIR = Path("./results")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# Bench setup: create DuckDB with TPC-H or TPC-DS
# =============================================================
def setup_benchmark_db(name: str, db_path: Path):
    """Create the DuckDB file with TPC-H or TPC-DS data at SCALE_FACTOR."""
    if db_path.exists():
        return
    print(f"  [setup] Generating {name.upper()} SF={SCALE_FACTOR} into {db_path}")
    con = duckdb.connect(str(db_path))
    if name == "tpch":
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")
        con.execute(f"CALL dbgen(sf={SCALE_FACTOR})")
    elif name == "tpcds":
        con.execute("INSTALL tpcds")
        con.execute("LOAD tpcds")
        con.execute(f"CALL dsdgen(sf={SCALE_FACTOR})")
    else:
        raise ValueError(name)
    con.close()


def table_sizes(con, schema) -> dict:
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in schema.tables}


# =============================================================
# One benchmark end-to-end
# =============================================================
def run_benchmark(name: str, splits: tuple) -> dict:
    n_train, n_calib, n_test = splits
    n_total = n_train + n_calib + n_test
    schema = get_schema(name)
    bench_dir = CACHE_DIR / name
    bench_dir.mkdir(exist_ok=True)
    db_path = bench_dir / f"{name}.duckdb"

    setup_benchmark_db(name, db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    sizes = table_sizes(con, schema)
    print(f"  [bench] {name} table sizes: " +
          ", ".join(f"{t}={n:,}" for t, n in sizes.items()))

    # 1) Workload
    print(f"  [step] Generating {n_total} queries ...")
    queries = generate_workload(
        con, schema, n_queries=n_total, max_tables=MAX_TABLES, seed=SEED,
        cache_path=bench_dir / "workload.pkl",
    )

    # 2) Plans + true intermediate cardinalities
    print(f"  [step] Enumerating plans and computing true intermediate cardinalities ...")
    plan_sets = enumerate_plans(
        con, queries, schema, max_orders=MAX_PLAN_ORDERS,
        cache_path=bench_dir / "plans.pkl",
    )
    con.close()

    # 3) Train estimator
    print(f"  [step] Training MSCN-style estimator on {n_train} training queries ...")
    train_qids = list(range(n_train))
    calib_qids = list(range(n_train, n_train + n_calib))
    test_qids  = list(range(n_train + n_calib, n_total))

    model_path = bench_dir / "mscn.pt"
    if model_path.exists():
        estimator = CardinalityEstimator.load(model_path, sizes, schema)
    else:
        estimator = train_estimator(
            queries, plan_sets, train_qids, sizes, schema, seed=SEED,
        )
        estimator.save(model_path)

    # 4) Per-plan estimates (one inference per intermediate per plan)
    print(f"  [step] Computing per-plan cardinality estimates ...")
    plan_cost_estimates = compute_plan_estimates(
        estimator, queries, plan_sets,
        cache_path=bench_dir / "plan_estimates.pkl",
    )

    # 5) Intermediate tables for cardinality-CP methods
    print(f"  [step] Building train/calib intermediate tables ...")
    train_tab = build_intermediate_table(
        queries, plan_sets, plan_cost_estimates, sizes, schema, train_qids)
    calib_tab = build_intermediate_table(
        queries, plan_sets, plan_cost_estimates, sizes, schema, calib_qids)
    test_tab  = build_intermediate_table(
        queries, plan_sets, plan_cost_estimates, sizes, schema, test_qids)
    print(f"     train: {len(train_tab.feat_arr):,} rows | "
          f"calib: {len(calib_tab.feat_arr):,} rows | "
          f"test: {len(test_tab.feat_arr):,} rows")

    # 6) Calibrate all methods
    print(f"  [step] Calibrating T1 (split CP) ...")
    q_t1 = calibrate_split_cp(calib_tab, ALPHAS)

    print(f"  [step] Calibrating T1+ (split CP by size) ...")
    q_t1s = calibrate_split_cp_by_size(calib_tab, ALPHAS)

    print(f"  [step] Calibrating T2 (LW-S-CP) -- training xgboost ...")
    g_hat = fit_difficulty_xgb(train_tab, seed=SEED)
    q_t2 = calibrate_lw_split_cp(calib_tab, g_hat, ALPHAS)

    print(f"  [step] Calibrating T3 (CQR) -- training one quantile net per alpha ...")
    q_t3 = {}
    cqr_per_alpha = {}
    for a in ALPHAS:
        m, xm, xs = fit_cqr(train_tab, alpha=a, seed=SEED)
        cqr_per_alpha[a] = (m, xm, xs)
        q_t3[a] = calibrate_cqr(calib_tab, m, xm, xs, alpha=a)

    print(f"  [step] Calibrating M1 (PR-CP) ...")
    tau_pr = calibrate_plan_regret(
        plan_sets, plan_cost_estimates, calib_qids, ALPHAS)

    # 7) Evaluate per-test-query plan regret
    print(f"  [step] Evaluating plan regret on {n_test} test queries ...")
    rows = []
    for qid in test_qids:
        qp = plan_sets[qid]
        oracle = qp.plans[qp.oracle_idx].cost

        pt_idx = picker_point_estimate(qid, plan_sets, plan_cost_estimates)
        pt_regret = plan_regret(plan_sets, qid, pt_idx)
        row = {
            "qid": qid,
            "n_tables": len(qp.plans[0].order),
            "oracle_cost": oracle,
            "pt_regret_lin": pt_regret,
            "pt_regret_log": float(np.log(pt_regret)),
            "is_trivial_oracle": int(pt_regret < 1.001),
        }
        for a in ALPHAS:
            t1_idx = picker_split_cp(qid, plan_sets, plan_cost_estimates, q_t1[a])
            row[f"t1_regret_{a}"] = plan_regret(plan_sets, qid, t1_idx)

            t1s_idx = picker_split_cp_by_size(qid, plan_sets, plan_cost_estimates,
                                              q_t1s[a], q_t1[a])
            row[f"t1s_regret_{a}"] = plan_regret(plan_sets, qid, t1s_idx)

            t2_idx = picker_lw_split_cp(
                qid, plan_sets, plan_cost_estimates, queries, schema, sizes,
                q_t2[a], g_hat,
            )
            row[f"t2_regret_{a}"] = plan_regret(plan_sets, qid, t2_idx)

            m, xm, xs = cqr_per_alpha[a]
            t3_idx = picker_cqr(qid, plan_sets, queries, schema, sizes,
                                q_t3[a], m, xm, xs)
            row[f"t3_regret_{a}"] = plan_regret(plan_sets, qid, t3_idx)

            row[f"m1_certified_{a}"] = int(row["pt_regret_log"] <= tau_pr[a])

        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"comparison_{name}.csv", index=False)

    # 8) Card-coverage on test intermediates (for paper Table I)
    print(f"  [step] Measuring cardinality coverage on test intermediates ...")
    cov_rows = []
    y_test = np.log(test_tab.true_arr + 1.0).astype(np.float32)
    log_est = np.log(test_tab.est_arr + 1.0).astype(np.float32)
    for a in ALPHAS:
        # T1
        L = log_est - q_t1[a]; U = log_est + q_t1[a]
        cov_t1 = float(((y_test >= L) & (y_test <= U)).mean())
        w_t1 = float(np.median(U - L))
        # T1+
        qs = q_t1s[a]
        L = log_est - np.array([qs.get(int(k), q_t1[a]) for k in test_tab.k_arr])
        U = log_est + np.array([qs.get(int(k), q_t1[a]) for k in test_tab.k_arr])
        cov_t1s = float(((y_test >= L) & (y_test <= U)).mean())
        w_t1s = float(np.median(U - L))
        # T2
        u_test = np.maximum(0.05, g_hat.predict(test_tab.feat_arr))
        L = log_est - q_t2[a] * u_test; U = log_est + q_t2[a] * u_test
        cov_t2 = float(((y_test >= L) & (y_test <= U)).mean())
        w_t2 = float(np.median(U - L))
        # T3
        import torch
        m, xm, xs = cqr_per_alpha[a]
        Xn = (test_tab.feat_arr - xm) / xs
        with torch.no_grad():
            lo, hi = m(torch.tensor(Xn))
            lo = lo.numpy().flatten(); hi = hi.numpy().flatten()
        L = lo - q_t3[a]; U = hi + q_t3[a]
        cov_t3 = float(((y_test >= L) & (y_test <= U)).mean())
        w_t3 = float(np.median(U - L))
        cov_rows.append({
            "alpha": a,
            "t1_cov": cov_t1, "t1_w": w_t1,
            "t1s_cov": cov_t1s, "t1s_w": w_t1s,
            "t2_cov": cov_t2, "t2_w": w_t2,
            "t3_cov": cov_t3, "t3_w": w_t3,
        })
    pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / f"card_coverage_{name}.csv", index=False)

    # 9) Aggregated summary for PR-CP bounds + plan-regret quantiles
    summary = {
        "label": name,
        "n_test": int(len(df)),
        "trivial_oracle_rate": float((df["is_trivial_oracle"] == 1).mean()),
        "plan_regret_cp_tau":      {str(a): float(tau_pr[a]) for a in ALPHAS},
        "plan_regret_cp_exp":      {str(a): float(np.exp(tau_pr[a])) for a in ALPHAS},
        "plan_regret_cp_coverage": {str(a): float(df[f"m1_certified_{a}"].mean())
                                    for a in ALPHAS},
        "plan_regret_quantiles": {},
    }
    for a in ALPHAS:
        d = {}
        for code, col in [("pt", "pt_regret_lin"),
                          ("t1", f"t1_regret_{a}"),
                          ("t1s", f"t1s_regret_{a}"),
                          ("t2", f"t2_regret_{a}"),
                          ("t3", f"t3_regret_{a}")]:
            v = df[col].values
            d[code] = {
                "P50": float(np.percentile(v, 50)),
                "P90": float(np.percentile(v, 90)),
                "P99": float(np.percentile(v, 99)),
                "mean": float(np.mean(v)),
                "max":  float(np.max(v)),
            }
        summary["plan_regret_quantiles"][str(a)] = d
    return summary


# =============================================================
# Main
# =============================================================
def main():
    print("=" * 72)
    print("Plan-Regret Conformal Prediction — end-to-end pipeline")
    print("=" * 72)

    all_summaries = {}

    print("\n>>> TPC-H")
    all_summaries["tpch"] = run_benchmark("tpch", TPCH_N)

    print("\n>>> TPC-DS")
    all_summaries["tpcds"] = run_benchmark("tpcds", TPCDS_N)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nWrote: {RESULTS_DIR}/summary.json")
    print(f"        {RESULTS_DIR}/comparison_tpch.csv")
    print(f"        {RESULTS_DIR}/comparison_tpcds.csv")
    print(f"        {RESULTS_DIR}/card_coverage_tpch.csv")
    print(f"        {RESULTS_DIR}/card_coverage_tpcds.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
