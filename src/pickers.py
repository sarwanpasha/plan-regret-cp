"""
Plan-picker implementations for each conformal method.

Each picker is a function that, given a query id and the prerequisite
calibration object(s), returns the index of the chosen plan within
`plan_sets[qid].plans`.

All pickers (except M1 PR-CP) follow the standard "upper-bound integration":
replace each cardinality estimate with its CP upper bound, sum to get a
cost upper bound, and choose argmin over plans. M1 PR-CP uses the
unwrapped point-estimate picker and adds a *certificate* on the chosen
plan's regret rather than changing the chosen plan.

Notation:
  plan_cost_estimates[qid] is a list-of-plans, each a list of per-intermediate
  cardinality estimates from the base estimator.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np
import torch

from src.benchmarks import BenchmarkSchema
from src.featurize import featurize_subquery


# =============================================================
# Point-estimate picker (baseline; also the M1 PR-CP picker)
# =============================================================
def picker_point_estimate(qid: int, plan_sets, plan_cost_estimates) -> int:
    est_costs = [sum(plan_cost_estimates[qid][i])
                 for i in range(len(plan_sets[qid].plans))]
    return int(np.argmin(est_costs))


# =============================================================
# T1: global split-CP upper-bound picker
# (provably == point estimate; included for empirical verification)
# =============================================================
def picker_split_cp(qid: int, plan_sets, plan_cost_estimates,
                    q_global: float) -> int:
    qp = plan_sets[qid]
    ub_costs = []
    for plan_idx, plan in enumerate(qp.plans):
        ests = plan_cost_estimates[qid][plan_idx]
        ub = sum(math.exp(math.log(c + 1.0) + q_global) - 1.0 for c in ests)
        ub_costs.append(ub)
    return int(np.argmin(ub_costs))


# =============================================================
# T1+: MUB-by-Size picker (size-stratified upper-bound)
# =============================================================
def picker_split_cp_by_size(qid: int, plan_sets, plan_cost_estimates,
                            q_by_size: Dict[int, float],
                            fallback: float) -> int:
    qp = plan_sets[qid]
    ub_costs = []
    for plan_idx, plan in enumerate(qp.plans):
        ests = plan_cost_estimates[qid][plan_idx]
        ub = 0.0
        for k in range(len(ests)):
            q = q_by_size.get(k + 1, fallback)
            ub += math.exp(math.log(ests[k] + 1.0) + q) - 1.0
        ub_costs.append(ub)
    return int(np.argmin(ub_costs))


# =============================================================
# T2: locally-weighted CP picker
# Needs to call the xgboost difficulty model per-intermediate.
# =============================================================
def picker_lw_split_cp(qid: int, plan_sets, plan_cost_estimates,
                       queries, schema: BenchmarkSchema,
                       table_sizes: Dict[str, int], q_lw: float, g_hat,
                       floor: float = 0.05) -> int:
    qp = plan_sets[qid]
    q = queries[qid]
    ub_costs = []
    for plan_idx, plan in enumerate(qp.plans):
        ests = plan_cost_estimates[qid][plan_idx]
        feats_per_inter = []
        for k in range(1, len(plan.order) + 1):
            tabs = plan.order[:k]
            sub_e = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                     if a in tabs and b in tabs]
            sub_p = [(t, p) for (t, p) in q.predicates if t in tabs]
            feats_per_inter.append(
                featurize_subquery(tabs, sub_e, sub_p, table_sizes,
                                   schema.tables, schema.edge_keys)
            )
        feats_per_inter = np.array(feats_per_inter)
        u = np.maximum(floor, g_hat.predict(feats_per_inter))
        ub = 0.0
        for k_i in range(len(ests)):
            ub += math.exp(math.log(ests[k_i] + 1.0) + q_lw * u[k_i]) - 1.0
        ub_costs.append(ub)
    return int(np.argmin(ub_costs))


# =============================================================
# T3: CQR picker
# =============================================================
def picker_cqr(qid: int, plan_sets, queries, schema: BenchmarkSchema,
               table_sizes: Dict[str, int],
               q_cqr: float, cqr_model, x_mean: np.ndarray, x_std: np.ndarray) -> int:
    qp = plan_sets[qid]
    q = queries[qid]
    ub_costs = []
    for plan_idx, plan in enumerate(qp.plans):
        feats_per_inter = []
        for k in range(1, len(plan.order) + 1):
            tabs = plan.order[:k]
            sub_e = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                     if a in tabs and b in tabs]
            sub_p = [(t, p) for (t, p) in q.predicates if t in tabs]
            feats_per_inter.append(
                featurize_subquery(tabs, sub_e, sub_p, table_sizes,
                                   schema.tables, schema.edge_keys)
            )
        feats_per_inter = np.array(feats_per_inter)
        Xn = (feats_per_inter - x_mean) / x_std
        with torch.no_grad():
            _, hi = cqr_model(torch.tensor(Xn).float())
            hi = hi.numpy().flatten()
        ub = sum(math.exp(hi[k_i] + q_cqr) - 1.0 for k_i in range(len(hi)))
        ub_costs.append(ub)
    return int(np.argmin(ub_costs))


# =============================================================
# Helper: regret of a chosen plan
# =============================================================
def plan_regret(plan_sets, qid: int, chosen_idx: int) -> float:
    qp = plan_sets[qid]
    chosen = qp.plans[chosen_idx].cost
    oracle = qp.plans[qp.oracle_idx].cost
    return (chosen + 1.0) / (oracle + 1.0)
