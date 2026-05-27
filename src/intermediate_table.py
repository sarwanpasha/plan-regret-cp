"""
Build a flat table over all (query, plan, intermediate) triples.

This is the shared input to the cardinality-CP calibration methods
(T1, T1+, T2, T3). M1 PR-CP works at the query level instead and does not
use this table.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.benchmarks import BenchmarkSchema
from src.conformal import IntermediateTable
from src.data_gen import Query, QueryPlans
from src.featurize import featurize_subquery


def build_intermediate_table(
    queries: List[Query],
    plan_sets: List[QueryPlans],
    plan_cost_estimates: Dict[int, List[List[float]]],
    table_sizes: Dict[str, int],
    schema: BenchmarkSchema,
    qids: Sequence[int],
) -> IntermediateTable:
    qid_list, plan_idx_list, k_list = [], [], []
    feat_list, est_list, true_list = [], [], []
    qid_set = set(qids)
    for q, qp in zip(queries, plan_sets):
        if q.qid not in qid_set:
            continue
        for plan_idx, plan in enumerate(qp.plans):
            for k in range(1, len(plan.order) + 1):
                tabs = plan.order[:k]
                sub_e = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                         if a in tabs and b in tabs]
                sub_p = [(t, p) for (t, p) in q.predicates if t in tabs]
                feat = featurize_subquery(
                    tabs, sub_e, sub_p, table_sizes,
                    schema.tables, schema.edge_keys,
                )
                feat_list.append(feat)
                est_list.append(plan_cost_estimates[q.qid][plan_idx][k - 1])
                true_list.append(plan.intermediate_cards[k - 1])
                qid_list.append(q.qid)
                plan_idx_list.append(plan_idx)
                k_list.append(k)
    return IntermediateTable(
        qid_arr=np.array(qid_list, dtype=np.int64),
        plan_idx_arr=np.array(plan_idx_list, dtype=np.int64),
        k_arr=np.array(k_list, dtype=np.int64),
        feat_arr=np.array(feat_list, dtype=np.float32),
        est_arr=np.array(est_list, dtype=np.float32),
        true_arr=np.array(true_list, dtype=np.int64),
    )
