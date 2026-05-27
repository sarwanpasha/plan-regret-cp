"""
Featurization of (sub-)queries for the MSCN-style cardinality estimator.

A feature vector has these components:
  - one-hot of all tables in the schema
  - one-hot of all join edges in the schema
  - per-table predicate count
  - log of cross-product of involved table sizes (a coarse scaling signal)

Both TPC-H and TPC-DS use this same template; the schema (table list + join graph)
is provided by the benchmark-specific configuration in src/benchmarks.py.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def featurize_subquery(
    tables_in_subquery: Sequence[str],
    edges_in_subquery: Sequence[Tuple[str, str, str, str]],
    predicates_in_subquery: Sequence[Tuple[str, str]],
    table_sizes: Dict[str, int],
    all_tables: Sequence[str],
    edge_keys: Sequence[Tuple[str, str]],
) -> np.ndarray:
    """
    Produce a numeric feature vector for an intermediate subquery.

    Args:
        tables_in_subquery: list of table names present in the subquery
        edges_in_subquery: list of (table_a, col_a, table_b, col_b) join edges
            active in the subquery
        predicates_in_subquery: list of (table, predicate_sql) tuples
        table_sizes: total row count per table (used for the cross-product feature)
        all_tables: ordered list of all tables in the schema (defines one-hot order)
        edge_keys: ordered list of (table_a, table_b) for all possible join edges

    Returns:
        A 1D numpy float32 array of length
            len(all_tables) + len(edge_keys) + len(all_tables) + 1.
    """
    feat: List[float] = []

    table_set = set(tables_in_subquery)
    for t in all_tables:
        feat.append(1.0 if t in table_set else 0.0)

    edge_set = set()
    for (a, _, b, _) in edges_in_subquery:
        if a in table_set and b in table_set:
            edge_set.add(tuple(sorted([a, b])))
    for (a, b) in edge_keys:
        feat.append(1.0 if tuple(sorted([a, b])) in edge_set else 0.0)

    pred_count = {t: 0 for t in all_tables}
    for (t, _) in predicates_in_subquery:
        if t in pred_count:
            pred_count[t] += 1
    for t in all_tables:
        feat.append(float(pred_count[t]))

    log_xprod = sum(
        math.log(max(1, table_sizes.get(t, 1))) for t in tables_in_subquery
    )
    feat.append(log_xprod)

    return np.array(feat, dtype=np.float32)
