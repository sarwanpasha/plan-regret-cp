"""
Generate a random multi-join workload over a benchmark schema and compute
ground-truth cardinalities for every (sub)query.

A workload entry is a Query with:
  - tables: list of table names
  - edges:  list of (table_a, col_a, table_b, col_b) join edges actually used
  - predicates: list of (table_name, predicate_sql) tuples
  - sql:    materialized SQL query
  - cardinality: true count of result rows

We restrict to left-deep plans for tractability (see paper Sec. VII).
"""
from __future__ import annotations

import itertools
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.benchmarks import BenchmarkSchema


@dataclass
class Query:
    qid: int
    tables: List[str]
    edges: List[Tuple[str, str, str, str]]
    predicates: List[Tuple[str, str]]
    sql: str
    cardinality: int = -1


@dataclass
class Plan:
    order: List[str]
    intermediate_sqls: List[str]
    intermediate_cards: List[int]
    cost: float    # C_out: sum of intermediate cardinalities


@dataclass
class QueryPlans:
    qid: int
    plans: List[Plan]
    oracle_idx: int   # index into plans of the cost-minimizing one


# =============================================================
# Workload sampling
# =============================================================
def _sample_connected_subgraph(
    rng: random.Random, schema: BenchmarkSchema, max_tables: int
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """Random connected subgraph of the join graph, 2..max_tables tables."""
    edges = list(schema.join_graph)
    rng.shuffle(edges)
    used_edges = [edges[0]]
    used_tables = {edges[0][0], edges[0][2]}
    target = rng.randint(2, max_tables)
    for e in edges[1:]:
        if len(used_tables) >= target:
            break
        if e[0] in used_tables or e[2] in used_tables:
            used_edges.append(e)
            used_tables.update([e[0], e[2]])
    return list(used_tables), used_edges


def _sample_predicate(rng: random.Random, table: str, schema: BenchmarkSchema):
    """Sample a single predicate for a given table; None if no template."""
    cols = schema.predicate_cols.get(table, [])
    if not cols:
        return None
    spec = rng.choice(cols)
    col, kind = spec[0], spec[1]
    if kind == "num":
        lo, hi = spec[2], spec[3]
        a = rng.uniform(lo, hi)
        width = (hi - lo) * rng.uniform(0.05, 0.6)
        b = a + width
        return f"{col} BETWEEN {a:.4f} AND {b:.4f}"
    elif kind == "cat":
        choices = spec[2]
        n = rng.randint(1, max(1, len(choices) // 2))
        picks = rng.sample(choices, k=n)
        quoted = ", ".join(f"'{p}'" for p in picks)
        return f"{col} IN ({quoted})"
    return None


def _build_sql(tables: List[str], edges, predicates: List[str]) -> str:
    """Assemble a SELECT COUNT(*) SQL string from a connected join graph."""
    sql = "SELECT COUNT(*) FROM " + tables[0]
    in_join = {tables[0]}
    remaining = list(edges)
    while remaining:
        progressed = False
        for e in list(remaining):
            ta, ca, tb, cb = e
            if ta in in_join and tb not in in_join:
                sql += f" JOIN {tb} ON {ta}.{ca} = {tb}.{cb}"
                in_join.add(tb)
                remaining.remove(e)
                progressed = True
            elif tb in in_join and ta not in in_join:
                sql += f" JOIN {ta} ON {ta}.{ca} = {tb}.{cb}"
                in_join.add(ta)
                remaining.remove(e)
                progressed = True
        if not progressed:
            break
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql


def generate_workload(
    duckdb_con,
    schema: BenchmarkSchema,
    n_queries: int,
    max_tables: int = 5,
    seed: int = 7,
    drop_empty_prob: float = 0.95,
    cache_path: Path | None = None,
) -> List[Query]:
    """Generate `n_queries` queries and execute each for ground-truth cardinality.

    `duckdb_con` is an already-open read-only DuckDB connection.
    If `cache_path` is provided and the file exists, the workload is loaded from disk.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            wl = pickle.load(f)
        if len(wl) >= n_queries:
            return wl[:n_queries]

    rng = random.Random(seed)
    queries: List[Query] = []
    attempts = 0
    t0 = time.time()
    while len(queries) < n_queries and attempts < n_queries * 8:
        attempts += 1
        tables, edges = _sample_connected_subgraph(rng, schema, max_tables)
        if len(tables) < 2:
            continue
        n_preds_target = rng.randint(1, 4)
        preds_typed: List[Tuple[str, str]] = []
        candidates = list(tables)
        rng.shuffle(candidates)
        for t in candidates:
            if len(preds_typed) >= n_preds_target:
                break
            for _ in range(2):
                p = _sample_predicate(rng, t, schema)
                if p is not None:
                    preds_typed.append((t, p))
                    if len(preds_typed) >= n_preds_target:
                        break
        sql = _build_sql(tables, edges, [p for _, p in preds_typed])
        try:
            card = duckdb_con.execute(sql).fetchone()[0]
        except Exception:
            continue
        if card == 0 and rng.random() < drop_empty_prob:
            continue
        queries.append(Query(
            qid=len(queries), tables=sorted(tables), edges=edges,
            predicates=preds_typed, sql=sql, cardinality=int(card),
        ))
        if len(queries) % 200 == 0:
            print(f"  [workload] {len(queries)}/{n_queries} "
                  f"(attempts={attempts}, {time.time()-t0:.1f}s)")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(queries, f)
    return queries
