"""
Left-deep plan enumeration.

Given a query, enumerate up to a few connected left-deep orderings, compute the
true cardinality of every intermediate, and compute the C_out cost of each plan.
"""
from __future__ import annotations

import itertools
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.benchmarks import BenchmarkSchema
from src.data_gen import Plan, Query, QueryPlans, _build_sql


def _enumerate_connected_left_deep(
    tables: List[str],
    edges: List[Tuple[str, str, str, str]],
    max_orders: int = 8,
    seed: int = 7,
) -> List[List[str]]:
    """All left-deep orderings where every prefix is connected via the join graph."""
    if len(tables) <= 1:
        return [list(tables)]
    adj: Dict[str, set] = {t: set() for t in tables}
    for ta, _, tb, _ in edges:
        if ta in adj and tb in adj:
            adj[ta].add(tb)
            adj[tb].add(ta)
    orders: List[List[str]] = []
    perms = list(itertools.permutations(tables))
    rng = random.Random(seed + len(tables))
    rng.shuffle(perms)
    for perm in perms:
        ok = True
        for i in range(2, len(perm) + 1):
            seen = {perm[0]}
            frontier = [perm[0]]
            while frontier:
                x = frontier.pop()
                for y in adj[x]:
                    if y in perm[:i] and y not in seen:
                        seen.add(y)
                        frontier.append(y)
            if seen != set(perm[:i]):
                ok = False
                break
        if ok:
            orders.append(list(perm))
            if len(orders) >= max_orders:
                break
    if not orders:
        orders = [list(tables)]
    return orders


def enumerate_plans(
    duckdb_con,
    queries: List[Query],
    schema: BenchmarkSchema,
    max_orders: int = 8,
    cache_path: Path | None = None,
) -> List[QueryPlans]:
    """For each query, enumerate plans and compute true intermediate cardinalities.

    Caches results on disk if `cache_path` is provided. Reuses an
    intermediate-SQL cache within a single call so that subqueries shared
    across plans are executed only once.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            qps = pickle.load(f)
        if len(qps) >= len(queries):
            return qps[:len(queries)]

    card_cache: Dict[str, int] = {}
    qps: List[QueryPlans] = []
    t0 = time.time()
    for q in queries:
        orders = _enumerate_connected_left_deep(q.tables, q.edges, max_orders)
        plans: List[Plan] = []
        for order in orders:
            inter_sqls: List[str] = []
            inter_cards: List[int] = []
            for k in range(1, len(order) + 1):
                tabs = order[:k]
                sub_edges = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                             if a in tabs and b in tabs]
                sub_preds = [p for (t, p) in q.predicates if t in tabs]
                sub_sql = _build_sql(tabs, sub_edges, sub_preds)
                if sub_sql in card_cache:
                    c = card_cache[sub_sql]
                else:
                    try:
                        c = int(duckdb_con.execute(sub_sql).fetchone()[0])
                    except Exception:
                        c = 0
                    card_cache[sub_sql] = c
                inter_sqls.append(sub_sql)
                inter_cards.append(c)
            cost = float(sum(inter_cards))
            plans.append(Plan(
                order=order, intermediate_sqls=inter_sqls,
                intermediate_cards=inter_cards, cost=cost,
            ))
        oracle_idx = int(np.argmin([p.cost for p in plans]))
        qps.append(QueryPlans(qid=q.qid, plans=plans, oracle_idx=oracle_idx))
        if (q.qid + 1) % 200 == 0:
            print(f"  [plans] {q.qid+1}/{len(queries)} "
                  f"(cache={len(card_cache):,}, {time.time()-t0:.1f}s)")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(qps, f)
    return qps
