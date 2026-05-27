"""
MSCN-style feed-forward neural cardinality estimator.

The architecture follows Kipf et al. (CIDR 2019) at small scale:
4 hidden layers, 256 units, ReLU activations, predicting log(c+1) from a
flat feature vector. Trained with mean-squared-error in log-space.

This is the base estimator shared across all six methods in the paper; only the
calibration wrapper differs between them.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.benchmarks import BenchmarkSchema
from src.data_gen import Query, QueryPlans
from src.featurize import featurize_subquery


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, x):
        return self.net(x)


class CardinalityEstimator:
    """Wrapper bundling a trained MLP + its feature normalization stats."""

    def __init__(self, model: _MLP, x_mean: np.ndarray, x_std: np.ndarray,
                 table_sizes: Dict[str, int], schema: BenchmarkSchema):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.table_sizes = table_sizes
        self.schema = schema
        self.model.eval()

    def estimate(self, sub_tables, sub_edges, sub_preds) -> float:
        """Return the estimated cardinality (linear scale) of a subquery."""
        feat = featurize_subquery(
            sub_tables, sub_edges, sub_preds,
            self.table_sizes, self.schema.tables, self.schema.edge_keys,
        )
        feat_n = (feat - self.x_mean.flatten()) / self.x_std.flatten()
        with torch.no_grad():
            log_pred = self.model(
                torch.tensor(feat_n).unsqueeze(0).float()
            ).item()
        return max(0.0, math.exp(log_pred) - 1.0)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean, "x_std": self.x_std,
            "d_in": self.x_mean.shape[1],
        }, path)

    @staticmethod
    def load(path: Path, table_sizes: Dict[str, int],
             schema: BenchmarkSchema) -> "CardinalityEstimator":
        ckpt = torch.load(path, weights_only=False)
        model = _MLP(ckpt["d_in"])
        model.load_state_dict(ckpt["state_dict"])
        return CardinalityEstimator(
            model=model, x_mean=ckpt["x_mean"], x_std=ckpt["x_std"],
            table_sizes=table_sizes, schema=schema,
        )


def train_estimator(
    queries: List[Query],
    plan_sets: List[QueryPlans],
    train_qids: Sequence[int],
    table_sizes: Dict[str, int],
    schema: BenchmarkSchema,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 7,
) -> CardinalityEstimator:
    """Train an MSCN-style estimator on all intermediates of training-split plans."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_qid_set = set(train_qids)
    X_list, y_list = [], []
    for q, qp in zip(queries, plan_sets):
        if q.qid not in train_qid_set:
            continue
        for plan in qp.plans:
            for k in range(1, len(plan.order) + 1):
                tabs = plan.order[:k]
                sub_e = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                         if a in tabs and b in tabs]
                sub_p = [(t, p) for (t, p) in q.predicates if t in tabs]
                feat = featurize_subquery(
                    tabs, sub_e, sub_p, table_sizes,
                    schema.tables, schema.edge_keys,
                )
                X_list.append(feat)
                y_list.append(math.log(plan.intermediate_cards[k - 1] + 1.0))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  [train] {X.shape[0]:,} training intermediates, dim={X.shape[1]}")

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - x_mean) / x_std

    Xn_t = torch.tensor(Xn)
    y_t = torch.tensor(y).unsqueeze(1)

    model = _MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = Xn_t.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xn_t[idx], y_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        if (ep + 1) % 20 == 0:
            print(f"  [train] epoch {ep+1:3d}/{n_epochs}  MSE={total/n:.4f}")
    return CardinalityEstimator(
        model=model, x_mean=x_mean, x_std=x_std,
        table_sizes=table_sizes, schema=schema,
    )


def compute_plan_estimates(
    estimator: CardinalityEstimator,
    queries: List[Query],
    plan_sets: List[QueryPlans],
    cache_path: Path | None = None,
) -> Dict[int, List[List[float]]]:
    """For each query and each candidate plan, estimate cardinality of every intermediate.

    Returns a dict[qid] -> list_of_plans of list_of_intermediate_estimates.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    estimates: Dict[int, List[List[float]]] = {}
    for q, qp in zip(queries, plan_sets):
        per_plan = []
        for plan in qp.plans:
            ests = []
            for k in range(1, len(plan.order) + 1):
                tabs = plan.order[:k]
                sub_e = [(a, ca, b, cb) for (a, ca, b, cb) in q.edges
                         if a in tabs and b in tabs]
                sub_p = [(t, p) for (t, p) in q.predicates if t in tabs]
                ests.append(estimator.estimate(tabs, sub_e, sub_p))
            per_plan.append(ests)
        estimates[q.qid] = per_plan
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(estimates, f)
    return estimates
