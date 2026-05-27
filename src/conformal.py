"""
Conformal calibration methods, side by side.

Implements all six methods compared in the paper:
  T1   split CP                              (Section IV.A, Prop. 1)
  T1+  MUB-by-Size (size-stratified split CP)  (Section VI)
  T2   locally-weighted split CP             (Section IV.B)
  T3   conformalized quantile regression     (Section IV.B)
  M1   Plan-Regret CP                        (Section V)

Each method exposes:
  - a calibration function that consumes an intermediate-level table and
    returns the calibrated quantile(s)
  - the calibrated objects are then consumed by the pickers in src/pickers.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


# =============================================================
# Generic split-conformal quantile
# =============================================================
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-CP finite-sample-corrected quantile."""
    n = len(scores)
    if n == 0:
        return float('inf')
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


# =============================================================
# Intermediate-level data shaping for cardinality CP methods
# =============================================================
@dataclass
class IntermediateTable:
    """Flat table over (query, plan, intermediate-position) triples."""
    qid_arr: np.ndarray
    plan_idx_arr: np.ndarray
    k_arr: np.ndarray          # number of tables in this intermediate
    feat_arr: np.ndarray       # (M, d) feature vectors
    est_arr: np.ndarray        # base-estimator cardinality estimates
    true_arr: np.ndarray       # true intermediate cardinalities


# =============================================================
# T1: split CP, global
# =============================================================
def calibrate_split_cp(tab: IntermediateTable, alphas: Sequence[float]) -> Dict[float, float]:
    scores = np.abs(np.log(tab.est_arr + 1.0) - np.log(tab.true_arr + 1.0))
    return {a: conformal_quantile(scores, a) for a in alphas}


# =============================================================
# T1+: MUB-by-Size (split CP stratified by #tables in intermediate)
# =============================================================
def calibrate_split_cp_by_size(
    tab: IntermediateTable, alphas: Sequence[float]
) -> Dict[float, Dict[int, float]]:
    qs: Dict[float, Dict[int, float]] = {a: {} for a in alphas}
    sizes = np.unique(tab.k_arr)
    for s in sizes:
        mask = tab.k_arr == s
        sc = np.abs(np.log(tab.est_arr[mask] + 1.0)
                    - np.log(tab.true_arr[mask] + 1.0))
        for a in alphas:
            qs[a][int(s)] = conformal_quantile(sc, a)
    return qs


# =============================================================
# T2: locally-weighted split CP. Uses xgboost to predict residual magnitude.
# =============================================================
def fit_difficulty_xgb(train_tab: IntermediateTable, seed: int = 7):
    """Train xgboost to predict |log_residual| from intermediate features."""
    import xgboost as xgb
    X = train_tab.feat_arr
    res = np.abs(np.log(train_tab.est_arr + 1.0)
                 - np.log(train_tab.true_arr + 1.0))
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=seed,
        verbosity=0, n_jobs=4,
    )
    model.fit(X, res)
    return model


def calibrate_lw_split_cp(
    calib_tab: IntermediateTable, g_hat, alphas: Sequence[float],
    floor: float = 0.05,
) -> Dict[float, float]:
    u = np.maximum(floor, g_hat.predict(calib_tab.feat_arr))
    raw = np.abs(np.log(calib_tab.est_arr + 1.0)
                 - np.log(calib_tab.true_arr + 1.0))
    sc = raw / u
    return {a: conformal_quantile(sc, a) for a in alphas}


# =============================================================
# T3: Conformalized Quantile Regression.
# Two-headed neural net trained with pinball loss at alpha/2 and 1-alpha/2.
# alpha-specific by construction; train one model per alpha.
# =============================================================
def _pinball_loss(pred, target, q):
    diff = target - pred
    return ((q - 1.0) * diff).clamp(min=0) + (q * diff).clamp(min=0)


class _CQRTwoHead(nn.Module):
    def __init__(self, d_in: int, d_h: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
        )
        self.head_lo = nn.Linear(d_h, 1)
        self.head_hi = nn.Linear(d_h, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_lo(h), self.head_hi(h)


def fit_cqr(train_tab: IntermediateTable, alpha: float,
            d_h: int = 256, n_epochs: int = 80, batch_size: int = 256,
            lr: float = 1e-3, seed: int = 7, verbose: bool = True):
    """Train a two-headed quantile network at heads alpha/2 and 1-alpha/2."""
    torch.manual_seed(seed)
    X = train_tab.feat_arr
    y = np.log(train_tab.true_arr + 1.0).astype(np.float32)
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - x_mean) / x_std

    Xn_t = torch.tensor(Xn)
    y_t = torch.tensor(y).unsqueeze(1)
    n = Xn_t.shape[0]
    model = _CQRTwoHead(X.shape[1], d_h)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

    for ep in range(n_epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xn_t[idx], y_t[idx]
            lo, hi = model(xb)
            loss = (_pinball_loss(lo, yb, q_lo).mean()
                    + _pinball_loss(hi, yb, q_hi).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        if verbose and (ep + 1) % 20 == 0:
            print(f"  [cqr alpha={alpha}] epoch {ep+1:3d}/{n_epochs}  "
                  f"pinball={total/n:.4f}")
    return model.eval(), x_mean, x_std


def calibrate_cqr(calib_tab: IntermediateTable,
                  cqr_model, x_mean: np.ndarray, x_std: np.ndarray,
                  alpha: float) -> float:
    X = calib_tab.feat_arr
    y = np.log(calib_tab.true_arr + 1.0).astype(np.float32)
    Xn = (X - x_mean) / x_std
    with torch.no_grad():
        lo, hi = cqr_model(torch.tensor(Xn))
        lo = lo.numpy().flatten()
        hi = hi.numpy().flatten()
    sc = np.maximum(lo - y, y - hi)
    return conformal_quantile(sc, alpha)


# =============================================================
# M1: Plan-Regret CP (this paper)
# =============================================================
def calibrate_plan_regret(
    plan_sets, plan_cost_estimates, calib_qids, alphas: Sequence[float],
) -> Dict[float, float]:
    """Calibrate the log-ratio of chosen-plan cost to oracle-plan cost.

    `plan_sets[qid]` exposes `.plans[i].cost` (true) and `.oracle_idx`.
    `plan_cost_estimates[qid]` is the per-plan estimated intermediate list
    (used to identify the point-estimate plan picker's choice).
    """
    scores = []
    for qid in calib_qids:
        qp = plan_sets[qid]
        est_costs = [sum(plan_cost_estimates[qid][i])
                     for i in range(len(qp.plans))]
        chosen_true = qp.plans[int(np.argmin(est_costs))].cost
        oracle = qp.plans[qp.oracle_idx].cost
        scores.append(math.log((chosen_true + 1.0) / (oracle + 1.0)))
    arr = np.array(scores)
    return {a: conformal_quantile(arr, a) for a in alphas}
