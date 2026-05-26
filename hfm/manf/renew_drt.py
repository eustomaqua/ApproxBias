# coding: utf-8

from typing import (
    Dict, Iterable, Mapping, Optional, Tuple, Union, Any)
import math
import numpy as np
from numba import njit  # ,prange
# from scipy.spatial import distance
from hfm.utils.verifiers import INF64, EPS64
from hfm.utils.decorators import fantasy_timer

# def njit(*args, **kwargs):
#     if args and callable(args[0]):
#         return args[0]
#     return lambda f: f
#
# prange = range
import pdb

IndexLike = Union[np.ndarray, Iterable[int]]  # ,Iterable[list]]
ArrayLike = Union[np.ndarray, Iterable[float]]
PType = Union[int, float, str]
# CONST_INF = float("inf")
DTY_FLT = np.float64
DTY_INT = np.int64
STRATEGIES = ['Approx', 'StratES', 'StratRA']  # 'Vacant'


# ------------------------------------------
# Dual exponent and dual-norm normalisation


def _as_float_p(p: PType) -> float:
    """Convert common p specifications into float, including 'inf'."""
    if isinstance(p, str):
        if p.lower() in {"inf", "infty", "infinity", "linf", "l_inf",
                         "l-infinity"}:
            return float('inf')   # return CONST_INF
        p = float(p)
    else:
        p = float(p)
    if p < 1.0:
        raise ValueError("L_p distance requires p >= 1.")
    return p  # float(p)


@njit(cache=True)
def dual_exponent(p: PType) -> float:
    """
    Return the dual exponent q of p, where 1/p + 1/q = 1.

    Cases:
        p = 1      -> q = inf
        p = 2      -> q = 2
        p = inf    -> q = 1
        1 < p < inf -> q = p/(p-1)
    """
    # p = _as_float_p(p)
    # if p < 1.0:
    #     raise ValueError("L_p norm requires p >= 1.")
    if p == 1.0:
        return float("inf")
    if math.isinf(p):
        return 1.0
    return p / (p - 1.0)


@njit(cache=True)
def lp_norm_vec(x: ArrayLike, p: PType) -> float:
    """Compute ||x||_p for p in [1, inf)."""
    # p = _as_float_p(p)
    x = np.asarray(x, dtype=DTY_FLT)
    if x.size == 0:
        return 0.0

    ax = np.abs(x)  # .astype(DTY_FLT)
    # if p < 1.0:
    #     raise ValueError("L_p norm requires p >= 1.")
    if p == 1.0:
        return float(np.sum(ax))
    if p == 2.0:
        return float(np.sqrt(np.dot(x, x)))
        # return float(np.linalg.norm(x))
    if math.isinf(p):
        return float(np.max(ax))  # if x.size else 0.0
    return float(np.sum(ax ** p) ** (1.0 / p))


@njit(cache=True)
def dual_norm_vec(w: ArrayLike, p: PType) -> float:
    q = dual_exponent(p)
    return lp_norm_vec(w, q)


def dual_normalize(w: ArrayLike, p: PType, *, eps=EPS64,
                   target_norm: float = 1.0) -> float:
    if not (0.0 < target_norm <= 1.0 + eps):
        raise ValueError("target_norm should be in (0, 1] to keep "
                         "projections 1-Lipschitz.")
    out = np.asarray(w, dtype=DTY_FLT)
    nrm = dual_norm_vec(out, p)
    if nrm <= eps:
        raise ValueError("Cannot normalise a near-zero projection vector.")
    return (target_norm / nrm) * out


# ------------------------------------------
# Distance between sets


def _validate_inputs(X: np.ndarray, A: np.ndarray, *,
                     require_cross_group: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=DTY_FLT)
    A = np.asarray(A)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (#samples, #feat).")
    if A.ndim != 1:
        raise ValueError("A must be a 1D sen-att vec for sing-SA functions.")
    if X.shape[0] != A.shape[0]:
        raise ValueError("X and A must have the same number of rows.")
    if require_cross_group and len(np.unique(A)) < 2:
        raise ValueError("At least two sen-att values are required for "
                         "cross-group distances.")
    return X, A


@njit
def _aggregate_dmin(d_min: np.ndarray) -> Tuple[float, float]:
    """Return (max nearest cross-group distance, average nearest
     cross-group distance)."""
    d_min = np.asarray(d_min, dtype=DTY_FLT)
    if np.any(~np.isfinite(d_min)):
        raise ValueError(
            "Some points have no finite cross-group candidate dist.")
    return float(np.max(d_min)), float(np.mean(d_min))


@njit(cache=True)
def _lp_distance_rows(X: np.ndarray, i: int, j: int, p: float) -> float:
    """Lp distance between rows X[i] and X[j], numba-compatible."""
    d = X.shape[1]
    if p == 1.0:
        s = 0.0
        for k in range(d):
            s += abs(X[i, k] - X[j, k])
        return s
    if p == 2.0:
        s = 0.0
        for k in range(d):
            t = X[i, k] - X[j, k]
            s += t * t
        return math.sqrt(s)
    if math.isinf(p):
        mx = 0.0
        for k in range(d):
            t = abs(X[i, k] - X[j, k])
            if t > mx:
                mx = t
        return mx
    s = 0.0
    for k in range(d):
        s += abs(X[i, k] - X[j, k])**p
    return s ** (1.0 / p)


@njit(cache=True)
def _direct_dmin_sing_sa(X: np.ndarray, A: np.ndarray, p: float
                         ) -> np.ndarray:
    """Exact O(n^2) nearest cross-group distances for validation."""
    n = X.shape[0]
    out = np.empty(n, dtype=DTY_FLT)
    for i in range(n):
        best = INF64
        ai = A[i]
        for j in range(n):
            if A[j] == ai:
                continue
            curr = _lp_distance_rows(X, i, j, p)
            if curr < best:
                best = curr
        out[i] = best
    return out


# Direct_nonbin

@njit
def _Direct_halfway_min(X: np.ndarray, ele_i: int, Si_c: IndexLike,
                        p: float) -> float:
    d_min = INF64
    for ele_j in Si_c:
        curr = _lp_distance_rows(X, ele_i, ele_j, p)
        if curr < d_min:
            d_min = curr
    return d_min


@njit(cache=True)
def _Direct_mediator(X: np.ndarray, idx_Si: IndexLike, p: float
                     ):  # -> np.ndarray:
    # Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    Sj = np.where(idx_Si)[0]     # Sj = np.nonzero(idx_Si)[0]
    Sj_c = np.where(~idx_Si)[0]  # Sj_c = np.nonzero(~idx_Si)[0]
    # np.where(idx_Sj==False)

    # d_min = INF64
    # for ele_i in Sj:
    #     curr = _Direct_halfway_min(X, ele_i, Sj_c, p)
    #     if curr < d_min:
    #         d_min = curr
    # return d_min

    elements = [_Direct_halfway_min(X, ele_i, Sj_c, p) for ele_i in Sj]
    return max(elements), sum(elements)


def _idx_marginalised(A_i: IndexLike, priv_val: int = 1):
    vAi = np.unique(A_i).tolist()
    vAi.remove(priv_val)
    idx_Si = [A_i == priv_val]
    for j in vAi:
        idx_Si.append(A_i == j)
    return idx_Si


# ------------------------------------------
# Public distance-computation APIs


@fantasy_timer
def Direct_bin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
               priv_val: int = 1):
    p = _as_float_p(p)
    idx_Si = A_i == priv_val
    half_1, half_1avg = _Direct_mediator(X_nA_y, idx_Si, p)
    half_2, half_2avg = _Direct_mediator(X_nA_y, ~idx_Si, p)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


@fantasy_timer
def Direct_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
                  priv_val: int = 1):
    """Exact O(n^2) nearest cross-group distance computation"""
    p = _as_float_p(p)
    idx_Sjs = _idx_marginalised(A_i, priv_val)
    half_mid = [_Direct_mediator(X_nA_y, idx_Si, p) for idx_Si in idx_Sjs]
    half_pl_max, half_pl_avg = zip(*half_mid)
    n = len(X_nA_y)
    return max(half_pl_max), sum(half_pl_avg) / n


# @fantasy_timer
# def Direct_multivar(X_nA_y: np.ndarray, A: np.ndarray, p: PType = 2.0,
#                     priv_val: int = 1):
#     return


@fantasy_timer
def direct_sing_sa(X: np.ndarray, A: np.ndarray, *, p: PType = 2.0):
    """Exact O(n^2) nearest cross-group distance computation for one SA."""
    p = _as_float_p(p)
    X, A = _validate_inputs(X, A)
    d_min = _direct_dmin_sing_sa(X, A, p)
    d_max, d_avg = _aggregate_dmin(d_min)
    return d_max, d_avg


# @fantasy_timer
# def direct_multivar(X: np.ndarray, A: np.ndarray, *, p: PType = 2.0):
#     n_a = A.shape[1]
#     half_mid = [direct_sing_sa(X, A[:, i], p=p) for i in range(n_a)]
#     half_mid, half_ut = zip(*half_mid)
#     half_pl_max, half_pl_avg = zip(*half_mid)
#     return max(half_pl_max), sum(half_pl_avg) / n_a, (
#         half_pl_max, half_pl_avg, half_ut)


# ------------------------------------------
