# coding: utf-8


# import pdb
from typing import (
    Dict, Iterable, Mapping, Optional, Tuple, Union, Any)
import math
import numpy as np
from numba import njit


INF64 = np.float64(1e308)
EPS64 = 1e-12  # np.float64(1e-12)
# INF64 = np.float64(np.inf)
STRATEGIES = ['Approx', 'StratES', 'StratRA']  # 'Vacant'


IndexLike = Union[np.ndarray, Iterable[int]]  # ,Iterable[list]]
ArrayLike = Union[np.ndarray, Iterable[float]]
PType = Union[int, float, str]
DTY_FLT = np.float64
DTY_INT = np.int64

# CONST_INF = float("inf")
hfmOUTCOME = Tuple[float, float]


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


@njit(inline='always', cache=True)
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


@njit(inline='always', cache=True, fastmath=True)
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
        # return s  # 平方距离，不sqrt
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
    # return s     # p次方和，不开1/p次根


# ------------------------------------------
# Approximation algorithm(s)


# @njit(cache=True, fastmath=True)
# def orthogonal_weight_prime(n_d, n_e=3) -> np.ndarray:
#     B = np.random.rand(n_d, n_d)
#     Q, _ = np.linalg.qr(B)
#     return np.ascontiguousarray(Q[:, :n_e].T)


@njit(cache=True)
def orthogonal_weight(n_d, n_e=3) -> np.ndarray:
    for _ in range(n_d):
        B = np.random.rand(n_d, n_d)
        tmp = np.linalg.det(B)
        if abs(tmp) > EPS64:
            break  # 数值稳定判断
    # 2. Gram-Schmidt (向量化版本)
    A_T = B.T
    eta = np.zeros((n_e, n_d))
    # 第一个向量归一化
    v = A_T[0]
    eta[0] = v / np.linalg.norm(v)
    # 后续向量
    for i in range(1, n_e):
        v = A_T[i].copy()
        # 投影部分向量化: proj=(eta[:i] @v)
        proj = eta[:i] @ v
        v = v - proj @ eta[:i]
        # 归一化
        v = v / np.linalg.norm(v)
        eta[i] = v
    return eta


# @njit(cache=True, inline='always')
def _determine_m2(n: int, m2: Optional[int] = None) -> int:
    if m2 is None:
        m2 = math.ceil(2.0 * math.log10(n))
        m2 = max(1, int(m2))
    # m2 = math.ceil(2.0 * math.log10(X_nA_y.shape[0]))
    return m2

# def _determine_m2(n: int, m2 = None) -> int:
#     if m2 is None:
#         m2 = math.ceil(2.0 * math.log10(n))
#         m2 = max(1, int(m2))
#     return int(m2)


# ------------------------------------------
#
