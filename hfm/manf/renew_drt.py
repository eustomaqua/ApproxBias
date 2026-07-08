# coding: utf-8


# from typing import Tuple
import math
import numpy as np
from numba import njit  # ,prange
# from scipy.spatial import distance
# from hfm.utils.verifiers import INF64, EPS64
from hfm.utils.decorators import fantasy_timer_prime

# def njit(*args, **kwargs):
#     if args and callable(args[0]):
#         return args[0]
#     return lambda f: f
#
# prange = range

from hfm.manf.renew_core import INF64, EPS64, DTY_FLT
from hfm.manf.renew_core import ArrayLike, PType, IndexLike
from hfm.manf.renew_core import (
    _lp_distance_rows, _as_float_p, _aggregate_dmin,
    _validate_inputs, hfmOUTCOME)  # ,_idx_marginalised)


# ------------------------------------------
# Distance between sets


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
                     ) -> hfmOUTCOME:  # -> np.ndarray:
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


def _idx_marginalised(A_i: IndexLike, priv_val: int = 1) -> list:
    vAi = np.unique(A_i).tolist()
    vAi.remove(priv_val)
    idx_Si = [A_i == priv_val]
    for j in vAi:
        idx_Si.append(A_i == j)
    return idx_Si


# ------------------------------------------
# Public distance-computation APIs


@fantasy_timer_prime
def Direct_bin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
               priv_val: int = 1) -> hfmOUTCOME:
    p = _as_float_p(p)
    idx_Si = A_i == priv_val
    half_1, half_1avg = _Direct_mediator(X_nA_y, idx_Si, p)
    half_2, half_2avg = _Direct_mediator(X_nA_y, ~idx_Si, p)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


@fantasy_timer_prime
def Direct_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
                  priv_val: int = 1) -> hfmOUTCOME:
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


@fantasy_timer_prime
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
