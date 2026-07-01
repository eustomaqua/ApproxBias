# coding: utf-8


from typing import Optional  # ,Tuple
import math
import numpy as np
from numba import njit, prange
import pdb

# from hfm.utils.verifiers import INF64, EPS64
from hfm.utils.decorators import fantasy_timer
# from hfm.manf.renew_drt import (
#     ArrayLike, PType, IndexLike, DTY_FLT, DTY_INT,
#     _as_float_p, _lp_distance_rows, _aggregate_dmin,
#     dual_norm_vec)

from hfm.manf.renew_core import INF64, DTY_FLT
from hfm.manf.renew_core import ArrayLike, PType, IndexLike
from hfm.manf.renew_core import (
    _as_float_p, _lp_distance_rows, _aggregate_dmin,
    orthogonal_weight, hfmOUTCOME)


# ------------------------------------------
# Approx_bin


@njit(inline='always', cache=True, fastmath=True)
def _sub_accelerator_dir(X_yddot: np.ndarray, Ai_order: IndexLike,
                         p: PType, order: np.ndarray, pos: int,
                         m2: int, direction: int) -> float:
    anchor = order[pos]
    ak = Ai_order[pos]  # ak = A_i[anchor]
    #                   # yxdot = X_yfx[anchor]

    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()?=g(xi,yi;w)
    count, best = 0, INF64
    n = X_yddot.shape[0]  # number of instances
    jp = pos + direction  # doesn't have to be compared with anchor
    while (count < m2) and (0 <= jp < n):
        j = order[jp]  # j_prev = order[j]

        # if ak == A_i[j]:  # set_belonging
        #     jp += direction
        #     continue
        # curr = _lp_distance_rows(X_yddot, anchor, j, p)
        # if curr < best:
        #     best = curr
        # count += 1
        # jp += direction

        # if A_i[j] != ak:  # set_belonging?
        if Ai_order[jp] != ak:
            curr = _lp_distance_rows(X_yddot, anchor, j, p)
            if curr < best:
                best = curr
            count += 1
        jp += direction
    return best


# @njit
# def projector(element, vec_w):
#     ans = np.dot(element, vec_w)
#     return float(ans)


@njit(cache=True, fastmath=True)
def AcceleCore_bin(X_yddot: np.ndarray, B_i: IndexLike, p: PType,
                   vec_w: np.ndarray, m2: int) -> hfmOUTCOME:
    # Project data points onto a one-dimensional space
    proj = X_yddot @ vec_w  # =[projector(ele, vec_w) for ele in X_yddot]
    order = np.argsort(proj)
    Ai_order = B_i[order]
    n = X_yddot.shape[0]  # number of instances
    # d_min = np.empty(n, dtype=DTY_FLT)
    d_max = d_sum = 0.0
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round
        # min_js = _sub_accelerator_dir(X_yddot, B_i, p, order, i, m2, -1)
        # min_jr = _sub_accelerator_dir(X_yddot, B_i, p, order, i, m2, +1)

        min_js = _sub_accelerator_dir(X_yddot, Ai_order, p, order, i, m2, -1)
        min_jr = _sub_accelerator_dir(X_yddot, Ai_order, p, order, i, m2, +1)
        # d_min[i] = min(min_js, min_jr)  # finally
        best = min_js if min_js < min_jr else min_jr
        if best > d_max:
            d_max = best
        d_sum += best
    # return d_min.max(), d_min.sum()
    # return _aggregate_dmin(d_min)
    return d_max, d_sum


@njit(inline='always', cache=True)
def rand_uniform(a, b):
    # vec_w[i] = np.random.uniform(-tmp, tmp)
    return a + (b - a) * np.random.random()


@njit(cache=True)
def weight_generator(n_d):
    vec_w = np.empty(n_d, dtype=DTY_FLT)
    tmp = 1.0
    for i in range(n_d - 1):
        vec_w[i] = rand_uniform(-tmp, tmp)
        tmp -= abs(vec_w[i])
    vec_w[n_d - 1] = tmp

    # for i in range(n_d - 1):
    #     val = rand_uniform(-tmp, tmp)
    #     vec_w[i] = val
    #     tmp -= math.fabs(val)
    # ss = 0.0  # compute last weight
    # for i in range(n_d - 1):
    #     ss += math.fabs(vec_w[i])
    # pdb.set_trace()
    # vec_w[n_d - 1] = 1. - ss
    return vec_w


@njit(parallel=True, cache=True)
def _Approx_bin_sub(X_nA_y: np.ndarray, B_i: IndexLike, p: PType,
                    m1: int, m2: int) -> hfmOUTCOME:
    n, n_d = X_nA_y.shape  # <class 'int'>
    d_max = np.empty(m1, dtype=DTY_FLT)
    d_avg = np.empty(m1, dtype=DTY_FLT)
    for k in prange(m1):
        vec_w = weight_generator(n_d)  # weights[k] # n_d-1:#non-sa
        # tmp = AcceleCore_bin(X_nA_y, B_i, p, vec_w, m2)
        # d_max[k] = tmp[0]
        # d_avg[k] = tmp[1]

        mx, sm = AcceleCore_bin(X_nA_y, B_i, p, vec_w, m2)
        d_max[k] = mx
        d_avg[k] = sm
    return d_max.min(), d_avg.min() / n


@fantasy_timer
def Approx_bin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
               m1: int = 25, m2: int = 11) -> hfmOUTCOME:
    p = _as_float_p(p)
    tmp = _Approx_bin_sub(X_nA_y, A_i, p, m1, m2)
    return list(map(float, tmp))
    # mx, sm = _Approx_bin_sub(X_nA_y, A_i, p, m1, m2)
    # return float(mx), float(sm)


# ------------------------------------------
# Approx_nonbin


@njit(parallel=True, cache=True)
def _Approx_nonbin_sub(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                       m1: int, m2: int, n_e: int) -> hfmOUTCOME:
    n, n_d = X_nA_y.shape  # n_d-1: number of non-sen-att(s)
    d_max = np.empty(m1, dtype=DTY_FLT)
    d_avg = np.empty(m1, dtype=DTY_FLT)
    for k in prange(m1):
        # Take two orthogonal vectors $w_0$ and $w_1$ where each $w_k
        # \in [-1,+1]^{1+n_x} (k=\{0,1\})$
        # Or take three orthogonal vectors. Your choice.
        W = orthogonal_weight(n_d, n_e)

        t_max = np.empty(n_e, dtype=DTY_FLT)
        t_avg = np.empty(n_e, dtype=DTY_FLT)
        for j in range(n_e):
            mx, sm = AcceleCore_bin(X_nA_y, A_i, p, W[j], m2)
            t_max[j] = mx
            t_avg[j] = sm
        d_max[k] = t_max.min()
        d_avg[k] = t_avg.min()
    return d_max.min(), d_avg.min() / n


@fantasy_timer
def Approx_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
                  m1: int = 20, m2: int = 8, n_e: int = 2) -> hfmOUTCOME:
    p = _as_float_p(p)
    tmp = _Approx_nonbin_sub(X_nA_y, A_i, p, m1, m2, n_e)
    return list(map(float, tmp))

# @fantasy_timer
# def Approx_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0, *,
#                   m1: int = 25, m2: Optional[int] = None, n_e: int = 2):
#     p = _as_float_p(p)
#     return


# ------------------------------------------
