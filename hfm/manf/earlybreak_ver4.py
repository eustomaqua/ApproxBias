# coding: utf-8
#           EarlyBreak extensions
#
# Reference:
# Taha AA, Hanbury A. An efficient algorithm for calculating the
# exact Hausdorff distance. IEEE transactions on pattern analysis
# and machine intelligence. 2015 Mar 3;37(11):2153-63.
#


import numpy as np
from numba import njit
from hfm.utils.decorators import fantasy_timer_prime

# from hfm.manf.earlybreak_ver1 import HDD_randomize
from hfm.manf.renew_core import ( 
    ArrayLike, PType, IndexLike, INF64, _lp_distance_rows,
    _as_float_p)  # lp_norm_vec,, _idx_marginalised


# renew_eff.py
# ==========================================
# An efficient algorithm for calculating the exact Hausdorff distance


# Algorithm 1. NAIVEHDD
# Straightfowardly computes the directed Hausdorff distance


# @njit(cache=True)
# def NaiveHDD(A: ArrayLike, B: ArrayLike, p: PType) -> float:
#     cmax = 0.0
#     for ele_x in A:
#         cmin = INF64
#         for ele_y in B:
#             d = lp_norm_vec(ele_x - ele_y, p)
#             if d < cmin:
#                 cmin = d
#         if cmin > cmax:
#             cmax = cmin
#     return cmax


@njit(cache=True)
def NaiveHDD(X: np.ndarray, iA: IndexLike, iB: IndexLike,
             p: float) -> float:
    cmax = 0.0
    for ele_i in iA:       # idx_A
        cmin = INF64
        for ele_ic in iB:  # idx_B
            d = _lp_distance_rows(X, ele_i, ele_ic, p)
            if d < cmin:
                cmin = d
        if cmin > cmax:
            cmax = cmin
    return cmax


# Early Breaking

# Algorithm 3. RANDOMIZE
# Finds a random order of a given point set

@njit(cache=True)
def HDD_randomize(S):
    m = len(S)
    ind = np.arange(m)
    for p in range(m):
        # 在 [0,m) 中随机选一个 q
        q = np.random.randint(0, m)
        q = ind[q]
        if q == p:
            continue
        # 交换 ind[p], ind[q]
        tmp = ind[p]
        ind[p] = ind[q]
        ind[q] = tmp
    return ind


# Algorithm 2. EARLYBREAK
# Computes the directed HDD using the Early Break technique and
# the Random Sampling

# @njit(cache=True)
# def HDD_earlybreak(A: ArrayLike, B: ArrayLike, p: PType) -> float:
#     cmax = 0
#     Er_ind = HDD_randomize(A)
#     Br_ind = HDD_randomize(B)
#     Er = A[Er_ind]
#     Br = B[Br_ind]
#
#     for ele_x in Er:
#         cmin = INF64
#         for ele_y in Br:
#             d = lp_norm_vec(ele_x - ele_y, p)
#             if d < cmin:
#                 cmin = d
#             if d < cmax:
#                 break
#
#         if cmin > cmax:
#             cmax = cmin
#     return cmax


@njit(cache=True)
def HDD_earlybreak(X: np.ndarray, iA: IndexLike, iB: IndexLike,
                   p: float) -> float:
    cmax = 0.0
    Er = iA[HDD_randomize(iA)]
    Br = iB[HDD_randomize(iB)]
    for ele_i in Er:
        cmin = INF64
        for ele_ic in Br:
            d = _lp_distance_rows(X, ele_i, ele_ic, p)
            if d < cmin:
                cmin = d
            if d < cmax:
                break
        if cmin > cmax:
            cmax = cmin
    return cmax


# ==========================================
# NaiveHDD & EarlyBreak (aka. EffHDD)


# @fantasy_timer
# def EffHD_bin(X_nA_y: ArrayLike, idx_Si: IndexLike, p: PType
#               ) -> float:
#     Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
#     half_1 = HDD_earlybreak(Sj, Sj_c, p)
#     half_2 = HDD_earlybreak(Sj_c, Sj, p)
#     return max(half_1, half_2)
#
#
# @fantasy_timer
# def EffHD_nonbin(X_nA_y: ArrayLike, idx_Sjs, p: PType) ->float:
#     cmax = 0
#     for idx_Si in idx_Sjs:
#         Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
#         half_1 = HDD_earlybreak(Sj, Sj_c, p)
#         if half_1 > cmax:
#             cmax = half_1
#     return cmax


@fantasy_timer_prime
def EffHD_bin(X_nA_y: ArrayLike, Ai: IndexLike, p: PType = 2.0,
              priv_val: int = 1) -> float:
    p = _as_float_p(p)
    Sj = np.where(Ai == priv_val)[0]
    Sj_c = np.where(Ai != priv_val)[0]
    half_1 = HDD_earlybreak(X_nA_y, Sj, Sj_c, p)
    half_2 = HDD_earlybreak(X_nA_y, Sj_c, Sj, p)
    return max(half_1, half_2)


@fantasy_timer_prime
def EffHD_nonbin(X_nA_y: ArrayLike, Ai: IndexLike, p: PType = 2.0,
                 priv_val: int = 1) -> float:
    p = _as_float_p(p)
    vAi = np.unique(Ai).tolist()
    vAi.remove(priv_val)
    vAi = [priv_val, ] + vAi

    cmax = 0.0
    for ai in vAi:
        Sj = np.where(Ai == ai)[0]
        Sj_c = np.where(Ai != ai)[0]
        half_1 = HDD_earlybreak(X_nA_y, Sj, Sj_c, p)
        if half_1 > cmax:
            cmax = half_1
    return cmax


# @fantasy_timer
# def Naive_bin(X_nA_y: ArrayLike, idx_Si: IndexLike, p: PType
#               ) -> float:
#     Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
#     half_1 = NaiveHDD(Sj, Sj_c, p)
#     half_2 = NaiveHDD(Sj_c, Sj, p)
#     return max(half_1, half_2)
#
#
# @fantasy_timer
# def Naive_nonbin(X_nA_y: ArrayLike, idx_Sjs, p: PType) -> float:
#     cmax = 0.0
#     for idx_Si in idx_Sjs:
#         Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
#         half_1 = NaiveHDD(Sj, Sj_c, p)
#         if half_1 > cmax:
#             cmax = half_1
#     return cmax


@fantasy_timer_prime
def Naive_bin(X_nA_y: ArrayLike, Ai: IndexLike, p: PType = 2.0,
              priv_val: int = 1) -> float:
    p = _as_float_p(p)
    Sj = np.nonzero(Ai == priv_val)[0]
    Sj_c = np.nonzero(Ai != priv_val)[0]
    # idx_Si = A_i == priv_val
    half_1 = NaiveHDD(X_nA_y, Sj, Sj_c, p)
    half_2 = NaiveHDD(X_nA_y, Sj_c, Sj, p)
    return max(half_1, half_2)


@fantasy_timer_prime
def Naive_nonbin(X_nA_y: ArrayLike, Ai: IndexLike, p: PType = 2.0,
                 priv_val: int = 1) -> float:
    p = _as_float_p(p)
    # idx_Sjs = _idx_marginalised(A_i, priv_val)
    vAi = np.unique(Ai).tolist()
    vAi.remove(priv_val)
    vAi = [priv_val, ] + vAi

    cmax = 0.0
    for ai in vAi:
        Sj = np.where(Ai == ai)[0]
        Sj_c = np.where(Ai != ai)[0]
        half_1 = NaiveHDD(X_nA_y, Sj, Sj_c, p)
        if half_1 > cmax:
            cmax = half_1
    return cmax
