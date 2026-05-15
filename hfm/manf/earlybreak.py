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
from hfm.manf.dist_internal import alter_intermediate, name_intermediate
from hfm.utils.decorators import fantasy_timer
from hfm.utils.verifiers import INF64


# ==========================================
# An efficient algorithm for calculating the exact Hausdorff distance


# Algorithm 1. NAIVEHDD
# Straightfowardly computes the directed Hausdorff distance

@njit
def NaiveHDD(A, B, func, p):
    cmax = 0.
    for ele_x in A:
        cmin = INF64  # float(np.finfo(np.float32).max)
        for ele_y in B:
            d = alter_intermediate(ele_x, ele_y, func, p)
            if d < cmin:
                cmin = d
        if cmin > cmax:
            cmax = cmin
    return cmax


# def NaiveHDD(A, B, func ='euclidean', p=3):
#     func_id = name_intermediate.index(func)
#     return NaiveHDD_core(A, B, func_id, p)


# Early Breaking

# Algorithm 3. RANDOMIZE
# Finds a random order of a given point set

# def HDD_randomize(S):
#     m = len(S)
#     ind = list(range(m))
#     for p in range(m):
#         q = np.random.choice(ind)
#         if q == p:
#             continue
#         tmp = ind[p]
#         ind[p] = ind[q]
#         ind[q] = tmp
#     return ind


@njit
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

@njit
def HDD_earlybreak(A, B, func, p):
    cmax = 0
    Er_ind = HDD_randomize(A)
    Br_ind = HDD_randomize(B)
    Er = A[Er_ind]
    Br = B[Br_ind]

    for ele_x in Er:
        cmin = INF64  # float(np.finfo(np.float32).max)
        for ele_y in Br:
            d = alter_intermediate(ele_x, ele_y, func, p)
            if d < cmin:
                cmin = d
            if d < cmax:
                break

        if cmin > cmax:
            cmax = cmin
    return cmax


# ==========================================
# NaiveHDD & EarlyBreak (aka. EffHDD)


@fantasy_timer
def EffHD_bin(X_nA_y, idx_Si, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    half_1 = HDD_earlybreak(Sj, Sj_c, func_id, p)
    half_2 = HDD_earlybreak(Sj_c, Sj, func_id, p)
    return max(half_1, half_2)


@fantasy_timer
def EffHD_nonbin(X_nA_y, idx_Sjs, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    cmax = 0
    for idx_Si in idx_Sjs:
        Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
        half_1 = HDD_earlybreak(Sj, Sj_c, func_id, p)
        if half_1 > cmax:
            cmax = half_1
    return cmax


@fantasy_timer
def EffHD_multiver(X_nA_y, idx_Ai_Sj, func='euclidean', p=3):
    half_mid = [EffHD_nonbin(
        X_nA_y, idx_Sjs, func, p) for idx_Sjs in idx_Ai_Sj]
    half_mid, half_ut = zip(*half_mid)
    return max(half_mid), (half_mid, half_ut)


@fantasy_timer
def Naive_bin(X_nA_y, idx_Si, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    half_1 = NaiveHDD(Sj, Sj_c, func_id, p)
    half_2 = NaiveHDD(Sj_c, Sj, func_id, p)
    return max(half_1, half_2)


@fantasy_timer
def Naive_nonbin(X_nA_y, idx_Sjs, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    cmax = 0
    for idx_Si in idx_Sjs:
        Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
        half_1 = NaiveHDD(Sj, Sj_c, func_id, p)
        if half_1 > cmax:
            cmax = half_1
    return cmax


@fantasy_timer
def Naive_multiver(X_nA_y, idx_Ai_Sj, func='euclidean', p=3):
    half_mid = [Naive_nonbin(
        X_nA_y, idx_Sjs, func, p) for idx_Sjs in idx_Ai_Sj]
    half_mid, half_ut = zip(*half_mid)
    return max(half_mid), (half_mid, half_ut)
