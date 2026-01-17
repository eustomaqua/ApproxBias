# coding: utf-8
# Usage: to approximate the distance quickly
#
# Author: Yj
# 2. Approximating Discrimination Within Models When Faced With Several
#    Non-Binary Sensitive Attributes [https://arxiv.org/abs/2408.06099]
#


import numpy as np
import numba
# import pdb

from hfm.utils.decorators import fantasy_timer
from hfm.dist_drt import DistDirect_Euclidean
from hfm.dist_est_bin import projector
from hfm.dist_est_nonbin import (
    set_belonging, orthogonal_weight, AcceleCore,
    AcceleCoreBack)

# from hfm.dist_est_nonbin import ApproxDist_nonbin as blank
# https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
from numba.typed import List


# ==========================================
# Estimated distance between sets
# (Converged version)


# # @numba.njit(inline='always')
# @numba.jit(nopython=True)
# def sep_helper(idx_y_fx, Ai, X_yfx, proj, i):
#     i_anchor = idx_y_fx[i]
#     A_anchor = Ai[i_anchor]
#     X_yfx_anchor = X_yfx[i_anchor]
#     g_anchor = proj[i_anchor]
#     return i_anchor, A_anchor, X_yfx_anchor, g_anchor


# ------------------------------------------
# ------------------------------------------
# Algorithm 3. Sub-routines


# @numba.jit(nopython=True)
# def separate_accel_smaler(X_yfx, proj, Ai, idx_y_fx, i):
#     i_anch, A_anch, X_yfx_anch, g_anch = sep_helper(
#         idx_y_fx, Ai, X_yfx, proj, i)
#
#     # Compute the distances d(anchor,\cdot) for at most m2 nearby
#     # data points that meets a!=ai and g()<=g(xi,yi;w)
#     j, num_j, min_js = i, 0, np.finfo(np.float32).max
#     j = i - 1  # doesn't have to be compared with the anchor
#
#     # while num_j < m2:
#     #     if j < 0:
#     #         break
#     #
#     #     idx_j = idx_y_fx[j]
#     #     if set_belonging(A_anchor, Ai[idx_j]):
#     #         j -= 1
#     #         continue
#     #
#     #     curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
#     #     if curr < min_js:
#     #         min_js = curr
#     #     # Find the minimum among them, recorded as d_min^s
#     #
#     #     num_j += 1
#     #     j -= 1
#
#     # curr_best = np.finfo(np.float32).max
#     p_best = float(min_js)  # curr_best
#     while j >= 0:
#         idx_j = idx_y_fx[j]
#         if set_belonging(A_anch, Ai[idx_j]):
#             j -= 1
#             continue
#
#         curr_p = abs(g_anch - proj[idx_j])
#         if curr_p >= p_best:
#             break
#         p_best = curr_p
#         curr = DistDirect_Euclidean(X_yfx_anch, X_yfx[idx_j])
#         if curr < min_js:
#             min_js = curr
#         num_j += 1
#         j -= 1
#     return min_js, num_j


@numba.jit(nopython=True)
def sub_accelerator_smaler(X_yfx, proj, Ai, idx_y_fx, i):
    # i_anch, A_anch, X_yfx_anch, g_anch = sep_helper(
    #     idx_y_fx, Ai, X_yfx, proj, i)
    i_anch = idx_y_fx[i]
    A_anch = Ai[i_anch]
    X_yfx_anch = X_yfx[i_anch]
    g_anch = proj[i_anch]

    j, num_j, min_js = i, 0, np.finfo(np.float32).max
    j = i - 1  # doesn't have to be compared with the anchor
    while j >= 0:
        idx_j = idx_y_fx[j]
        if set_belonging(A_anch, Ai[idx_j]):
            j -= 1
            continue

        if abs(g_anch - proj[idx_j]) >= min_js:
            break
        curr = DistDirect_Euclidean(X_yfx_anch, X_yfx[idx_j])
        if curr < min_js:
            min_js = curr
        num_j += 1
        j -= 1
    return min_js, num_j


@numba.jit(nopython=True)
def sub_accelerator_larger(X_yfx, proj, Ai, idx_y_fx, i):
    # i_anchor = idx_y_fx[i]
    # A_anchor = Ai[i_anchor]
    # X_yfx_anchor = X_yfx[i_anchor]
    # g_anchor = proj[i_anchor]
    # i_anch, A_anch, X_yfx_anch, g_anch = sep_helper(
    #     idx_y_fx, Ai, X_yfx, proj, i)
    i_anch = idx_y_fx[i]
    A_anch = Ai[i_anch]
    X_yfx_anch = X_yfx[i_anch]
    g_anch = proj[i_anch]

    j, num_j, min_jr = i, 0, np.finfo(np.float32).max
    j = i + 1  # doesn't have to be compared with the anchor
    n = len(X_yfx)
    while j < n:
        idx_j = idx_y_fx[j]
        if set_belonging(A_anch, Ai[idx_j]):
            j += 1
            continue

        if abs(proj[idx_j] - g_anch) >= min_jr:
            break
        curr = DistDirect_Euclidean(X_yfx_anch, X_yfx[idx_j])
        if curr < min_jr:
            min_jr = curr
        num_j += 1
        j += 1
    return min_jr, num_j


# Algorithm 3. AcceleDist

@fantasy_timer
def AcceleDist_nonbin(X_nA_y, A_j, vec_w):
    # def AcceleDist_nonbin(X_nA_y, A_j, m2, vec_w):
    proj = List()
    [proj.append(projector(ele, vec_w)) for ele in X_nA_y]

    # proj = [projector(ele, vec_w) for ele in X_nA_y]
    idx_y_fx = np.argsort(proj)

    n = X_nA_y.shape[0]  # number of instances
    d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round

        min_js = sub_accelerator_smaler(
            X_nA_y, proj, A_j, idx_y_fx, i)
        min_jr = sub_accelerator_larger(
            X_nA_y, proj, A_j, idx_y_fx, i)

        # finally,
        tmp = min(min_js[0], min_jr[0])
        d_min.append(tmp)
    return max(d_min), sum(d_min)


"""
@fantasy_timer
def AcceleDist_nonbin(X_nA_y, A_j, vec_w):
    n = X_nA_y.shape[0]  # number of instances
    # proj = np.empty(n, dtype='float')
    # for i, ele in enumerate(X_nA_y):
    #     proj[i] = projector(ele, vec_w)
    proj = [projector(ele, vec_w) for ele in X_nA_y]
    idx_y_fx = np.argsort(proj)

    d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round
        # min_js = sub_accelerator_smaler(
        #     X_nA_y, proj, A_j, idx_y_fx, i)
        # min_jr = sub_accelerator_larger(
        #     X_nA_y, proj, A_j, idx_y_fx, i)

        # main loop
        i_anch = idx_y_fx[i]
        A_anch = A_j[i_anch]
        X_yfx_anch = X_nA_y[i_anch]
        g_anch = proj[i_anch]
        best = np.inf
        l, r = i - 1, i + 1  # two pointers
        while l >= 0 or r < n:
            # choose nearer side in projection space
            go_left = False
            if l >= 0 and r < n:
                go_left = (g_anch - proj[
                    idx_y_fx[l]]) <= (proj[idx_y_fx[r]] - g_anch)
            elif l >= 0:
                go_left = True
            if go_left:
                j = idx_y_fx[l]
                gap = g_anch - proj[j]
                l -= 1
            else:
                j = idx_y_fx[r]
                gap = proj[j] - g_anch
                r += 1
            # early stop
            if gap >= best:  # if gap * gap >= best:
                break
            if A_j[j] != A_anch:
                d2 = DistDirect_Euclidean(X_yfx_anch, X_nA_y[j])
                if d2 < best:
                    best = d2
        d_min.append(best)  # d_min[i] = best

        # finally,
        # tmp = min(min_js[0], min_jr[0])
        # d_min.append(tmp)
    return max(d_min), sum(d_min)
"""


# Algorithm 2. ApproxDist
# The early-stop strategy

@fantasy_timer
def ApproxDist_nonbin(X_nA_y, A_j, n_e=2):
    # def ApproxDist_nonbin(X_nA_y, A_j, m1=1,m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1=#nonsa
    # n_d-1: number of non-sensitive attributes

    # '''
    # d_max, d_avg = [], []
    # for _ in range(m1):
    #     W, _ = orthogonal_weight(n_d, n_e)
    #
    #     tmp = [AcceleDist_nonbin(
    #         X_nA_y, A_j,  # m2,
    #         W[k]) for k in range(n_e)]
    #     tmp, _ = zip(*tmp)
    #     t_max, t_avg = zip(*tmp)
    #     d_max.append(min(t_max))
    #     d_avg.append(min(t_avg))
    # return min(d_max), min(d_avg) / float(n)
    # '''

    W, _ = orthogonal_weight(n_d, n_e)
    tmp = [AcceleDist_nonbin(
        X_nA_y, A_j, W[k]) for k in range(n_e)]
    tmp, _ = zip(*tmp)
    t_max, t_avg = zip(*tmp)
    return min(t_max), min(t_avg) / float(n)


@fantasy_timer
def ApproxDist_nonbin_mpver(X_nA_y, A_j, n_e=2, pool=None):
    # def ApproxDist_nonbin_mpver(X_nA_y, A_j, m1=1, m2=8,
    #                         n_e=2, pool=None):
    n, n_d = X_nA_y.shape
    d_max, d_avg = [], []
    X_nA_y_map = [X_nA_y] * n_e
    A_j_map = [A_j] * n_e
    # m2_map = [m2] * n_e

    '''
    if pool is None:
        W = list(map(orthogonal_weight, [n_d] * m1, [n_e] * m1))
        W, _ = zip(*W)  # ignoring time cost
        for j in range(m1):
            tmp = list(map(AcceleDist_nonbin,
                           X_nA_y_map, A_j_map,  # m2_map,
                           W[j]))
            tmp, _ = zip(*tmp)
            t_max, t_avg = zip(*tmp)
            d_max.append(min(t_max))
            d_avg.append(min(t_avg))

    else:
        W = pool.map(orthogonal_weight, [n_d] * m1, [n_e] * m1)
        W, _ = zip(*W)
        for j in range(m1):
            tmp = pool.map(AcceleDist_nonbin,
                           X_nA_y_map, A_j_map,  # m2_map,
                           W[j])
            tmp, _ = zip(*tmp)
            t_max, t_avg = zip(*tmp)
            d_max.append(min(t_max))
            d_avg.append(min(t_avg))

    return min(d_max), min(d_avg) / float(n)
    '''

    W = orthogonal_weight(n_d, n_e)
    j = 0  # int(np.random.choice(W.shape[0]))
    if pool is None:
        tmp = list(map(AcceleDist_nonbin,
                       X_nA_y_map, A_j_map, W[j]))
    else:
        tmp = pool.map(AcceleDist_nonbin,
                       X_nA_y_map, A_j_map, W[j])
    tmp, _ = zip(*tmp)
    t_max, t_avg = zip(*tmp)
    d_max.append(min(t_max))
    d_avg.append(min(t_avg))
    return min(d_max), min(d_avg) / float(n)


# ------------------------------------------
# Optimised Algorithm 2. ApproxDist
# ------------------------------------------
# The rearrange strategy


@fantasy_timer  # blank
def StratVacant(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape
    d_max, d_avg = [], []
    for _ in range(m1):
        # Take two orthogonal vectors $w_k$
        W, _ = orthogonal_weight(n_d, n_e)
        tmp = [AcceleCore(
            X_nA_y, A_j, m2, W[k]) for k in range(n_e)]
        tmp = [[max(k), sum(k)] for k in tmp]  # converge
        t_max, t_avg = zip(*tmp)
        d_max.append(min(t_max))
        d_avg.append(min(t_avg))
    return min(d_max), min(d_avg) / float(n)

# StratVacant = blank


'''
@fantasy_timer
def StratRearrange_ver1(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1 #non-sen-att
    d_max, d_avg = [], []
    for _ in range(m1):
        # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
        W, _ = orthogonal_weight(n_d, n_e)
        tmp = [AcceleCoreBack(
            X_nA_y, A_j, m2, W[k]) for k in range(n_e)]
        dt_min = []
        for i in range(n):
            dt = [tmp[k][i] for k in range(n_e)]
            dt_min.append(min(dt))
        # pdb.set_trace()
        d_max.append(max(dt_min))
        d_avg.append(sum(dt_min))
    return min(d_max), min(d_avg) / float(n)
# INCORRECT!


@fantasy_timer
def StratRearrange_ver2(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1 #non-sen-att

    # dt_min = []  # d_max, d_avg = [], []
    # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
    W, _ = orthogonal_weight(n_d, n_e)
    tmp = [AcceleCoreBack(X_nA_y, A_j, m2,
                          W[k]) for k in range(n_e)]
    tmp = list(zip(*tmp))
    dt_min = [min(i) for i in tmp]

    for _ in range(1, m1):
        W, _ = orthogonal_weight(n_d, n_e)
        tmp = [AcceleCoreBack(X_nA_y, A_j, m2, W[k]
                              ) for k in range(n_e)]
        tmp = list(zip(*tmp))
        for i, tk in enumerate(tmp):
            dt = min(tk)  # tmp[i])
            if dt < dt_min[i]:
                dt_min[i] = dt
    return max(dt_min), sum(dt_min) / float(n)
# INCORRECT!
# StratRearrange = StratRearrange_ver2
'''


@fantasy_timer
def StratRearrange_ver3(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1 #non-sen-att
    dt_min = np.full(n, np.inf).tolist()
    for _ in range(m1):
        # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
        W, _ = orthogonal_weight(n_d, n_e)
        tmp = [AcceleCoreBack(X_nA_y, A_j, m2,
                              W[k]) for k in range(n_e)]
        tmp = list(zip(*tmp))
        for i, tk in enumerate(tmp):
            dt = min(tk)
            if dt < dt_min[i]:
                dt_min[i] = dt
    return max(dt_min), sum(dt_min) / float(n)


@fantasy_timer
def StratRearrange_ver4(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1 #non-sen-att
    dt_min = []
    for _ in range(m1):
        # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
        W, _ = orthogonal_weight(n_d, n_e)
        tmp = [AcceleCoreBack(X_nA_y, A_j, m2, W[k]
                              ) for k in range(n_e)]
        tmp = list(zip(*tmp))  # (n_e,n) --> (n,n_e)
        dt_min.append([min(i) for i in tmp])
    dt_min = list(zip(*dt_min))  # (m1,n) -> (n,m1)
    dt_min = [min(i) for i in dt_min]
    return max(dt_min), sum(dt_min) / float(n)


StratRearrange = StratRearrange_ver4
StratEarlyStop = ApproxDist_nonbin


@fantasy_timer
def EffExact(X_nA_y, A, Strat, m1=1, m2=8, n_e=3, pool=None):
    _, n_a = A.shape  # n: #inst, n_a: #sen-att
    X_nA_y_map = [X_nA_y] * n_a
    A_i_map = [A[:, j].copy() for j in range(n_a)]
    m1_map = [m1] * n_a
    m2_map = [m2] * n_a
    ne_map = [n_e] * n_a

    if Strat == StratEarlyStop:
        kw = (ne_map,)
    else:
        kw = (m1_map, m2_map, ne_map)
    del m1_map, m2_map, ne_map
    if pool is None:
        tmp = list(map(Strat, X_nA_y_map, A_i_map, *kw))
    else:
        tmp = pool.map(Strat, X_nA_y_map, A_i_map, *kw)
    del X_nA_y_map, A_i_map, kw

    tmp, half_ut = zip(*tmp)
    d_max, d_avg = zip(*tmp)
    return max(d_max), sum(d_avg) / float(n_a), (
        d_max, d_avg, half_ut)


# ------------------------------------------
# ------------------------------------------
# Algorithm 1. ExtendDist


@fantasy_timer
def ExtendDist_multiver_mp(X_nA_y, A, n_e=3, pool=None):
    # def ExtendDist_multiver_mp(X_nA_y, A, m1, n_e=3, pool=None):
    # def ExtendDist_multiver_mp(X_nA_y, A, m1=1, m2=8,
    #                        n_e=3, pool=None):
    _, n_a = A.shape  # n: #instance, n_a: #sen-att

    X_nA_y_map = [X_nA_y] * n_a
    A_i_map = [A[:, j].copy() for j in range(n_a)]
    # m1_map = [m1] * n_a
    # m2_map = [m2] * n_a
    ne_map = [n_e] * n_a

    if pool is None:
        tmp = list(map(ApproxDist_nonbin,
                       X_nA_y_map, A_i_map,  # m1_map, m2_map,
                       ne_map))
    else:
        tmp = pool.map(ApproxDist_nonbin,
                       X_nA_y_map, A_i_map,  # m1_map, m2_map,
                       ne_map)
    del X_nA_y_map, A_i_map, ne_map  # , m1_map, m2_map

    tmp, half_ut = zip(*tmp)
    d_max, d_avg = zip(*tmp)
    return max(d_max), sum(d_avg) / float(n_a), (
        d_max, d_avg, half_ut)


# ------------------------------------------
#
