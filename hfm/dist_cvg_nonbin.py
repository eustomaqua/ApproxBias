# coding: utf-8
# Usage: to approximate the distance quickly
#
# Author: Yj
# 2. Approximating Discrimination Within Models When Faced With Several
#    Non-Binary Sensitive Attributes [https://arxiv.org/abs/2408.06099]
#


import numpy as np
import numba
import pdb

from hfm.utils.decorators import fantasy_timer
from hfm.dist_drt import DistDirect_Euclidean
from hfm.dist_est_bin import projector
from hfm.dist_est_nonbin import (
    set_belonging, orthogonal_weight)


# ==========================================
# Estimated distance between sets
# (Converged version)


@numba.jit(nopython=True)
def sep_helper(idx_y_fx, Ai, X_yfx, proj, i):
    i_anchor = idx_y_fx[i]
    A_anchor = Ai[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]
    g_anchor = proj[i_anchor]
    return i_anchor, A_anchor, X_yfx_anchor, g_anchor


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
    i_anch, A_anch, X_yfx_anch, g_anch = sep_helper(
        idx_y_fx, Ai, X_yfx, proj, i)

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
    i_anch, A_anch, X_yfx_anch, g_anch = sep_helper(
        idx_y_fx, Ai, X_yfx, proj, i)

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


# ------------------------------------------
# Algorithm 3. AcceleDist


@fantasy_timer
def AcceleDist_nonbin(X_nA_y, A_j, vec_w):
    # def AcceleDist_nonbin(X_nA_y, A_j, m2, vec_w):
    proj = [projector(ele, vec_w) for ele in X_nA_y]
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


# ------------------------------------------
# Algorithm 2. ApproxDist


@fantasy_timer
def ApproxDist_nonbin(X_nA_y, A_j, n_e=2):
    # def ApproxDist_nonbin(X_nA_y, A_j, m1=1,m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1=#nonsa
    # n_d-1: number of non-sensitive attributes

    '''
    d_max, d_avg = [], []
    for _ in range(m1):
        W, _ = orthogonal_weight(n_d, n_e)

        tmp = [AcceleDist_nonbin(
            X_nA_y, A_j,  # m2,
            W[k]) for k in range(n_e)]
        tmp, _ = zip(*tmp)
        t_max, t_avg = zip(*tmp)
        d_max.append(min(t_max))
        d_avg.append(min(t_avg))
    return min(d_max), min(d_avg) / float(n)
    '''

    W, _ = orthogonal_weight(n_d, n_e)
    tmp = [AcceleDist_nonbin(
        X_nA_y, A_j, W[k]) for k in range(n_e)]
    tmp, _ = zip(*tmp)
    t_max, t_avg = zip(*tmp)
    return min(t_max), min(t_avg) / float(n)


@fantasy_timer
def ApproxDist_nonbin_mpver(X_nA_y, A_j, n_e=2, pool=None):
    # def ApproxDist_nonbin_mpver(X_nA_y,A_j, m1=1,m2,n_e=2,
    #                             pool=None):
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
# Algorithm 1. ExtendDist


@fantasy_timer
def ExtendDist_multiver_mp(X_nA_y, A, n_e=3, pool=None):
    # def ExtendDist_multiver_mp(X_nA_y, A, m1, n_e=3, pool=None):
    # def ExtendDist_multiver_mp(X_nA_y,A, m1,m2,n_e=3,pool=None):
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
