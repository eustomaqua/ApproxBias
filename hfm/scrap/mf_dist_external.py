# coding: utf-8
# manf/dist_external.py


import numpy as np
from numba import njit, prange
import math
import pdb
from hfm.utils.decorators import fantasy_timer_prime
from hfm.utils.verifiers import CONST_ZERO, INF64, DTY_FLT


from hfm.scrap.mf_dist_internal import (
    # from hfm.manf.dist_internal import (
    name_intermediate, alter_intermediate,  # projector,
    # sub_accelerator_smaler, sub_accelerator_larger,
    AcceleCore_bin, _sub_accelerator_dir)
# from hfm.manf.parm_hfm import dual_normalize


# ------------------------------------------
# Approximation (first)


@njit
def AcceleCore_nonbin(X_nA_y, A_j, m2, vec_w, func=0, p=3):
    # proj = [projector(ele, vec_w) for ele in X_nA_y]
    proj = X_nA_y @ vec_w
    idx_y_fx = np.argsort(proj)
    n = X_nA_y.shape[0]  # number of instances

    d_min = np.empty(n, dtype=DTY_FLT)  # d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round
        min_js = sub_accelerator_smaler(
            X_nA_y, A_j, idx_y_fx, i, m2, func, p)
        min_jr = sub_accelerator_larger(
            X_nA_y, A_j, idx_y_fx, i, m2, func, p)
        # finally
        # tmp = min(min_js, min_jr)
        # d_min.append(tmp)
        d_min[i] = min(min_js, min_jr)
    # return max(d_min), sum(d_min)
    return d_min  # 4convergence


# @njit
# def _sub_accelerator_dir(X_yfx, A, idx_y_fx, pos, m2,
#                          func, p, direction):
#     i_anch = idx_y_fx[pos]  # anchor's location after projection
#     A_anchor = A[i_anch]
#     X_yfx_anch = X_yfx[i_anch]
#     n = X_yfx.shape[0]
#     # Compute the distance d(anchor,\cdot) for at most m2 nearby
#     # data points that meets a!=ai and g()?g(xi,yi;w)
#     num_j, min_js_jr = 0, INF64  # count,best
#     j = pos + direction  # doesn't have to be compared with anchor
#     while (num_j < m2) and (0 <= j < n):
#         idx_j = idx_y_fx[j]
#         if A_anchor == A[idx_j]:  # set_belonging
#             j += direction
#             continue
#
#         curr = alter_intermediate(X_yfx_anch, X_yfx[idx_j], func, p)
#         # Find the minimum among them, recorded as d_min
#         if curr < min_js_jr:
#             min_js_jr = curr
#         num_j += 1
#         j += direction
#     return min_js_jr


@njit
def AcceleCoreBack(X_nA_y, A_j, m2, vec_w, func=0, p=3):
    # proj = [projector(ele, vec_w) for ele in X_nA_y]
    proj = X_nA_y @ vec_w
    idx_y_fx = np.argsort(proj)
    n = X_nA_y.shape[0]

    # dt_min = np.full(n, np.inf, dtype=DTY_FLT)
    dt_min = np.empty(n, dtype=DTY_FLT)
    for pos in range(n):
        i = idx_y_fx[pos]  # original/initial sample id
        # before pos<>i # dt_min[pos] = d_min[i]
        # min_js = sub_accelerator_smaler(
        #     X_nA_y, A_j, idx_y_fx, pos, m2, func, p)
        # min_jr = sub_accelerator_larger(
        #     X_nA_y, A_j, idx_y_fx, pos, m2, func, p)

        min_js = _sub_accelerator_dir(
            X_nA_y, A_j, idx_y_fx, pos, m2, func, p, -1)
        min_jr = _sub_accelerator_dir(
            X_nA_y, A_j, idx_y_fx, pos, m2, func, p, +1)
        dt_min[i] = min(min_js, min_jr)  # pos
    return dt_min  # 4convergence


# @fantasy_timer
@njit
def orthogonal_weight(n_d, n_e=3):
    # # 生成 n_d x n_e 的随机矩阵（只生成需要的列）
    # B = np.random.randn(n_d, n_e)
    # # QR 分解得到正交基
    # Q, _ = np.linalg.qr(B)
    # # 返回前 n_e 个正交向量
    # return Q.T

    # B = np.random.randn(n_d, n_e)
    # Q = np.zeros((n_d, n_e))
    # for i in range(n_e):
    #     v = B[:, i].copy()
    #     for j in range(i):
    #
    #         v -= np.dot(v, Q[:, j]) * Q[:, j]
    #         # dot = 0.0
    #         # for k in range(n_d):
    #         #     dot += v[k] * Q[k, j]
    #         # for k in range(n_d):
    #         #     v[k] -= dot * Q[k, j]
    #     # normalize
    #     v /= np.linalg.norm(v)
    #     # norm = math.sqrt(np.sum(v * v))
    #     # for k in range(n_d):
    #     #     v[k] /= norm
    #
    #     Q[:, i] = v
    # return Q.T

    for _ in range(n_d):  # while True:
        B = np.random.rand(n_d, n_d)
        tmp = np.linalg.det(B)
        if abs(tmp) > CONST_ZERO:
            break  # 数值稳定判断
    # 2. Gram-Schmidt （向量化版本）
    A_T = B.T
    eta = np.zeros((n_e, n_d))
    # 第一个向量归一化
    v = A_T[0]
    eta[0] = v / np.linalg.norm(v)
    # 后续向量
    for i in range(1, n_e):
        v = A_T[i].copy()
        # 投影部分向量化：proj=(eta[:i] @v)
        proj = eta[:i] @ v
        v = v - proj @ eta[:i]
        # 归一化
        v = v / np.linalg.norm(v)
        eta[i] = v
    return eta


# @fantasy_timer
@njit(parallel=True)
def Approx_nonbin_sub(X_nA_y, A_j, m1, m2, n_e, func_id, p):
    n, n_d = X_nA_y.shape  # n_d-1: number of non-sen-att(s)
    d_max = np.empty(m1, dtype=DTY_FLT)
    d_avg = np.empty(m1, dtype=DTY_FLT)
    # d_max, d_avg = [], []
    for k in prange(m1):
        # Take two orthogonal vectors $w_0$ and $w_1$ where each $w_k
        # \in [-1,+1]^{1+n_x} (k=\{0,1\})$
        # Or take three orthogonal vectors. Your choice.
        W = orthogonal_weight(n_d, n_e)

        # tmp = [AcceleCore_bin(X_nA_y, A_j, m2, W[i]) for i in range(n_e)]
        # #  tmp, _ = zip(*tmp)
        # t_max, t_avg = zip(*tmp)
        t_max = np.empty(n_e, dtype=DTY_FLT)  # np.full(n_e, np.inf)
        t_avg = np.empty(n_e, dtype=DTY_FLT)  # np.full(n_e, np.inf)
        for j in range(n_e):
            mx, sm = AcceleCore_bin(X_nA_y, A_j, m2, W[j], func_id, p)
            t_max[j] = mx  # tmp[0]
            t_avg[j] = sm  # tmp[1]

        d_max[k] = t_max.min()  # min(t_max) # d_max.append(min(t_max))
        d_avg[k] = t_avg.min()  # min(t_avg) # d_avg.append(min(t_avg))
    # return min(d_max), min(d_avg) / float(n)
    return d_max.min(), d_avg.min() / n


@fantasy_timer_prime
def Approx_nonbin(X_nA_y, A_j, m1, m2, n_e=2, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    tmp = Approx_nonbin_sub(X_nA_y, A_j, m1, m2, n_e, func_id, p)
    return list(map(float, tmp))


# @njit
# def Extend_multiver_sub(X_nA_y, A, m1, m2, n_e, func_id, p):
#     # tmp, half_ut = zip(*tmp)
#     # d_max, d_avg = zip(*tmp)
#     # return max(d_max), sum(d_avg) / float(n_a), (
#     #     d_max, d_avg, half_ut)
#
#     # d_max = d_max.max()   # d_max.tolist()
#     # d_avg = d_avg.sum()   # d_avg.tolist()
#     # # # return max(d_max), sum(d_avg) / float(n_a), (d_max, d_avg)
#     # return float(d_max), float(d_avg) / n_a
#
#     # return d_max.max(), d_avg.sum() / n_a
#     # return max(d_max), sum(d_avg) / float(n_a)
#     return d_max.max(), d_avg.mean()


@fantasy_timer_prime
def Extend_multivar(X_nA_y, A, m1, m2, n_e=3, func='euclidean', p=3):
    _, n_a = A.shape  # n= #instances, n_a: number of sen-att
    func_id = name_intermediate.index(func)

    # # tmp = []
    # d_max = np.empty(n_a)  # []
    # d_avg = np.empty(n_a)  # []
    # for j in range(n_a):
    #     (t0, t1) = Approx_nonbin_sub(
    #         X_nA_y, A[:, j], m1, m2, n_e, func_id, p)
    #     # tmp.append(tj)
    #     d_max[j] = t0  # tmp[0]  # d_max.append(tmp[0])
    #     d_avg[j] = t1  # tmp[1]  # d_avg.append(tmp[1])
    # return float(d_max.max()), float(d_avg.mean())

    # tmp = []
    # for j in range(n_a):
    #     t = Approx_nonbin_sub(X_nA_y, A[:, j], m1, m2, n_e, func_id, p)
    #     tmp.append(t)
    # tmp, half_ut = zip(*tmp)
    # d_max, d_avg = zip(*tmp)
    # return max(d_max), sum(d_avg) / float(n_a), (
    #     d_max, d_avg, half_ut)

    d_max = -np.inf
    d_avg = 0.0
    for j in range(n_a):
        tmp, t1 = Approx_nonbin_sub(
            X_nA_y, A[:, j], m1, m2, n_e, func_id, p)
        if tmp > d_max:
            d_max = tmp
        d_avg += t1
    return float(d_max), float(d_avg) / n_a  # slow, bottleneck


# ------------------------------------------
# Approximation (second)


@njit
def _cvg_accelerator_dir(X_yfx, proj, Ai, idx_y_fx, i, func, p,
                         direction):
    i_anchor = idx_y_fx[i]
    A_anchor = Ai[i_anchor]
    X_yfx_anch = X_yfx[i_anchor]
    g_anchor = proj[i_anchor]

    n = X_yfx.shape[0]
    num_j, min_j_sr = 0, INF64
    j = i + direction
    while 0 <= j < n:
        idx_j = idx_y_fx[j]
        if A_anchor == Ai[idx_j]:
            j += direction
            continue

        # tmp = (proj[idx_j] - g_anchor) * direction
        # tmp = g_anchor - proj[
        #     idx_j] if direction < 0 else proj[idx_j] - g_anchor
        if direction < 0:
            tmp = g_anchor - proj[idx_j]
        else:
            tmp = proj[idx_j] - g_anchor

        if abs(tmp) >= min_j_sr:
            break
        curr = alter_intermediate(X_yfx_anch, X_yfx[idx_j], func, p)
        if curr < min_j_sr:
            min_j_sr = curr
        num_j += 1
        j += direction
    return min_j_sr, num_j


# @njit
# def cvg_accelerator_smaler(X_yfx, proj, Ai, idx_y_fx, i,
#                            func=0, p=3):
#     i_anchor = idx_y_fx[i]
#     A_anchor = Ai[i_anchor]
#     X_yfx_anch = X_yfx[i_anchor]
#     g_anchor = proj[i_anchor]
#
#     num_j, min_js = 0, INF64  # np.finfo(np.float64).max
#     j = i - 1  # doesn't have to be compared with the anchor
#     while j >= 0:
#         idx_j = idx_y_fx[j]
#         if A_anchor == Ai[idx_j]:
#             j -= 1
#             continue
#
#         tmp = g_anchor - proj[idx_j]
#         if abs(tmp) >= min_js:
#             break
#         curr = alter_intermediate(X_yfx_anch, X_yfx[idx_j], func, p)
#         if curr < min_js:
#             min_js = curr
#         num_j += 1
#         j -= 1
#     return min_js, num_j


# @njit
# def cvg_accelerator_larger(X_yfx, proj, Ai, idx_y_fx, i,
#                            func=0, p=3):
#     i_anchor = idx_y_fx[i]
#     A_anchor = Ai[i_anchor]
#     X_yfx_anch = X_yfx[i_anchor]
#     g_anchor = proj[i_anchor]
#
#     num_j, min_jr = 0, INF64  # np.finfo(np.float64).max
#     j = i + 1  # doesn't have to be compared with the anchor
#     n = len(X_yfx)
#     while j < n:
#         idx_j = idx_y_fx[j]
#         if A_anchor == Ai[idx_j]:
#             j += 1
#             continue
#
#         tmp = proj[idx_j] - g_anchor
#         if abs(tmp) >= min_jr:
#             break
#         curr = alter_intermediate(X_yfx_anch, X_yfx[idx_j], func, p)
#         if curr < min_jr:
#             min_jr = curr
#         num_j += 1
#         j += 1
#     return min_jr, num_j


@njit(parallel=True)
def _StratES_subproc(X_nA_y, A_j, vec_w, func, p):
    # proj = List()
    # [proj.append(projector(ele, vec_w)) for ele in X_nA_y]
    proj = X_nA_y @ vec_w
    idx_y_fx = np.argsort(proj)
    n = X_nA_y.shape[0]  # number of instances

    d_min = np.empty(n, dtype=DTY_FLT)  # []
    for i in prange(n):
        # Set the anchor data point (xi,yi) in this round
        # min_js, _ = cvg_accelerator_smaler(
        #     X_nA_y, proj, A_j, idx_y_fx, i, func, p)
        # min_jr, _ = cvg_accelerator_larger(
        #     X_nA_y, proj, A_j, idx_y_fx, i, func, p)

        min_js, _ = _cvg_accelerator_dir(
            X_nA_y, proj, A_j, idx_y_fx, i, func, p, -1)
        min_jr, _ = _cvg_accelerator_dir(
            X_nA_y, proj, A_j, idx_y_fx, i, func, p, +1)

        # tmp = min(min_js[0], min_jr[0])
        # d_min.append(tmp)
        d_min[i] = min(min_js, min_jr)
    # return max(d_min), sum(d_min)
    return d_min.max(), d_min.sum()


# @njit
# def _StratES_core(X_nA_y, A_j, n_e, func_id, p):
#     n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
#     W = orthogonal_weight(n_d, n_e)
#     # tmp = [_StratES_subproc(
#     #     X_nA_y, A_j, W[k], func_id, p) for k in range(n_e)]
#     # # tmp, _ = zip(*tmp)
#     # t_max, t_avg = zip(*tmp)
#     # return min(t_max), min(t_avg) / float(n)
#
#     t_max = np.empty(n_e, dtype=DTY_FLT)
#     t_avg = np.empty(n_e, dtype=DTY_FLT)
#     for k in range(n_e):
#         mx, sm = _StratES_subproc(X_nA_y, A_j, W[k], func_id, p)
#         t_max[k] = mx
#         t_avg[k] = sm
#     return t_max.min(), t_avg.min() / n
#
#
# @fantasy_timer
# def StratES_nonbin(X_nA_y, A_j, n_e=2, func='euclidean', p=3):
#     func_id = name_intermediate.index(func)
#     tmp = _StratES_core(X_nA_y, A_j, n_e, func_id, p)
#     return list(map(float, tmp))


@fantasy_timer_prime
def StratES_nonbin(X_nA_y, A_j, n_e=2, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
    W = orthogonal_weight(n_d, n_e)
    # vec_w = dual_normalize(W[0], p)
    t_max, t_avg = _StratES_subproc(X_nA_y, A_j, W[0], func_id, p)
    return float(t_max), float(t_avg) / n


@njit
def reduce_min_axis0(arr):
    # def np_min_2d(arr, axis=0):
    m, n = arr.shape
    out = np.empty(n, dtype=DTY_FLT)
    for j in range(n):
        # mn = arr[0, j]
        # for i in range(1, m):
        #     if arr[i, j] < mn:
        #         mn = arr[i, j]
        # out[j] = mn
        out[j] = arr[:, j].min()
    return out


@njit(parallel=True)
def _StratRA_core(X_nA_y, A_j, m1, m2, n_e, func, p):
    n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
    dt_min = np.empty((m1, n), dtype=DTY_FLT)  # []
    for v in prange(m1):
        # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
        W = orthogonal_weight(n_d, n_e)
        # tmp = [AcceleCoreBack(
        #     X_nA_y, A_j, m2, W[k], func_id, p) for k in range(n_e)]
        # tmp = list(zip(*tmp))  # (n_e,n) --> (n,n_e)
        # dt_min.append([min(i) for i in tmp])

        tmp = np.empty((n_e, n), dtype=DTY_FLT)
        for k in range(n_e):
            tmp[k] = AcceleCoreBack(X_nA_y, A_j, m2, W[k], func, p)
        # dt_min[v] = np.min(tmp, axis=0)  # tmp.min(axis=0)
        # for ji in range(n):
        #     dt_min[v, ji] = np.min(tmp[:, ji])
        dt_min[v] = reduce_min_axis0(tmp)

    # dt_min = list(zip(*dt_min))
    # dt_min = [min(i) for i in dt_min]
    # return max(dt_min), sum(dt_min) / float(n)

    # finale = np.min(dt_min, axis=0)  # dt_min.min(axis=0)
    # fin = np.empty(n)
    # for ji in range(n):
    #     fin[ji] = np.min(dt_min[:, ji])
    fin = reduce_min_axis0(dt_min)
    return fin.max(), fin.sum() / n


@fantasy_timer_prime
def StratRA_nonbin(X_nA_y, A_j, m1, m2, n_e=2, func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    tmp = _StratRA_core(X_nA_y, A_j, m1, m2, n_e, func_id, p)
    return list(map(float, tmp))


# ------------------------------------------
# Approximation


@fantasy_timer_prime
def EffExact_multivar(X_nA_y, A, Strat, m1=10, m2=4, n_e=3,
                      func='euclidean', p=3):
    _, n_a = A.shape  # n #inst, n_a #sen-att
    if Strat.endswith('ES'):
        tmp = [StratES_nonbin(X_nA_y, A[
            :, i], n_e, func, p) for i in range(n_a)]
    elif Strat.endswith('RA'):
        tmp = [StratRA_nonbin(X_nA_y, A[
            :, i], m1, m2, n_e, func, p) for i in range(n_a)]
    else:  # Strat == 'Vacant':
        tmp = [Approx_nonbin(X_nA_y, A[
            :, i], m1, m2, n_e, func, p) for i in range(n_a)]
    tmp, half_ut = zip(*tmp)
    d_max, d_avg = zip(*tmp)
    return max(d_max), sum(d_avg) / float(n_a), (
        d_max, d_avg, half_ut)


name_effexact = ['Approx', 'StratES', 'StratRA']
