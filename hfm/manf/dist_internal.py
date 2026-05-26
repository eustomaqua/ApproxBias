# coding: utf-8

from scipy.spatial import distance
import numpy as np
from numba import njit, prange
# import numba
# import pdb
import math
from hfm.utils.decorators import fantasy_timer
from hfm.utils.verifiers import INF64, DTY_FLT
# from hfm.manf.parm_hfm import dual_normalize


# ------------------------------------------
# Minkowski distance

@fantasy_timer
def avbl_Euclidean(vec):  # ele_i, ele_ic):
    # alt = np.linalg.norm(ele_i - ele_ic)
    alt = np.linalg.norm(vec)
    return float(alt)


@fantasy_timer
def avbl_Manhattan(vec):  # ele_i, ele_ic):
    alt = np.linalg.norm(vec, ord=1)
    return float(alt)


@fantasy_timer
def avbl_Chebyshev(vec):  # ele_i, ele_ic):
    alt = np.linalg.norm(vec, ord=np.inf)
    return float(alt)


@fantasy_timer
def avbl_Minkowski(ele_i, ele_ic, p=3):
    alt = distance.minkowski(ele_i, ele_ic, p=p)
    return float(alt)


@njit
def avbl_cos_sim(ele_i, ele_ic):
    # cosine similarity
    norm_a = np.linalg.norm(ele_i)
    norm_b = np.linalg.norm(ele_ic)
    ans = np.dot(ele_i, ele_ic) / (norm_a * norm_b)
    return float(1. - ans)


@njit
def avbl_corr_pearson(X, Y):
    ans = 1. - np.corrcoef(X, Y)[1, 0]
    return float(ans)


# Euclidean metric

@njit
def dist_Euclidean(vec):
    # ans = np.linalg.norm(vec)
    ans = math.sqrt(np.sum(vec ** 2))
    return float(ans)


@njit
def dist_Manhattan(vec):
    # vec = ele_i - ele_ic
    # l1_norm_alt =
    ans = np.sum(np.abs(vec))
    return float(ans)


@njit
def dist_Chebyshev(vec):
    # vec = ele_i - ele_ic
    # alt=np.max(np.abs(a - b))
    ans = np.max(np.abs(vec))
    return float(ans)


@njit
def dist_Minkowski(vec, p=3):
    ans = np.sum(np.abs(vec ** p))
    ans = np.power(ans, 1. / p)
    return float(ans)


@njit
def dist_cos_sim(ele_i, ele_ic):
    na = math.sqrt(np.sum(ele_i * ele_i))
    nb = math.sqrt(np.sum(ele_ic * ele_ic))
    ele_i = np.ascontiguousarray(ele_i)
    ele_ic = np.ascontiguousarray(ele_ic)
    ans = np.dot(ele_i, ele_ic) / (na * nb)
    return float(1. - ans)


# @fantasy_timer
# def dist_Mahalanobis(ele_i, ele_ic):
#     cov_matrix = np.cov()
#     return


@njit
def Pearson_correlation(X, Y):
    # or np.corrcoef(X, Y)[1, 0]
    # or np.corrcoef(Y, X)[1, 0]

    # Covariance 协方差
    Xi_bar = X - X.mean()  # np.array(X) - np.mean(X)
    Yi_bar = Y - Y.mean()  # np.array(Y) - np.mean(Y)
    tmp = np.multiply(Xi_bar, Yi_bar)  # =Xi_bar*Yi_bar
    n = tmp.shape[0]  # n = len(tmp)
    cov = tmp.sum() / (n - 1.)

    numerator = tmp.sum()     # ↓ denominator
    denom_X = (Xi_bar * Xi_bar).sum()  # sum(Xi_bar**2)
    denom_Y = (Yi_bar * Yi_bar).sum()  # sum(Yi_bar**2)
    denominator = math.sqrt(denom_X) * math.sqrt(denom_Y)
    denominator = denominator if denominator != 0. else 1.
    return numerator / denominator, cov


@njit
def dist_corr_pearson(ele_i, ele_ic):
    ans = Pearson_correlation(ele_i, ele_ic)[0]
    return float(1. - ans)


# ------------------------------------------
# Distance between sets
# intermediate


name_intermediate = ['euclidean', 'manhattan', 'chebyshev',
                     'minkowski', 'cos_sim', 'correla', ]
# Mahalanobis dist, Hamming dist, Jaccard similarity/dist

STRATEGIES = ['Approx', 'StratES', 'StratRA']  # 'Vacant'
curr_intermediate = name_intermediate[-2:] + name_intermediate[:-2]


dist_intermediate = {
    'euclidean': dist_Euclidean,
    'manhattan': dist_Manhattan,
    'chebyshev': dist_Chebyshev,
    'minkowski': dist_Minkowski,
    'cos_sim': dist_cos_sim,
    'correla': dist_corr_pearson,
}


# @njit
# def alter_intermediate(ele_i, ele_ic, func='euclidean', p=3):
#     if func == 'euclidean':
#         return dist_Euclidean(ele_i - ele_ic)
#     elif func == 'manhattan':
#         return dist_Manhattan(ele_i - ele_ic)
#     elif func == 'chebyshev':
#         return dist_Chebyshev(ele_i - ele_ic)
#
#     # if func in ['euclidean', 'manhattan', 'chebyshev']:
#     #     return dist_intermediate[func](ele_i - ele_ic)
#     elif func == 'minkowski':
#         return dist_Minkowski(ele_i - ele_ic, p)
#     return dist_cos_sim(ele_i, ele_ic)

# name_inter_alt = (dist_Euclidean, dist_Manhattan, dist_Chebyshev,
#                   dist_Minkowski, dist_cos_sim, dist_corr_pearson)
# name_inter_alt = (dist_Euclidean, dist_Manhattan, dist_Chebyshev)
# name_inter_alt_sep = (dist_cos_sim, dist_corr_pearson)


@njit
def alter_intermediate(ele_i, ele_ic, func_id=0, p=3):
    # if func_id == 0:
    #     return dist_Euclidean(ele_i - ele_ic)
    # elif func_id == 1:
    #     return dist_Manhattan(ele_i - ele_ic)
    # elif func_id == 2:
    #     return dist_Chebyshev(ele_i - ele_ic)
    # elif func_id == 3:
    #     return dist_Minkowski(ele_i - ele_ic, p)
    # elif func_id == 4:
    #     return dist_cos_sim(ele_i, ele_ic)
    # return dist_corr_pearson(ele_i, ele_ic)

    # if func_id >= 4:
    #     return name_inter_alt[func_id](ele_i, ele_i)
    # vec_diff = ele_i - ele_ic
    # if func_id == 3:
    #     return name_inter_alt[func_id](vec_diff, p)
    # return name_inter_alt[func_id](vec_diff)

    # if func_id >= 4:
    #     return name_inter_alt_sep[func_id - 4](ele_i, ele_ic)
    # vec_diff = ele_i - ele_ic  # np.ascontiguousarray()
    # if func_id <= 2:
    #     return name_inter_alt[func_id](vec_diff)
    # return dist_Minkowski(vec_diff, p)
    if func_id == 4:
        return dist_cos_sim(ele_i, ele_ic)
    elif func_id == 5:
        return dist_corr_pearson(ele_i, ele_ic)
    vec_diff = ele_i - ele_ic  # np.ascontiguousarray()
    if func_id == 0:
        return dist_Euclidean(vec_diff)
    elif func_id == 1:
        return dist_Manhattan(vec_diff)
    elif func_id == 2:
        return dist_Chebyshev(vec_diff)
    return dist_Minkowski(vec_diff, p)


@njit
def Direct_halfway_min(ele_i, Si_c, func=0, p=3):  # 'euclidean'
    elements = [alter_intermediate(
        ele_i, ele_ic, func, p) for ele_ic in Si_c]
    return min(elements)


@njit
def Direct_mediator(X_nA_y, idx_Si, func=0, p=3):  # 'euclidean'
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    if len(Sj) == 0 or len(Sj_c) == 0:
        return 0., 0.  # default if Sj is an empty set
    # Sj = np.ascontiguousarray(Sj)
    # Sj_c = np.ascontiguousarray(Sj_c)
    elements = [Direct_halfway_min(
        ele_i, Sj_c, func, p) for ele_i in Sj]
    return max(elements), sum(elements)


# # @fantasy_timer
# # @numba.jit(nopython=True)
# def Direct_halfway_min(ele_i, Si_c, func='euclidean', p=3):
#     dist_fn = dist_intermediate[func]
#     if func in ['euclidean', 'manhattan', 'chebyshev']:
#         elements = [dist_fn(ele_i - ele_ic) for ele_ic in Si_c]
#     elif func == 'minkowski':
#         elements = [dist_fn(ele_i - ele_ic, p) for ele_ic in Si_c]
#     elif func == 'cos_sim':
#         elements = [dist_fn(ele_i, ele_ic) for ele_ic in Si_c]
#     return min(elements)
#
#
# def Direct_mediator(X_nA_y, idx_Si, func='euclidean', p=3):
#     Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
#     if len(Sj) == 0 or len(Sj_c) == 0:
#         return 0., 0.  # default if Sj is an empty set
#     elements = [
#         Direct_halfway_min(ele_i, Sj_c, func, p) for ele_i in Sj]
#     return max(elements), sum(elements)


# @fantasy_timer
def idx_marginalised(A_i, priv_val=1):
    vAi = np.unique(A_i).tolist()
    vAi.remove(priv_val)
    idx_Si = [A_i == priv_val]
    for j in vAi:
        idx_Si.append(A_i == j)
    return idx_Si
# def Direct_marginalised(A, priv_val=1):
#     _, na = A.shape
#     vA = np.unique(A).tolist()
#     nai = len(vA)
#     vA.remove(priv_val)
#     indices = []
#     for i in range(na):
#         tmp = [A[:, i] == priv_val]
#         for j in vA:
#             tmp.append(A[:, i] == j)
#         indices.append(tmp)
#     return indices


@fantasy_timer
def Direct_bin(X_nA_y, A_i, priv_val=1, idx_Si=None,
               func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    if idx_Si is None:
        idx_Si = A_i == priv_val
    half_1, half_1avg = Direct_mediator(X_nA_y, idx_Si, func_id, p)
    half_2, half_2avg = Direct_mediator(X_nA_y, ~idx_Si, func_id, p)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


@fantasy_timer
def Direct_nonbin(X_nA_y, A_i, priv_val=1, idx_Sjs=None,
                  func='euclidean', p=3):
    func_id = name_intermediate.index(func)
    if idx_Sjs is None:
        idx_Sjs = idx_marginalised(A_i, priv_val)
    half_mid = [Direct_mediator(
        X_nA_y, idx_Si, func_id, p) for idx_Si in idx_Sjs]
    half_pl_max, half_pl_avg = zip(*half_mid)
    n = len(X_nA_y)
    return max(half_pl_max), sum(half_pl_avg) / n


@fantasy_timer
def Direct_multivar(X_nA_y, A, priv_val=1, indices=None,
                    func='euclidean', p=3):
    # func_id = name_intermediate.index(func)
    n_a = A.shape[1]
    if indices is None:  # that is, idx_Ai_Sjs
        indices = [
            idx_marginalised(A[:, i], priv_val) for i in range(n_a)]
    half_mid = [Direct_nonbin(X_nA_y, A[
        :, i], priv_val, indices[i], func, p) for i in range(n_a)]
    half_mid, half_ut = zip(*half_mid)
    half_pl_max, half_pl_avg = zip(*half_mid)
    return max(half_pl_max), sum(half_pl_avg) / n_a, (
        half_pl_max, half_pl_avg, half_ut)


# ------------------------------------------
# Approximation


@njit
def set_belonging(A, i_anchor, idx_j):
    if A[i_anchor] == A[idx_j]:
        return True
    return False


@njit
def _sub_accelerator_dir(X_yfx, A, idx_y_fx, pos, m2, func, p,
                         direction):
    i_anchor = idx_y_fx[pos]  # anchor's location after projection
    A_anchor = A[i_anchor]
    X_yfx_anch = X_yfx[i_anchor]
    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()?=g(xi,yi;w)
    num_j, best = 0, INF64
    n = X_yfx.shape[0]
    j = pos + direction  # doesn't have to be compared with anchor
    while (num_j < m2) and (0 <= j < n):
        idx_j = idx_y_fx[j]
        if A_anchor == A[idx_j]:  # set_belonging
            j += direction
            continue
        curr = alter_intermediate(X_yfx_anch, X_yfx[idx_j], func, p)
        if curr < best:
            best = curr
        num_j += 1
        j += direction
    return best


@njit
def sub_accelerator_smaler(X_yfx, A, idx_y_fx, i, m2, func=0, p=3):
    return _sub_accelerator_dir(X_yfx, A, idx_y_fx, i, m2, func, p, -1)


@njit
def sub_accelerator_larger(X_yfx, A, idx_y_fx, i, m2, func=0, p=3):
    return _sub_accelerator_dir(X_yfx, A, idx_y_fx, i, m2, func, p, +1)


# @njit
# def sub_accelerator_smaler(X_yfx, A, idx_y_fx, i, m2,
#                            func=0, p=3):  # 'euclidean'
#     i_anchor = idx_y_fx[i]  # anchor's location after projection
#     A_anchor = A[i_anchor]
#     X_yfx_anchor = X_yfx[i_anchor]
#     # tmp = []
#
#     # Compute the distance d(anchor,\cdot) for at most m2 nearby
#     # data points that meets a!=ai and g()<=g(xi,yi;w)
#     num_j, min_js = 0, INF64  # np.finfo(np.float64).max
#     j = i - 1  # doesn't have to be compared with the anchor
#     while num_j < m2 and j >= 0:
#         # if j < 0:
#         #     break
#         idx_j = idx_y_fx[j]
#         if A_anchor == A[idx_j]:  # set_belonging(A, i_anchor, idx_j):
#             j -= 1
#             continue
#
#         curr = alter_intermediate(X_yfx_anchor, X_yfx[idx_j], func, p)
#         if curr < min_js:
#             min_js = curr
#         num_j += 1
#         j -= 1
#         # tmp.append(int(idx_j))
#     # Find the minimum among them, recorded as d_min^s
#     return min_js  # , tmp


# @njit
# def sub_accelerator_larger(X_yfx, A, idx_y_fx, i, m2,
#                            func=0, p=3):  # 'euclidean'
#     i_anchor = idx_y_fx[i]  # anchor's location after projection
#     A_anchor = A[i_anchor]
#     X_yfx_anchor = X_yfx[i_anchor]
#     # tmp = []
#
#     # Compute the distance d(anchor,\cdot) for at most m2 nearby
#     # data points that meets a!=ai and g()>=g(xi,yi;w)
#     num_j, min_jr = 0, INF64  # np.finfo(np.float64).max
#     j = i + 1  # doesn't have to be compared with the anchor
#     n = len(X_yfx)
#     while num_j < m2 and j < n:
#         # if j >= n:
#         #     break
#         idx_j = idx_y_fx[j]
#         if A_anchor == A[idx_j]:  # set_belonging(A, i_anchor, idx_j):
#             j += 1
#             continue
#
#         curr = alter_intermediate(X_yfx_anchor, X_yfx[idx_j], func, p)
#         if curr < min_jr:
#             min_jr = curr
#         num_j += 1
#         j += 1
#         # tmp.append(int(idx_j))
#     # Find the minimum among them, recorded as d_min^r
#     return min_jr  # , tmp


@njit
def projector(element, vec_w):
    ans = np.dot(element, vec_w)
    return float(ans)


@njit
def projector_alt(element, vec_w):
    ans = 0.0
    # for i, ele in enumerate(element):
    #     ans += ele * vec_w[i]
    for i in range(element.size):
        ans += element[i] * vec_w[i]
    return float(ans)


# @fantasy_timer
@njit
def AcceleCore_bin(X_yddot, B_i, m2, vec_w, func=0, p=3):  # 'euclidean'
    # Project data points onto a one-dimensional space
    # proj = [projector(ele, vec_w) for ele in X_yddot]
    proj = X_yddot @ vec_w
    idx_y_fx = np.argsort(proj)

    n = X_yddot.shape[0]  # number of instances
    d_min = np.empty(n, dtype=DTY_FLT)   # d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round
        # min_js = sub_accelerator_smaler(
        #     X_yddot, B_i, idx_y_fx, i, m2, func, p)
        # min_jr = sub_accelerator_larger(
        #     X_yddot, B_i, idx_y_fx, i, m2, func, p)

        min_js = _sub_accelerator_dir(
            X_yddot, B_i, idx_y_fx, i, m2, func, p, -1)
        min_jr = _sub_accelerator_dir(
            X_yddot, B_i, idx_y_fx, i, m2, func, p, +1)

        # finally,
        # tmp = min(min_js, min_jr)
        # d_min.append(tmp)
        d_min[i] = min(min_js, min_jr)
    # return max(d_min), sum(d_min)
    # return float(d_min.max()), float(d_min.sum())
    return d_min.max(), d_min.sum()


@njit
def rand_uniform(a, b):
    # vec_w[i] = np.random.uniform(-tmp, tmp)
    return a + (b - a) * np.random.random()


# @fantasy_timer
@njit
def weight_generator(n_d):
    vec_w = np.empty(1 + n_d, dtype=DTY_FLT)
    tmp = 1.0
    for i in range(n_d):
        val = rand_uniform(-tmp, tmp)
        vec_w[i] = val
        tmp -= math.fabs(val)
    ss = 0.0  # compute last weight
    for i in range(n_d):
        ss += math.fabs(vec_w[i])
    vec_w[n_d] = 1. - ss
    return vec_w


# uniform sampling over L1 sphere

# # @fantasy_timer
# @njit(parallel=True)
# def weight_gen_many(n, n_d):
#     out = np.empty((n, 1 + n_d))
#     for i in prange(n):
#         out[i] = weight_generator(n_d)
#     return out


@njit(parallel=True)
def Approx_bin_sub(X_nA_y, B_i, m1, m2, func_id, p):
    n, n_d = X_nA_y.shape  # <class 'int'>
    d_max = np.empty(m1, dtype=DTY_FLT)
    d_avg = np.empty(m1, dtype=DTY_FLT)
    # weights = weight_gen_many(m1, n_d - 1)
    # d_max, d_avg = [], []
    for k in prange(m1):
        vec_w = weight_generator(n_d - 1)  # weights[k]
        tmp = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, func_id, p)
        # d_max.append(tmp[0])
        # d_avg.append(tmp[1])
        d_max[k] = tmp[0]
        d_avg[k] = tmp[1]
    # return min(d_max), min(d_avg) / float(n)
    # return float(d_max.min()), float(d_avg.min()) / n
    return d_max.min(), d_avg.min() / n


@fantasy_timer
def Approx_bin(X_nA_y, B_i, m1, m2, func='euclidean', p=3):
    # def Approx_bin(X_nA_y, A_i,priv_val=1,idx_Si=None,
    #                func='euclidean', m1=20, m2=8):
    #     if idx_Si is None:
    #         idx_Si = A_i == priv_val
    func_id = name_intermediate.index(func)
    tmp = Approx_bin_sub(X_nA_y, B_i, m1, m2, func_id, p)
    return list(map(float, tmp))
