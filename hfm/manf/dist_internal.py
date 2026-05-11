# coding: utf-8

from scipy.spatial import distance
import numpy as np
from numba import njit  # , prange
# import numba
import pdb
import math
from hfm.utils.decorators import fantasy_timer


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


# Euclidean metric

@njit
def dist_Euclidean(vec):
    # ans = np.linalg.norm(vec)
    ans = np.sqrt(np.sum(vec ** 2))
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


# @fantasy_timer
@njit
def dist_cos_sim(ele_i, ele_ic):
    # cosine similarity
    norm_a = np.linalg.norm(ele_i)
    norm_b = np.linalg.norm(ele_ic)
    ans = np.dot(ele_i, ele_ic) / (norm_a * norm_b)
    return float(ans)


# @fantasy_timer
# def dist_Mahalanobis(ele_i, ele_ic):
#     cov_matrix = np.cov()
#     return


# ------------------------------------------
# Distance between sets
# intermediate


name_intermediate = ['euclidean', 'manhattan', 'chebyshev',
                     'minkowski', 'cos_sim']
# Mahalanobis dist, Hamming dist, Jaccard similarity/dist


dist_intermediate = {
    'euclidean': dist_Euclidean,
    'manhattan': dist_Manhattan,
    'chebyshev': dist_Chebyshev,
    'minkowski': dist_Minkowski,
    'cos_sim': dist_cos_sim,
}


@njit
def alter_intermediate(ele_i, ele_ic, func='euclidean', p=3):
    if func == 'euclidean':
        return dist_Euclidean(ele_i - ele_ic)
    elif func == 'manhattan':
        return dist_Manhattan(ele_i - ele_ic)
    elif func == 'chebyshev':
        return dist_Chebyshev(ele_i - ele_ic)

    # if func in ['euclidean', 'manhattan', 'chebyshev']:
    #     return dist_intermediate[func](ele_i - ele_ic)
    elif func == 'minkowski':
        return dist_Minkowski(ele_i - ele_ic, p)
    return dist_cos_sim(ele_i, ele_ic)


# @fantasy_timer
# @numba.jit(nopython=True)
def Direct_halfway_min(ele_i, Si_c, func='euclidean', p=3):
    dist_fn = dist_intermediate[func]
    if func in ['euclidean', 'manhattan', 'chebyshev']:
        elements = [dist_fn(ele_i - ele_ic) for ele_ic in Si_c]
    elif func == 'minkowski':
        elements = [dist_fn(ele_i - ele_ic, p) for ele_ic in Si_c]
    elif func == 'cos_sim':
        elements = [dist_fn(ele_i, ele_ic) for ele_ic in Si_c]
    return min(elements)


def Direct_mediator(X_nA_y, idx_Si, func='euclidean', p=3):
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    if len(Sj) == 0 or len(Sj_c) == 0:
        return 0., 0.  # default if Sj is an empty set
    elements = [
        Direct_halfway_min(ele_i, Sj_c, func, p) for ele_i in Sj]
    return max(elements), sum(elements)


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
    if idx_Si is None:
        idx_Si = A_i == priv_val
    half_1, half_1avg = Direct_mediator(X_nA_y, idx_Si, func, p)
    half_2, half_2avg = Direct_mediator(X_nA_y, ~idx_Si, func, p)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


@fantasy_timer
def Direct_nonbin(X_nA_y, A_i, priv_val=1, idx_Sjs=None,
                  func='euclidean', p=3):
    if idx_Sjs is None:
        idx_Sjs = idx_marginalised(A_i, priv_val)
    half_mid = [Direct_mediator(
        X_nA_y, idx_Si, func, p) for idx_Si in idx_Sjs]
    half_pl_max, half_pl_avg = zip(*half_mid)
    n = len(X_nA_y)
    return max(half_pl_max), sum(half_pl_avg) / n


@fantasy_timer
def Direct_multiver(X_nA_y, A, priv_val=1, indices=None,
                    func='euclidean', p=3):
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
def sub_accelerator_smaler(X_yfx, A, idx_y_fx, i, m2,
                           func='euclidean', p=3):
    i_anchor = idx_y_fx[i]  # anchor's location after projection
    A_anchor = A[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]
    tmp = []

    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()<=g(xi,yi;w)
    j, num_j, min_js = i - 1, 0, np.finfo(np.float32).max
    # j = i - 1  # doesn't have to be compared with the anchor
    while num_j < m2:
        if j < 0:
            break
        idx_j = idx_y_fx[j]
        if A_anchor == A[idx_j]:  # set_belonging(A, i_anchor, idx_j):
            j -= 1
            continue

        curr = alter_intermediate(X_yfx_anchor, X_yfx[idx_j], func, p)
        if curr < min_js:
            min_js = curr
        num_j += 1
        j -= 1
        tmp.append(idx_j)
    # Find the minimum among them, recorded as d_min^s
    return min_js, tmp


@njit
def sub_accelerator_larger(X_yfx, A, idx_y_fx, i, m2,
                           func='euclidean', p=3):
    i_anchor = idx_y_fx[i]  # anchor's location after projection
    A_anchor = A[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]

    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()>=g(xi,yi;w)
    j, num_j, min_jr = i + 1, 0, np.finfo(np.float32).max
    # j = i + 1  # doesn't have to be compared with the anchor
    n = len(X_yfx)
    while num_j < m2:
        if j >= n:
            break
        idx_j = idx_y_fx[j]
        if A_anchor == A[idx_j]:  # set_belonging(A, i_anchor, idx_j):
            j += 1
            continue

        curr = alter_intermediate(X_yfx_anchor, X_yfx[idx_j], func, p)
        if curr < min_jr:
            min_jr = curr
        num_j += 1
        j += 1
    # Find the minimum among them, recorded as d_min^r
    return min_jr


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


@fantasy_timer
def AcceleCore_bin(X_yddot, A_i, m2, vec_w, func='euclidean', p=3):
    # Project data points onto a one-dimensional space
    proj = [projector(ele, vec_w) for ele in X_yddot]
    idx_y_fx = np.argsort(proj)

    n = X_yddot.shape[0]  # number of instances
    d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round
        min_js = sub_accelerator_smaler(
            X_yddot, A_i, idx_y_fx, i, m2, func, p)
        min_jr = sub_accelerator_larger(
            X_yddot, A_i, idx_y_fx, i, m2, func, p)
        # finally,
        tmp = min(min_js, min_jr)
        d_min.append(tmp)
    return max(d_min), sum(d_min)


@njit
def rand_uniform(a, b):
    # vec_w[i] = np.random.uniform(-tmp, tmp)
    return a + (b - a) * np.random.random()


# @fantasy_timer
@njit
def weight_generator(n_d):
    vec_w = np.empty(1 + n_d)
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

# @fantasy_timer
# @njit(parallel=True)
# def weight_gen_many(n, n_d):
#     out = np.empty((n, 1 + n_d))
#     for i in prange(n):
#         out[i] = weight_generator(n_d)
#     return out


@fantasy_timer
def Approx_bin(X_nA_y, A_i, priv_val=1, idx_Si=None,
               func='euclidean', m1=20, m2=8):
    if idx_Si is None:
        idx_Si = A_i == priv_val
    n, n_d = X_nA_y.shape
    d_max, d_avg = [], []
    for _ in range(m1):
        vec_w = weight_generator(n_d - 1)
    return
