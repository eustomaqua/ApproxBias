# coding: utf-8

import numpy as np
import numba
from hfm.utils.decorators import fantasy_timer
from scipy.spatial import distance


# ------------------------------------------
# Euclidean metric

def dist_Euclidean(vec):
    ans = np.sqrt(np.sum(vec ** 2))
    # ans = np.linalg.norm(vec)
    return float(ans)


def dist_Manhattan(vec):
    # vec = ele_i - ele_ic
    # l1_norm_alt =
    ans = np.sum(np.abs(vec))
    return float(ans)


def dist_Chebyshev(vec):
    # vec = ele_i - ele_ic
    # alt=np.max(np.abs(a - b))
    ans = np.max(np.abs(vec))
    return float(ans)


def dist_Minkowski(vec, p=3):
    ans = np.sum(np.abs(vec ** p))
    ans = np.power(ans, 1. / p)
    return float(ans)


# @fantasy_timer
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


# ------------------------------------------
# Distance between sets
# intermediate


dist_intermediate = {
    'euclidean': dist_Euclidean,
    'manhattan': dist_Manhattan,
    'chebyshev': dist_Chebyshev,
    'minkowski': dist_Minkowski,
    'cos_sim': dist_cos_sim,
}


# @fantasy_timer
# @numba.jit(nopython=True)
def Direct_halfway_min(ele_i, Si_c, func='euclidean'):
    dist_fn = dist_intermediate[func]
    if func == 'cos_sim':
        elements = [dist_fn(ele_i, ele_ic) for ele_ic in Si_c]
        return min(elements)
    elements = [dist_fn(ele_i - ele_ic) for ele_ic in Si_c]
    return min(elements)


def Direct_mediator(X_nA_y, idx_Si, func='euclidean'):
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    if len(Sj) == 0 or len(Sj_c) == 0:
        return 0., 0.  # default if Sj is an empty set
    elements = [Direct_halfway_min(ele_i, Sj_c, func) for ele_i in Sj]
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
               func='euclidean'):
    if idx_Si is None:
        idx_Si = A_i == priv_val
    half_1, half_1avg = Direct_mediator(X_nA_y, idx_Si, func)
    half_2, half_2avg = Direct_mediator(X_nA_y, ~idx_Si, func)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


@fantasy_timer
def Direct_nonbin(X_nA_y, A_i, priv_val=1, idx_Sjs=None,
                  func='euclidean'):
    if idx_Sjs is None:
        idx_Sjs = idx_marginalised(A_i, priv_val)
    half_mid = [Direct_mediator(
        X_nA_y, idx_Si, func) for idx_Si in idx_Sjs]
    half_pl_max, half_pl_avg = zip(*half_mid)
    n = len(X_nA_y)
    return max(half_pl_max), sum(half_pl_avg) / n


@fantasy_timer
def Direct_multiver(X_nA_y, A, priv_val=1, indices=None):
    n_a = A.shape[1]
    if indices is None:  # that is, idx_Ai_Sjs
        indices = [
            idx_marginalised(A[:, i], priv_val) for i in range(n_a)]
    half_mid = [Direct_nonbin(
        X_nA_y, A[:, i], priv_val, indices[i]) for i in range(n_a)]
    half_mid, half_ut = zip(*half_mid)
    half_pl_max, half_pl_avg = zip(*half_mid)
    return max(half_pl_max), sum(half_pl_avg) / n_a, (
        half_pl_max, half_pl_avg, half_ut)


# ------------------------------------------
#
