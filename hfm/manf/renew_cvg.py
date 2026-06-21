# coding: utf-8


from typing import Optional, Tuple
import math
import numpy as np
from numba import njit, prange
import pdb

# from hfm.utils.verifiers import INF64, EPS64 #,CONST_ZERO
from hfm.utils.decorators import fantasy_timer
# from hfm.manf.renew_drt import (  # CONST_INF,
#     ArrayLike, PType, IndexLike, DTY_FLT, DTY_INT,
#     _as_float_p, dual_norm_vec, _lp_distance_rows,
#     _aggregate_dmin, dual_normalize)


from hfm.manf.renew_core import INF64, EPS64, DTY_FLT, DTY_INT
from hfm.manf.renew_core import ArrayLike, PType, IndexLike
from hfm.manf.renew_core import (
    _as_float_p, _lp_distance_rows, _aggregate_dmin,
    dual_normalize, orthogonal_weight, _determine_m2,
    hfmOUTCOME)  # ,dual_norm_vec)


# ------------------------------------------
# StratES


# @njit(cache=True)
# def orthogonal_weight(n_d, n_e=3):
#     for _ in range(n_d):
#         B = np.random.rand(n_d, n_d)
#         tmp = np.linalg.det(B)
#         if abs(tmp) > EPS64:
#             break  # 数值稳定判断
#     # 2. Gram-Schmidt (向量化版本)
#     A_T = B.T
#     eta = np.zeros((n_e, n_d))
#     # 第一个向量归一化
#     v = A_T[0]
#     eta[0] = v / np.linalg.norm(v)
#     # 后续向量
#     for i in range(1, n_e):
#         v = A_T[i].copy()
#         # 投影部分向量化: proj=(eta[:i] @v)
#         proj = eta[:i] @ v
#         v = v - proj @ eta[:i]
#         # 归一化
#         v = v / np.linalg.norm(v)
#         eta[i] = v
#     return eta


# @njit(cache=True, parallel=True)
# def _StratES_subproc_ver1(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                           vec_w: np.ndarray
#                           ) -> Tuple[np.ndarray, np.ndarray]:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]  # number of instances
#
#     d_min = np.empty(n, dtype=DTY_FLT)
#     scanned = np.zeros(n, dtype=DTY_INT)  # 'int')
#     for pos in prange(n):
#         # Set the anchor data point (xi,yi) in this round
#         k = order[pos]
#         ak = A_i[k]   # ak = Ai_ord[pos]
#         gk = proj[k]  # gk = proj_ord[pos]
#
#         best = INF64
#         left, righ = pos - 1, pos + 1
#         gap_left = gk - proj[order[left]] if left >= 0 else INF64
#         gap_righ = proj[order[righ]] - gk if righ < n else INF64
#         while gap_left < best or gap_righ < best:
#             # # If the smallest possible projected gap is already no
#             # # better than current best, then no unscanned point can
#             # # improve the answer.
#             # if gap_left >= best and gap_righ >= best:
#             #     break
#
#             # Move pointer now; same-group points can still certify
#             # that all further points on this side are even farther
#             # in projection.
#             if gap_left <= gap_righ:
#                 j = order[left]
#                 left -= 1
#                 gap_left = gk - proj[order[left]] if left >= 0 else INF64
#             else:
#                 j = order[righ]
#                 righ += 1
#                 gap_righ = proj[order[righ]] - gk if righ < n else INF64
#
#             if A_i[j] == ak:
#                 continue
#             curr = _lp_distance_rows(X_nA_y, k, j, p)
#             scanned[k] += 1
#             if curr < best:
#                 best = curr
#         d_min[k] = best
#     return d_min, scanned


@njit(cache=True, parallel=True)
def _StratES_subproc_ver2(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                          vec_w: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    proj = X_nA_y @ vec_w
    order = np.argsort(proj)
    n = X_nA_y.shape[0]  # number of instances
    proj_ord = proj[order]
    Ai_ord = A_i[order]

    d_min = np.empty(n, dtype=DTY_FLT)
    scanned = np.zeros(n, dtype=DTY_INT)  # 'int')
    for pos in prange(n):
        # Set the anchor data point (xi,yi) in this round
        k = order[pos]
        ak = Ai_ord[pos]    # ak = A_i[k]
        gk = proj_ord[pos]  # gk = proj[k]

        best = INF64
        left, righ = pos - 1, pos + 1
        gap_left = gk - proj_ord[left] if left >= 0 else INF64
        gap_righ = proj_ord[righ] - gk if righ < n else INF64
        while gap_left < best or gap_righ < best:
            if gap_left <= gap_righ:
                jp = left  # j = order[left]
                left -= 1
                gap_left = gk - proj_ord[left] if left >= 0 else INF64
            else:
                jp = righ  # j = order[righ]
                righ += 1
                gap_righ = proj_ord[righ] - gk if righ < n else INF64

            if Ai_ord[jp] == ak:  # if A_i[j] == ak:
                continue
            j = order[jp]
            curr = _lp_distance_rows(X_nA_y, k, j, p)
            scanned[k] += 1
            if curr < best:
                best = curr
        d_min[k] = best
    return d_min, scanned


# @fantasy_timer
# def StratES_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0
#                    ) -> hfmOUTCOME:
#     p = _as_float_p(p)
#     n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
#     W = orthogonal_weight(n_d, n_e=1)
#     vec_w = dual_normalize(W[0], p)
#     # d_min, _ = _StratES_subproc_ver1(X_nA_y, A_i, p, vec_w)
#     # d_min, _ = _StratES_subproc_ver2(X_nA_y, A_i, p, W[0])
#     d_min, _ = _StratES_subproc_ver2(X_nA_y, A_i, p, vec_w)
#     t_max, t_avg = _aggregate_dmin(d_min)
#     return t_max, t_avg


# @njit(parallel=True, cache=True)
# def _StratES_subproc_ver3(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                           vec_w: np.ndarray) -> np.ndarray:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]  # number of instances
#     Ai_ord = A_i[order]
#     proj_ord = proj[order]
#
#     d_min = np.empty(n, dtype=DTY_FLT)
#     for pos in prange(n):
#         # Set the anchor data point (xi, yi) in this round
#         k = order[pos]
#         ak, gk = Ai_ord[pos], proj_ord[pos]  # A_i[k], proj[k]
#         min_js = min_jr = INF64
#
#         for jp in range(pos - 1, -1, -1):
#             if Ai_ord[jp] != ak:                 # A_i[order[jp]]
#                 if gk - proj_ord[jp] >= min_js:  # proj[order[jp]]
#                     break
#                 j = order[jp]
#                 curr = _lp_distance_rows(X_nA_y, k, j, p)
#                 if curr < min_js:
#                     min_js = curr
#
#         for jp in range(pos + 1, n, 1):
#             if Ai_ord[jp] != ak:                 # A_i[order[jp]]
#                 if proj_ord[jp] - gk >= min_jr:  # proj[order[jp]]
#                     break
#                 j = order[jp]
#                 curr = _lp_distance_rows(X_nA_y, k, j, p)
#                 if curr < min_jr:
#                     min_jr = curr
#         d_min[k] = min(min_js, min_jr)
#     return d_min


# @njit(parallel=True, cache=True)
# def _StratES_subproc_ver5(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                           vec_w: np.ndarray) -> np.ndarray:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]  # number of instances
#     Ai_ord = A_i[order]
#     proj_ord = proj[order]
#
#     d_min = np.empty(n, dtype=DTY_FLT)
#     for pos in prange(n):
#         # Set the anchor data point (xi, yi) in this round
#         k = order[pos]
#         ak, gk = Ai_ord[pos], proj_ord[pos]  # A_i[k], proj[k]
#         min_js = min_jr = INF64
#
#         for jp in range(pos - 1, -1, -1):
#             # if Ai_ord[jp] != ak:           # A_i[order[jp]]
#             if Ai_ord[jp] == ak:
#                 continue
#             if gk - proj_ord[jp] >= min_js:  # proj[order[jp]]
#                 break
#             curr = _lp_distance_rows(X_nA_y, k, order[jp], p)
#             if curr < min_js:
#                 min_js = curr
#
#         for jp in range(pos + 1, n, 1):
#             # if Ai_ord[jp] != ak:           # A_i[order[jp]]
#             if Ai_ord[jp] == ak:
#                 continue
#             if proj_ord[jp] - gk >= min_jr:  # proj[order[jp]]
#                 break
#             curr = _lp_distance_rows(X_nA_y, k, order[jp], p)
#             if curr < min_jr:
#                 min_jr = curr
#         d_min[k] = min(min_js, min_jr)
#     return d_min


@njit(parallel=True, cache=True)
def _StratES_subproc_ver4(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                          vec_w: np.ndarray) -> np.ndarray:
    proj = X_nA_y @ vec_w
    order = np.argsort(proj)
    n = X_nA_y.shape[0]  # number of instances
    Ai_ord = A_i[order]
    proj_ord = proj[order]

    d_min = np.empty(n, dtype=DTY_FLT)
    for pos in prange(n):
        # Set the anchor data point (xi, yi) in this round
        k = order[pos]
        ak, gk = Ai_ord[pos], proj_ord[pos]  # A_i[k], proj[k]
        min_js = min_jr = INF64

        for jp in range(pos - 1, -1, -1):
            if Ai_ord[jp] == ak:
                continue
            # if Ai_ord[jp] != ak:           # A_i[order[jp]]
            if gk - proj_ord[jp] >= min_js:  # proj[order[jp]]
                break
            j = order[jp]
            curr = _lp_distance_rows(X_nA_y, k, j, p)
            if curr < min_js:
                min_js = curr

        for jp in range(pos + 1, n, 1):
            if Ai_ord[jp] == ak:
                continue
            # if Ai_ord[jp] != ak:           # A_i[order[jp]]
            if proj_ord[jp] - gk >= min_jr:  # proj[order[jp]]
                break
            j = order[jp]
            curr = _lp_distance_rows(X_nA_y, k, j, p)
            if curr < min_jr:
                min_jr = curr
        # d_min[k] = min(min_js, min_jr)
        d_min[pos] = min(min_js, min_jr)
    return d_min


@fantasy_timer
def StratES_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0
                   )->hfmOUTCOME:
    p = _as_float_p(p)
    n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
    W = orthogonal_weight(n_d, n_e=1)
    vec_w = dual_normalize(W[0], p)
    # d_min = _StratES_subproc_ver3(X_nA_y, A_i, p, vec_w)
    d_min = _StratES_subproc_ver4(X_nA_y, A_i, p, vec_w)
    # d_min = _StratES_subproc_ver5(X_nA_y, A_i, p, vec_w)
    return _aggregate_dmin(d_min)


# ------------------------------------------
# StratRA


@njit(cache=True)
def _reduce_min_axis0(arr):
    m, n = arr.shape
    out = np.empty(n, dtype=DTY_FLT)
    for j in range(n):
        out[j] = arr[:, j].min()

    # for j in range(n):
    #     out[j] = arr[0, j]
    # for i in range(1, m):
    #     for j in range(n):
    #         v = arr[i, j]
    #         if v < out[j]:
    #             out[j] = v
    return out


# @njit(cache=True)
# def AcceleCoreBack_ver1(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                         vec_w: np.ndarray, m2: int) -> np.ndarray:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]  # number of instances
#
#     dt_min = np.empty(n, dtype=DTY_FLT)
#     for pos in range(n):
#         k = order[pos]
#         ak = A_i[k]
#         best = INF64
#
#         # smaller /left side
#         count = 0
#         jp = pos - 1
#         while jp >= 0 and count < m2:
#             j = order[jp]
#             if A_i[j] != ak:
#                 curr = _lp_distance_rows(X_nA_y, k, j, p)
#                 if curr < best:
#                     best = curr
#                 count += 1
#             jp -= 1
#
#         # larger /right side
#         count = 0
#         jp = pos + 1
#         while jp < n and count < m2:
#             j = order[jp]
#             if A_i[j] != ak:
#                 curr = _lp_distance_rows(X_nA_y, k, j, p)
#                 if curr < best:
#                     best = curr
#                 count += 1
#             jp += 1
#
#         dt_min[k] = best
#     return dt_min


@njit(cache=True)
def AcceleCoreBack_ver2(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                        vec_w: np.ndarray, m2: int,
                        best_all: np.ndarray) -> None:
    proj = X_nA_y @ vec_w
    order = np.argsort(proj)
    n = X_nA_y.shape[0]  # number of instances

    Ai_ord = A_i[order]
    # proj_ord = proj[order]
    for pos in range(n):
        k = order[pos]
        ak = Ai_ord[pos]
        # gk = proj_ord[pos]

        # Start from the global best found by previous projections.
        # This does NOT create early stopping because we still scan
        # the fixed m2 candidates. It only avoids unnecessary assignments.
        best = INF64  # best = best_all[k]

        count = 0
        jp = pos - 1
        while jp >= 0 and count < m2:
            # gap = gk - proj_ord[jp]
            # if gap >= best:
            #     break
            if Ai_ord[jp] != ak:
                j = order[jp]
                curr = _lp_distance_rows(X_nA_y, k, j, p)
                if curr < best:
                    best = curr
                count += 1
            jp -= 1

        count = 0
        jp = pos + 1
        while jp < n and count < m2:
            # gap = proj_ord[jp] - gk
            # if gap >= best:
            #     break
            if Ai_ord[jp] != ak:
                j = order[jp]
                curr = _lp_distance_rows(X_nA_y, k, j, p)
                if curr < best:
                    best = curr
                count += 1
            jp += 1

        best_all[k] = best


@njit(cache=True)
def _build_prev_next_outlier(Ai_ord: np.ndarray, n_grp: int
                             )-> Tuple[np.ndarray, np.ndarray]:
    n = Ai_ord.shape[0]
    prev_not = np.empty((n_grp, n), dtype=DTY_INT)
    next_not = np.empty((n_grp, n), dtype=DTY_INT)
    for g in range(n_grp):
        last = -1
        for pos in range(n):
            prev_not[g, pos] = last
            if Ai_ord[pos] != g:
                last = pos
        nxt = n
        for pos in range(n - 1, -1, -1):
            next_not[g, pos] = nxt
            if Ai_ord[pos] != g:
                nxt = pos
    return prev_not, next_not


# @njit(cache=True)  # ,parallel=True)
# def AcceleCoreBack_ver3(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                         vec_w: np.ndarray, m2: int,
#                         best_all: np.ndarray, n_grp: int) -> None:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]
#
#     Ai_ord = A_i[order]
#     prev_not, next_not = _build_prev_next_outlier(Ai_ord, n_grp)
#     for pos in range(n):
#         k = order[pos]
#         ak = Ai_ord[pos]
#         best = best_all[k]
#
#         # left side: jump directly to cross-group candidates
#         count = 0
#         jp = prev_not[ak, pos]
#         while jp >= 0 and count < m2:
#             j = order[jp]
#             curr = _lp_distance_rows(X_nA_y, k, j, p)
#             if curr < best:
#                 best = curr
#             count += 1
#             jp = prev_not[ak, jp]
#
#         # right side: jump directly to cross-group candidates
#         count = 0
#         jp = next_not[ak, pos]
#         while jp < n and count < m2:
#             j = order[jp]
#             curr = _lp_distance_rows(X_nA_y, k, j, p)
#             if curr < best:
#                 best = curr
#             count += 1
#             jp = next_not[ak, jp]
#
#         if best < best_all[k]:
#             best_all[k] = best


@njit(cache=True)
def AcceleCoreBack_ver4(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                        vec_w: np.ndarray, m2: int, n_grp: int,
                        best_all: np.ndarray, row: int) -> None:
    proj = X_nA_y @ vec_w
    order = np.argsort(proj)
    n = X_nA_y.shape[0]

    Ai_ord = A_i[order]
    prev_not, next_not = _build_prev_next_outlier(Ai_ord, n_grp)
    # local = np.empty(n, dtype=DTY_FLT)
    # # for i in range(n):
    # #     local[i] = INF64
    for pos in range(n):
        k = order[pos]
        ak = Ai_ord[pos]
        best = INF64

        count = 0
        jp = prev_not[ak, pos]
        while jp >= 0 and count < m2:
            j = order[jp]
            curr = _lp_distance_rows(X_nA_y, k, j, p)
            if curr < best:
                best = curr
            count += 1
            jp = prev_not[ak, jp]

        count = 0
        jp = next_not[ak, pos]
        while jp < n and count < m2:
            j = order[jp]
            curr = _lp_distance_rows(X_nA_y, k, j, p)
            if curr < best:
                best = curr
            count += 1
            jp = next_not[ak, jp]

        best_all[row, k] = best
    #     local[k] = best
    # return local


@njit(parallel=True, cache=True)
def _StratRA_core(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                  m1: int, m2: int, n_e: int, n_grp: int) -> hfmOUTCOME:
    n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
    # dt_min = np.empty((m1, n), dtype=DTY_FLT)
    # for v in prange(m1):
    #     # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
    #     W = orthogonal_weight(n_d, n_e)
    #     tmp = np.empty((n_e, n), dtype=DTY_FLT)
    #     for k in range(n_e):
    #         vec_w = W[k] / dual_norm_vec(W[k], p)
    #         # vec_w = dual_normalize(W[k], p)
    #         tmp[k] = AcceleCoreBack_ver1(X_nA_y, A_i, p, vec_w, m2)
    #     dt_min[v] = _reduce_min_axis0(tmp)
    # fin = _reduce_min_axis0(dt_min)
    # return _aggregate_dmin(fin)

    # best_all = np.empty(n, dtype=DTY_FLT)
    # for i in range(n):
    #     best_all[i] = INF64
    # for v in prange(m1):
    #     W = orthogonal_weight(n_d, n_e)
    #     for k in range(n_e):
    #         # vec_w = W[k]  # vec_w = W[k] / dual_norm_vec(W[k], p)
    #         # AcceleCoreBack_ver2(X_nA_y, A_i, p, vec_w, m2, best_all)
    #
    #         # Pure StratRA does not need dual normalisation. Positive
    #         # rescaling of w does not change argsort(X @ w).
    #         AcceleCoreBack_ver3(X_nA_y, A_i, p, W[k], m2, best_all, n_grp)
    # return _aggregate_dmin(best_all)

    W_all = np.empty((m1, n_e, n_d), dtype=DTY_FLT)
    for v in range(m1):
        W_all[v] = orthogonal_weight(n_d, n_e)
    total = m1 * n_e
    best_all = np.empty((total, n), dtype=DTY_FLT)
    # for v in prange(m1):
    #     W = orthogonal_weight(n_d, n_e)
    #     for k in range(n_e):
    #         t = v * n_e + k  # v = t//n_e, k=t-v*n_e
    for t in prange(total):
        v = t // n_e
        k = t % n_e
        vec_w = W_all[v, k]  # W[k]
        # pdb.set_trace()
        # best_all[t] = AcceleCoreBack_ver4(X_nA_y, A_i, p, vec_w, m2, n_grp)
        AcceleCoreBack_ver4(X_nA_y, A_i, p, vec_w, m2, n_grp, best_all, t)

    fin = np.empty(n, dtype=DTY_FLT)
    for i in range(n):
        best = best_all[0, i]  # INF64
        for t in range(1, total):
            val = best_all[t, i]
            if val < best:
                best = val
        fin[i] = best
    # fin = _reduce_min_axis0(best_all)
    return _aggregate_dmin(fin)


@fantasy_timer
def StratRA_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
                   # m1: int = 25, m2: Optional[int] = None,
                   # n_e: int = 3) -> hfmOUTCOME:  # *,
                   m1: int = 20, m2: int = 8, n_e: int = 2) -> hfmOUTCOME:
    p = _as_float_p(p)
    # if m2 is None:
    #     m2 = math.ceil(2.0 * math.log10(X_nA_y.shape[0]))
    #     m2 = max(1, int(m2))
    # # m2 = _determine_m2(X_nA_y.shape[0], m2)

    # # if m1 <= 0:
    # #     raise ValueError("m1 must be positive.")
    # # if m2 <= 0:
    # #     raise ValueError("m2 must be positive.")
    # # if n_e <= 0:
    # #     raise ValueError("n_e must be positive.")
    # return _StratRA_core(X_nA_y, A_i, p, m1, m2, n_e)

    B_i = np.asarray(A_i).reshape(-1)
    _, A_code = np.unique(B_i, return_inverse=True)
    A_code = A_code.astype(DTY_INT)
    n_grp = int(A_code.max()) + 1
    # pdb.set_trace()
    # n_grp = len(np.unique(A_i))
    # if n_groups < 2:
    #     raise ValueError("At least two sensitive groups are required.")
    return _StratRA_core(X_nA_y, A_code, p, m1, m2, n_e, n_grp)
    # return _StratRA_core(X_nA_y, A_i, p, m1, m2, n_e)  # , n_grp)


# @njit(parallel=True, cache=True)
# def _StratRA_core_alt(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                       m1: int, m2: int, n_e: int) -> hfmOUTCOME:
#     n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
#     dt_min = np.empty((m1, n), dtype=DTY_FLT)
#     for v in prange(m1):
#         # Take two orthogonal vectors $w_k\in[-1,+1]^{1+n_x}$
#         W = orthogonal_weight(n_d, n_e)
#
#         tmp = np.empty((n_e, n), dtype=DTY_FLT)
#         # tmp = np.full((n_e, n), INF64, dtype=DTY_FLT)
#         for k in range(n_e):
#             AcceleCoreBack_ver2(X_nA_y, A_i, p, W[k], m2, tmp[k])
#             # tmp[k] = AcceleCoreBack_ver1(X_nA_y, A_i, p, W[k], m2)
#         dt_min[v] = _reduce_min_axis0(tmp)
#     fin = _reduce_min_axis0(dt_min)
#     return _aggregate_dmin(fin)
#
#
# @fantasy_timer
# def StratRA_nonbin_alt(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
#                        m1: int = 20, m2: int = 8, n_e: int = 2)->hfmOUTCOME:
#     p = _as_float_p(p)
#     return _StratRA_core_alt(X_nA_y, A_i, p, m1, m2, n_e)
#
#
# # @njit(parallel=True, cache=True)
# # def _StratRA_core_alt(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
# #                        m1: int, m2: int, n_e: int, n_grp: int)->hfmOUTCOME:
# #     n, n_d = X_nA_y.shape
# #     W_all = np.empty((m1, n_e, n_d), dtype=DTY_FLT)
# #     for v in range(m1):
# #         W_all[v] = orthogonal_weight(n_d, n_e)
# #     total = m1 * n_e
# #     dt_min = np.empty((total, n), dtype=DTY_FLT)
# #     for t in prange(total):
# #         v = t // n_e
# #         k = t % n_e
# #         vec_w = W_all[v, k]
# #         AcceleCoreBack_ver2(X_nA_y, A_i, p, vec_w, m2, dt_min[t])
# #
# #     # fin = np.empty(n, dtype=DTY_FLT)
# #     # for i in range(n):
# #     #     best = dt_min[0, i]
# #     #     for t in range(1, total):
# #     #         val = dt_min[t, i]
# #     #         if val < best:
# #     #             best = val
# #     #     fin[i] = best
# #     fin = _reduce_min_axis0(dt_min)
# #     return _aggregate_dmin(fin)
# #
# #
# # @fantasy_timer
# # def StratRA_nonbin_alt2(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
# #                         m1: int=20, m2: int=8, n_e: int = 2)->hfmOUTCOME:
# #     p = _as_float_p(p)
# #     # B_i = np.asarray(A_i)  # .reshape(-1)
# #     _, A_code = np.unique(A_i, return_inverse=True)  # B_i,
# #     A_code = A_code.astype(DTY_INT)
# #     n_grp = int(A_code.max()) + 1
# #     return _StratRA_core_alt(X_nA_y, A_code, p, m1, m2, n_e, n_grp)


# @njit
# def _build_outlier_alt(A_i: np.ndarray  # , n_grp: int
#                        )-> Tuple[np.ndarray, np.ndarray]:
#     n = A_i.shape[0]  # Ai_ord.shape[0]
#     prev_not = np.empty(n, dtype=DTY_INT)
#     next_not = np.empty(n, dtype=DTY_INT)
#     for pos in range(n):
#         ak = A_i[pos]
#         last = pos - 1
#         while last >= 0 and A_i[last] == ak:
#             last -= 1
#         prev_not[pos] = last
#         nxt = pos + 1
#         while nxt < n and A_i[nxt] == ak:
#             nxt += 1
#         next_not[pos] = nxt
#     return prev_not, next_not
#
#
# @njit
# def AcceleCoreBack_ver5(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                         vec_w: np.ndarray, m2: int,  # n_grp: int,
#                         best_all: np.ndarray, row: int) -> None:
#     proj = X_nA_y @ vec_w
#     order = np.argsort(proj)
#     n = X_nA_y.shape[0]
#     Ai_ord = A_i[order]
#     prev_not, next_not = _build_outlier_alt(Ai_ord)
#     # pdb.set_trace()
#     for pos in range(n):
#         k = order[pos]
#         ak = Ai_ord[pos]
#         best = INF64
#         # count = 0
#         # jp = prev_not[pos]
#         # while jp >= 0 and count < m2:
#         #     j = order[jp]
#         #     curr = _lp_distance_rows(X_nA_y, k, j, p)
#         #     if curr < best:
#         #         best = curr
#         #     count += 1
#         #     jp = prev_not[jp]
#         # count = 0
#         # jp = next_not[pos]
#         # while jp < n and count < m2:
#         #     j = order[jp]
#         #     curr = _lp_distance_rows(X_nA_y, k, j, p)
#         #     if curr < best:
#         #         best = curr
#         #     count += 1
#         #     jp = next_not[jp]
#         best_all[row, k] = best
# # @njit(parallel=True)
# def _StratRA_core_alt(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
#                       m1: int, m2: int, n_e: int)->hfmOUTCOME:
#     n, n_d = X_nA_y.shape  # n_d-1: #non-sen-att
#     W_all = np.empty((m1, n_e, n_d), dtype=DTY_FLT)
#     for v in range(m1):
#         W_all[v] = orthogonal_weight(n_d, n_e)
#     total = m1 * n_e
#     dt_min = np.empty((total, n), dtype=DTY_FLT)
#     for t in range(total):
#         v = t // n_e
#         k = t % n_e
#         vec_w = W_all[v, k]
#         AcceleCoreBack_ver5(X_nA_y, A_i, p, vec_w, m2, dt_min, t)
#     fin = _reduce_min_axis0(dt_min)
#     return _aggregate_dmin(fin)
# @fantasy_timer
# def StratRA_nonbin_alt(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
#                        m1: int = 20, m2: int = 8, n_e: int = 2)->hfmOUTCOME:
#     p = _as_float_p(p)
#     return _StratRA_core_alt(X_nA_y, A_i, p, m1, m2, n_e)


# ------------------------------------------
