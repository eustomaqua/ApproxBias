# coding: utf-8


from typing import Optional
import math
import numpy as np
from numba import njit, prange
import pdb
from hfm.utils.verifiers import INF64, EPS64
from hfm.utils.decorators import fantasy_timer
from hfm.manf.renew_drt import (
    ArrayLike, PType, IndexLike, DTY_FLT, DTY_INT,
    _as_float_p, _lp_distance_rows, _aggregate_dmin,
    dual_norm_vec)


# ------------------------------------------
# Approx_bin


def Approx_bin():
    return


# ------------------------------------------
# Approx_nonbin


@njit(cache=True, parallel=True)
def _Approx_nonbin_sub(X_nA_y: np.ndarray, A_i: IndexLike, p: PType,
                       m1: int, m2: int, n_e: int):
    n, n_d = X_nA_y.shape  # n_d-1: number of non-sen-att(s)
    d_max = np.empty(m1, dtype=DTY_FLT)
    d_avg = np.empty(m1, dtype=DTY_FLT)
    return


@fantasy_timer
def Approx_nonbin(X_nA_y: np.ndarray, A_i: IndexLike, p: PType = 2.0,
                  *, m1: int = 25, m2: Optional[int] = None, n_e: int = 2):
    p = _as_float_p(p)
    if m2 is None:
        m2 = math.ceil(2.0 * math.log10(X_nA_y.shape[0]))
        m2 = max(1, int(m2))

    return


# ------------------------------------------
