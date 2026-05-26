# coding: utf-8

import numpy as np
# import numba


# ---------------------
# Constants


CONST_ZERO = 1e-13
CONST_DIFF = 1e-7


DTY_FLT = 'float'
DTY_INT = 'int'
DTY_BOL = 'bool'
DTY_PLT = '.pdf'

GAP_INF = 2 ** 31 - 1
GAP_MID = 1e8  # 1e16
GAP_NAN = 1e-16

# INF64 = np.finfo(np.float32).max
INF64 = np.float64(1e308)
EPS64 = 1e-12  # np.float64(1e-12)
# INF64 = np.float64(np.inf)


# ---------------------
# Helper functions


def check_belong(tmp, *args):
    for v in args:
        if isinstance(tmp, v):
            return True
    return False


def check_zero(tmp, diff=CONST_ZERO):
    if not check_belong(tmp, list, tuple):
        return tmp if tmp != 0. else diff
    tmp = [check_zero(i, diff) for i in tmp]
    return all(tmp)


def check_sign(x, diff=CONST_ZERO):
    if abs(x) > CONST_ZERO:
        return x
    elif x > 0:
        return CONST_ZERO
    elif x < 0:
        return -CONST_ZERO
    return 0.0


def non_negative(tmp):
    if check_belong(tmp, list, tuple, np.ndarray):
        return [non_negative(i) for i in tmp]
    return tmp if tmp >= 0 else 0.


def check_equal(tmp_a, tmp_b, diff=CONST_DIFF):
    flag_a = check_belong(tmp_a, list, tuple, set)
    flag_b = check_belong(tmp_b, list, tuple, set)
    if not (flag_a or flag_b):
        return True if abs(tmp_a - tmp_b) < diff else False

    if flag_a and check_belong(tmp_b, int, float):
        tmp = [abs(i - tmp_b) < diff for i in tmp_a]
    elif flag_b and check_belong(tmp_a, int, float):
        tmp = [abs(tmp_a - i) < diff for i in tmp_b]
    elif flag_a and flag_b:
        tmp = [abs(i - j) < diff for i, j in zip(tmp_a, tmp_b)]
    return all(tmp)


# ---------------------
# Partial ordering


def poset_nolessthan(ga, gb):    # >=
    tmp = [i >= j or check_equal(
        i, j) for i, j in zip(ga, gb)]
    return all(tmp)


def poset_nomorethan(ga, gb):    # <=
    tmp = [i <= j or check_equal(
        i, j) for i, j in zip(ga, gb)]
    return all(tmp)


# ---------------------
#


def unique_column(nb_col, alphabet=None):
    # i.e., generate_unique_column_name
    # alphabet = [chr(i) for i in range(97, 123)]  # 'a' etc.
    # alphabet = [chr(i) for i in range(65, 91)]  ## 'A' etc.
    # import string
    # string.ascii_lowercase

    if alphabet is None:
        alphabet = [chr(i) for i in range(65, 91)]
    if nb_col <= 26:
        return alphabet[: nb_col]

    double = [i + j for i in alphabet for j in alphabet]
    index = nb_col - 26
    if index <= 26**2:
        return alphabet + double[: index]

    triple = [i + j + k for i in alphabet for j in alphabet for k in alphabet]
    index = nb_col - 26 - 26**2
    if index <= 26**3:
        return alphabet + double + triple[: index]

    return list()


# @numba.jit(nopython=True)
# def judge_transform_need(y):
#     vY = sorted(set(y))  # list(set(y))
#     dY = len(vY)
#     if dY == 2 and (-1 in vY) and (1 in vY):
#         dY = 1
#     return vY, dY  # 2, or ...


# def judge_mathcal_Y(nc=1):
#     # vY: list(range(nc)) if nc >= 2 else [-1, +1]
#     if nc == 1:
#         return [-1, +1]
#     return list(range(nc))


# def random_seed_generator(psed='fixed_tseed'):  # _tim
#     if (psed is not None) or (not isinstance(psed, int)):
#         import time
#         psed = int(time.time() * GAP_MID % GAP_INF)
#     prng = np.random.RandomState(seed=psed)
#     return psed, prng


# ---------------------
#
