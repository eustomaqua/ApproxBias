# coding: utf-8

import numpy as np
import pdb
from hfm.utils.verifiers import check_equal


def dist_vector(ele_i, ele_ic):
    from hfm.manf.dist_internal import (
        dist_Euclidean, dist_Manhattan, dist_Chebyshev, dist_Minkowski,
        avbl_Euclidean, avbl_Manhattan, avbl_Chebyshev, avbl_Minkowski,
        dist_cos_sim)

    vec = ele_i - ele_ic
    my1 = dist_Euclidean(vec)
    my2 = dist_Manhattan(vec)
    my3 = dist_Chebyshev(vec)
    my4 = dist_Minkowski(vec)

    ex1 = avbl_Euclidean(vec)  # ele_i, ele_ic)
    ex2 = avbl_Manhattan(vec)  # ele_i, ele_ic)
    ex3 = avbl_Chebyshev(vec)  # ele_i, ele_ic)
    ex4 = avbl_Minkowski(ele_i, ele_ic)

    # my1 = dist_Euclidean(ele_i, ele_ic)
    # my2 = dist_Manhattan(ele_i, ele_ic)  # faster
    # my3 = dist_Chebyshev(ele_i, ele_ic)  # faster
    # my4 = dist_Minkowski(ele_i, ele_ic)  # faster

    assert check_equal(ex1[0], my1)  # [0])
    assert check_equal(ex2[0], my2)  # [0])
    assert check_equal(ex3[0], my3)  # [0])
    assert check_equal(ex4[0], my4)  # [0])

    ans = dist_cos_sim(ele_i, ele_ic)  # ans, _ =
    assert 0 <= ans <= 1
    return


def dist_direct_part1(Si, Si_c):
    from hfm.manf.dist_internal import Direct_halfway_min
    t1 = Direct_halfway_min(Si[0], Si_c, 'euclidean')
    t2 = Direct_halfway_min(Si[0], Si_c, 'manhattan')
    t3 = Direct_halfway_min(Si[0], Si_c, 'chebyshev')
    t4 = Direct_halfway_min(Si[0], Si_c, 'minkowski')
    t5 = Direct_halfway_min(Si[0], Si_c, 'cos_sim')
    # t1, t2, t3, t4 = t1[0], t2[0], t3[0], t4[0]
    assert t1 >= 0 and t2 >= 0 and t3 >= 0 and t4 >= 0
    assert 0 <= t5 <= 1
    return


def dist_direct_part2(X_nA_y, A):
    from hfm.manf.dist_internal import Direct_marginalised
    _, na = A.shape
    nai = len(np.unique(A))
    indices = [[A[:, i] == j + 1 for j in range(nai)] for i in range(na)]
    ind_alt = [Direct_marginalised(A[:, i]) for i in range(A.shape[1])]
    # ind_alt = Direct_marginalised(A, priv_val=1)
    assert np.equal(indices, ind_alt).all()
    pdb.set_trace()
    return


def test_internal():
    n = 324  # 101,11
    ele_i = np.random.rand(n)
    ele_ic = np.random.rand(n)
    dist_vector(ele_i, ele_ic)

    nd = 17 + 240
    Si = np.random.rand(n, nd)
    Si_c = np.random.rand(n, nd)
    dist_direct_part1(Si, Si_c)

    nc = na = nai = 2
    nai = 3
    nd = 17
    X = np.random.rand(n, nd) * 10
    y = np.random.randint(nc, size=n)  # binary classification
    A = np.random.randint(nai, size=(n, na)) + 1
    X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
    dist_direct_part2(X_nA_y, A)
    return
