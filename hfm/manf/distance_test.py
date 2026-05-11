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


def dist_direct_part2(X_nA_y, A, indices):
    # from hfm.manf.dist_internal import Direct_marginalised
    from hfm.manf.dist_internal import (
        Direct_mediator, idx_marginalised, Direct_bin)

    ind_alt = [idx_marginalised(A[:, i]) for i in range(A.shape[1])]
    # ind_alt = Direct_marginalised(A, priv_val=1)
    assert np.equal(indices, ind_alt).all()

    idx = indices[1][0]
    t1 = Direct_mediator(X_nA_y, idx, 'euclidean')
    t2 = Direct_mediator(X_nA_y, idx, 'manhattan')
    t3 = Direct_mediator(X_nA_y, idx, 'chebyshev')
    t4 = Direct_mediator(X_nA_y, idx, 'minkowski')
    t5 = Direct_mediator(X_nA_y, idx, 'cos_sim')
    # t1, t2, t3, t4, t5 = t1[0], t2[0], t3[0], t4[0], t5[0]
    assert all([i[0] >= 0 for i in [t1, t2, t3, t4]])
    assert 0 <= t5[0] <= 1

    priv = 1
    t1 = Direct_bin(X_nA_y, A[:, 1], priv, func='euclidean')
    t2 = Direct_bin(X_nA_y, A[:, 1], priv, func='manhattan')
    t3 = Direct_bin(X_nA_y, A[:, 1], priv, func='chebyshev')
    t4 = Direct_bin(X_nA_y, A[:, 1], priv, func='minkowski')
    t5 = Direct_bin(X_nA_y, A[:, 1], priv, func='cos_sim')

    m1 = Direct_bin(X_nA_y, A[:, 1], priv, ~idx, 'euclidean')
    m2 = Direct_bin(X_nA_y, A[:, 1], priv, ~idx, 'manhattan')
    m3 = Direct_bin(X_nA_y, A[:, 1], priv, ~idx, 'chebyshev')
    m4 = Direct_bin(X_nA_y, A[:, 1], priv, ~idx, 'minkowski')
    m5 = Direct_bin(X_nA_y, A[:, 1], priv, ~idx, 'cos_sim')

    assert np.equal(t1[0], m1[0]).all()
    assert np.equal(t2[0], m2[0]).all()
    assert np.equal(t3[0], m3[0]).all()
    assert np.equal(t4[0], m4[0]).all()
    assert np.equal(t5[0], m5[0]).all()
    return


def dist_direct_part3(X_nA_y, A, indices):
    from hfm.manf.dist_internal import Direct_nonbin, Direct_multiver
    priv = 1

    t1 = Direct_nonbin(X_nA_y, A[:, 1], priv, func='euclidean')
    t2 = Direct_nonbin(X_nA_y, A[:, 1], priv, func='manhattan')
    t3 = Direct_nonbin(X_nA_y, A[:, 1], priv, func='chebyshev')
    t4 = Direct_nonbin(X_nA_y, A[:, 1], priv, func='minkowski')
    t5 = Direct_nonbin(X_nA_y, A[:, 1], priv, func='cos_sim')

    m1 = Direct_nonbin(X_nA_y, A[:, 1], priv, indices[1], 'euclidean')
    m2 = Direct_nonbin(X_nA_y, A[:, 1], priv, indices[1], 'manhattan')
    m3 = Direct_nonbin(X_nA_y, A[:, 1], priv, indices[1], 'chebyshev')
    m4 = Direct_nonbin(X_nA_y, A[:, 1], priv, indices[1], 'minkowski')
    m5 = Direct_nonbin(X_nA_y, A[:, 1], priv, indices[1], 'cos_sim')

    assert np.equal(t1[0], m1[0]).all()
    assert np.equal(t2[0], m2[0]).all()
    assert np.equal(t3[0], m3[0]).all()
    assert np.equal(t4[0], m4[0]).all()
    assert np.equal(t5[0], m5[0]).all()

    t1 = Direct_multiver(X_nA_y, A, priv, func='euclidean')
    t2 = Direct_multiver(X_nA_y, A, priv, func='manhattan')
    t3 = Direct_multiver(X_nA_y, A, priv, func='chebyshev')
    t4 = Direct_multiver(X_nA_y, A, priv, func='minkowski')
    t5 = Direct_multiver(X_nA_y, A, priv, func='cos_sim')

    m1 = Direct_multiver(X_nA_y, A, priv, indices, 'euclidean')
    m2 = Direct_multiver(X_nA_y, A, priv, indices, 'manhattan')
    m3 = Direct_multiver(X_nA_y, A, priv, indices, 'chebyshev')
    m4 = Direct_multiver(X_nA_y, A, priv, indices, 'minkowski')
    m5 = Direct_multiver(X_nA_y, A, priv, indices, 'cos_sim')

    # tn faster
    assert np.equal(t1[0][:2], m1[0][:2]).all()
    assert np.equal(t2[0][:2], m2[0][:2]).all()
    assert np.equal(t3[0][:2], m3[0][:2]).all()
    assert np.equal(t4[0][:2], m4[0][:2]).all()
    assert np.equal(t5[0][:2], m5[0][:2]).all()
    return


def drt_part4_bin(X_nA_y, A_i, idx, m2, vec_w):
    from hfm.manf.dist_internal import (
        sub_accelerator_smaler, sub_accelerator_larger, projector)
    from hfm.dist_est_bin import sub_accelerator_smaler as smaler
    from hfm.dist_est_bin import sub_accelerator_larger as larger

    proj = [projector(ele, vec_w) for ele in X_nA_y]
    idx_y_fx = np.argsort(proj)
    i = 11
    alt = (A_i == 1).astype('int')
    n1 = smaler(X_nA_y, A_i, ~idx, idx, idx_y_fx, i, m2)
    n2 = larger(X_nA_y, A_i, ~idx, idx, idx_y_fx, i, m2)
    n3 = smaler(X_nA_y, A_i, ~alt, alt, idx_y_fx, i, m2)
    n4 = larger(X_nA_y, A_i, ~alt, alt, idx_y_fx, i, m2)
    assert n1 == n3 and n2 == n4

    t1 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2, 'euclidean')
    t2 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2, 'manhattan')
    t3 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2, 'chebyshev')
    t4 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2, 'minkowski')
    t5 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2, 'cos_sim')

    if t1[0] != n1[0]:
        pdb.set_trace()
    return


def dist_direct_part4(X_nA_y, A, indices):
    from hfm.manf.dist_internal import (
        weight_generator,  # , weight_gen_many)
        projector, projector_alt, AcceleCore_bin)
    from hfm.dist_est_bin import weight_generator as weight
    from hfm.dist_est_bin import AcceleDist_bin as Accele

    n_d = X_nA_y.shape[1] - 1
    vec_w = weight_generator(n_d)
    tmp = weight(n_d)
    assert check_equal(1., [np.sum(np.abs(tmp)), np.sum(np.abs(vec_w))])
    # assert np.sum(np.abs(tmp)) == 1. == np.sum(np.abs(vec_w))
    # tmp = weight_gen_many(2, n_d)

    t1 = projector(X_nA_y[0], vec_w)
    t2 = projector_alt(X_nA_y[0], vec_w)
    t3 = projector(X_nA_y[0], vec_w)
    assert check_equal(t2, [t1, t3])

    m2 = 8
    A_i, idx = A[:, 1], indices[1][0]  # .astype('int')
    drt_part4_bin(X_nA_y, A_i, idx, m2, vec_w)

    n1 = Accele(X_nA_y, A_i, ~idx, idx, m2, vec_w)
    n2 = Accele(X_nA_y, A_i, ~idx, idx, m2, vec_w)
    n3 = Accele(X_nA_y, A_i, idx, ~idx, m2, vec_w)

    t1 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'euclidean')
    t2 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'manhattan')
    t3 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'chebyshev')
    t4 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'minkowski')
    t5 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'cos_sim')

    t6 = AcceleCore_bin(X_nA_y, A_i, m2, vec_w, 'euclidean')
    assert np.equal(n1[0], n2[0]).all()
    assert np.equal(n1[0], n3[0]).all()
    assert np.equal(t1[0], t6[0]).all()
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
    # _, na = A.shape
    # nai = len(np.unique(A))
    indices = [[A[:, i] == j + 1 for j in range(
        nai)] for i in range(na)]

    # dist_direct_part2(X_nA_y, A, indices)
    # dist_direct_part3(X_nA_y, A, indices)
    dist_direct_part4(X_nA_y, A, indices)
    return
