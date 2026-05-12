# coding: utf-8

import numpy as np
import pdb
from hfm.utils.verifiers import check_equal, poset_nolessthan


def dist_vector(ele_i, ele_ic):
    from hfm.manf.dist_internal import (
        dist_Euclidean, dist_Manhattan, dist_Chebyshev, dist_Minkowski,
        avbl_Euclidean, avbl_Manhattan, avbl_Chebyshev, avbl_Minkowski,
        dist_cos_sim, avbl_cos_sim,   # dist_cos_sim_alt)
        dist_corr_pearson, avbl_corr_pearson)

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

    alt = avbl_cos_sim(ele_i, ele_ic)  # dist_cos_sim_alt
    ans = dist_cos_sim(ele_i, ele_ic)  # ans, _ =
    assert 0 <= ans <= 1
    assert check_equal(alt, ans)

    alt = avbl_corr_pearson(ele_i, ele_ic)
    ans = dist_corr_pearson(ele_i, ele_ic)
    assert 0 <= ans <= 2
    assert check_equal(alt, ans)
    return


def dist_direct_part1(Si, Si_c):
    from hfm.manf.dist_internal import Direct_halfway_min
    t1 = Direct_halfway_min(Si[0], Si_c, 0)  # 'euclidean')
    t2 = Direct_halfway_min(Si[0], Si_c, 1)  # 'manhattan')
    t3 = Direct_halfway_min(Si[0], Si_c, 2)  # 'chebyshev')
    t4 = Direct_halfway_min(Si[0], Si_c, 3)  # 'minkowski')
    t5 = Direct_halfway_min(Si[0], Si_c, 4)  # 'cos_sim')
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
    t1 = Direct_mediator(X_nA_y, idx, 0)  # 'euclidean')
    t2 = Direct_mediator(X_nA_y, idx, 1)  # 'manhattan')
    t3 = Direct_mediator(X_nA_y, idx, 2)  # 'chebyshev')
    t4 = Direct_mediator(X_nA_y, idx, 3)  # 'minkowski')
    t5 = Direct_mediator(X_nA_y, idx, 4)  # 'cos_sim')
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


def approx_part4_sub(X_nA_y, A_i, idx, m2, vec_w):
    from hfm.manf.dist_internal import (
        sub_accelerator_smaler, sub_accelerator_larger, projector)
    from hfm.dist_est_bin import sub_accelerator_smaler as smaler
    from hfm.dist_est_bin import sub_accelerator_larger as larger

    proj = [projector(ele, vec_w) for ele in X_nA_y]
    tt = X_nA_y @ vec_w
    assert all([check_equal(i, j) for i, j in zip(proj, tt)])
    # pdb.set_trace()
    idx_y_fx = np.argsort(proj)
    i = 11
    alt = (A_i == 1).astype('bool')  # 'int')
    n1 = smaler(X_nA_y, A_i, ~idx, idx, idx_y_fx, i, m2)
    n2 = larger(X_nA_y, A_i, ~idx, idx, idx_y_fx, i, m2)
    n3 = smaler(X_nA_y, A_i, ~alt, alt, idx_y_fx, i, m2)
    n4 = larger(X_nA_y, A_i, ~alt, alt, idx_y_fx, i, m2)
    assert n1 == n3 and n2 == n4

    alt = alt.astype('int')  # /idx
    t1 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2, 0)  # 'euclidean')
    t2 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2, 1)  # 'manhattan')
    t3 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2, 2)  # 'chebyshev')
    t4 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2, 3)  # 'minkowski')
    t5 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2, 4)  # 'cos_sim')
    assert (0 <= t5 <= 1) and check_equal(t1, n1)  # (t1 == n1 >=0) and
    assert t2 >= 0 and t3 >= 0 and t4 >= 0

    t1 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2, 0)  # 'euclidean')
    t2 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2, 1)  # 'manhattan')
    t3 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2, 2)  # 'chebyshev')
    t4 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2, 3)  # 'minkowski')
    t5 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2, 4)  # 'cos_sim')
    assert (0 <= t5 <= 1) and check_equal(t1, n2)
    assert t2 >= 0 and t3 >= 2 and t4 >= 0

    t1 = sub_accelerator_smaler(X_nA_y, alt, idx_y_fx, i, m2)
    t6 = sub_accelerator_smaler(X_nA_y, A_i, idx_y_fx, i, m2)
    assert isinstance(t1, float) and isinstance(t6, float)  # [0]
    t1 = sub_accelerator_larger(X_nA_y, alt, idx_y_fx, i, m2)
    t6 = sub_accelerator_larger(X_nA_y, A_i, idx_y_fx, i, m2)
    assert isinstance(t1, float) and isinstance(t6, float)  # [0]
    # if t1[0] != n1[0]:
    return


def approx_part4_bin(X_nA_y, A_i, idx, m2, vec_w):
    from hfm.manf.dist_internal import AcceleCore_bin
    from hfm.dist_est_bin import AcceleDist_bin as Accele

    n1 = Accele(X_nA_y, A_i, ~idx, idx, m2, vec_w)
    n2 = Accele(X_nA_y, A_i, ~idx, idx, m2, vec_w)
    n3 = Accele(X_nA_y, A_i, idx, ~idx, m2, vec_w)

    B_i = (A_i == 1).astype('int')
    t6 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 0)  # 'euclidean')

    t1 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 0)  # 'euclidean')
    t2 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 1)  # 'manhattan')
    t3 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 2)  # 'chebyshev')
    t4 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 3)  # 'minkowski')
    t5 = AcceleCore_bin(X_nA_y, B_i, m2, vec_w, 4)  # 'cos_sim')

    assert np.equal(n1[0], n2[0]).all()
    assert np.equal(n1[0], n3[0]).all()
    assert np.equal(t1, t6).all()  # [0]
    assert check_equal(t1, n1[0])  # [0]
    return


def approx_part4_drt(X_nA_y, A_i, idx, m1, m2):
    from hfm.manf.dist_internal import Direct_bin, Approx_bin
    from hfm.dist_est_bin import ApproxDist_bin as Approx
    from hfm.dist_drt import DirectDist_bin as Direct

    n1 = Approx(X_nA_y, A_i, idx, m1, m2)
    # n2 = Approx(X_nA_y, A_i, idx, m1, m2)
    # n3 = Approx(X_nA_y, A_i, ~idx, m1, m2)
    B_i = idx.astype('int')  # (A_i == 1).astype('int')
    t6 = Approx_bin(X_nA_y, B_i, m1, m2, 'euclidean')

    t1 = Approx_bin(X_nA_y, B_i, m1, m2, 'euclidean')
    # t2 = Approx_bin(X_nA_y, B_i, m1, m2, 'manhattan')
    # t3 = Approx_bin(X_nA_y, B_i, m1, m2, 'chebyshev')
    # t4 = Approx_bin(X_nA_y, B_i, m1, m2, 'minkowski')
    # t5 = Approx_bin(X_nA_y, B_i, m1, m2, 'cos_sim')

    n2 = Approx(X_nA_y, B_i, idx, m1, m2)
    n3 = Direct(X_nA_y, idx)
    n4 = Direct_bin(X_nA_y, B_i, 1, idx_Si=idx, func='euclidean')
    n5 = Direct_bin(X_nA_y, A_i, 1, idx_Si=idx, func='euclidean')

    assert n1[0] >= n3[0][0] and n2[0] >= n3[0][0]
    assert check_equal(n3[0], n4[0]) and check_equal(n3[0], n5[0])
    assert poset_nolessthan(t6[0], n4[0])
    assert poset_nolessthan(t1[0], n4[0])
    return


def dist_direct_part4(X_nA_y, A, indices):
    from hfm.manf.dist_internal import (
        weight_generator,  # , weight_gen_many)
        projector, projector_alt)
    from hfm.dist_est_bin import weight_generator as weight

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
    approx_part4_sub(X_nA_y, A_i, idx, m2, vec_w)
    approx_part4_bin(X_nA_y, A_i, idx, m2, vec_w)
    m1 = 20
    approx_part4_drt(X_nA_y, A_i, idx, m1, m2)
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

    dist_direct_part2(X_nA_y, A, indices)
    dist_direct_part3(X_nA_y, A, indices)
    dist_direct_part4(X_nA_y, A, indices)
    return


def dist_approx_part5(X_nA_y, A, indices):
    from hfm.manf.dist_external import orthogonal_weight
    from hfm.dist_est_nonbin import orthogonal_weight as weight
    n_d = X_nA_y.shape[1] - 1
    tt = orthogonal_weight(n_d, n_e=3)
    vec_w = orthogonal_weight(n_d, n_e=3)
    tmp = weight(n_d, n_e=3)
    assert check_equal([np.dot(tmp[0][0], tmp[0][1]),
                        np.dot(vec_w[0], vec_w[1]),
                        np.dot(tt[0], tt[1])], 0.)
    return


def approx_part5_drt(X_nA_y, A_i, idx_Sjs, m1, m2, n_e=3):
    from hfm.manf.dist_internal import Direct_nonbin
    from hfm.manf.dist_external import Approx_nonbin
    from hfm.dist_est_nonbin import ApproxDist_nonbin as Approx
    from hfm.dist_drt import DirectDist_nonbin as Direct

    n1 = Approx(X_nA_y, A_i, m1, m2, n_e)
    n2 = Approx(X_nA_y, A_i, m1, m2, n_e)
    t6 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'euclidean')

    t1 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'euclidean')
    # t2 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'manhattan')
    # t3 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'chebyshev')
    # t4 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'minkowski')
    # t5 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, 'cos_sim')

    alt = [idx_Sjs[0], ~idx_Sjs[0]]
    n5 = Direct_nonbin(X_nA_y, (A_i == 1).astype('int'), 1, alt, 'euclidean')
    n3 = Direct(X_nA_y, alt)
    n4 = Direct(X_nA_y, idx_Sjs)
    n6 = Direct_nonbin(X_nA_y, A_i, 1, idx_Sjs, 'euclidean')

    assert poset_nolessthan(t1[0], n6[0])
    assert check_equal(n4[0], n6[0])
    assert check_equal(n3[0], n5[0])
    assert poset_nolessthan(n1[0], n4[0])
    assert poset_nolessthan(n2[0], n4[0])
    return


def approx_part5_app(X_nA_y, A, indices, m1, m2, n_e=3):
    from hfm.manf.dist_internal import Direct_multiver
    from hfm.manf.dist_external import Extend_multiver
    from hfm.dist_est_nonbin import ExtendDist_multiver_mp as Extend
    from hfm.dist_drt import DirectDist_multiver as Direct

    n1 = Extend(X_nA_y, A, m1, m2, n_e)
    n2 = Extend(X_nA_y, A, m1, m2, n_e)
    t6 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'euclidean')

    t1 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'euclidean')
    # t2 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'manhattan')
    # t3 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'chebyshev')
    # t4 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'minkowski')
    # t5 = Extend_multiver(X_nA_y, A, m1, m2, n_e, 'cos_sim')

    n3 = Direct(X_nA_y, indices)
    n4 = Direct(X_nA_y, indices)
    n5 = Direct_multiver(X_nA_y, A, 1, indices, 'euclidean')
    n6 = Direct_multiver(X_nA_y, A, 1, indices, 'euclidean')
    # pdb.set_trace()

    assert poset_nolessthan(n1[0][:2], n3[0][:2])
    assert poset_nolessthan(n2[0][:2], n4[0][:2])
    assert check_equal(n3[0][:2], n5[0][:2])
    assert check_equal(n4[0][:2], n6[0][:2])
    assert poset_nolessthan(t6[0], n5[0][:2])
    assert poset_nolessthan(t1[0], n6[0][:2])
    return


def approx_part5_sub(X_nA_y, A_i, idx_Sjs, m2, vec_w):
    from hfm.manf.dist_external import (
        cvg_accelerator_smaler, cvg_accelerator_larger)
    # from hfm.manf.dist_internal import projector
    # from hfm.manf.dist_internal import sub_accelerator_smaler as smaler
    # from hfm.manf.dist_internal import sub_accelerator_larger as larger
    from hfm.dist_est_nonbin import sub_accelerator_smaler as smaler
    from hfm.dist_est_nonbin import sub_accelerator_larger as larger

    proj = X_nA_y @ vec_w
    idx_y_fx = np.argsort(proj)
    i = 11
    alt = (A_i == 1).astype('int')
    n3 = smaler(X_nA_y, A_i, idx_y_fx, i, m2)
    n4 = larger(X_nA_y, A_i, idx_y_fx, i, m2)
    n1 = smaler(X_nA_y, alt, idx_y_fx, i, m2)  # , 0)
    n2 = larger(X_nA_y, alt, idx_y_fx, i, m2)  # , 0)
    assert isinstance(n3, float) and isinstance(n4, float)

    t1 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 0)
    # t2 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 1)
    # t3 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 2)
    # t4 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 3)
    # t5 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 4)
    # t6 = cvg_accelerator_smaler(X_nA_y, proj, alt, idx_y_fx, i, 5)
    assert check_equal(n1, t1[0]) or n1 >= t1[0]
    # assert check_equal(t1[1], [i[1] for i in [t2, t3, t4, t5, t6]])

    t1 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 0)
    # t2 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 1)
    # t3 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 2)
    # t4 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 3)
    # t5 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 4)
    # t6 = cvg_accelerator_larger(X_nA_y, proj, alt, idx_y_fx, i, 5)
    assert check_equal(n2, t1[0]) or n2 >= t1[0]

    t7 = cvg_accelerator_smaler(X_nA_y, proj, A_i, idx_y_fx, i, 0)
    assert check_equal(n3, t7[0]) or n3 >= t7[0]
    t7 = cvg_accelerator_larger(X_nA_y, proj, A_i, idx_y_fx, i, 0)
    assert check_equal(n3, t7[0]) or n3 >= t7[0]
    return


def conver_part6_drt(X_nA_y, A_i, idx_Sjs, n_e=3):
    from hfm.manf.dist_internal import Direct_nonbin
    from hfm.manf.dist_external import StratES_nonbin, StratRA_nonbin
    from hfm.dist_cvg_nonbin import ApproxDist_nonbin as Approx
    from hfm.dist_drt import DirectDist_nonbin as Direct

    n1 = Approx(X_nA_y, A_i, n_e)
    n2 = Approx(X_nA_y, A_i, n_e)

    # alt = [idx_Sjs[0], ~idx_Sjs[0]]
    # n5 = Direct_nonbin(X_nA_y, (A_i == 1).astype('int'), 1, alt, 'euclidean')
    # n3 = Direct(X_nA_y, alt)
    n4 = Direct(X_nA_y, idx_Sjs)
    n6 = Direct_nonbin(X_nA_y, A_i, 1, idx_Sjs, 'euclidean')
    assert check_equal(n4[0], n6[0])
    assert poset_nolessthan(n1[0], n4[0])
    assert poset_nolessthan(n2[0], n4[0])

    t7 = StratES_nonbin(X_nA_y, A_i, n_e, 'euclidean')
    t1 = StratES_nonbin(X_nA_y, A_i, n_e, 'euclidean')
    # t2 = StratES_nonbin(X_nA_y, A_i, n_e, 'manhattan')
    # t3 = StratES_nonbin(X_nA_y, A_i, n_e, 'chebyshev')
    # t4 = StratES_nonbin(X_nA_y, A_i, n_e, 'minkowski')
    # t5 = StratES_nonbin(X_nA_y, A_i, n_e, 'cos_sim')
    # t6 = StratES_nonbin(X_nA_y, A_i, n_e, 'correla')
    assert check_equal(t1[0], n6[0])

    m1, m2 = 20, 8
    t7 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'euclidean')
    t1 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'euclidean')
    # t2 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'manhattan')
    # t3 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'chebyshev')
    # t4 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'minkowski')
    # t5 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'cos_sim')
    # t6 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, 'correla')
    assert poset_nolessthan(t1[0], n6[0])

    assert isinstance(t7[-1], float)  # pdb.set_trace()
    return


def conver_part6_app(X_nA_y, A, indices, m1, m2, n_e):
    from hfm.manf.dist_external import EffExact_multiver
    from hfm.manf.dist_internal import Direct_multiver
    from hfm.dist_drt import DirectDist_multiver as Direct
    from hfm.dist_cvg_nonbin import EffExact as Extend

    from hfm.dist_cvg_nonbin import StratVacant as SV1
    from hfm.dist_cvg_nonbin import StratEarlyStop as SES
    from hfm.dist_cvg_nonbin import StratRearrange as SRA
    n1 = Extend(X_nA_y, A, SV1, m1, m2, n_e)
    # t7 = EffExact_multiver(X_nA_y, A, 'Vacant', m1, m2, n_e)

    t1 = EffExact_multiver(X_nA_y, A, 'Vacant', m1, m2, n_e, 'euclidean')
    n4 = Direct(X_nA_y, indices)
    n6 = Direct_multiver(X_nA_y, A, 1, indices, 'euclidean')
    assert check_equal(n4[0][:2], n6[0][:2])
    assert poset_nolessthan(t1[0][:2], n6[0][:2])
    assert poset_nolessthan(n1[0][:2], n4[0][:2])

    n2 = Extend(X_nA_y, A, SES, m1, m2, n_e)
    t2 = EffExact_multiver(X_nA_y, A, 'StratES', m1, m2, n_e, 'euclidean')
    assert check_equal(t2[0][:2], n6[0][:2])
    assert check_equal(n2[0][:2], n6[0][:2])
    n3 = Extend(X_nA_y, A, SRA, m1, m2, n_e)
    t3 = EffExact_multiver(X_nA_y, A, 'StratRA', m1, m2, n_e, 'euclidean')
    assert poset_nolessthan(t3[0][:2], n6[0][:2])
    assert poset_nolessthan(n3[0][:2], n6[0][:2])
    return


def test_external():
    n, nd = 324, 17
    nc = na = nai = 2
    nai = 3
    X = np.random.rand(n, nd) * 10
    y = np.random.randint(nc, size=n)  # binary classification
    A = np.random.randint(nai, size=(n, na)) + 1
    X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
    indices = [[A[:, i] == j + 1 for j in range(
        nai)] for i in range(na)]

    dist_approx_part5(X_nA_y, y, indices)
    m1, m2, n_e = 20, 8, 3
    approx_part5_drt(X_nA_y, A[:, 1], indices[1], m1, m2, n_e)
    approx_part5_app(X_nA_y, A, indices, m1, m2, n_e)

    from hfm.manf.dist_internal import weight_generator
    vec_w = weight_generator(n_d=nd)
    # approx_part5_sub(X_nA_y, A[:, 1], indices[1], m2, vec_w)

    conver_part6_drt(X_nA_y, A[:, 1], indices[1], n_e)
    conver_part6_app(X_nA_y, A, indices, m1, m2, n_e)
    return
