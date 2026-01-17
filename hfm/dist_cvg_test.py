# coding: utf-8
#
# Usage: to approximate the distance quickly
#        when faced with one bi-valued sensitive attribute
#


from hfm.dist_est_bin import (
    AcceleDist_bin, ApproxDist_bin, ApproxDist_bin_revised)
# from hfm.dist_est_bin import (weight_generator, AcceleDist_bin,
#                               ApproxDist_bin,
#                               ApproxDist_bin_revised)
from hfm.dist_est_nonbin import sub_accelerator_larger as larger
from hfm.dist_est_nonbin import sub_accelerator_smaler as smaler
from hfm.dist_cvg_nonbin import (AcceleDist_nonbin,
                                 sub_accelerator_smaler,
                                 sub_accelerator_larger,
                                 ApproxDist_nonbin,
                                 ApproxDist_nonbin_mpver)

from hfm.dist_cvg_nonbin import (
    StratVacant, StratRearrange, StratEarlyStop, EffExact)
from hfm.dist_est_nonbin import ApproxDist_nonbin as StratPrev
from hfm.earlybreak import (
    EffHD_bin, EffHD_nonbin, EffHD_multivar,
    Naive_bin, Naive_nonbin, Naive_multivar)
from hfm.dist_est_nonbin import ExtendDist_multiver_mp as DistExtend

import numpy as np
from pathos import multiprocessing as pp
from hfm.dist_drt import (DirectDist_bin, DirectDist_nonbin,
                          DirectDist_multiver)
from hfm.dist_drt_test import no_less_than_check

from hfm.dist_est_test import generate_dat
from hfm.dist_est_bin import projector
from hfm.utils.verifiers import check_equal


def test_compare_subproc():
    X_nA_y, A, indices, vec_w = generate_dat(40, 5, 2, 3)
    k, m2, i = 0, 6, 11
    idx_S1, Ap = indices[k][1], A[:, k]
    proj = [projector(ele, vec_w) for ele in X_nA_y]
    idx_y_fx = np.argsort(proj)

    min_js = smaler(X_nA_y, Ap, idx_y_fx, i, m2)
    min_jr = larger(X_nA_y, Ap, idx_y_fx, i, m2)
    tmp_js = sub_accelerator_smaler(X_nA_y, proj, Ap, idx_y_fx, i)
    tmp_jr = sub_accelerator_larger(X_nA_y, proj, Ap, idx_y_fx, i)

    assert min_js >= tmp_js[0]  # and tmp_js[1] >= m2
    assert min_jr >= tmp_jr[0]  # and tmp_jr[1] >= m2
    # cmp_js = separate_accel_smaler(X_nA_y, proj, Ap, idx_y_fx, i)
    # pdb.set_trace()
    return


def compare_accele(nai, m1, m2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    k = 0
    idx_S1, Ap = indices[k][1], A[:, k]
    idx_S0 = ~idx_S1

    res_1 = DirectDist_bin(X_nA_y, idx_S1)
    index_alt = [~indices[k][1], indices[k][1]]
    res_2 = DirectDist_nonbin(X_nA_y, index_alt)
    assert res_1[0][0] == res_2[0][0]  # max
    assert res_1[0][1] == res_2[0][1]  # avg

    Aq = Ap.copy()
    Aq[Ap > 1] = 0  # binarisation
    tmp_1 = AcceleDist_bin(X_nA_y, Aq, idx_S0, idx_S1, m2, vec_w)
    tmp_2 = AcceleDist_nonbin(X_nA_y, Aq, vec_w)  # m2,
    tmp_3 = AcceleDist_bin(X_nA_y, Ap, idx_S0, idx_S1, m2, vec_w)
    tmp_4 = AcceleDist_nonbin(X_nA_y, Ap, vec_w)  # m2,

    # pdb.set_trace()
    tmp_1, _ = tmp_1
    tmp_2, _ = tmp_2
    tmp_3, _ = tmp_3
    tmp_4, _ = tmp_4

    check_equal(tmp_1, tmp_3)
    check_equal(tmp_2, tmp_4)
    assert tmp_1[0] >= tmp_2[0] and tmp_1[1] >= tmp_2[1]
    assert tmp_3[0] >= tmp_4[0] and tmp_3[1] >= tmp_4[1]
    return


def compare_approx(nai, m1, m2, n_e=2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, _ = generate_dat(n, nd, na, nai)
    k = 0
    idx_S1, Ap = indices[k][1], A[:, k]
    Aq = Ap.copy()
    Aq[Ap > 1] = 0  # binarisation

    res_1 = DirectDist_bin(X_nA_y, idx_S1)
    index_alt = [~indices[k][1], indices[k][1]]
    res_2 = DirectDist_nonbin(X_nA_y, index_alt)
    res_3 = DirectDist_nonbin(X_nA_y, indices[k])
    assert res_1[0][0] == res_2[0][0]  # max
    assert res_1[0][1] == res_2[0][1]  # avg

    ans_1 = ApproxDist_bin(X_nA_y, Aq, idx_S1, m1, m2)
    ans_3 = ApproxDist_nonbin(X_nA_y, Aq, n_e)  # m1,m2,
    ans_4 = ApproxDist_bin(X_nA_y, Ap, idx_S1, m1, m2)
    ans_6 = ApproxDist_nonbin(X_nA_y, Ap, n_e)  # m1,m2,
    # ans_2 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # Aq,
    # ans_5 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # Ap,

    # pdb.set_trace()
    ans_1, _ = ans_1
    ans_3, _ = ans_3
    ans_4, _ = ans_4
    ans_6, _ = ans_6
    # ans_2, _ = ans_2
    # ans_5, _ = ans_5
    assert ans_1 >= ans_3[0] and ans_4 >= ans_6[0]
    return


def compare_multiver(nai, m1, m2, n_e=2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    pool = pp.ProcessingPool(nodes = 3)  # mp_cores)
    # W, tim = orthogonal_weight(nd + 1, n_e)
    # assert np.dot(W[0], W[1]) < 10**8  # sum(W[0]*W[1])
    assert abs(1 - sum(vec_w)) < 10**8

    k = 0
    A_j = A[:, k]
    idx_S1 = A_j == 1
    idx_S0 = ~idx_S1
    tmp_1 = DirectDist_bin(X_nA_y, idx_S1)
    tmp_3 = DirectDist_nonbin(X_nA_y, [idx_S0, idx_S1])
    tmp_4 = DirectDist_nonbin(X_nA_y, indices[k])
    tmp_7, tmp_8 = DirectDist_multiver(X_nA_y, indices)

    tmp_1, _ = tmp_1
    tmp_3, _ = tmp_3
    tmp_4, _ = tmp_4
    _, _, tmp_7 = tmp_7
    tmp_7 = tmp_7[: -1]
    assert tmp_1[0] == tmp_3[0] and tmp_4[0] == tmp_7[0][k]  # max
    assert tmp_1[1] == tmp_3[1] and tmp_4[1] == tmp_7[1][k]

    # res_2 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  #A_j,
    res_1 = ApproxDist_bin(X_nA_y, A_j, idx_S1, m1, m2)
    res_4 = ApproxDist_nonbin(X_nA_y, A_j, n_e)  # m1,m2,
    res_5 = ApproxDist_nonbin_mpver(X_nA_y, A_j, n_e)  # m1,m2,
    res_6 = ApproxDist_nonbin_mpver(X_nA_y, A_j,  # m1,  # m2,
                                    n_e, pool)

    # pdb.set_trace()
    # res_2, _ = res_2
    res_1, _ = res_1
    res_4, _ = res_4
    res_5, _ = res_5
    res_6, _ = res_6
    for i in [0, 1]:  # max, avg
        # no_less_than_check(res_2[i], tmp_1[i])
        assert tmp_1[i] == tmp_3[i]
        no_less_than_check(res_4[i], tmp_4[i])
        no_less_than_check(res_5[i], tmp_4[i])
        no_less_than_check(res_6[i], tmp_4[i])
    no_less_than_check(res_1, tmp_1[0])
    assert tmp_7[0][k] == tmp_4[0]  # max
    assert tmp_7[1][k] == tmp_4[1]  # avg
    return


def test_approx_dist():
    m1, m2 = 3, 5
    compare_accele(2, m1, m2)
    compare_accele(3, m1, m2)
    compare_approx(2, m1, m2)
    compare_approx(3, m1, m2)
    compare_multiver(2, m1, m2)
    compare_multiver(3, m1, m2)
    return


# ------------------------------------------
# strategies


def test_strategy():
    m1, m2, n_e, k = 3, 5, 2, 0
    n, nd, na, nai = 30, 4, 2, 3
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    idx_S1, Ap = indices[k][1], A[:, k]
    S0 = [indices[k][1], indices[k][0], indices[k][2]]
    i_alt = [~idx_S1, idx_S1]  # idx_S0=~idx_S1

    res_1 = DirectDist_bin(X_nA_y, idx_S1)
    res_2 = DirectDist_nonbin(X_nA_y, i_alt)
    res_3 = DirectDist_nonbin(X_nA_y, S0)
    res_4 = DirectDist_nonbin(X_nA_y, indices[k])
    # res_5 = Naive_bin(X_nA_y, idx_S1)
    # res_6 = Naive_nonbin(X_nA_y, indices[k])
    # res_7 = EffHD_bin(X_nA_y, idx_S1)
    # res_8 = EffHD_nonbin(X_nA_y, indices[k])
    assert check_equal(res_1[0], res_2[0])
    assert check_equal(res_3[0], res_4[0])

    ans_0 = StratPrev(X_nA_y, Ap, m1, m2, n_e)
    ans_1 = StratVacant(X_nA_y, Ap, m1, m2, n_e)
    ans_2 = StratRearrange(X_nA_y, Ap, m1, m2, n_e)
    ans_3 = StratEarlyStop(X_nA_y, Ap, n_e)
    assert check_equal(ans_3[0], res_3[0])

    B_j = idx_S1.astype('int')
    cmp_0 = StratPrev(X_nA_y, B_j, m1, m2, n_e)
    cmp_1 = StratVacant(X_nA_y, B_j, m1, m2, n_e)  # idx_S1
    cmp_2 = StratRearrange(X_nA_y, B_j, m1, m2, n_e)
    cmp_3 = StratEarlyStop(X_nA_y, B_j, n_e)
    assert check_equal(cmp_3[0], res_2[0])
    assert check_equal(cmp_3[0], res_1[0])

    # from hfm.dist_est_nonbin import AcceleCore
    # from hfm.dist_est_nonbin import orthogonal_weight
    # from hfm.dist_cvg_nonbin import (
    #     # StratRearrange_ver1, StratRearrange_ver2,
    #     StratRearrange_ver3, StratRearrange_ver4)
    # W, _ = orthogonal_weight(nd + 1, n_e)
    # T_min = AcceleCore(X_nA_y, Ap, m2, W[0])
    # T_alt = AcceleCore(X_nA_y, Ap, m2, W[1])
    # T_v1 = StratRearrange_ver1(X_nA_y, Ap, m1, m2, n_e)
    # T_v2 = StratRearrange_ver2(X_nA_y, Ap, m1, m2, n_e)
    # T_v3 = StratRearrange_ver3(X_nA_y, Ap, m1, m2, n_e)
    # T_v4 = StratRearrange_ver4(X_nA_y, Ap, m1, m2, n_e)
    # if not check_equal(ans_2[0], res_3[0]):
    #     pdb.set_trace()
    # assert check_equal(ans_2[0], res_3[0])
    assert check_equal(cmp_2[0], res_2[0])

    kw = {'m1': m1, 'm2': m2, 'n_e': n_e}
    lst_1 = EffExact(X_nA_y, A, StratVacant, **kw)
    lst_2 = EffExact(X_nA_y, A, StratRearrange, **kw)
    lst_3 = EffExact(X_nA_y, A, StratEarlyStop, **kw)
    lst_6 = Naive_multivar(X_nA_y, indices)
    lst_7 = EffHD_multivar(X_nA_y, indices)
    lst_8 = DirectDist_multiver(X_nA_y, indices)
    lst_9 = DistExtend(X_nA_y, A, m1, m2, n_e)
    # pdb.set_trace()
    return


def test_German():
    from experiment.utils_empirical import DataSetup
    from experiment.datasets import (
        transform_unpriv_tag)  # transform_X_and_y,
    # from experiment.preprocessing_bin import normalise_disturb_prime
    from experiment.preprocessing_nonbin import (
        renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
        check_marginalised_indices,  # renewed_transform_disturb,
        renewed_normalise_disturb, renewed_normalise_separate)

    cs = DataSetup('german')
    ratio = .97
    origin_dat, processed_dat, process_mult, disturbed_dat, \
        disturb_mult = renewed_prep_and_adversarial(
            cs._dataset, cs._data_frame, ratio, logger=None)
    processed_Xy = process_mult['numerical-multisen']
    # disturbed_Xy = disturb_mult['numerical-multisen']
    X, A, y, _ = renewed_transform_X_A_and_y(
        cs._dataset, processed_Xy, with_joint=False)
    # _, Aq, _, _ = renewed_transform_X_A_and_y(
    #         cs._dataset, disturbed_Xy, with_joint=False)

    X_nA_y = np.concatenate([
        y.values.reshape(-1, 1).astype('float'),
        X.values], axis=1)  # .flags
    sens_att = cs._dataset.get_sensitive_attrs_with_joint()[: 2]
    priv_val = cs._dataset.get_privileged_group_with_joint('')[: 2]
    marginalised_group = origin_dat['marginalised_groups']
    g1m_indices = check_marginalised_indices(
        processed_dat['original'], sens_att, priv_val,
        marginalised_group)
    k, m1, m2, n_e = 1, 3, 5, 2
    non_sa = g1m_indices[k][0]
    A = A.values
    Ap = A[:, k]

    res_5 = Naive_bin(X_nA_y, non_sa)
    res_6 = Naive_nonbin(X_nA_y, g1m_indices[k])
    res_1 = DirectDist_bin(X_nA_y, non_sa)
    res_2 = DirectDist_nonbin(X_nA_y, [~non_sa, non_sa])
    res_4 = DirectDist_nonbin(X_nA_y, g1m_indices[k])
    res_7 = EffHD_bin(X_nA_y, non_sa)
    res_8 = EffHD_nonbin(X_nA_y, g1m_indices[k])
    assert check_equal(res_1[0], res_2[0])

    B_j = non_sa.astype('int')
    cmp_0 = StratPrev(X_nA_y, B_j, m1, m2, n_e)
    cmp_1 = StratVacant(X_nA_y, B_j, m1, m2, n_e)
    cmp_2 = StratRearrange(X_nA_y, B_j, m1, m2, n_e)
    cmp_3 = StratEarlyStop(X_nA_y, B_j, n_e)
    assert check_equal(cmp_3[0], res_2[0])
    assert check_equal(cmp_3[0], res_1[0])

    ans_0 = StratPrev(X_nA_y, Ap, m1, m2, n_e)
    ans_1 = StratVacant(X_nA_y, Ap, m1, m2, n_e)
    ans_2 = StratRearrange(X_nA_y, Ap, m1, m2, n_e)
    ans_3 = StratEarlyStop(X_nA_y, Ap, n_e)
    assert check_equal(ans_3[0], res_4[0])
    assert check_equal(cmp_2[0], res_2[0])

    kw = {'m1': m1, 'm2': m2, 'n_e': n_e}
    lst_1 = EffExact(X_nA_y, A, StratVacant, **kw)
    lst_2 = EffExact(X_nA_y, A, StratRearrange, **kw)
    lst_3 = EffExact(X_nA_y, A, StratEarlyStop, **kw)
    lst_6 = Naive_multivar(X_nA_y, g1m_indices)
    lst_7 = EffHD_multivar(X_nA_y, g1m_indices)
    lst_8 = DirectDist_multiver(X_nA_y, g1m_indices)
    lst_9 = DistExtend(X_nA_y, A, m1, m2, n_e)

    # pdb.set_trace()
    return
