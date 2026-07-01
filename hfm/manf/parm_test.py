# coding: utf-8
# parm_hfm_test.py

import numpy as np
import pdb
from hfm.utils.verifiers import check_equal, poset_nolessthan

from hfm.manf.parm_hfm import (
    direct_single_sa, strat_es_single_sa, strat_ra_single_sa)
from hfm.manf.renew_core import (  # .renew_drt import (
    lp_norm_vec, dual_norm_vec, dual_normalize, _as_float_p)


from hfm.manf.renew_drt import direct_sing_sa as re_sing_sa
from hfm.manf.renew_drt import Direct_bin as re_bin
from hfm.manf.renew_drt import Direct_nonbin as re_nonbin
from hfm.manf.renew_cvg import StratES_nonbin as re_StratES
from hfm.manf.renew_cvg import StratRA_nonbin as re_StratRA
from hfm.manf.renew_app import Approx_bin as re_Approx_bin
from hfm.manf.renew_app import Approx_nonbin as re_Approx_nonbin

from hfm.manf.dist_internal import (Direct_bin, Direct_nonbin)
from hfm.manf.dist_external import StratES_nonbin, StratRA_nonbin
from hfm.manf.dist_internal import Approx_bin, weight_generator
from hfm.manf.dist_external import Approx_nonbin
from hfm.dist_est_bin import ApproxDist_bin as prev_Approx_bin
from hfm.dist_est_nonbin import ApproxDist_nonbin as prev_Approx_nonbin
from hfm.manf.renew_core import dual_exponent, orthogonal_weight


n, nd = 324, 17
nc = na = nai = 2
nai = 3
X = np.random.rand(n, nd) * 10
y = np.random.randint(nc, size=n)  # binary classification
A = np.random.randint(nai, size=(n, na)) + 1
X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
indices = [[A[:, i] == j + 1 for j in range(nai)] for i in range(na)]
m1, m2, n_e = 20, 8, 3

A_i = A[:, 1]
B_i = (A_i == 1).astype('int')
B_index = [indices[1][0], ~indices[1][0]]
priv = 1
curr_p = [7, 2, 1, float("inf"), 3, ]
curr_d = ['cos_sim', 'correla',   # 'minkowski (p=3)'
          'euclidean', 'manhattan', 'chebyshev', 'minkowski']
ele_i, ele_ic = X_nA_y[:2]

n_e = 2
m1, m2 = 8, 20  # m2, m1


def dist():
    from hfm.manf.dist_internal import alter_intermediate
    d0 = [alter_intermediate(ele_i, ele_ic, p) for p in [4, 5, 0, 1, 3, 2]]
    d1 = [lp_norm_vec(ele_i - ele_ic, _as_float_p(p)
                      ) for p in [4, 7, 2, 1, 3, 'inf']]
    assert check_equal(d0[2:], d1[2:])

    # from hfm.manf.renew_drt import dual_exponent
    # from hfm.manf.renew_cvg import orthogonal_weight
    W = orthogonal_weight(1 + nd, 1)
    curr = [4, 7, 1, 2, float('inf'), 3, ]
    assert check_equal(lp_norm_vec(W[0], 2), 1.0)

    d1 = [dual_normalize(W[0], p) for p in curr]
    d2 = [dual_exponent(p) for p in curr]
    d3 = [lp_norm_vec(dp, p) for dp, p in zip(d1, d2)]
    assert check_equal(d3, 1.0)
    # d4 = [lp_norm_vec(dp, p) for dp, p in zip(d1, curr)]
    # d1 = [lp_norm_vec(ele_i - ele_ic, _as_float_p(p)) for p in curr]
    # d2 = [dual_normalize(t, p) for t, p in zip(d1, curr_p)]

    vec_w = weight_generator(nd + 1)
    assert check_equal(np.abs(vec_w).sum(), 1.0)
    # pdb.set_trace()

    # from hfm.manf.renew_core import orthogonal_weight_prime
    # Wp = orthogonal_weight_prime(1 + nd, 1)
    # assert check_equal(1., (W[0] ** 2).sum().tolist())
    # assert check_equal(1., (Wp[0] ** 2).sum().tolist())
    # pdb.set_trace()
    return


def drt_bin():
    d0 = [direct_single_sa(X_nA_y, B_i, p=p) for p in curr_p]
    d1 = [strat_es_single_sa(X_nA_y, B_i, p=p, seed=42) for p in curr_p]
    d2 = [strat_ra_single_sa(X_nA_y, B_i, p=p, m1=m1, m2=m2, n_e=n_e,
                             seed=42) for p in curr_p]
    d0, d1, d2 = d0[1:], d1[1:], d2[1:]
    # t1 = [Direct_bin(X_nA_y, A_i, priv, indices[1][0], func=p) for p in curr_d]
    # t2 = [Direct_nonbin(X_nA_y, A_i, priv, indices[1], func=p) for p in curr_d]
    t1 = [Direct_bin(X_nA_y, B_i, priv, B_index[0], func=p) for p in curr_d]
    t2 = [Direct_nonbin(X_nA_y, B_i, priv, B_index, func=p) for p in curr_d]
    t1, t2 = t1[2:], t2[2:]

    for curr, ct1, ct2 in zip(d0, t1, t2):
        assert check_equal(curr[0], ct1[0]) and check_equal(curr[0], ct2[0])

    t3 = [StratES_nonbin(X_nA_y, B_i, n_e, func=p) for p in curr_d]
    t4 = [StratRA_nonbin(X_nA_y, B_i, m1, m2, n_e, func=p) for p in curr_d]
    t3, t4 = t3[2:], t4[2:]
    # for curr, ct1, ct2 in zip(d0, d1, t3):
    #     assert check_equal(curr[0], ct1[0]) and check_equal(curr[0], ct2[0])
    # for curr, ct1, ct2 in zip(d0, t2, t4):
    #     assert check_equal(curr[0], ct1[0]) and check_equal(curr[0], ct2[0])

    r0 = [re_sing_sa(X_nA_y, B_i, p=p) for p in curr_p]
    r1 = [re_bin(X_nA_y, B_i, p=p, priv_val=priv) for p in curr_p]
    r2 = [re_nonbin(X_nA_y, B_i, p=p, priv_val=priv) for p in curr_p]
    r0, r1, r2 = r0[1:], r1[1:], r2[1:]
    for curr, ct1, ct2 in zip(r0, r1, r2):
        assert check_equal(curr[0], ct1[0]) and check_equal(curr[0], ct2[0])

    r3 = [re_StratES(X_nA_y, B_i, p=p) for p in curr_p]
    r4 = [re_StratRA(X_nA_y, B_i, p=p, m1=m1, m2=m2, n_e=n_e) for p in curr_p]
    r3, r4 = r3[1:], r4[1:]
    # pdb.set_trace()
    for d, t, r in zip(t3, r3, r2):
        assert check_equal(t[0], r[0])  # and check_equal(d[0], r[0])
    # for d, t, r in zip(t4, r4, r2):
    #     pass
    return


def drt_nonbin():
    d0 = [direct_single_sa(X_nA_y, A_i, p=p) for p in curr_p]
    d1 = [strat_es_single_sa(X_nA_y, A_i, p=p, seed=42) for p in curr_p]
    d2 = [strat_ra_single_sa(X_nA_y, A_i, p=p, m1=m1, m2=m2, n_e=n_e,
                             seed=42) for p in curr_p]
    d0, d1, d2 = d0[1:], d1[1:], d2[1:]
    t1 = [Direct_bin(X_nA_y, A_i, priv, indices[1][0], func=p) for p in curr_d]
    t2 = [Direct_nonbin(X_nA_y, A_i, priv, indices[1], func=p) for p in curr_d]
    t1, t2 = t1[2:], t2[2:]

    for curr, ct1, ct2 in zip(d0, t1, t2):
        assert check_equal(curr[0], ct2[0])  # and check_equal(curr[0], ct1[0])

    t3 = [StratES_nonbin(X_nA_y, A_i, n_e, func=p) for p in curr_d]
    t4 = [StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, func=p) for p in curr_d]
    t3, t4 = t3[2:], t4[2:]
    r3 = [re_StratES(X_nA_y, A_i, p=p) for p in curr_p]
    r4 = [re_StratRA(X_nA_y, A_i, p=p, m1=m1, m2=m2, n_e=n_e) for p in curr_p]
    r3, r4 = r3[1:], r4[1:]

    r0 = [re_sing_sa(X_nA_y, A_i, p=p) for p in curr_p]
    r1 = [re_bin(X_nA_y, A_i, p=p, priv_val=priv) for p in curr_p]
    r2 = [re_nonbin(X_nA_y, A_i, p=p, priv_val=priv) for p in curr_p]
    r0, r1, r2 = r0[1:], r1[1:], r2[1:]

    for d, r in zip(d0, r0):
        assert check_equal(d[0], r[0])
    for d, r in zip(t1, r1):
        assert check_equal(d[0], r[0])
    for d, r in zip(t2, r2):
        assert check_equal(d[0], r[0])

    # pdb.set_trace()
    for d, t, r in zip(t3, r3, r2):
        assert check_equal(t[0], r[0])  # and check_equal(d[0], r[0])
    # for d, t, r in zip(t4, r4, r2):
    #     pass
    return


def app_bin():
    # d0 = [direct_single_sa(X_nA_y, B_i, p=p) for p in curr_p]
    # d0 = d0[1:]
    t1 = [Direct_bin(X_nA_y, B_i, priv, B_index[0], func=p) for p in curr_d]
    t2 = [Direct_nonbin(X_nA_y, B_i, priv, B_index, func=p) for p in curr_d]
    t3 = [StratES_nonbin(X_nA_y, B_i, n_e, func=p) for p in curr_d]
    t4 = [StratRA_nonbin(X_nA_y, B_i, m1, m2, n_e, func=p) for p in curr_d]
    t1, t2, t3, t4 = t1[2:], t2[2:], t3[2:], t4[2:]

    r0 = [re_sing_sa(X_nA_y, B_i, p=p) for p in curr_p]
    r1 = [re_bin(X_nA_y, B_i, p=p, priv_val=priv) for p in curr_p]
    r2 = [re_nonbin(X_nA_y, B_i, p=p, priv_val=priv) for p in curr_p]
    r0, r1, r2 = r0[1:], r1[1:], r2[1:]
    r3 = [re_StratES(X_nA_y, B_i, p=p) for p in curr_p]
    r4 = [re_StratRA(X_nA_y, B_i, p=p, m1=m1, m2=m2, n_e=n_e) for p in curr_p]
    r3, r4 = r3[1:], r4[1:]

    r5 = [re_Approx_bin(X_nA_y, B_i, p, m1, m2) for p in curr_p]
    r6 = [Approx_bin(X_nA_y, B_i, m1, m2, func=p) for p in curr_d]
    r7 = prev_Approx_bin(X_nA_y, B_i, B_index[0], m1, m2)
    r5, r6 = r5[1:], r6[2:]
    r8 = [re_Approx_nonbin(X_nA_y, B_i, p, m1, m2, n_e) for p in curr_p]
    r9 = [Approx_nonbin(X_nA_y, B_i, m1, m2, n_e, func=p) for p in curr_d]
    r10 = prev_Approx_nonbin(X_nA_y, B_i, m1, m2, n_e)
    r8, r9 = r8[1:], r9[2:]
    # pdb.set_trace()

    for n1, n2, n3, n4 in zip(t1, t2, t3, t4):
        assert check_equal(n1[0], n2[0])  # and check_equal(n2[0], n3[0])
    for n1, n2, n3, n4 in zip(t1, r0, r1, r2):
        assert check_equal(n1[0], n2[0]) and check_equal(n1[0], n3[0])
        assert check_equal(n3[0], n4[0])
    for n1, n2 in zip(r0, r1):
        assert check_equal(n2[0], n1[0])
    for n2, n3, n4 in zip(r1, r5, r6):
        assert poset_nolessthan(n3[0], n2[0])
        assert poset_nolessthan(n4[0], n2[0])
    for n2, n3, n4 in zip(r2, r8, r9):
        assert poset_nolessthan(n3[0], n2[0])
        assert poset_nolessthan(n4[0], n2[0])
    assert r7[0] >= r1[0][0][0] and poset_nolessthan(r10[0], r1[0][0])
    return


def app_nonbin():
    t1 = [Direct_bin(X_nA_y, A_i, priv, indices[1][0], func=p) for p in curr_d]
    t2 = [Direct_nonbin(X_nA_y, A_i, priv, indices[0], func=p) for p in curr_d]
    t3 = [StratES_nonbin(X_nA_y, A_i, n_e, func=p) for p in curr_d]
    t4 = [StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, func=p) for p in curr_d]
    t1, t2, t3, t4 = t1[2:], t2[2:], t3[2:], t4[2:]

    r0 = [re_sing_sa(X_nA_y, A_i, p=p) for p in curr_p]
    r1 = [re_bin(X_nA_y, A_i, p=p, priv_val=priv) for p in curr_p]
    r2 = [re_nonbin(X_nA_y, A_i, p=p, priv_val=priv) for p in curr_p]
    r0, r1, r2 = r0[1:], r1[1:], r2[1:]
    r3 = [re_StratES(X_nA_y, A_i, p=p) for p in curr_p]
    r4 = [re_StratRA(X_nA_y, A_i, p=p, m1=m1, m2=m2, n_e=n_e) for p in curr_p]
    r3, r4 = r3[1:], r4[1:]

    r5 = [re_Approx_bin(X_nA_y, A_i, p, m1, m2) for p in curr_p]
    r6 = [Approx_bin(X_nA_y, A_i, m1, m2, func=p) for p in curr_d]
    r7 = prev_Approx_bin(X_nA_y, A_i, indices[1][0], m1, m2)
    r5, r6 = r5[1:], r6[2:]
    r8 = [re_Approx_nonbin(X_nA_y, A_i, p, m1, m2, n_e) for p in curr_p]
    r9 = [Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, func=p) for p in curr_d]
    r10 = prev_Approx_nonbin(X_nA_y, A_i, m1, m2, n_e)
    r8, r9 = r8[1:], r9[2:]
    # pdb.set_trace()

    # for n1, n2, n3, n4 in zip(t1, t2, t3, t4):
    #     assert check_equal(n1[0], n2[0])  # and check_equal(n2[0], n3[0])
    for n1, n2, n3, n4 in zip(t1, r0, r1, r2):
        assert check_equal(n1[0], n3[0]) and check_equal(n2[0], n4[0])
    for n2, n3, n4 in zip(r1, r5, r6):
        assert poset_nolessthan(n3[0], n2[0])
        assert poset_nolessthan(n4[0], n2[0])
    for n2, n3, n4 in zip(r2, r8, r9):
        assert poset_nolessthan(n3[0], n2[0])
        assert poset_nolessthan(n4[0], n2[0])
    assert r7[0] >= r1[0][0][0] and poset_nolessthan(r10[0], r1[0][0])
    return


def together(curr, ind):
    from hfm.dist_drt import DirectDist_bin as prev_Direct_bin
    from hfm.dist_drt import DirectDist_nonbin as prev_Direct_nonbin
    # from hfm.dist_cvg_nonbin import ApproxDist_nonbin as Approx_nonv2
    from hfm.dist_cvg_nonbin import StratVacant, StratEarlyStop, StratRearrange

    from hfm.earlybreak import EffHD_bin as prev_Eff_bin
    from hfm.earlybreak import EffHD_nonbin as prev_Eff_nonbin
    # from hfm.manf.earlybreak import EffHD_bin, EffHD_nonbin

    from hfm.manf.earlybreak_ver3 import EffHD_bin, EffHD_nonbin
    from hfm.manf.earlybreak_ver4 import EffHD_bin as re_Eff_bin
    from hfm.manf.earlybreak_ver4 import EffHD_nonbin as re_Eff_nonbin

    prev_Eff_bin(X_nA_y, ind[0])
    prev_Eff_nonbin(X_nA_y, ind)
    prev_Direct_bin(X_nA_y, ind[0])
    prev_Approx_bin(X_nA_y, curr, ind[0], m1, m2)
    prev_Direct_nonbin(X_nA_y, ind)
    prev_Approx_nonbin(X_nA_y, curr, m1, m2, n_e)
    StratVacant(X_nA_y, curr, m1, m2, n_e)
    StratEarlyStop(X_nA_y, curr, n_e)
    StratRearrange(X_nA_y, curr, m1, m2, n_e)

    nb_v1_eff_1 = prev_Eff_bin(X_nA_y, ind[0])
    nb_v1_eff_2 = prev_Eff_nonbin(X_nA_y, ind)
    nb_v32_ef1 = [EffHD_bin(X_nA_y, ind[0], p) for p in curr_d][2:]
    nb_v32_ef2 = [EffHD_nonbin(X_nA_y, ind, p) for p in curr_d][2:]
    # nb_v4_ef1 = [re_Eff_bin(X_nA_y, ind[0], p) for p in curr_p][1:]
    # nb_v4_ef2 = [re_Eff_nonbin(X_nA_y, ind, p) for p in curr_p][1:]
    nb_v4_ef1 = [re_Eff_bin(X_nA_y, curr, p, priv) for p in curr_p][1:]
    nb_v4_ef2 = [re_Eff_nonbin(X_nA_y, curr, p, priv) for p in curr_p][1:]

    na_v1_drt = prev_Direct_bin(X_nA_y, ind[0])
    nb_v1_drt = prev_Direct_nonbin(X_nA_y, ind)
    na_v1_app = prev_Approx_bin(X_nA_y, curr, ind[0], m1, m2)
    nb_v1_app = prev_Approx_nonbin(X_nA_y, curr, m1, m2, n_e)
    nb_v2_app = StratVacant(X_nA_y, curr, m1, m2, n_e)
    nb_v2_cvg_es = StratEarlyStop(X_nA_y, curr, n_e)
    nb_v2_cvg_ra = StratRearrange(X_nA_y, curr, m1, m2, n_e)

    nb_v3_dt1 = [Direct_bin(X_nA_y, curr, priv, ind[0], p) for p in curr_d][2:]
    nb_v3_dt2 = [Direct_nonbin(X_nA_y, curr, priv, ind, p) for p in curr_d][2:]
    nb_v3_ap1 = [Approx_bin(X_nA_y, curr, m1, m2, p) for p in curr_d][2:]
    nb_v3_ap2 = [Approx_nonbin(X_nA_y, curr, m1, m2, n_e, p) for p in curr_d][2:]
    nb_v3_es = [StratES_nonbin(X_nA_y, curr, n_e, p) for p in curr_d][2:]
    nb_v3_ra = [StratRA_nonbin(X_nA_y, curr, m1, m2, n_e, p) for p in curr_d][2:]

    nb_v4_d1 = [re_bin(X_nA_y, curr, p, priv) for p in curr_p][1:]
    nb_v4_d2 = [re_nonbin(X_nA_y, curr, p, priv) for p in curr_p][1:]
    nb_v4_a1 = [re_Approx_bin(X_nA_y, curr, p, m1, m2) for p in curr_p][1:]
    nb_v4_a2 = [re_Approx_nonbin(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
    nb_v4_es = [re_StratES(X_nA_y, curr, p) for p in curr_p][1:]
    nb_v4_ra = [re_StratRA(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]

    # assert nb_v4_d1[0][-1] > nb_v4_a1[0][-1]
    # assert poset_nolessthan([nb_v4_d2[0][-1]] * 3, [
    #     nb_v4_a2[0][-1], nb_v4_es[0][-1], nb_v4_ra[0][-1], ])

    # from hfm.manf.renew_cvg import StratES_nonbin_alt2 as alt2_StratES
    # from hfm.manf.renew_cvg import StratES_nonbin_alt as alt_StratES
    # tmp_es_v4 = [alt_StratES(X_nA_y, curr, p) for p in curr_p][1:]
    # tmp_es_v5 = [alt2_StratES(X_nA_y, curr, p) for p in curr_p][1:]
    # from hfm.manf.renew_cvg import StratRA_nonbin_alt2 as alt2_StratRA
    # from hfm.manf.renew_cvg import StratRA_nonbin_alt as alt_StratRA
    # tmp_ra_v6 = [alt_StratRA(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
    # tmp_ra_v7 = [alt2_StratRA(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
    # tmp_ra_v8 = [re_StratRA(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
    pdb.set_trace()

    assert nb_v3_dt1[0][-1] > nb_v3_ap1[0][-1]
    assert poset_nolessthan([nb_v3_dt2[0][-1]] * 3, [
        nb_v3_ap2[0][-1], nb_v3_es[0][-1], nb_v3_ra[0][-1], ])

    assert na_v1_drt[-1] > nb_v3_dt1[0][-1] > nb_v4_d1[0][-1]
    assert na_v1_app[-1] > nb_v3_ap1[0][-1] > nb_v4_a1[0][-1]
    assert nb_v1_drt[-1] > nb_v3_dt2[0][-1] > nb_v4_d2[0][-1]
    assert nb_v1_app[-1] > nb_v3_ap2[0][-1] > nb_v4_a2[0][-1]
    assert nb_v2_cvg_es[-1] > nb_v3_es[0][-1] > nb_v4_es[0][-1]
    assert nb_v2_cvg_ra[-1] > nb_v3_ra[0][-1] > nb_v4_ra[0][-1]

    assert nb_v1_eff_1[-1] > nb_v32_ef1[0][-1] > nb_v4_ef1[0][-1]
    assert nb_v1_eff_2[-1] > nb_v32_ef2[0][-1] > nb_v4_ef2[0][-1]
    return


def test_drt():
    dist()
    # drt_bin()
    # drt_nonbin()
    # app_bin()
    # app_nonbin()
    together(B_i, B_index)
    together(A_i, indices[1])

    # w0 = np.random.rand(5)
    # from hfm.manf.parm_hfm import dual_normalized_vec, dual_normalize
    # w1 = dual_normalized_vec(w0, 3)
    # w3 = dual_normalized_vec(w0, 3)
    # w2 = dual_normalize(w0, 3)
    # pdb.set_trace()
    return
