# coding: utf-8

import numpy as np
import pdb
from hfm.utils.verifiers import check_equal, poset_nolessthan

from hfm.manf.parm_hfm import (
    direct_single_sa, strat_es_single_sa, strat_ra_single_sa)
from hfm.manf.renew_drt import (
    lp_norm_vec, dual_norm_vec, dual_normalize, _as_float_p)


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
curr_p = [7, 1, 2, float("inf"), 3, ]
curr_d = ['cos_sim', 'correla',   # 'minkowski (p=3)'
          'manhattan', 'euclidean', 'chebyshev', 'minkowski']
ele_i, ele_ic = X_nA_y[:2]


def dist():
    from hfm.manf.dist_internal import alter_intermediate
    d0 = [alter_intermediate(ele_i, ele_ic, p) for p in [4, 5, 0, 1, 3, 2]]
    d1 = [lp_norm_vec(ele_i - ele_ic, _as_float_p(p)) for p in [4, 7, 2, 1, 3, 'inf']]
    assert check_equal(d0[2:], d1[2:])

    from hfm.manf.renew_drt import dual_exponent
    from hfm.manf.renew_cvg import orthogonal_weight
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
    # pdb.set_trace()
    return


def drt_bin():
    from hfm.manf.dist_internal import (Direct_bin, Direct_nonbin)
    from hfm.manf.dist_external import StratES_nonbin, StratRA_nonbin
    from hfm.manf.renew_drt import Direct_bin as re_bin
    from hfm.manf.renew_drt import Direct_nonbin as re_nonbin
    from hfm.manf.renew_drt import direct_sing_sa as re_sing_sa

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

    pdb.set_trace()
    return


def drt_nonbin():
    from hfm.manf.dist_internal import (Direct_bin, Direct_nonbin)
    from hfm.manf.dist_external import StratES_nonbin, StratRA_nonbin
    from hfm.manf.renew_drt import Direct_bin as re_bin
    from hfm.manf.renew_drt import Direct_nonbin as re_nonbin
    from hfm.manf.renew_drt import direct_sing_sa as re_sing_sa
    from hfm.manf.renew_cvg import StratES_nonbin as re_StratES
    from hfm.manf.renew_cvg import StratRA_nonbin as re_StratRA

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

    pdb.set_trace()
    for d, t, r in zip(t3, r3, r2):
        assert check_equal(t[0], r[0])  # and check_equal(d[0], r[0])
    return


def test_drt():
    dist()
    # drt_bin()
    drt_nonbin()

    # w0 = np.random.rand(5)
    # from hfm.manf.parm_hfm import dual_normalized_vec, dual_normalize
    # w1 = dual_normalized_vec(w0, 3)
    # w3 = dual_normalized_vec(w0, 3)
    # w2 = dual_normalize(w0, 3)
    # pdb.set_trace()
    return
