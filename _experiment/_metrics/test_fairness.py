# coding: utf-8


import numpy as np
# import pdb
from hfm.utils.simulator import synthetic_clf, synthetic_set
from hfm.utils.verifiers import judge_transform_need, check_equal

from hfm.metrics.fairness_grp import (
    marginalised_pd_mat, prev_unpriv_manual, prev_unpriv_unaware,
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)
from hfm.metrics.fair_grp_ext import (
    marginalised_np_mat, unpriv_manual, unpriv_unaware,
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    zero_division, calc_fair_group, StatsParity_sing,
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing)
from experiment.utils.fair_rev_group import (
    UD_grp1_DP, UD_grp2_EO, UD_grp3_PQP)


nb_spl, nb_lbl, nb_clf = 371, 3, 2  # nb_clf=7
y_bin, _, _ = synthetic_set(2, nb_spl, nb_clf)
y_non, _, _ = synthetic_set(nb_lbl, nb_spl, nb_clf)
ht_bin = synthetic_clf(y_bin, nb_clf, err=.4)
ht_non = synthetic_clf(y_non, nb_clf, err=.4)

idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
idx_Sjs = [idx_priv == 1, idx_priv == 0]
multi_pv = np.random.randint(3, size=nb_spl, dtype='int')
Sjs_bin = [multi_pv == 1, multi_pv != 1]
Sjs_non = [multi_pv == 1, multi_pv == 0, multi_pv == 2]


def test_group_fair():
    def subroutine(y, hx, pos, priv):
        vY, dY = judge_transform_need(y)
        vY = vY[:: -1]
        z, ht = np.array(y), np.array(hx)
        g1M, g0M = marginalised_np_mat(z, ht, pos, priv)
        _, _, c1, c0 = marginalised_pd_mat(z, ht, pos, priv)

        just_one = unpriv_group_one(g1M, g0M)
        just_two = unpriv_group_two(g1M, g0M)
        just_thr = unpriv_group_thr(g1M, g0M)
        just_zero = unpriv_unaware(g1M, g0M)
        just_four = unpriv_manual(g1M, g0M)
        assert check_equal(just_one, prev_unpriv_grp_one(c1, c0))
        assert check_equal(just_two, prev_unpriv_grp_two(c1, c0))
        assert check_equal(just_thr, prev_unpriv_grp_thr(c1, c0))
        assert check_equal(just_zero, prev_unpriv_unaware(c1, c0))
        assert check_equal(just_four, prev_unpriv_manual(c1, c0))

        assert zero_division(0., 0.) == 0.
        assert zero_division(1., 0.) == 10
        assert zero_division(1.5, 0.2) == 7.5
        ans = calc_fair_group(*just_one)
        res = StatsParity_sing(ht, idx_Sjs, pos)[0]
        tmp = extGrp1_DP_sing(z, ht, idx_Sjs, pos)[0][: -1]
        assert check_equal(res, tmp)
        assert 0. <= ans <= 1.

        # pdb.set_trace()
    subroutine(y_bin, ht_bin[0], 1, idx_priv)
    subroutine(y_non, ht_non[0], 1, idx_priv)
    return


def test_fair_update():
    vY, dY = judge_transform_need(y_bin)
    vY = vY[:: -1]
    z, ht = np.array(y_bin), np.array(ht_bin[0])
    g1, g0 = marginalised_np_mat(z, ht, 1, Sjs_bin[0])

    just_one = unpriv_group_one(g1, g0)
    just_two = unpriv_group_two(g1, g0)
    just_thr = unpriv_group_thr(g1, g0)
    ans_1 = calc_fair_group(*just_one)
    ans_2 = calc_fair_group(*just_two)
    ans_3 = calc_fair_group(*just_thr)
    val_inA = [1, 0, 2]

    def sub_update(ans, pos, priv, func, extcls, grpstr):
        tmp_1 = func(z, ht, Sjs_bin, pos)[0]
        tmp_2 = func(z, ht, Sjs_non, pos)[0]
        res_0 = extcls.bival(z, ht, Sjs_bin[0], pos)[0]
        res_1 = extcls.mu_sp(z, ht, priv, 1, pos)[0]
        res_2 = extcls.mu_cx(z, ht, priv, 1, pos)[0]
        res_3 = extcls.yev_sp(z, ht, priv, val_inA, pos)[0]
        res_4 = extcls.yev_cx(z, ht, priv, val_inA, pos)[0]

        assert len(tmp_1[-1]) == 2 and len(tmp_2[-1]) == 3
        assert ans == res_0[0] == res_1
        # pdb.set_trace()
        # if grpstr in ('DP'):
        #     assert check_equal(res_1, res_2[1])

    sub_update(ans_1, 1, multi_pv, extGrp1_DP_sing, UD_grp1_DP, 'DP')
    sub_update(ans_2, 1, multi_pv, extGrp2_EO_sing, UD_grp2_EO, 'EO')
    sub_update(ans_3, 1, multi_pv, extGrp3_PQP_sing, UD_grp3_PQP, 'PP')
    return
