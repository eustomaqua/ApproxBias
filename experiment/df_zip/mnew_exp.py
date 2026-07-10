# coding: utf-8


import pdb
import time
import numpy as np
from hfm.utils.verifiers import unique_column, DTY_FLT, DTY_INT

from hfm.scrap.mf_dist_internal import STRATEGIES
from hfm.manf.renew_core import curr_intermediate
from hfm.manf.renew_drt import Direct_bin, Direct_nonbin
from hfm.manf.renew_app import Approx_bin, Approx_nonbin
from hfm.manf.renew_cvg import StratES_nonbin, StratRA_nonbin
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4
from hfm.manf.earlybreak_ver4 import EffHD_bin, EffHD_nonbin
from hfm.manf.earlybreak_ver4 import Naive_bin, Naive_nonbin


from pyfair.facil.metric_cont import contg_tab_mu_type2 as contingency_tab
from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_specificity, imba_geometric_mean, imba_discriminant_power)
from pyfair.marble.metric_fair import (
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing, alterGrps_sing,
    marginalised_pd_mat)
from hfm.discriminative_risk import hat_L_fair, hat_L_loss

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# from experiment.utils_learner import (
#     INDIVIDUALS, LGBMClassifier, FairGBMClassifier, AdaFair)
from pyfair.marble.metric_fair import prev_unpriv_grp_one as sa_grp_dp
from pyfair.marble.metric_fair import prev_unpriv_grp_two as sa_grp_eo
from pyfair.marble.metric_fair import prev_unpriv_grp_thr as sa_grp_pp


# =====================================
# 4convergance


# -------------------------------------
#


class DistPerformance:
    def __init__(self, priv_val, omitted=True):
        self._omit = omitted
        self._priv_val = priv_val


class cvgExp1C_take(DistPerformance):
    def single_sen_att(self, X_nA_y, A_i, m1, m2, n_e, func):
        # indices,
        ans_max, ans_avg, ans_tim = [], [], []
        tmp = Naive_bin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0])
        ans_tim.append(tmp[1])
        tmp = EffHD_bin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0])
        ans_tim.append(tmp[1])
        tmp = Direct_bin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])
        tmp = Approx_bin(X_nA_y, A_i, func, m1, m2)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])

        tmp = Naive_nonbin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0])
        ans_tim.append(tmp[1])
        tmp = EffHD_nonbin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0])
        ans_tim.append(tmp[1])
        tmp = Direct_nonbin(X_nA_y, A_i, func, self._priv_val)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])
        tmp = Approx_nonbin(X_nA_y, A_i, func, m1, m2, n_e)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])
        tmp = StratES_nonbin(X_nA_y, A_i, func)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])
        tmp = StratRA_nonbin(X_nA_y, A_i, func, m1, m2, n_e)
        ans_max.append(tmp[0][0])
        ans_avg.append(tmp[0][1])
        ans_tim.append(tmp[1])
        return ans_tim + ans_max + ans_avg  # (26=10+10+6,)

    def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2, n_e, func):
        #                  verbose=False):
        X_nA_y = np.concatenate([  # X_yfx
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_a = len(g1m_indices)     # A.shape[1]

        # if verbose:
        #     self.single_sen_att(X_nA_y, A[:, 0], m1, m2, n_e, func)

        res_curr = []   # non_sa = g1m_indices[0][0]
        res_curr += self.single_sen_att(
            X_nA_y, A[:, 0], m1, m2, n_e, func)  # g1m_indices[0],
        if n_a == 1:
            return res_curr + [''] * 26 + res_curr
        res_curr += self.single_sen_att(
            X_nA_y, A[:, 1], m1, m2, n_e, func)  # g1m_indices[0],

        tmp_sa1, tmp_sa2 = res_curr[:26], res_curr[26:]
        tmp_tim = np.add(tmp_sa1[:10], tmp_sa2[:10])
        tmp_max = np.maximum(tmp_sa1[10:-6], tmp_sa2[10:-6])
        tmp_avg = np.add(tmp_sa1[-6:], tmp_sa2[-6:]) / n_a
        tmp = np.concatenate([tmp_tim, tmp_max, tmp_avg], axis=0).tolist()
        del tmp_max, tmp_avg, tmp_tim
        del tmp_sa1, tmp_sa2
        return res_curr + tmp  # shape=(52+26,)

    def prepare_trial(self):
        csv_row_1 = unique_column(9 + 52)
        tmp_bin = ['Naive_bin', 'EffHD_bin', 'Direct_bin', 'Approx_bin']
        tmp_nonbin = ['Naive_nonbin', 'EffHD_nonbin', 'Direct_nonbin',
                      'Approx_nonbin', 'StratES_nonbin', 'StratRA_nonbin']
        csv_r4c = (tmp_bin + tmp_nonbin) * 2 + tmp_bin[2:] + tmp_nonbin[2:]
        csv_r4c = csv_r4c * 3
        # csv_r3c = ['tim'] + [''] * 9 + [
        #     'd^max'] + [''] * 9 + ['d^avg'] + [''] * 5
        # csv_r3c = csv_r3c * 3
        # csv_r3c[-1] = '[END]'
        # csv_r2c = ['sa#1'] + [''] * 25 + ['sa#2'] + [''] * 25 + [
        #     'together'] + [''] * 24 + ['[END]']

        csv_r2c = (['sa#1'] + [''] * 9) * 2 + ['sa#1'] + [''] * 5 + (
            ['sa#2'] + [''] * 9) * 2 + ['sa#2'] + [''] * 5 + (
            ['together'] + [''] * 9) * 2 + ['together'] + [''] * 4 + ['[END]']
        # csv_r3c = (['tim'] + [''] * 3 + ['tim'] + [''] * 5 + [
        #     'd^max'] + [''] * 3 + ['d^max'] + [''] * 5 + [
        #     'd^avg', '', 'd^avg', '', '', '[rim]']) * 3
        csv_r3c = (['tim'] + [''] * 3 + ['tim'] + [''] * 5 + [
            'd^max'] + [''] * 3 + ['d^max'] + [''] * 5 + [
            'd^avg', '', 'd^avg', '', '', '']) * 3
        csv_r3c[-1] = '[rim]'  # '[END]'
        return csv_row_1, csv_r2c, csv_r3c, csv_r4c


class cvgExp1D_take(DistPerformance):
    def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2, n_e, func):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_a = len(g1m_indices)

        Naive_bin(X_nA_y, A)
        return


class cvgExp1A_anal(DistPerformance):
    _m2_set = list(range(2, 14, 1))  # not 23, len=12

    def schedule_content(self, X, A, y_fx, g1m_indices, m1, n_e, n_p, func):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_l, n_a = len(self._m2_set), len(g1m_indices)

        # curr_res = [self.subproc_core_alt(
        #     X_yfx, A[:, i], g1m_indices[i], m1, n_e, n_l,
        #     func, n_p) for i in range(n_a)]
        res_tmp = {'wh': {'max': [], 'avg': [], 'tim': []},
                   0: {'max': [], 'avg': [], 'tim': []},
                   1: {'max': [], 'avg': [], 'tim': []}}

        (Ds, Ds_avg, Ds_midtmp), t_Ds = Direct_multivar(
            X_yfx, A, self._priv_val, g1m_indices, func, n_p)
        res_tmp['wh']['max'].append(Ds)
        res_tmp['wh']['avg'].append(Ds_avg)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            res_tmp[i]['max'].append(Ds_midtmp[0][i])
            res_tmp[i]['avg'].append(Ds_midtmp[1][i])
            res_tmp[i]['tim'].append(Ds_midtmp[2][i])
        if n_a == 1:
            res_tmp[1]['max'].append('')
            res_tmp[1]['avg'].append('')
            res_tmp[1]['tim'].append('')
        del Ds, Ds_avg, Ds_midtmp, t_Ds

        kw = dict(m1=m1, n_e=n_e, func=func, p=n_p)
        Strat = 'Vacant'  # STRATEGIES[0]
        ans_approx = [EffExact_multivar(
            X_yfx, A, Strat, m2=m2, **kw) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx
        res_tmp['wh']['max'].extend(hat_Ds)
        res_tmp['wh']['avg'].extend(hat_Ds_avg)
        res_tmp['wh']['tim'].extend(ans_ut)
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)  # .transpose(2, 1, 0)
        for i in range(n_a):
            res_tmp[i]['max'].extend(hat_Ds_midtmp[:, 0, i].tolist())
            res_tmp[i]['avg'].extend(hat_Ds_midtmp[:, 1, i].tolist())
            res_tmp[i]['tim'].extend(hat_Ds_midtmp[:, 2, i].tolist())
        if n_a == 1:
            res_tmp[1]['max'].extend([''] * n_l)
            res_tmp[1]['avg'].extend([''] * n_l)
            res_tmp[1]['tim'].extend([''] * n_l)

        Strat = 'ES'      # STRATEGIES[1]
        (hat_Ds, hat_Ds_avg, hat_Ds_midtmp), t_Ds = EffExact_multivar(
            X_yfx, A, Strat, m2=0, **kw)
        res_tmp['wh']['max'].append(hat_Ds)
        res_tmp['wh']['avg'].append(hat_Ds_avg)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            res_tmp[i]['max'].append(hat_Ds_midtmp[0][i])
            res_tmp[i]['avg'].append(hat_Ds_midtmp[1][i])
            res_tmp[i]['tim'].append(hat_Ds_midtmp[2][i])
        if n_a == 1:
            res_tmp[1]['max'].append('')
            res_tmp[1]['avg'].append('')
            res_tmp[1]['tim'].append('')

        Strat = 'RA'      # STRATEGIES[2]
        ans_approx = [EffExact_multivar(
            X_yfx, A, Strat, m2=m2, **kw) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx
        res_tmp['wh']['max'].extend(hat_Ds)
        res_tmp['wh']['avg'].extend(hat_Ds_avg)
        res_tmp['wh']['tim'].extend(ans_ut)
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)  # .transpose(2, 1, 0)
        for i in range(n_a):
            res_tmp[i]['max'].extend(hat_Ds_midtmp[:, 0, i].tolist())
            res_tmp[i]['avg'].extend(hat_Ds_midtmp[:, 1, i].tolist())
            res_tmp[i]['tim'].extend(hat_Ds_midtmp[:, 2, i].tolist())
        if n_a == 1:
            res_tmp[1]['max'].extend([''] * n_l)
            res_tmp[1]['avg'].extend([''] * n_l)
            res_tmp[1]['tim'].extend([''] * n_l)

        # res_curr = {'wh': res_tmp['wh'][
        #     'tim'] + res_tmp['wh']['max'] + res_tmp['wh']['avg'],
        #     0: res_tmp[0]['tim'] + res_tmp[0]['max'] + res_tmp[0]['avg'],
        #     1: res_tmp[1]['tim'] + res_tmp[1]['max'] + res_tmp[1]['avg']}
        # return res_curr['wh'] + res_curr[0] + res_curr[1]
        res_curr = {
            'tim': res_tmp['wh']['tim'] + res_tmp[0]['tim'] + res_tmp[1]['tim'],
            'max': res_tmp['wh']['max'] + res_tmp[0]['max'] + res_tmp[1]['max'],
            'avg': res_tmp['wh']['avg'] + res_tmp[0]['avg'] + res_tmp[1]['avg']}
        return res_curr['max'] + res_curr['avg'] + res_curr['tim']

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 78 * 3)
        csv_r4c = [f'm2={i}' for i in self._m2_set]  # n_l=12
        csv_r4c = [''] + csv_r4c + [''] + csv_r4c    # (1+n_l)*2   =26
        csv_r4c = csv_r4c * 3                        # (1+n_l)*2*3 =78
        n_l = len(self._m2_set) - 1
        csv_r3c = (['Direct_nonbin', 'Approx_nonbin'] + [''] * n_l + [
            'StratES', 'StratRA'] + [''] * n_l) * 2
        csv_r3c = ['Direct_multiver', 'ExtendDist'] + [''] * n_l + [
            'ExactDist(StratES)', 'ExactDist(StratRA)'] + [''] * n_l + csv_r3c
        n_l = (1 + len(self._m2_set)) * 2 - 1
        csv_r2c = ['d^max (*_multiver)'] + [''] * n_l + ['d^max sa#1'] + [
            ''] * n_l + ['d^max sa#2'] + [''] * n_l + ['d^avg (*_multiver)'] + [
            ''] * n_l + ['d^avg sa#1'] + [''] * n_l + ['d^avg sa#2'] + [
            ''] * n_l + ['T(*_multiver)'] + [''] * n_l + ['T() sa#1'] + [
            ''] * n_l + ['T() sa#2'] + [''] * n_l
        csv_r2c[-1] = '[END]'
        return csv_row_1, csv_r2c, csv_r3c * 3, csv_r4c * 3


class cvgExp1B_anal(DistPerformance):
    _m1_set = list(range(3, 34, 2))  # not 50, len=16

    def schedule_content(self, X, A, y_fx, g1m_indices, m2, n_e, n_p, func):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_l, n_a = len(self._m1_set), len(g1m_indices)
        res_tmp = {'wh': {'max': [], 'avg': [], 'tim': []},
                   0: {'max': [], 'avg': [], 'tim': []},
                   1: {'max': [], 'avg': [], 'tim': []}}

        (Ds, Ds_avg, Ds_midtmp), t_Ds = Direct_multivar(
            X_yfx, A, self._priv_val, g1m_indices, func, n_p)
        res_tmp['wh']['max'].append(Ds)
        res_tmp['wh']['avg'].append(Ds_avg)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            res_tmp[i]['max'].append(Ds_midtmp[0][i])
            res_tmp[i]['avg'].append(Ds_midtmp[1][i])
            res_tmp[i]['tim'].append(Ds_midtmp[2][i])
        if n_a == 1:
            res_tmp[1]['max'].append('')
            res_tmp[1]['avg'].append('')
            res_tmp[1]['tim'].append('')
        del Ds, Ds_avg, Ds_midtmp, t_Ds

        kw = dict(m2=m2, n_e=n_e, func=func, p=n_p)
        Strat = 'Vacant'  # STRATEGIES[0]
        ans_approx = [EffExact_multivar(
            X_yfx, A, Strat, m1=m1, **kw) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx
        res_tmp['wh']['max'].extend(hat_Ds)
        res_tmp['wh']['avg'].extend(hat_Ds_avg)
        res_tmp['wh']['tim'].extend(ans_ut)
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)
        for i in range(n_a):
            res_tmp[i]['max'].extend(hat_Ds_midtmp[:, 0, i].tolist())
            res_tmp[i]['avg'].extend(hat_Ds_midtmp[:, 1, i].tolist())
            res_tmp[i]['tim'].extend(hat_Ds_midtmp[:, 2, i].tolist())
        if n_a == 1:
            res_tmp[1]['max'].extend([''] * n_l)
            res_tmp[1]['avg'].extend([''] * n_l)
            res_tmp[1]['tim'].extend([''] * n_l)

        Strat = 'ES'      # STRATEGIES[1]
        (hat_Ds, hat_Ds_avg, hat_Ds_midtmp), t_Ds = EffExact_multivar(
            X_yfx, A, Strat, m1=0, **kw)
        res_tmp['wh']['max'].append(hat_Ds)
        res_tmp['wh']['avg'].append(hat_Ds_avg)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            res_tmp[i]['max'].append(hat_Ds_midtmp[0][i])
            res_tmp[i]['avg'].append(hat_Ds_midtmp[1][i])
            res_tmp[i]['tim'].append(hat_Ds_midtmp[2][i])
        if n_a == 1:
            res_tmp[1]['max'].append('')
            res_tmp[1]['avg'].append('')
            res_tmp[1]['tim'].append('')

        Strat = 'RA'      # STRATEGIES[2]
        ans_approx = [EffExact_multivar(
            X_yfx, A, Strat, m1=m1, **kw) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx
        res_tmp['wh']['max'].extend(hat_Ds)
        res_tmp['wh']['avg'].extend(hat_Ds_avg)
        res_tmp['wh']['tim'].extend(ans_ut)
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)  # .transpose(2, 1, 0)
        for i in range(n_a):
            res_tmp[i]['max'].extend(hat_Ds_midtmp[:, 0, i].tolist())
            res_tmp[i]['avg'].extend(hat_Ds_midtmp[:, 1, i].tolist())
            res_tmp[i]['tim'].extend(hat_Ds_midtmp[:, 2, i].tolist())
        if n_a == 1:
            res_tmp[1]['max'].extend([''] * n_l)
            res_tmp[1]['avg'].extend([''] * n_l)
            res_tmp[1]['tim'].extend([''] * n_l)

        res_curr = {
            'tim': res_tmp['wh']['tim'] + res_tmp[0]['tim'] + res_tmp[1]['tim'],
            'max': res_tmp['wh']['max'] + res_tmp[0]['max'] + res_tmp[1]['max'],
            'avg': res_tmp['wh']['avg'] + res_tmp[0]['avg'] + res_tmp[1]['avg']}
        return res_curr['max'] + res_curr['avg'] + res_curr['tim']

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 102 * 3)
        csv_r4c = [f'm1={i}' for i in self._m1_set]  # n_l=16
        csv_r4c = [''] + csv_r4c + [''] + csv_r4c    # (1+n_l)*2   =34
        csv_r4c = csv_r4c * 3                        # (1+n_l)*2*3 =102
        n_l = len(self._m1_set) - 1
        csv_r3c = (['Direct_nonbin', 'Approx_nonbin'] + [''] * n_l + [
            'StratES', 'StratRA'] + [''] * n_l) * 2
        csv_r3c = ['Direct_multiver', 'ExtendDist'] + [''] * n_l + [
            'ExactDist(StratES)', 'ExactDist(StratRA)'] + [''] * n_l + csv_r3c
        n_l = (1 + len(self._m1_set)) * 2 - 1
        csv_r2c = ['d^max (*_multiver)'] + [''] * n_l + ['d^max sa#1'] + [
            ''] * n_l + ['d^max sa#2'] + [''] * n_l + ['d^avg (*_multiver)'] + [
            ''] * n_l + ['d^avg sa#1'] + [''] * n_l + ['d^avg sa#2'] + [
            ''] * n_l + ['T(*_multiver)'] + [''] * n_l + ['T() sa#1'] + [
            ''] * n_l + ['T() sa#2'] + [''] * n_l
        csv_r2c[-1] = '[END]'
        return csv_row_1, csv_r2c, csv_r3c * 3, csv_r4c * 3


# -------------------------------------
#


# =====================================
#
