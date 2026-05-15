# coding: utf-8


import pdb
import time
import numpy as np
from hfm.utils.verifiers import unique_column, DTY_FLT, DTY_INT

from hfm.manf.dist_internal import (
    Direct_bin, Direct_nonbin, Direct_multiver, Approx_bin, STRATEGIES)
from hfm.manf.dist_external import (
    Approx_nonbin, StratES_nonbin, StratRA_nonbin, EffExact_multiver)
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4


from pyfair.facil.metric_cont import contg_tab_mu_type2 as contingency_tab
from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_specificity, imba_geometric_mean, imba_discriminant_power)
from pyfair.marble.metric_fair import (
    # prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr,
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing, alterGrps_sing,
    marginalised_pd_mat,)
from hfm.discriminative_risk import hat_L_fair, hat_L_loss

# from hfm.earlybreak import Naive_bin as NaiveHD_bin
# from hfm.earlybreak import Naive_nonbin as NaiveHD_nonbin
# from hfm.earlybreak import Naive_multivar as Naive_multivar
# from hfm.earlybreak import EffHD_bin, EffHD_nonbin, EffHD_multivar
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# from experiment.utils_learner import (
#     INDIVIDUALS, LGBMClassifier, FairGBMClassifier, AdaFair)


# import torch
# import torch.nn as nn
# import torch.optim as optim
from pyfair.marble.metric_fair import prev_unpriv_grp_one as sa_grp_dp
from pyfair.marble.metric_fair import prev_unpriv_grp_two as sa_grp_eo
from pyfair.marble.metric_fair import prev_unpriv_grp_thr as sa_grp_pp
from hfm.manf.earlybreak import EffHD_multiver  # EffHD_bin,EffHD_nonbin,


# =====================================
# 4convergance


# -------------------------------------
#


class DistPerformance:
    # _m2_set = list(range(2, 23, 1))  # len=21
    # _m1_set = list(range(3, 50, 2))  # len=24

    def __init__(self, priv_val, omitted=True):
        self._omit = omitted  # pass
        self._priv_val = priv_val

    # def subproc_core_alt(self, X_yfx, A_j, idx_Sjs, m1, m2, n_e,
    #                      n_p=3, func='euclidean', priv_val=1):
    #     non_sa = idx_Sjs[0]
    #     B_j = (A_j == priv_val).astype(DTY_INT)
    #     t1 = Direct_bin(X_yfx, B_j, priv_val, non_sa, func, n_p)
    #     t2 = Direct_nonbin(X_yfx, B_j, priv_val, [non_sa, ~non_sa], func, n_p)
    #     t3 = Direct_nonbin(X_yfx, A_j, priv_val, [non_sa, ~non_sa], func, n_p)
    #     t4 = Direct_bin(X_yfx, A_j, priv_val, non_sa, func, n_p)
    #     t5 = Direct_nonbin(X_yfx, A_j, priv_val, idx_Sjs, func, n_p)
    #     pdb.set_trace()
    #     return
    #
    # def subproc_core_sup(self, X_yfx, A, indices, m1, m2, n_e,
    #                      n_p=3, func='euclidean', priv_val=1):
    #     # alt = Direct_multiver(X_yfx, A, priv_val, [
    #     #     [idx[0], ~idx[0]] for idx in indices], func, n_p)
    #     tmp = Direct_multiver(X_yfx, A, priv_val, indices, func, n_p)
    #     pdb.set_trace()
    #     return


class cvgExp1C_take(DistPerformance):
    def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2, n_e, n_p,
                         func='euclidean'):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_a = len(g1m_indices)     # A.shape[1]
        # priv_val = set(self._priv_val).pop()
        # res_wh = []
        # res_sa = {0: [], 1: []}
        # res_tim = {'wh': [], 0: [], 1: []}
        res_tmp = {'wh': {'max': [], 'avg': [], 'tim': []},
                   0: {'max': [], 'avg': [], 'tim': []},
                   1: {'max': [], 'avg': [], 'tim': []}}

        non_sa = g1m_indices[0][0]
        (Ds, Ds_avg, half_tmp), t_Ds = Direct_multiver(
            X_yfx, A, self._priv_val, g1m_indices, func, n_p)
        # half_tmp = list(zip(*half_tmp))
        # res_wh.extend([Ds, Ds_avg, t_Ds])
        res_tmp['wh']['max'].append(Ds)
        res_tmp['wh']['avg'].append(Ds_avg)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            # res_sa[i].extend(half_tmp[i])
            res_tmp[i]['max'].append(half_tmp[0][i])
            res_tmp[i]['avg'].append(half_tmp[1][i])
            res_tmp[i]['tim'].append(half_tmp[2][i])
        if n_a == 1:
            # res_sa[1].extend([''] * 3)
            res_tmp[1]['max'].append('')
            res_tmp[1]['avg'].append('')
            res_tmp[1]['tim'].append('')

        (Ds, half_tmp), t_Ds = EffHD_multiver(X_yfx, g1m_indices, func, n_p)
        # (Ds, half_tmp), t_Ds = EffHD_multivar(X_yfx, g1m_indices)
        res_tmp['wh']['max'].append(Ds)
        res_tmp['wh']['tim'].append(t_Ds)
        for i in range(n_a):
            res_tmp[i]['max'].append(half_tmp[0][i])
            res_tmp[i]['tim'].append(half_tmp[1][i])
        if n_a == 1:
            res_tmp[1]['max'].append('')
            res_tmp[1]['tim'].append('')

        for Strat in ['Vacant', 'StratES', 'StratRA']:
            (Ds, Ds_avg, half_tmp), t_Ds = EffExact_multiver(
                X_yfx, A, Strat, m1, m2, n_e, func, n_p)
            # half_tmp = list(zip(*half_tmp))
            res_tmp['wh']['max'].append(Ds)
            res_tmp['wh']['avg'].append(Ds_avg)
            res_tmp['wh']['tim'].append(t_Ds)
            for i in range(n_a):
                res_tmp[i]['max'].append(half_tmp[0][i])
                res_tmp[i]['avg'].append(half_tmp[1][i])
                res_tmp[i]['tim'].append(half_tmp[2][i])
            if n_a == 1:
                res_tmp[1]['max'].append('')
                res_tmp[1]['avg'].append('')
                res_tmp[1]['tim'].append('')

        res_curr = {  # (3+2+3*3)*3 =(5+9)*3 =14*3=42
            'max': res_tmp['wh']['max'] + res_tmp[0]['max'] + res_tmp[1]['max'],
            'avg': res_tmp['wh']['avg'] + res_tmp[0]['avg'] + res_tmp[1]['avg'],
            'tim': res_tmp['wh']['tim'] + res_tmp[0]['tim'] + res_tmp[1]['tim']}
        # if func in ['cos_sim', 'correla']:
        #     pdb.set_trace()
        return res_curr['tim'] + res_curr['max'] + res_curr['avg']

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 42)
        csv_r2c = ['tim'] + [''] * 14 + ['d^max'] + [''] * 14 + ['d^avg'] + [''] * 11
        csv_r3c = ['wh'] + [''] * 4 + ['sa#1'] + [''] * 4 + ['sa#2'] + [''] * 4 + [
            'wh'] + [''] * 4 + ['sa#1'] + [''] * 4 + ['sa#2'] + [''] * 4 + [
            'wh'] + [''] * 3 + ['sa#1'] + [''] * 3 + ['sa#2'] + [''] * 3
        csv_r4c = ['Direct_multiver', 'EarlyBreak', 'ExtendDist',
                   'ExactDist(StratES)', 'ExactDist(StratRA)'] + [
            'Direct_nonbin', 'EarlyBreak', 'Approx_nonbin', 'StratES', 'StratRA'
        ] * 2 + ['Direct_multiver', 'EarlyBreak', 'ExtendDist',
                 'ExactDist(StratES)', 'ExactDist(StratRA)'] + [
            'Direct_nonbin', 'EarlyBreak', 'Approx_nonbin', 'StratES', 'StratRA'
        ] * 2 + ['Direct_multiver', 'ExtendDist', 'ExactDist(StratES)',
                 'ExactDist(StratRA)'] + [
            'Direct_nonbin', 'Approx_nonbin', 'StratES', 'StratRA'] * 2
        csv_r2c[-1] = '[END]'
        csv_r3c[-1] = '[END]'
        return csv_row_1, csv_r2c, csv_r3c, csv_r4c


# class cvgExp1D_take(DistPerformance):
#     def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2, n_e, n_p,
#                          func='euclidean'):
#         X_yfx = np.concatenate([
#             y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
#         n_a = len(g1m_indices)
#         # priv_val = set(self._priv_val).pop()
#
#         pdb.set_trace()
#         return


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

        (Ds, Ds_avg, Ds_midtmp), t_Ds = Direct_multiver(
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
        ans_approx = [EffExact_multiver(
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
        (hat_Ds, hat_Ds_avg, hat_Ds_midtmp), t_Ds = EffExact_multiver(
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
        ans_approx = [EffExact_multiver(
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

        (Ds, Ds_avg, Ds_midtmp), t_Ds = Direct_multiver(
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
        ans_approx = [EffExact_multiver(
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
        (hat_Ds, hat_Ds_avg, hat_Ds_midtmp), t_Ds = EffExact_multiver(
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
        ans_approx = [EffExact_multiver(
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


# def subproc_core_alt(self, X_yfx, A_j, idx_Sjs, m1, n_e, n_l, func, n_p):
#     return
# def subproc_core_alt(self, X_yfx, A_j, non_sa, m2, n_e, n_l):
#     return


# class learner_fair_ens:
#     def __init__(self):
#         pass
#     def get_member_clf


# class learner_norm_cls:
#     def __init__(self):
#         pass


# -------------------------------------
#


# =====================================
#
