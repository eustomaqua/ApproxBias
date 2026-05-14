# coding: utf-8


import pdb
import time
import numpy as np
from hfm.utils.verifiers import unique_column, DTY_FLT, DTY_INT

from hfm.manf.dist_internal import (
    Direct_bin, Direct_nonbin, Direct_multiver, Approx_bin)
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
from hfm.earlybreak import EffHD_bin, EffHD_nonbin, EffHD_multivar
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# from experiment.utils_learner import (
#     INDIVIDUALS, LGBMClassifier, FairGBMClassifier, AdaFair)


# import torch
# import torch.nn as nn
# import torch.optim as optim
from pyfair.marble.metric_fair import prev_unpriv_grp_one as sa_grp_dp
from pyfair.marble.metric_fair import prev_unpriv_grp_two as sa_grp_eo
from pyfair.marble.metric_fair import prev_unpriv_grp_thr as sa_grp_pp


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

    def subproc_core_alt(self, X_yfx, A_j, idx_Sjs, m1, m2, n_e,
                         n_p=3, func='euclidean', priv_val=1):

        return


class cvgExp1C_take(DistPerformance):
    def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2, n_e, n_p):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)

        func = 'euclidean'
        non_sa = g1m_indices[0][0]  # idx_Sjs[0]  # .astype(DTY_INT)
        (Ds, Ds_avg), t_Ds = Direct_bin(X_yfx, A_j, priv_val, non_sa, func, n_p)
        t1 = Direct_nonbin(X_yfx, non_sa.astype(DTY_INT), priv_val, [non_sa, ~non_sa], func, n_p)
        t2 = Direct_nonbin(X_yfx, A_j, priv_val, idx_Sjs, func, n_p)
        # Approx_bin()
        pdb.set_trace()
        return


class cvgExp1A_anal(DistPerformance):
    _m2_set = list(range(2, 14, 1))  # not 23, len=12

    def schedule_content(self, X, A, y_fx, g1m_indices, m1, n_e, n_p):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_l, n_a = len(self._m2_set), len(g1m_indices)

        self.subproc_core_alt(X_yfx, A[:, 0], g1m_indices[0], m1, 4, n_e, n_p, 'euclidean', self._priv_val[0])
        return

    # def subproc_core_alt(self, X_yfx, A_j, non_sa, m1, n_e, n_l,
    #                      func='euclidean', n_p=3):
    #     return


class cvgExp1B_anal(DistPerformance):
    _m1_set = list(range(3, 34, 2))  # not 50, len=16

    def schedule_content(self, X, A, y_fx, g1m_indices, m2, n_e, n_p):
        X_y_fx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_l, n_a = len(self._m1_set), len(g1m_indices)
        return

    def subproc_core_alt(self, X_yfx, A_j, non_sa, m2, n_e, n_l):
        return


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
