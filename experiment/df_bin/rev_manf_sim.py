# coding: utf-8


from copy import deepcopy
import csv
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import pdb

from hfm.utils.decorators import (
    elegant_dated, elegant_durat, elegant_durat_core)
from hfm.utils.recorders import elegant_print, get_elogger

from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    manual_repetitive, scale_normalize_helper)
from pyfair.datasets import preprocess
from pyfair.preprocessing_dr import (
    adversarial, transform_X_and_y, transform_unpriv_tag)
from pyfair.preprocessing_hfm import binarized_data_set
# from pyfair.preprocessing_hfm import (
#     binarized_data_set,  transform_X_A_and_y,
#     transform_disturb_prime, normalise_disturb_prime,
#     normalise_disturb_whole, renewed_transform_disturb,
#     renewed_normalise_disturb, renewed_normalise_separate)

from experiment.utils_empirical import DataSetup
from experiment.utils_learner import AVAILABLE_ENSF
# from experiment.datasets import (
#     preprocess, adversarial, transform_X_and_y,
#     transform_unpriv_tag)
from experiment.preprocessing_bin import (
    #     binarized_data_set, transform_X_A_and_y,
    #     transform_disturb_prime, normalise_disturb_prime,
    normalise_disturb_whole, transform_X_A_and_y)
from experiment.preprocessing_nonbin import (
    renewed_transform_disturb, renewed_normalise_disturb,
    renewed_normalise_separate)


from experiment.df_bin.manf_exp import (
    ComparisonB1_withDirectComput, ComparisonB2_withDirectComput,
    ComparisonC2_withDirectComput, ComparisonC4_withDirectComput,
    ComparisonC5_withDirectComput,)

from experiment.utils_learner import (
    INDIVIDUALS, LGBMClassifier, FairGBMClassifier, AdaFair)
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.dist_drt import DirectDist_bin as DirectDist
from hfm.dist_est_bin import ApproxDist_bin as ApproxDist
from hfm.hfm_df import bias_degree as fair_degree
from hfm.discriminative_risk import hat_L_fair, hat_L_loss
from hfm.earlybreak import EffHD_bin

from pyfair.facil.metric_cont import contg_tab_mu_type2 as contingency_table
from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    # calc_fpr, calc_fnr, calc_sensitivity, imba_Matthew_s_cc, imba_Cohen_s_kappa
    calc_specificity, imba_geometric_mean, imba_discriminant_power)
from pyfair.marble.metric_fair import (
    marginalised_pd_mat,  # prev_unpriv_unaware, prev_unpriv_manual,
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)

unpriv_group_one = prev_unpriv_grp_one
unpriv_group_two = prev_unpriv_grp_two
unpriv_group_thr = prev_unpriv_grp_thr
# unpriv_unaware = prev_unpriv_unaware
# unpriv_manual = prev_unpriv_manual
# del prev_unpriv_unaware, prev_unpriv_manual
del prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr


# ===============================
# Empirical

CURR_CLFS = ['bagging', 'adaboost', 'lightgbm',
             'fairgbm : FPR', 'fairgbm : FNR', 'fairgbm : FPR,FNR',
             'adafair']


# -------------------------------
# RQ2. How efficient is ApproxDist compared with the direct
#      computation of distances in Eq.(5)?


class PartH_efficient:
    _abbr_clfs = CURR_CLFS

    def __init__(self, nb_cls=1, saIndex=tuple(), saValue=tuple):
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        # self._abbr_clfs = CURR_CLFS

    def count_sing_part1(self, y, y_hat, pos_label):
        # NB. must be np.ndarray
        tp, fp, fn, tn = contingency_table(y, y_hat, pos_label)
        res_indi = []

        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        sen = calc_recall(tp, fp, fn, tn)  # calc_sensitivity()
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe])
        res_indi.append(calc_f1_score(tp, fp, fn, tn))

        res_indi.append(imba_geometric_mean(sen, spe))
        res_indi.append(imba_discriminant_power(sen, spe))
        # res_indi.append(imba_Matthew_s_cc(tp, fp, fn, tn))
        # res_indi.extend(imba_Cohen_s_kappa(tp, fp, fn, tn))
        return res_indi  # shape=(7,)=(5+2,)

    def count_sing_part2(self, y, y_hat,  # y_qtb,
                         non_sa, pos_label=1):
        _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(
            y, y_hat, pos_label, non_sa)
        cmp_fair = []
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        # cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        # cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        # cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair

    def count_sing_part3(self,  # X, A, y, y_hat,
                         X_and_y, X_and_y_hat, A,
                         non_sa, m1=20, m2=8):
        cmp_fair = []
        # X_and_y = np.concatenate([
        #     y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        # X_and_y_hat = np.concatenate([
        #     y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)

        Ds, t_Ds = DirectDist(X_and_y, non_sa)
        Df, t_Df = DirectDist(X_and_y_hat, non_sa)
        Ds, Df = Ds[0], Df[0]
        ddf, t_ddf = fair_degree(Ds, Df)
        cmp_fair.extend((Ds, Df) + ddf + (t_Ds, t_Df, t_Ds + t_Df + t_ddf))

        Ds, t_Ds = EffHD_bin(X_and_y, non_sa)
        Df, t_Df = EffHD_bin(X_and_y_hat, non_sa)
        ddf, t_ddf = fair_degree(Ds, Df)
        cmp_fair.extend((Ds, Df) + ddf + (t_Ds, t_Df, t_Ds + t_Df + t_ddf))

        Ds, t_Ds = ApproxDist(X_and_y, A, non_sa, m1, m2)
        Df, t_Df = ApproxDist(X_and_y_hat, A, non_sa, m1, m2)
        ddf, t_ddf = fair_degree(Ds, Df)
        cmp_fair.extend((Ds, Df) + ddf + (t_Ds, t_Df, t_Ds + t_Df + t_ddf))
        return cmp_fair  # shape=(21,)=(7*3,)

    def count_single_member(self, X, A, y, y_hat, y_qtb, g1m, idx_jt,
                            pos_label=1, m1=20, m2=8):
        res_ans = []
        tmp = self.count_sing_part1(y, y_hat, pos_label)
        adv = self.count_sing_part1(y, y_qtb, pos_label)
        res_ans.extend(tmp)
        res_ans.extend([abs(i - j) for i, j in zip(tmp, adv)])

        # res_ans.extend(self.count_sing_part2(y, y_hat, y_qtb, g1))
        res_ans.extend(self.count_sing_part2(y, y_hat, g1m[0]))
        if len(g1m) == 1:
            res_ans.extend([''] * 3 * 3)
        else:
            res_ans.extend(self.count_sing_part2(y, y_hat, g1m[1]))
            res_ans.extend(self.count_sing_part2(y, y_hat, idx_jt[0]))
            res_ans.append(self.count_sing_part2(y, y_hat, idx_jt[1]))
        res_ans.append(hat_L_fair(y_hat, y_qtb))
        res_ans.append(hat_L_loss(y_hat, y))

        X_and_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_and_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        res_ans.extend(self.count_sing_part3(
            X_and_y, X_and_y_hat, A, g1m[0], m1, m2))
        if len(idx_jt) == 0:
            res_ans.extend([''] * 21 * 3)
        else:
            res_ans.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, A, g1m[1], m1, m2))
            res_ans.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, A, idx_jt[0], m1, m2))
            res_ans.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, A, idx_jt[1], m1, m2))
        return res_ans

    def count_scores(self,
                     X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                     X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                     positive_label=1, m1=20, m2=8):
        # X_nA_y_trn = np.concatenate([
        #     y_trn.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        # X_nA_y_tst = np.concatenate([
        #     y_tst.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        # X_nA_y_hat_trn = np.concatenate([
        #     y_insp.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        # X_nA_y_hat_tst = np.concatenate([
        #     y_pred.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        ans_trn = self.count_single_member(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            positive_label, m1, m2)
        ans_tst = self.count_single_member(
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        # pdb.set_trace()
        return ans_trn + ans_tst


class PartH1_earlybreak(PartH_efficient):
    def subroute_one_fair_ens(self, name_ens, nb_cls,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst=None,
                              constraint='FPR,FNR', sa_idx=None, sa_val=None):
        ut = time.time()

        if name_ens == 'bagging':
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'adaboost':
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'lightgbm':
            clf = LGBMClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)

        elif name_ens == 'fairgbm':
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint)
            clf.fit(X_A_trn, y_trn, constraint_group=~nsa_trn)
        elif name_ens == 'adafair':
            clf = AdaFair(n_estimators=nb_cls,
                          saIndex=sa_idx, saValue=sa_val)
            clf.fit(X_A_trn, y_trn)

        ut = time.time() - ut
        y_insp = clf.predict(X_A_trn)
        y_pred = clf.predict(X_A_tst)
        yq_insp = clf.predict(X_Aq_trn)
        yq_pred = clf.predict(X_Aq_tst)
        return y_insp, y_pred, yq_insp, yq_pred, ut

    def subroute_one_norm_att(self,  # name_ens, nb_cls,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              # sa_idx, sa_val,
                              positive_label=1, m1=20, m2=8,
                              X_trn=None, A_trn=None, g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None, g1_tst=None, jt_tst=None):
        res_attr = []
        nsa_trn = None
        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,  # nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,  # nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,  # nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,  # nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,  # nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,  # nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)
        return res_attr

    def subroute_one_sens_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              sa_idx, sa_val, positive_label=1, m1=20, m2=8,
                              X_trn=None, A_trn=None, g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None, g1_tst=None, jt_tst=None):
        res_attr = []

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                constraint=constraint_type)
            tmp = self.count_scores(  # clf,
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst,
            sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        return res_attr

    # def schedule_content(self, X, A, y_fx, idx_S0, idx_S1, m1):
    #     X_yfx = np.concatenate([
    #         y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
    #     return

    def schedule_content(self, logger,
                         X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                         X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                         X_trn, A_trn, X_tst, A_tst, m1, m2, pos_label):
        res_iter = []
        pms = {'positive_label': pos_label, 'm1': m1, 'm2': m2,
               'X_trn': X_trn, 'A_trn': A_trn, 'X_tst': X_tst, 'A_tst': A_tst,
               'g1_trn': g1_trn, 'jt_trn': jt_trn,
               'g1_tst': g1_tst, 'jt_tst': jt_tst}
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, None,
            X_A_tst, y_tst, X_Aq_tst, None, **pms)
        res_iter.extend(tmp)

        tmp = self.subroute_one_sens_att(
            X_A_trn, y_trn, X_Aq_trn, g1_trn[0],
            X_A_tst, y_tst, X_Aq_tst, g1_tst[0],
            self.saIndex[0], self.saValue[0], **pms)
        res_iter.extend(tmp)
        if len(jt_trn) == 0:
            return res_iter
        tmp = self.subroute_one_sens_att(
            X_A_trn, y_trn, X_Aq_trn, g1_trn[1],
            X_A_tst, y_tst, X_Aq_tst, g1_tst[1],
            self.saIndex[1], self.saValue[1], **pms)
        res_iter.extend(tmp)
        return res_iter

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 1 + 112 * 2)
        csv_row_2c = ['Ensem'] + ['Training set'] + [''] * 13 + [
            'Grp fairness'] + [''] * 11 + ['fairvote', '', 'fairmanf'] + [
            ''] * 27 + ['Test set'] + [''] * 13 + ['Grp fairness'] + [
            ''] * 11 + ['fairvote', '', 'fairmanf'] + [''] * 27

        tmp_3_1 = ['Normal'] + [''] * 6 + ['abs()'] + [''] * 6
        tmp_3_2 = ['sa#1,sa#2,jt*2', '', ''] + [''] * 3 * 3 + ['DR(loss)', '']
        tmp_3_3 = ['HFM sa#1'] + [''] * 6 + ['HFM sa#2'] + [''] * 6 + [
            'HFM jt#&'] + [''] * 6 + ['HFM jt#|'] + [''] * 6
        tmp_3 = tmp_3_1 + tmp_3_2 + tmp_3_3
        csv_row_3c = ['Time cost (sec)'] + tmp_3 + tmp_3
        del tmp_3, tmp_3_1, tmp_3_2, tmp_3_3

        tmp_4_1 = ['Accuracy', 'Precision', 'Recall/sensitivity',
                   'Specificity', 'f1_score', 'g_mean', 'dp']
        tmp_4_2 = ['g1', 'g0'] * 3 * 4 + ['hat_L(fair)', 'hat_L(loss)']
        tmp_4_3 = ['Ds', 'Df', 'ddf', '', 't_Ds', 't_Df', 't(...)']  # *4
        tmp_4 = tmp_4_1 * 2 + tmp_4_2 + tmp_4_3 * 4  # 7*2+14+28=56
        csv_row_4c = ['ut'] + tmp_4 + tmp_4_3
        del tmp_4, tmp_4_1, tmp_4_2, tmp_4_3  # 

        pdb.set_trace()
        # return [], [], [], []
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class PartH2_earlybreak(PartH1_earlybreak):
    _abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                  'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def subroute_one_gene_clf(self, abbr_cls,  # name_ens, nb_cls,
                              X_A_trn, y_trn, X_Aq_trn,   # nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst):  # , nsa_tst):
        ut = time.time()
        clf = INDIVIDUALS[abbr_cls]
        clf.fit(X_A_trn, y_trn)
        ut = time.time() - ut
        y_insp = clf.predict(X_A_trn)
        y_pred = clf.predict(X_A_tst)
        yq_insp = clf.predict(X_Aq_trn)
        yq_pred = clf.predict(X_Aq_tst)
        return y_insp, y_pred, yq_insp, yq_pred, ut

    def schedule_content(self, logger,
                         X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                         X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                         X_trn, A_trn, X_tst, A_tst, m1, m2, pos_label):
        res_iter = []
        pms = {'positive_label': pos_label, 'm1': m1, 'm2': m2,
               'X_trn': X_trn, 'A_trn': A_trn, 'g1_trn': g1_trn, 'jt_trn': jt_trn,
               'X_tst': X_tst, 'A_tst': A_tst, 'g1_tst': g1_tst, 'jt_tst': jt_tst}
        for abbr_cls in self._abbr_clfs:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_gene_clf(
                abbr_cls, X_A_trn, y_trn, X_Aq_trn, X_A_tst, y_tst, X_Aq_tst)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                pos_label, m1, m2)
            res_iter.append([ut] + tmp)
        return res_iter


# -------------------------------
# Maniford empirical


class RevisedManfEmpirical(DataSetup):
    def __init__(self, trial_type, data_type,  # abbr_cls='DT',
                 nb_iter=5, m1=30, m2=10, m2_fixed=False, ratio=.5,
                 prep=False, nb_cls=1, gen=False, rep=False,
                 screen=True, logged=False):  # constraint_type='FPR,FNR'
        super().__init__(data_type)
        self._ratio = ratio
        # self.preparing_iterator(
        #     trial_type, nb_iter, nb_cls, gen, rep,
        #     m1, m2, prep, screen, logged)
        self._m2_fixed = m2_fixed
        self.preparing_iterator(trial_type, nb_iter, m1, m2, prep,
                                nb_cls, gen, rep, screen, logged)

    def preparing_iterator(self, trial_type, nb_iter, m1, m2,
                           prep, nb_cls, gen, rep,  # constraint_type,
                           screen=True, logged=False):
        self._trial_type = trial_type
        # self._abbr_cls = abbr_cls
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep     # cv-split
        self._prep = prep        # pre-processing data

        self._log_document = "_".join([
            trial_type,
            "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
            f'{prep}_cls{nb_cls}', self._log_document,
            'pms', f'rat{int(self._ratio * 100)}'])
        self._screen, self._logged = screen, logged
        self._m1, self._m2 = m1, m2

        self._nb_cls = nb_cls
        # self._constraint_type = constraint_type
        # self.saIndex = [-1] if self._data_type == 'ricci' else [-2, -1]

        if trial_type.endswith('expt2a'):
            self._iterator = ComparisonB1_withDirectComput(
                nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2b'):
            self._iterator = ComparisonB2_withDirectComput(
                nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2c'):
            self._iterator = ComparisonC2_withDirectComput(
                nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2d'):
            self._iterator = ComparisonC4_withDirectComput(
                nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2e'):
            self._iterator = ComparisonC5_withDirectComput(
                nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt8a'):
            self._iterator = PartH1_earlybreak(nb_cls, self.saIndex, self.saValue)
        elif trial_type.endswith('expt8b'):
            self._iterator = PartH2_earlybreak(nb_cls, self.saIndex, self.saValue)

        self._log_document += ('_gen' * gen + '_rep' * rep)
        return

    def preparing_current_data(self, logger=None):
        processed_dat = preprocess(self._dataset, self._data_frame, logger)
        disturbed_dat = adversarial(
            self._dataset, self._data_frame, self._ratio, logger)
        processed_Xy = processed_dat['numerical-binsensitive']
        disturbed_Xy = disturbed_dat['numerical-binsensitive']
        # binarized_Xy = binarized_data_set(processed_Xy)

        X, A, _, _ = transform_X_A_and_y(self._dataset, processed_Xy)
        _, Aq, _, _ = transform_X_A_and_y(self._dataset, disturbed_Xy)

        new_attr = self._dataset.sensitive_attrs
        new_attr = '-'.join(new_attr) if len(new_attr) > 1 else ''
        if new_attr:
            processed_Xy.drop(new_attr, axis=1, inplace=True)
            disturbed_Xy.drop(new_attr, axis=1, inplace=True)
            # binarized_Xy.drop(new_attr, axis=1, inplace=True)

        X_A, y = transform_X_and_y(self._dataset, processed_Xy)
        X_Aq, y = transform_X_and_y(self._dataset, disturbed_Xy)
        if not self._m2_fixed:
            self._m2 = np.ceil(2 * np.log10(len(y)))
            self._m2 = int(self._m2)
        elegant_print(["Due to #inst = {}".format(len(y)),
                       "self._m2 = {}".format(self._m2),
                       "self._m1 = {}".format(self._m1)], logger)

        tmp = processed_dat['original'][self._dataset.label_name]
        elegant_print(["\t BINARY? Y= {}".format(set(y.values)),
                       "\t ori.label= {}".format(set(tmp.values)),
                       ""], logger)
        del tmp
        elegant_print(["\t X_A .shape {}".format(X_A.shape),
                       "\t X_Aq.shape {}".format(X_Aq.shape),
                       "\t y   .shape {}".format(y.shape), ""], logger)

        sens_att = self._dataset.get_sensitive_attrs_with_joint()[:2]
        priv_val = self._dataset.get_privileged_group_with_joint('')[:2]
        # marginalised_group =
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_dat['original'], 'both')

        tmp_cls = tmp_ens = ''
        res_aux = [[
            self._dataset.dataset_name, len(y.unique()), len(sens_att),
            self._nb_iter, self._gen_iter, self._rep_iter,
            self._m1, self._m2, '', ''], sens_att, priv_val]
        # pdb.set_trace()
        # return (X_A.values, y.values, X_Aq.values, new_attr, res_aux,
        #         belongs_priv, ptb_with_joint)
        return (X_A.values, X_Aq.values, y.values,  # X_A, X_Aq, y,
                belongs_priv, ptb_with_joint, new_attr,
                res_aux, sens_att, priv_val, X, A, Aq)

    def trial_one_process(self, mode='w'):
        since = time.time()
        csv_t = open(self._log_document + '.csv', mode)
        csv_w = csv.writer(csv_t)
        if mode == 'a':
            csv_w.writerows([[''], [''], [''], ['']])

        if not (self._screen or self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + '.log', 'w')
            sys.stdout = fsock
        if self._logged:
            if os.path.exists(self._log_document + '.txt'):
                os.remove(self._log_document + '.txt')
            logger = get_elogger('fairmanf', self._log_document + '.txt')
        else:
            logger = None

        elegant_print([
            "[BEGAN AT {}]".format(elegant_dated(since)),
            "EXPERIMENT",
            "\t   trail = {}".format(self._trial_type),
            "\t dataset = {}".format(self._data_type),
            "\t binary? = {}".format(not self._trial_type.startswith('mu')),
            "\t nb_iter = {}, gen {}, rep/cvs {}".format(
                self._nb_iter,
                str(self._gen_iter)[0], str(self._rep_iter)[0]),
            "\tdata prep= {}".format(self._prep),
            "PARAMETERS",
            "\t  m1, m2 = {}, {}".format(self._m1, self._m2),
            "\t  nb_cls = {}".format(self._nb_cls),
            # "\t  constr = {}".format(self._constraint_type),
            "HYPER-PARAMS", ""], logger)

        # START
        self.coding_per_procedure(csv_w, logger)
        # END

        tim_elapsed = time.time() - since
        elegant_print(["",
                       "Duration /TimeCost: {}".format(
                           elegant_durat(tim_elapsed)),
                       "[ENDED AT {:s}]".format(
                           elegant_dated(time.time()))], logger)
        del logger
        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    def coding_per_procedure(self, csv_w, logger):
        csv_row_2a = ['dat_name', 'binary', '#sen-att', 'nb_iter',
                      'gen', 'rep/cvs', 'm1', 'm2', ]
        csv_row_2b = ['fair_ens', '#iter']  # '#eval'
        csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
        csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
        csv_row_2[-1] = '[END]'
        csv_w.writerows([csv_row_1, csv_row_2,
                         [''] * 10 + csv_r3c, [''] * 10 + csv_r4c])
        del csv_r4c, csv_r3c, csv_r2c, csv_row_2b
        del csv_row_2a, csv_row_2, csv_row_1

        # START
        res_data, res_aux = self.coding_per_dataset(logger)
        json_saver = json.dumps({
            'res_aux': res_aux, 'res_data': res_data})
        json_w = open(self._log_document + '.json', 'w')
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w

        csv_w.writerow(res_aux[0])
        sens_att = res_aux[1]  # not res_aux[3][:2]
        if self._trial_type[-6:] in ['expt8a', 'expt8b']:
            abbr_clf = res_aux[-1]
            if self._trial_type.endswith('expt8a'):
                for t_b, tmp_b in enumerate(abbr_clf[:3]):
                    csv_w.writerow([''] * 7 + ['', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
                for t_a, tmp_a in enumerate(sens_att):
                    for t_b, tmp_b in enumerate(abbr_clf[3:]):
                        csv_w.writerow([''] * 7 + [
                            tmp_a, tmp_b, 0] + res_data[0][t_b + 3 + t_a * 4])
                        for k in range(1, self._nb_iter):
                            csv_w.writerow([''] * 7 + [
                                '', '', k] + res_data[k][t_b + 3 + t_a * 4])
            elif self._trial_type.endswith('expt8b'):
                for t_b, tmp_b in enumerate(abbr_clf):
                    csv_w.writerow([''] * 7 + ['---', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
            # pdb.set_trace()
            del sens_att, abbr_clf

        elif self._trial_type[-6:] in ['expt2a', 'expt2b']:
            # sens_att = res_aux[3][: 2]
            fair_ens = res_aux[-1]  # i.e., res_aux[4]
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens):
                    k = 0
                    csv_w.writerow([''] * 7 + [tmp_a, tmp_b, k] + res_data[k][t_a][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_a][t_b])
            del sens_att, fair_ens
        elif self._trial_type[-6:] in ['expt2c']:
            # sens_att = res_aux[3][: 2]
            # siz= (5, 14+4*?, 199)= (5, 11+3+4+4, 199)
            fair_ens = res_aux[-2]  # i.e. res_aux[4]; len=3+4
            abbr_clf = res_aux[-1]  # i.e. res_aux[5]; len=11
            for t_b, tmp_b in enumerate(abbr_clf):
                csv_w.writerow([''] * 7 + ['---', tmp_b, 0] + res_data[0][t_b])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
            for t_b, tmp_b in enumerate(fair_ens[: 3]):
                csv_w.writerow([''] * 7 + ['', tmp_b, 0] + res_data[0][t_b + 11])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b + 11])
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens[3:]):
                    csv_w.writerow([''] * 7 + [
                        tmp_a, tmp_b, 0] + res_data[0][t_b + 14 + t_a * 4])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + [
                            '', '', k] + res_data[k][t_b + 14 + t_a * 4])
            del sens_att, fair_ens, abbr_clf
        elif self._trial_type[-6:] in ('expt2d', 'expt2e'):
            # sens_att = res_aux[3][: 2]
            abbr_clf = res_aux[-1]
            if self._trial_type.endswith('expt2d'):
                for t_b, tmp_b in enumerate(abbr_clf):
                    csv_w.writerow([''] * 7 + ['---', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
            elif self._trial_type.endswith('expt2e'):
                for t_b, tmp_b in enumerate(abbr_clf[: 3]):
                    csv_w.writerow([''] * 7 + ['', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
                for t_a, tmp_a in enumerate(sens_att):
                    for t_b, tmp_b in enumerate(abbr_clf[3:]):
                        csv_w.writerow([''] * 7 + [
                            tmp_a, tmp_b, 0] + res_data[0][t_b + 3 + t_a * 4])
                        for k in range(1, self._nb_iter):
                            csv_w.writerow([''] * 7 + [
                                '', '', k] + res_data[k][t_b + 3 + t_a * 4])
            del sens_att, abbr_clf

        # END
        return

    def coding_per_dataset(self, logger):
        (X_and_A, X_and_Aq, y, idx_g1, idx_jt, new_attr, res_aux,
         _, priv_val, X, A, Aq) = self.preparing_current_data(logger)
        n_a = len(idx_g1)
        g1m_indices = [[idx_g1[i]] for i in range(n_a)]  # non sa_indices
        # g1m_indices = [[i, ~i] for i, p in zip(idx_g1, priv_val)]

        if 'expt2' in self._trial_type:
            res_aux.append(CURR_CLFS)
            if 'expt2c' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs)
            elif 'expt2d' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs + [
                    'bagging', 'adaboost'])
            elif 'expt2e' in self._trial_type:
                res_aux.append(CURR_CLFS)  # + CURR_CLFS[3:])
        elif self._trial_type[-6:] in ('expt8a',):
            res_aux.append(self._iterator._abbr_clfs)

        elegant_print('Not-repetitively, via cv_split?: {}'.format(
            'Yes' if self._rep_iter else 'Noe'), logger)
        if not self._rep_iter:
            elegant_print('nb_iter={}, repetitive'.format(
                self._nb_iter), logger)
            split_idx = manual_repetitive(self._nb_iter, y, self._gen_iter)
            res_aux = []
            for k, idx in enumerate(split_idx):
                elegant_print('Iteration {}-th starts.'.format(
                    k + 1), logger)
                XA_p, _, yp, XAq_p, g1_p, jt_p = renewed_transform_disturb(
                    X_and_A, None, y, X_and_Aq, idx, g1m_indices, idx_jt)
                g1_p = [t[0] for t in g1_p]
                if self._prep != 'none':
                    scaler = scale_normalize_helper(self._prep)
                    # scaler, Xp, Ap = normalise_disturb_whole(scaler, X, A)
                    scaler, XA_p, _, _, Xp, Ap, _, _, _, _ = renewed_normalise_disturb(
                        scaler, XA_p, [], XA_p, self.saIndex)
                else:
                    Xp, Ap, _, _, _, _ = renewed_normalise_separate(
                        XA_p, [], XA_p, self.saIndex)
                pdb.set_trace()
                res_iter = self.coding_per_iteration_as_whole(
                    logger, Xp, Ap, yp, g1_p, jt_p)
                res_ans.append(res_iter)
                elegant_print('Iteration {}-th done.'.format(k + 1), logger)
            return res_ans, res_aux

        elegant_print('nb_iter={}, cross_valid'.format(self._nb_iter), logger)
        if 'mCV' in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif 'KFS' in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y,
                                         X_and_A.values.astype(DTY_FLT))
        elif 'KF' in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print("\t CrossValid  {}\n".format(
            self._trial_type[:3]), logger)
        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (X_A_trn, _, y_trn, X_Aq_trn, s1_trn,
             jt_trn) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_trn, g1m_indices, idx_jt)
            (X_A_tst, _, y_tst, X_Aq_tst, s1_tst,
             jt_tst) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_tst, g1m_indices, idx_jt)
            s1_trn = [t[0] for t in s1_trn]  # g1_trn, non_sa
            s1_tst = [t[0] for t in s1_tst]  # g1_tst, non_sa
            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                (scaler, X_A_trn, _, X_A_tst, X_trn, A_trn, _, _,
                 X_tst, A_tst) = renewed_normalise_disturb(
                    scaler, X_A_trn, [], X_A_tst, self.saIndex)
            else:
                X_trn, A_trn, _, _, X_tst, A_tst = renewed_normalise_separate(
                    X_A_trn, [], X_A_tst, self.saIndex)
            # i-th K-Fold
            elegant_print('Iteration {}-th'.format(k + 1), logger)
            res_iter = self.coding_per_iteration_cv_split(
                logger, k,
                X_A_trn, y_trn, X_Aq_trn, s1_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, s1_tst, jt_tst,
                X_trn, A_trn, X_tst, A_tst)
            res_ans.append(res_iter)
            del X_trn, A_trn, y_trn, s1_trn, X_A_trn, X_Aq_trn, jt_trn
            del X_tst, A_tst, y_tst, s1_tst, X_A_tst, X_Aq_tst, jt_tst
        del X_and_A, y, X_and_Aq, idx_g1, idx_jt, g1m_indices, new_attr
        # pdb.set_trace()
        return res_ans, res_aux

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst):
        since = time.time()
        res_iter = []

        pm = {'m1': self._m1, 'm2': self._m2}
        pos_label = self._dataset.get_positive_class_val(
            'numerical-binsensitive')

        if 'expt2' in self._trial_type:
            pm['positive_label'] = pos_label
            res_iter = self._iterator.schedule_content_prime(
                logger,
                X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                X_trn, A_trn, X_tst, A_tst, **pm)
        elif 'expt8' in self._trial_type:
            pm['pos_label'] = pos_label
            res_iter = self._iterator.schedule_content(
                logger,
                X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                X_trn, A_trn, X_tst, A_tst, **pm)

        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter


# -------------------------------
