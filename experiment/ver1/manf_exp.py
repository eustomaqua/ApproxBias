# coding: utf-8
# Experiments


import time
import numpy as np

from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier)
import lightgbm
from fairgbm import FairGBMClassifier
# from experiment.utils.pkgs_AdaFair_mod import AdaFair
# from blbm.fairgbm import FairGBMClassifier
# from blbm.AdaFair_mod3 import AdaFair
from experiment.classifiers import INDIVIDUALS  # RelativeFairClsf,

from hfm.dist_drt import DirectDist_bin as DirectDist
from hfm.dist_est_bin import ApproxDist_bin as ApproxDist
from hfm.hfm_df import bias_degree as fair_degree
# from hfm.discriminative_risk import (
#     E_rho_L_fair_f, hat_L_fair, E_rho_L_loss_f, hat_L_loss)
from hfm.discriminative_risk import hat_L_fair, hat_L_loss

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.metrics.contingency_mat import \
    contg_tab_mu_type2 as contingency_table
from hfm.metrics.performance import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_fpr, calc_fnr, calc_sensitivity, calc_specificity,
    imba_geometric_mean, imba_discriminant_power,
    imba_Matthew_s_cc, imba_Cohen_s_kappa)  # calc_tpr,
from hfm.metrics.fairness_group import (
    marginalised_pd_mat, unpriv_unaware, unpriv_manual,
    unpriv_group_one, unpriv_group_two, unpriv_group_thr)

import sklearn.__version__ as skl_ver
if skl_ver.startswith('1.3.0'):
    from experiment.utils.pkgs_AdaFair_py36 import AdaFair
elif skl_ver.startswith('1.5.1'):
    pass
del skl_ver


# -------------------------------
# RelativeFairClsf
#


# -------------------------------
# fvote_empiric.py


ENSEM_NAMES = [
    "bagging", "adaboost", "rforest", "extrats", "gradbst",
]
ALG_NAMES = [
    "DT", "NB", "SVM", "linSVM", "MLP",
    "LR1", "LR2", "LM1", "LM2", "kNNu", "kNNd",
]


class ComparisonB_setup:
    def __init__(self):
        pass

    def get_individual(self, abbr_cls):
        return INDIVIDUALS[abbr_cls]

    def get_ensemble(self, name_ens, abbr_cls='DT', nb_cls=21):
        if (name_ens == 'bagging') and (abbr_cls != 'DT'):
            clf = INDIVIDUALS[abbr_cls]  # indv
            return BaggingClassifier(clf, n_estimators=nb_cls)
        elif name_ens == 'bagging':
            return BaggingClassifier(n_estimators=nb_cls)
        elif name_ens == 'adaboost':
            return AdaBoostClassifier(n_estimators=nb_cls)
        elif name_ens == 'rforest':
            return RandomForestClassifier(n_estimators=nb_cls)
        elif name_ens == 'extrats':
            return ExtraTreesClassifier(n_estimators=nb_cls)
        elif name_ens == 'gradbst':
            return GradientBoostingClassifier(n_estimators=nb_cls)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    def get_fair_ens(self, name_ens, nb_cls=2, constraint='',
                     saIndex=None, saValue=None):
        if name_ens == 'lightgbm':
            return lightgbm.LGBMClassifier(n_estimators=nb_cls)
        elif name_ens == 'fairgbm':
            return FairGBMClassifier(n_estimators=nb_cls,
                                     constraint_type=constraint)
        elif name_ens == 'adafair':
            return AdaFair(n_estimators=nb_cls,
                           saIndex=saIndex, saValue=saValue)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    # fvote_empiric.py
    def count_single_member(self, y, y_hat, X, A, y_qtb,
                            non_sa, positive_label=1, m1=20, m2=8):
        # NB. must be np.ndarray
        res_indi = self.count_sing_part1(y, y_hat, positive_label)
        cmp_fair = self.count_sing_part2(y, y_hat, y_qtb,
                                         non_sa, positive_label)
        res_fair = self.count_sing_part3(X, A, y, y_hat, non_sa,
                                         m1, m2)
        cmp_fair.extend(res_fair)

        # res_indi.shape: (13,)= (6+7,)
        # cmp_fair.shape: (17,)= (5*2+5+2,)
        # res_fair.shape: (14,)= (6*2+2,)
        # # cmp_fair.shape: (23+6+2,)= (5*2+5+2+3*2+2,)
        return res_indi + cmp_fair  # (44,)= (36+6+2,)

    def count_sing_part1(self, y, y_hat, positive_label):
        # NB. must be np.ndarray
        tp, fp, fn, tn = contingency_table(y, y_hat, positive_label)

        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        res_indi.append(calc_recall(tp, fp, fn, tn))
        res_indi.append(calc_f1_score(tp, fp, fn, tn))

        res_indi.append(calc_fpr(tp, fp, fn, tn))
        res_indi.append(calc_fnr(tp, fp, fn, tn))

        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, imba_geometric_mean(sen, spe)])
        res_indi.append(imba_discriminant_power(sen, spe))
        res_indi.append(imba_Matthew_s_cc(tp, fp, fn, tn))
        res_indi.extend(imba_Cohen_s_kappa(tp, fp, fn, tn))

        return res_indi  # shape= (13,)= (6+3+2+2,)

    def count_sing_part2(self, y, y_hat, y_qtb,
                         non_sa, positive_label=1):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, y_hat, positive_label, non_sa)
        cmp_fair = []

        tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)

        cmp_fair.extend(tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4)
        cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair  # shape= (17,)= (5*2+5+2,)

    def count_sing_part3(self, X, A, y, y_hat, non_sa, m1=20, m2=8):
        cmp_fair = []
        X_and_y = np.concatenate([y.reshape(-1, 1), X], axis=1,
                                 dtype=DTY_FLT)
        X_and_y_hat = np.concatenate([y_hat.reshape(-1, 1), X],
                                     axis=1, dtype=DTY_FLT)
        '''
        idx_sa = ~non_sa  # non_priv

        ut_a = time.time()
        Ds_01, t_Ds = DirectDist(X_and_y, idx_sa, non_sa)
        Df_01, t_Df = DirectDist(X_and_y_hat, idx_sa, non_sa)
        ddf, t_ddf = fair_degree(Ds_01, Df_01)
        ut_a = time.time() - ut_a
        cmp_fair.extend((Ds_01, Df_01) + ddf + (t_Ds, t_Df, t_ddf))  # siz=7

        ut_b = time.time()
        Ds_01, t_Ds = ApproxDist(X_and_y, A, idx_sa, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist(X_and_y_hat, A, idx_sa, non_sa, m1, m2)
        ddf, t_ddf = fair_degree(Ds_01, Df_01)
        ut_b = time.time() - ut_b
        cmp_fair.extend((Ds_01, Df_01) + ddf + (t_Ds, t_Df, t_ddf))  # siz=7
        '''

        ut_a = time.time()
        Ds_01, t_Ds = DirectDist(X_and_y, non_sa)
        Df_01, t_Df = DirectDist(X_and_y_hat, non_sa)
        Ds_01, Df_01 = Ds_01[0], Df_01[0]
        ddf, t_ddf = fair_degree(Ds_01, Df_01)
        ut_a = time.time() - ut_a
        cmp_fair.extend((Ds_01, Df_01) + ddf + (t_Ds, t_Df, t_ddf))
        ut_b = time.time()
        Ds_01, t_Ds = ApproxDist(X_and_y, A, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist(X_and_y_hat, A, non_sa, m1, m2)
        ddf, t_ddf = fair_degree(Ds_01, Df_01)
        ut_b = time.time() - ut_b
        cmp_fair.extend((Ds_01, Df_01) + ddf + (t_Ds, t_Df, t_ddf))

        cmp_fair.extend([ut_a, ut_b])
        return cmp_fair  # shape= (16,) =(7*2+2,)

    def count_scores(self,  # clf,
                     X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
                     X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
                     positive_label=1, m1=20, m2=8):
        ans_trn = self.count_single_member(
            y_trn, y_insp, X_trn, A_trn, yq_insp, nsa_trn,
            positive_label, m1, m2)
        ans_tst = self.count_single_member(
            y_tst, y_pred, X_tst, A_tst, yq_pred, nsa_tst,
            positive_label, m1, m2)
        return ans_trn + ans_tst  # (92,) = (46*2,) =((13+17+16)*2,)


class ComparisonB1_withDirectComput(ComparisonB_setup):
    def __init__(self, nb_cls=1,
                 saIndex=list(), saValue=list()):
        # super().__init__(
        #     abbr_cls, nb_cls, constraint_type, saIndex, saValue)
        # self._abbr_cls = abbr_cls  # useless
        self._nb_cls = nb_cls
        self.saIndex = saIndex  # self._saIndex
        self.saValue = saValue  # self._saValue

    # def schedule_content(self, logger,
    #                      X_trn, A_trn, y_trn, g1_trn, jt_trn,
    #                      X_tst, A_tst, y_tst, g1_tst, jt_tst,
    #                      m1=20, m2=8, Aq_trn=None, Aq_tst=None,
    #                      positive_label=None):
    def schedule_content(self, logger,
                         X_trn, A_trn, y_trn, Aq_trn, g1_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1_tst, jt_tst,
                         m1=20, m2=8, positive_label=None):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst, m1, m2, positive_label)

    def schedule_content_prime(self, logger,
                               X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                               X_trn, A_trn, X_tst, A_tst, m1, m2,
                               positive_label):
        res_iter = []
        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[0],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                X_trn, A_trn, X_tst, A_tst)
            res_iter.append(tmp)
            return res_iter  # shape= (#attr= 1, 7, 89)

        sa_len = len(g1_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[i],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[i],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                X_trn, A_trn, X_tst, A_tst)
            res_iter.append(tmp)
        return res_iter  # shape= (#attr= 2, 7, 89)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 92 + 1)  # 11+77+12
        csv_row_2c = ['Ensem'] + [
            'Training set'] + [''] * 45 + ['Test set'] + [''] * 45  # 44*2+1
        tmp_4_1 = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                   'fpr', 'fnr', 'sensitivity', 'specificity',
                   'g_mean', 'dp', 'Matthew', 'Cohen', '(random_acc)']
        tmp_4_2 = ['g1', 'g0'] * 5 + ['abs'] * 5 + ['hat_L(fair)', 'hat_L(loss)']
        # tmp_4_3 = ['Ds_01', 'Df_01', 'df'] * 2 + ['DirectDist', 'ApproxDist']
        tmp_4_3 = ['Ds_01', 'Df_01', 'df', '',
                   'T(Ds)', 'T(Df)', 'T(df)'] * 2 + ['DirectDist', 'ApproxDist']
        tmp_4 = tmp_4_1 + tmp_4_2 + tmp_4_3  # 13+17+16 =44+2 =46
        tmp_3 = ['Normal'] + [''] * 12 + [
            'Fairness'] + [''] * 9 + ['Group'] + [''] * 4 + ['fairvote', ''] + [
            'fairmanf: Direct', '', '', '', 'TimeCost(Direct)', '', '',
            'fairmanf: Approx', '', '', '', 'TimeCost(Approx)', '', '',
            'fairmanf: ut', '']  # Adversarial  # 13+(10+5+2)+(7*2+2) =13+17+16 =46
        csv_row_3c = ['Time Cost (sec)'] + tmp_3 + tmp_3  # 1+46*2 =89+4 =93
        csv_row_4c = ['ut'] + tmp_4 + tmp_4
        del tmp_4_1, tmp_4_2, tmp_4_3, tmp_4, tmp_3
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    # def subroute_one_sens_att(self,  # logger,
    #                           X_trn, A_trn, y_trn, Aq_trn, nsa_trn,
    #                           X_tst, A_tst, y_tst, Aq_tst, nsa_tst,
    #                           sa_idx, sa_val, positive_label=1,
    #                           m1=20, m2=8):
    #   X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
    #   X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
    #   X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
    #   X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)

    def subroute_one_sens_att(self,  # logger,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              sa_idx, sa_val, positive_label=1,
                              m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              X_tst=None, A_tst=None):

        res_attr = []  # shape= (7,89= 77+12)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                constraint=constraint_type)
            tmp = self.count_scores(  # clf,
                X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst,
            sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(  # clf,
            X_trn, A_trn, y_trn, y_insp, yq_insp, nsa_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, nsa_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        return res_attr  # shape= (7,1+88+4)

    def subroute_one_fair_ens(self, name_ens, nb_cls,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst=None,
                              # positive_label=1,
                              constraint='FPR,FNR',
                              sa_idx=None, sa_val=None):

        ut = time.time()

        if name_ens == 'bagging':
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'adaboost':
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'lightgbm':
            clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
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


class ComparisonB2_withDirectComput(ComparisonB_setup):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list()):
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue

    def count_sing_part2(self, y, y_hat, y_qtb,
                         non_sa, positive_label=1):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, y_hat, positive_label, non_sa)
        cmp_fair = []

        tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)
        # cmp_fair.extend(tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4)

        cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair  # shape= (7,)= (5+2,)

    def count_single_member(self, y, y_hat, X, A, y_qtb, g1, jt,
                            positive_label=1, m1=20, m2=8):
        # NB. must be np.ndarray
        res_indi = self.count_sing_part1(y, y_hat, positive_label)
        fair_2, fair_3 = [], []

        fair_2.extend(
            self.count_sing_part2(y, y_hat, y_qtb, g1[0], positive_label))
        if len(g1) > 1:
            fair_2.extend(
                self.count_sing_part2(y, y_hat, y_qtb, g1[1], positive_label))
            fair_2.extend(
                self.count_sing_part2(y, y_hat, y_qtb, jt[0], positive_label))
            fair_2.extend(
                self.count_sing_part2(y, y_hat, y_qtb, jt[1], positive_label))
        else:
            fair_2.extend([''] * 7 * 3)

        fair_3.extend(self.count_sing_part3(X, A, y, y_hat, g1[0],
                                            m1, m2))
        if len(g1) > 1:
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, g1[1],
                                                m1, m2))
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, jt[0],
                                                m1, m2))
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, jt[1],
                                                m1, m2))
        else:
            fair_3.extend([''] * 16 * 3)  # 8+6+2

        # return res_indi + fair_2 + fair_3  # shape= ( 97,)= (13+28+56,)
        return res_indi + fair_2 + fair_3    # shape= (105,)= (13+28+64,)

    def count_scores(self,
                     X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                     X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                     positive_label=1, m1=20, m2=8):
        ans_trn = self.count_single_member(
            y_trn, y_insp, X_trn, A_trn, yq_insp, g1_trn, jt_trn,
            positive_label, m1, m2)
        ans_tst = self.count_single_member(
            y_tst, y_pred, X_tst, A_tst, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        # return ans_trn + ans_tst  # shape= ( 97*2 =194,)
        return ans_trn + ans_tst    # shape= (105*2 =210,)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 210 + 1)  # + 147 + 6 * 4 * 2)
        # csv_row_2c = ['Ensem'] + [
        #     'Training set'] + [''] * 72 + ['Test set'] + [''] * 72
        csv_row_2c = ['Ensem'] + [
            'Training set'] + [''] * 12 + ['fairvote'] + [''] * 27 + [
            'fairmanf'] + [''] * 63 + [
            'Test set'] + [''] * 12 + ['fairvote'] + [''] * 27 + [
            'fairmanf'] + [''] * 63  # 13+28+64 =105, 105*2+1 =211

        tmp_3_1 = ['Normal'] + [''] * 12
        tmp_3_2 = ['sens_att#1'] + [''] * 6 + ['sens_att#2'] + [''] * 6 + [
            'joint_and'] + [''] * 6 + ['joint_or'] + [''] * 6  # 'att 1&|2'
        tmp_3_3 = ['sens_att#1'] + [''] * 15 + ['sens_att#2'] + [''] * 15 + [
            'att_#1&#2'] + [''] * 15 + ['att_#1|#2'] + [''] * 15
        tmp_3 = tmp_3_1 + tmp_3_2 + tmp_3_3  # 13+7*4+16*4 =13+23*4 =105
        csv_row_3c = ['Time Cost (sec)'] + tmp_3 + tmp_3  # 1+105*2 =211
        del tmp_3_1, tmp_3_2, tmp_3_3, tmp_3

        tmp_4_1 = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                   'fpr', 'fnr', 'sensitivity', 'specificity',
                   'g_mean', 'dp', 'Matthew', 'Cohen', '(random_acc)']
        tmp_4_2 = ['Group fairness: abs(g1-g0)'] + [
            ''] * 4 + ['hat_L(fair)', 'hat_L(loss)']  # 5+2 =7
        # tmp_4_3 = ['Ds_01', 'Df_01', 'df'] * 2 + ['DirectDist', 'ApproxDist']
        tmp_4_3 = [
            'Direct: Ds_01, Df_01, df(v3.4)', '', '', '',
            'T-Direct: Ds,Df,df', '', ''] + [
            'Approx: Ds_01, Df_01, df(v3.4)', '', '', '',
            'T-Approx: Ds,Df,df', '', ''] + [
            '*Dist: ut', '']  # 3*2+2 =8  #--> 6*2+2 =14  #--> (4+3)*2+2 =16
        tmp_4 = tmp_4_1 + tmp_4_2 * 4 + tmp_4_3 * 4  # =13+(7+16)*4 =105
        csv_row_4c = ['ut'] + tmp_4 + tmp_4
        del tmp_4_1, tmp_4_2, tmp_4_3, tmp_4

        # tmp_3 = ['Normal'] + [''] * 12 + [
        #     'Group fairness'] + [''] * 4 + ['fairvote', ''] + [
        #     'fairmanf: Direct', '', '', 'fairmanf: Approx', '', '',
        #     'fairmanf: ut', '']  # Adversarial  # 13+7+8 =28
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, logger,
                         X_trn, A_trn, y_trn, Aq_trn, g1_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1_tst, jt_tst,
                         m1=20, m2=8, positive_label=None):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)

        return self.schedule_content_prime(
            logger,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst, m1, m2, positive_label)

    def schedule_content_prime(self, logger,
                               X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                               X_trn=None, A_trn=None,
                               X_tst=None, A_tst=None,
                               m1=20, m2=8, positive_label=1):
        res_iter = []
        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[0],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                X_trn, A_trn, g1_trn, jt_trn,
                X_tst, A_tst, g1_tst, jt_tst)
            res_iter.append(tmp)
            return res_iter  # shape= (#att =1, 7, 194+1)

        sa_len = len(g1_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[i],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[i],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                X_trn, A_trn, g1_trn, jt_trn,
                X_tst, A_tst, g1_tst, jt_tst)
            res_iter.append(tmp)
        return res_iter    # shape= (#att =2, 7, 194+1)

    def subroute_one_sens_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              sa_idx, sa_val, positive_label=1,
                              m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None,
                              g1_tst=None, jt_tst=None):

        res_attr = []

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst, constraint=constraint_type)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst, sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        return res_attr  # shape= (7, 1+210)  # 1+194)

    def subroute_one_fair_ens(self, name_ens, nb_cls,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              constraint='FPR,FNR',
                              sa_idx=None, sa_val=None):
        ut = time.time()

        if name_ens == 'bagging':
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'adaboost':
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif name_ens == 'lightgbm':
            clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
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


# -------------------------------
#


class ComparisonC2_withDirectComput(ComparisonB2_withDirectComput):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list()):
        super().__init__(nb_cls, saIndex, saValue)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def count_single_member(self, y, y_hat, X, A, y_qtb, g1, jt,
                            positive_label=1, m1=20, m2=8):

        acc_norm = self.count_sing_part1(y, y_hat, positive_label)
        acc_irre = self.count_sing_part1(y, y_qtb, positive_label)
        tmp = [abs(i - j) for i, j in zip(acc_norm, acc_irre)]
        res_acc = acc_norm + acc_irre + tmp  # 13*3 =39
        del acc_norm, acc_irre, tmp
        fair_2, fair_3 = [], []

        fair_2.extend(self.count_sing_part2(y, y_hat, y_qtb, g1[0],
                                            positive_label))
        if len(g1) > 1:
            fair_2.extend(self.count_sing_part2(y, y_hat, y_qtb, g1[1],
                                                positive_label))
            fair_2.extend(self.count_sing_part2(y, y_hat, y_qtb, jt[0],
                                                positive_label))
            fair_2.extend(self.count_sing_part2(y, y_hat, y_qtb, jt[1],
                                                positive_label))
        else:
            fair_2.extend([''] * 7 * 3)

        fair_3.extend(self.count_sing_part3(X, A, y, y_hat, g1[0],
                                            m1, m2))
        if len(g1) > 1:
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, g1[1],
                                                m1, m2))
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, jt[0],
                                                m1, m2))
            fair_3.extend(self.count_sing_part3(X, A, y, y_hat, jt[1],
                                                m1, m2))
        else:
            fair_3.extend([''] * 16 * 3)

        # return res_acc + fair_2 + fair_3  # shape= (123,)= (39+21*4,)
        return res_acc + fair_2 + fair_3    # shape= (131,)= (39+23*4,)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 131 * 2 + 1)  # + 199 + 24 * 2)
        csv_row_2c = ['/Ensem'] + [
            'Training set'] + [''] * 38 + ['fairvote'] + [''] * 27 + [
            'fairmanf'] + [''] * 63 + [
            'Test set'] + [''] * 38 + ['fairvote'] + [''] * 27 + [
            'fairmanf'] + [''] * 63  # 39+28+64 =131, 131*2+1 =263

        tmp_3_1 = ['Normal'] + [''] * 12 + ['Adversarial'] + [''] * 12 + [
            'abs()'] + [''] * 12
        tmp_3_2 = ['sens_att#1'] + [''] * 6 + ['sens_att#2'] + [''] * 6 + [
            'joint_and'] + [''] * 6 + ['joint_or'] + [''] * 6  # 'att 1&|2'
        tmp_3_3 = ['sens_att#1'] + [''] * 15 + ['sens_att#2'] + [''] * 15 + [
            'att_#1&#2'] + [''] * 15 + ['att_#1|#2'] + [''] * 15
        tmp_3 = tmp_3_1 + tmp_3_2 + tmp_3_3  # 13*3+7*4+16*4 =39+28+(56+8)
        csv_row_3c = ['Time Cost (sec)'] + tmp_3 + tmp_3  # 1+131*2
        del tmp_3_1, tmp_3_2, tmp_3_3, tmp_3

        tmp_4_1 = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                   'fpr', 'fnr', 'sensitivity', 'specificity',
                   'g_mean', 'dp', 'Matthew', 'Cohen', '(random_acc)']
        tmp_4_2 = ['abs({})'.format(i) for i in [
            'Acc', 'P', 'R', 'f1', 'fpr', 'fnr', 'sen', 'spe',
            'gm', 'dp', 'Matthew', 'Cohen', '(ra)']]  # 8+5=13
        tmp_4_3 = ['Group fairness: abs(g1-g0)'
                   ] + [''] * 4 + ['hat_L(fair)', 'hat_L(loss)']  # 5+2=7
        tmp_4_4 = [
            'Direct: Ds_01, Df_01, df(v3.4)', '', '', '',
            'T-Direct: Ds,Df,df', '', ''] + [
            'Approx: Ds_01, Df_01, df(v3.4)', '', '', '',
            'T-Approx: Ds,Df,df', '', ''] + [
            '*Dist: ut', '']  # 8+6 =14  #--> 7*2+2 =16
        tmp_4 = tmp_4_1 * 2 + tmp_4_2 + tmp_4_3 * 4 + tmp_4_4 * 4
        # csv_row_4c = ['ut'] + tmp_4 + tmp_4  # ↑ 13*2+13 +7*4+14*4 =39+60+24
        csv_row_4c = ['ut'] + tmp_4 + tmp_4  # ↑ 13*2+13 +7*4+16*4 =39+28+64
        del tmp_4_1, tmp_4_2, tmp_4_3, tmp_4_4, tmp_4

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, logger,
                         X_trn, A_trn, y_trn, Aq_trn, g1_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1_tst, jt_tst,
                         m1=20, m2=8, positive_label=None):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst, m1, m2, positive_label)

    def schedule_content_prime(self, logger,
                               X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                               X_trn, A_trn, X_tst, A_tst,
                               m1, m2, positive_label):
        res_iter = []
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, None,
            X_A_tst, y_tst, X_Aq_tst, None, positive_label, m1, m2,
            X_trn, A_trn, g1_trn, jt_trn,
            X_tst, A_tst, g1_tst, jt_tst)  # shape= (11+3, 1+131*2)
        res_iter.extend(tmp)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[0],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                X_trn, A_trn, g1_trn, jt_trn,
                X_tst, A_tst, g1_tst, jt_tst)  # shape= (3+1, 1+131*2)
            res_iter.extend(tmp)
            return res_iter  # shape= (14+4*1, 1+131*2)

        sa_len = len(g1_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1_trn[i],
                X_A_tst, y_tst, X_Aq_tst, g1_tst[i],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                X_trn, A_trn, g1_trn, jt_trn,
                X_tst, A_tst, g1_tst, jt_tst)
            res_iter.extend(tmp)
        return res_iter    # shape= (14+4*2, 1+131*2)

    def subroute_one_norm_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              positive_label=1, m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None,
                              g1_tst=None, jt_tst=None):
        res_attr = []

        for abbr_cls in self._abbr_clfs:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_gene_clf(
                abbr_cls, X_A_trn, y_trn, X_Aq_trn, X_A_tst, y_tst, X_Aq_tst)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)  # shape= (247,)= (1+131*2,)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        # return res_attr  # shape= (14,247)= (11+3, 1+123*2)
        return res_attr    # shape= (14,263)= (11+3, 1+131*2)

    def subroute_one_sens_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              sa_idx, sa_val, positive_label=1,
                              m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None,
                              g1_tst=None, jt_tst=None):
        res_attr = []

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst, constraint=constraint_type)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst, sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        # return res_attr  # shape= (4,247)= (3+1, 1+123*2)
        return res_attr    # shape= (4,263)= (3+1, 1+131*2)

    def subroute_one_gene_clf(self, abbr_cls,
                              X_A_trn, y_trn, X_Aq_trn,
                              X_A_tst, y_tst, X_Aq_tst):
        ut = time.time()
        clf = INDIVIDUALS[abbr_cls]
        clf.fit(X_A_trn, y_trn)
        ut = time.time() - ut

        y_insp = clf.predict(X_A_trn)
        y_pred = clf.predict(X_A_tst)
        yq_insp = clf.predict(X_Aq_trn)
        yq_pred = clf.predict(X_Aq_tst)

        return y_insp, y_pred, yq_insp, yq_pred, ut


class ComparisonC4_withDirectComput(ComparisonC2_withDirectComput):

    def schedule_content(self, logger,
                         X_trn, A_trn, y_trn, Aq_trn, g1_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1_tst, jt_tst,
                         m1=20, m2=8, positive_label=None):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst, m1, m2, positive_label)

    def schedule_content_prime(self, logger,
                               X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                               X_trn, A_trn, X_tst, A_tst, m1, m2,
                               positive_label):
        res_iter = []
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, None,
            X_A_tst, y_tst, X_Aq_tst, None, positive_label, m1, m2,
            X_trn, A_trn, g1_trn, jt_trn,
            X_tst, A_tst, g1_tst, jt_tst)  # shape= (11+3, 1+99*2)
        res_iter.extend(tmp)
        return res_iter    # shape= (14+4*2, 1+99*2)

    def subroute_one_norm_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              positive_label=1, m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None,
                              g1_tst=None, jt_tst=None):
        res_attr = []

        for abbr_cls in self._abbr_clfs:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_gene_clf(
                abbr_cls, X_A_trn, y_trn, X_Aq_trn, X_A_tst, y_tst, X_Aq_tst)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
                positive_label, m1, m2)
            res_attr.append([ut] + tmp)  # shape= (199,)= (1+99*2,)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        return res_attr  # shape= (14,199)= (11+3, 1+99*2)


class ComparisonC5_withDirectComput(ComparisonC2_withDirectComput):

    def subroute_one_norm_att(self,
                              X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                              X_A_tst, y_tst, X_Aq_tst, nsa_tst,
                              positive_label=1, m1=20, m2=8,
                              X_trn=None, A_trn=None,
                              g1_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None,
                              g1_tst=None, jt_tst=None):
        res_attr = []

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1_tst, jt_tst,
            positive_label, m1, m2)
        res_attr.append([ut] + tmp)

        return res_attr  # shape= (14,199)= (11+3, 1+99*2)


# -------------------------------
#


# -------------------------------
#
