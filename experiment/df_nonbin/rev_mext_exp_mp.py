# coding: utf-8
# rev_mext_exp_mp.py

import time
import pdb
import numpy as np

from hfm.utils.verifiers import unique_column, DTY_FLT, DTY_INT


from hfm.dist_drt import DirectDist_bin as DistDirect_bin
from hfm.dist_drt import DirectDist_nonbin as DistDirect_nonbin
from hfm.dist_drt import DirectDist_multiver as DistDirect_multivar
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4
# from hfm.dist_est_nonbin import AcceleDist_nonbin as DistAccele
from hfm.dist_est_nonbin import ApproxDist_nonbin_mpver as DistApprox
from hfm.dist_est_nonbin import ExtendDist_multiver_mp as DistExtend
from hfm.dist_est_bin import ApproxDist_bin
from hfm.dist_est_bin import ApproxDist_bin_revised as ApproxDist_alter


from pyfair.facil.metric_cont import \
    contg_tab_mu_type2 as contingency_tab
from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_fpr, calc_fnr, calc_sensitivity, calc_specificity,
    imba_geometric_mean, imba_discriminant_power,
    imba_Matthew_s_cc, imba_Cohen_s_kappa)
from pyfair.marble.metric_fair import (
    marginalised_pd_mat, prev_unpriv_unaware, prev_unpriv_manual,
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)
from hfm.discriminative_risk import hat_L_fair, hat_L_loss
# from experiment.utils.fair_grp_ext import (
# from hfm.metrics.fair_grp_ext import (
from pyfair.marble.metric_fair import (
    # StatsParity_sing, StatsParity_mult,
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing,
    alterGrps_sing)


# Convergence
from hfm.dist_cvg_nonbin import (
    # StratVacant, StratEarlyStop, EffExact)  # ,StratRearrange
    StratVacant, StratEarlyStop, StratRearrange, EffExact)


from hfm.earlybreak import Naive_bin as NaiveHD_bin
from hfm.earlybreak import Naive_nonbin as NaiveHD_nonbin
from hfm.earlybreak import Naive_multivar as NaiveHD_multivar
from hfm.earlybreak import EffHD_bin as EffHDD_bin
from hfm.earlybreak import EffHD_nonbin as EffHDD_nonbin
from hfm.earlybreak import EffHD_multivar as EffHDD_multivar
# from hfm.earlybreak import EffHD_bin, EffHD_nonbin, EffHD_multivar
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier)
from experiment.utils_learner import (
    INDIVIDUALS, LGBMClassifier, FairGBMClassifier, AdaFair)

import torch
import torch.nn as nn
import torch.optim as optim
# from experiment.utils.fair_rev_group import (
#     UD_grp1_DP, UD_grp2_EO, UD_grp3_PQP)
DistApprox_nonbin = DistApprox
DistExtend_multivar = DistExtend

unpriv_group_one = prev_unpriv_grp_one
unpriv_group_two = prev_unpriv_grp_two
unpriv_group_thr = prev_unpriv_grp_thr
unpriv_unaware = prev_unpriv_unaware
unpriv_manual = prev_unpriv_manual
del prev_unpriv_unaware, prev_unpriv_manual
del prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr


# =====================================
# fairmanf_ext
# mext_exp5_mp.py


# -------------------------------------
# expt2*
# class ComparisonBCD_setup:


class ComparisonD_setup:
    def __init__(self):
        pass

    def count_scores(
            self,
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst, jt_tst,
            positive_label=1, m1=20, m2=8, n_e=3, pool=None):
        ans_trn = self.count_single_member(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn, jt_trn,
            positive_label, m1, m2, n_e, pool)
        ans_tst = self.count_single_member(
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst, jt_tst,
            positive_label, m1, m2, n_e, pool)
        return ans_trn + ans_tst

    def count_single_member(self,
                            X, A, y, y_hat, y_qtb, g1m_indices, jt,
                            positive_label=1, m1=20, m2=8, n_e=3,
                            pool=None):  # *,omitted=True):
        raise NotImplementedError

    def count_sing_part3_df_bin(self, X_y, X_y_hat, non_sa, A_j,
                                m1=20, m2=8, n_e=2, pool=None):
        ans_fair = []

        ut_a = time.time()
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_y, non_sa)
        (Df_01, Df_avg), t_Df = DistDirect_bin(X_y_hat, non_sa)
        df_ecai, _ = fair_degree_v3(Ds_01, Df_01)  # ut_ddf_ecai
        df_nips, _ = fair_degree_v4(Ds_01, Df_01)  # ut_ddf_nips
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_a = time.time() - ut_a
        ans_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         df_ecai, df_nips, v3_df_avg, v4_df_avg])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del df_ecai, df_nips, v3_df_avg, v4_df_avg

        ut_b = time.time()
        '''
        idx_sa = ~non_sa  # actually, doesn't need A here
        Ds_01, t_Ds = ApproxDist_bin(X_y, A_j, idx_sa, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist_bin(X_y_hat, A_j, idx_sa, non_sa, m1, m2)
        '''
        Ds_01, t_Ds = ApproxDist_bin(X_y, A_j, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist_bin(X_y_hat, A_j, non_sa, m1, m2)
        df_ecai, _ = fair_degree_v3(Ds_01, Df_01)
        df_nips, _ = fair_degree_v4(Ds_01, Df_01)
        # TODO: 这里应该加上 df_nips, avg，也不用，这是时间耗时，
        # 这个实验虽然没直接算出来，但是我可以在绘图代码里手动计算
        ut_b = time.time() - ut_b
        ans_fair.extend([Ds_01, t_Ds, Df_01, t_Df, df_ecai, df_nips])
        del Ds_01, t_Ds, Df_01, t_Df, df_ecai, df_nips

        B_j = non_sa.astype(DTY_INT)
        ut_c = time.time()
        (Ds_01, Ds_avg), t_Ds = DistApprox(X_y, B_j, m1, m2, n_e, pool)
        (Df_01, Df_avg), t_Df = DistApprox(X_y_hat, B_j, m1, m2, n_e, pool)
        df_ecai, _ = fair_degree_v3(Ds_01, Df_01)
        df_nips, _ = fair_degree_v4(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_c = time.time() - ut_c
        ans_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         df_ecai, df_nips, v3_df_avg, v4_df_avg])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del df_ecai, df_nips, v3_df_avg, v4_df_avg

        ans_fair.extend([ut_a, ut_b, ut_c])
        return ans_fair  # shape=(29,) =(10+6+10+3,)

    def count_sing_part4_df_non(self, X_y, X_y_hat, g1m_indices, A,
                                m1=20, m2=8, n_e=2, pool=None):
        ans_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(
            X_y, g1m_indices)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistDirect_multivar(
            X_y_hat, g1m_indices)
        v3_df_max, _ = fair_degree_v3(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_max, _ = fair_degree_v4(Ds_01, Df_01)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ans_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        n_a = len(g1m_indices)

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            ans_fair.extend([Ds_01[i], Ds_avg[i], t_Ds[i],
                             Df_01[i], Df_avg[i], t_Df[i]])
            v3_df_max, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            v3_df_avg, _ = fair_degree_v3(Ds_avg[i], Df_avg[i])
            v4_df_max, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            v4_df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            ans_fair.extend([v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        if n_a == 1:
            ans_fair.extend([''] * 10)
        del v3_df_max, v3_df_avg, v4_df_max, v4_df_avg
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        del Ds_midtmp, Df_midtmp
        ut_a = time.time() - ut_a

        ut_b = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistExtend(
            X_y, A, m1, m2, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistExtend(
            X_y_hat, A, m1, m2, n_e, pool)
        v3_df_max, _ = fair_degree_v3(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_max, _ = fair_degree_v4(Ds_01, Df_01)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ans_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del v3_df_max, v4_df_max, v3_df_avg, v4_df_avg

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            ans_fair.extend([
                Ds_01[i], Ds_avg[i], t_Ds[i], Df_01[i], Df_avg[i], t_Df[i]])
            v3_df_max, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            v3_df_avg, _ = fair_degree_v3(Ds_avg[i], Df_avg[i])
            v4_df_max, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            v4_df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            ans_fair.extend([v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        if n_a == 1:
            cmp_fair.extend([''] * 10)
        del v3_df_max, v3_df_avg, v4_df_max, v4_df_avg
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        del Ds_midtmp, Df_midtmp
        ut_b = time.time() - ut_b

        cmp_fair.extend([ut_a, ut_b])
        return ans_fair  # shape=(62,) =(10+10*2+10+10*2+2,)


class ComparisonDp_setup:
    def __init__(self, *, omitted=True):
        self._metric_part1 = [
            'Accuracy', 'Precision', 'Recall', 'f1_score',
            'fpr', 'fnr', 'sensitivity', 'specificity', 'g_mean', 'dp',
            'Matthew', 'Cohen', '(random_acc)']
        if omitted:
            self._metric_part1.remove('fpr')
            self._metric_part1.remove('fnr')
            self._metric_part1.remove('Matthew')
            self._metric_part1.remove('Cohen')
            self._metric_part1.remove('(random_acc)')

        self._metric_part2 = [
            'unaware', 'grp_one', 'grp_two', 'grp_thr', 'manual',
            'hat_L(fair)', 'hat_L(loss)']
        if omitted:
            self._metric_part2.remove('unaware')
            self._metric_part2.remove('manual')

        self._metric_part4 = [
            'Ds', 'Ds_avg', 't(Ds)', 'Df', 'Df_avg', 't(Df)',
            'df.ecai', 'df.nips', 'v3:df_avg', 'v4:df_avg']  # siz=10
        self._metric_part3 = [
            'Ds', '', 't(Ds)', 'Df', '', 't(Df)',
            'df.ecai', 'df.nips', 'v3:avg', 'v4:avg',        # 1.#10
            'Ds', 't(Ds)', 'Df', 't(Df)', 'df.v3', 'df.v4',  # 2.#6
            'Ds', '', 't(Ds)', 'Df', '', 't(Df)',
            'df.ecai', 'df.nips', 'v3:avg', 'v4:avg',        # 3.#10
            'T(DistDirect_bin)', 'T(ApproxDist_bin)', 'T(DistApprox)']

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
            return LGBMClassifier(n_estimators=nb_cls)
        elif name_ens == 'fairgbm':
            return FairGBMClassifier(n_estimators=nb_cls,
                                     constraint_type=constraint)
        elif name_ens == 'adafair':
            return AdaFair(n_estimators=nb_cls,
                           saIndex=saIndex, saValue=saValue)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    # fvote_empiric.py
    # fairmanf/manf_exp3.py
    # def count_single_member(self, X, A, y, y_hat, y_qtb, g1m_indices, jt,
    #                         positive_label=1, m1=20, m2=8, n_e=3, *, omitted=True):
    #   # NB. must be np.ndarray
    #   raise NotImplementedError

    def count_sing_part1(self, y, y_hat, positive_label,
                         *, omitted=True):
        # NB. must be np.ndarray
        tp, fp, fn, tn = contingency_tab(y, y_hat, positive_label)
        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        res_indi.append(calc_recall(tp, fp, fn, tn))
        res_indi.append(calc_f1_score(tp, fp, fn, tn))

        if not omitted:
            res_indi.append(calc_fpr(tp, fp, fn, tn))
            res_indi.append(calc_fnr(tp, fp, fn, tn))
        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, imba_geometric_mean(sen, spe)])
        res_indi.append(imba_discriminant_power(sen, spe))
        if not omitted:
            res_indi.append(imba_Matthew_s_cc(tp, fp, fn, tn))
            res_indi.extend(imba_Cohen_s_kappa(tp, fp, fn, tn))

        # if not omitted:  shape= (13,)= (6+3+2+2,)  # no sen
        return res_indi  # shape= ( 8,)= (4+1+1+2,)

    def count_sing_part2(self, y, y_hat, y_qtb, non_sa,
                         positive_label=1, *, omitted=True):
        _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(
            y, y_hat, positive_label, non_sa)
        # g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
        #     marginalised_pd_mat(y, y_hat, positive_label, non_sa)
        cmp_fair = []
        if not omitted:
            tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
            cmp_fair.extend(tmp_0)

        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)
        if not omitted:
            tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)
            cmp_fair.extend(tmp_4)

        if not omitted:
            cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        if not omitted:
            cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))

        # if not omitted:  shape= (17,)= (5*2+5+2,)
        return cmp_fair  # shape= (11,)= (3*2+3+2,)


# -------------------------------------


# class Comparison_BCD2_withDirectComput(ComparisonBCD_setup):
class ComparisonD2_withDirectComput(ComparisonD_setup):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3, *, omitted=True):
        super().__init__(omitted=omitted)
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        self._n_e = n_e

    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
            m1, m2, positive_label,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None,
                               X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []

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
            clf = LGBMClassifier(n_estimators=nb_cls)
            clf.fit(X_A_trn, y_trn)

        elif name_ens == 'fairgbm':  # TODO
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
        return y_insp, y_pred, yq_insp, yq_pred, ut, clf

    def subroute_one_norm_att(self,
                              X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
                              X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
                              positive_label=1, m1=20, m2=8, pool=None,
                              X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                              *, omitted=True):
        res_attr = []

        y_insp, y_pred, yq_insp, yq_pred, ut, _ = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,
            X_A_tst, y_tst, X_Aq_tst, None)

        return

    def subroute_one_sens_att(self):
        pass

    def count_single_member(self, X, A, y, y_hat, y_qtb, g1m_indices, jt,
                            positive_label=1, m1=20, m2=8, n_e=3, pool=None,
                            *, omitted=True):
        ut_c = time.time()
        return


# -------------------------------------

# -------------------------------------


# =====================================
# Revision


# -------------------------------------
# An efficient algorithm for calculating the exact Hausdorff
# distance


class RevCompZ_setup:
    # def __init__(self, abbr_cls, *, omitted=True):
    #   # self._metric_part1 = []
    #   self._member = INDIVIDUALS[abbr_cls]

    def __init__(self, *, omitted=True):
        self._omit = omitted

    def subproc_bin(self, X_nA_y, A_j, non_sa, m1, m2, n_e):
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        '''
        app_0 = ApproxDist_bin(X_nA_y, A_j, ~non_sa, non_sa, m1, m2)
        app_1 = ApproxDist_alter(X_nA_y, A_j, non_sa, m1, m2)
        '''
        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)
        app_1 = ApproxDist_alter(X_nA_y, non_sa, m1, m2)  # ~non_sa
        app_2 = DistApprox_nonbin(
            X_nA_y, non_sa.astype('int'), m1, m2, n_e)

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])

        ans_tim = [hdd_1[1], hdd_2[1], eff_1[1], eff_2[1],
                   luo_1[1], luo_2[1], app_0[1], app_1[1], app_2[1]]
        ans_max = [hdd_1[0], hdd_2[0], eff_1[0], eff_2[0],
                   luo_1[0][0], luo_2[0][0],
                   app_0[0], app_1[0][0], app_2[0][0]]
        ans_avg = [luo_1[0][1], luo_2[0][1], app_2[0][1]]
        return ans_tim + ans_max + ans_avg  # (21,) =(9+9+3,)

    def subproc_nonbin(self, X_nA_y, A_j, g1m, m1, m2, n_e, pool):
        non_sa = g1m[0]  # non_sa, n_ai = g1m[0], len(g1m)
        Aj_bin = non_sa.astype('int')
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        luo_3 = DistDirect_nonbin(X_nA_y, g1m)
        '''
        app_0 = ApproxDist_bin(X_nA_y, A_j, ~non_sa, non_sa, m1, m2)
        app_1 = ApproxDist_alter(X_nA_y, A_j, non_sa, m1, m2)
        '''
        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)
        app_1 = ApproxDist_alter(X_nA_y, non_sa, m1, m2)  # ~non_sa
        app_2 = DistApprox_nonbin(X_nA_y, Aj_bin, m1, m2, n_e)
        app_3 = DistApprox_nonbin(X_nA_y, A_j, m1, m2, n_e)

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        hdd_3 = NaiveHD_nonbin(X_nA_y, g1m)
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_3 = EffHDD_nonbin(X_nA_y, g1m)

        ans_tim = [hdd_1[1], hdd_2[1], hdd_3[1],
                   eff_1[1], eff_2[1], eff_3[1],
                   luo_1[1], luo_2[1], luo_3[1],
                   app_0[1], app_1[1], app_2[1], app_3[1]]
        ans_max = [hdd_1[0], hdd_2[0], hdd_3[0],
                   eff_1[0], eff_2[0], eff_3[0],
                   luo_1[0][0], luo_2[0][0], luo_3[0][0],
                   app_0[0], app_1[0][0], app_2[0][0], app_3[0][0]]
        ans_avg = [luo_1[0][1], luo_2[0][1], luo_3[0][1],
                   app_2[0][1], app_3[0][1]]
        # pdb.set_trace()
        return ans_tim + ans_max + ans_avg  # (31,) =(13+13+5,)

    def subproc_multivar(self, X_nA_y, A, g1m_ind, m1, m2, n_e, pool):
        luo_4 = DistDirect_multivar(X_nA_y, g1m_ind)
        app_4 = DistExtend_multivar(X_nA_y, A, m1, m2, n_e)
        luo_mid, app_mid = luo_4[0][2], app_4[0][2]
        luo_4 = (luo_4[0][: 2], luo_4[1])
        app_4 = (app_4[0][: 2], app_4[1])

        hdd_4 = NaiveHD_multivar(X_nA_y, g1m_ind)
        eff_4 = EffHDD_multivar(X_nA_y, g1m_ind) 
        hdd_mid, eff_mid = hdd_4[0][1], eff_4[0][1]
        hdd_4 = (hdd_4[0][0], hdd_4[1])
        eff_4 = (eff_4[0][0], eff_4[1])

        ans_tim = [hdd_4[1], eff_4[1], luo_4[1], app_4[1]]
        ans_max = [hdd_4[0], eff_4[0], luo_4[0][0], app_4[0][0]]
        ans_avg = [luo_4[0][1], app_4[0][1]]
        result = ans_tim + ans_max + ans_avg  # (10,) =(4+4+2,)
        n_a = len(g1m_ind)
        for i in range(n_a):
            tmp_tim = [hdd_mid[1][i], eff_mid[1][i],
                       luo_mid[2][i], app_mid[2][i]]
            tmp_max = [hdd_mid[0][i], eff_mid[0][i],
                       luo_mid[0][i], app_mid[0][i]]
            tmp_avg = [luo_mid[1][i], app_mid[1][i]]
            result.extend(tmp_tim + tmp_max + tmp_avg)  # (10,)
        if n_a == 1:
            result.extend([''] * 10)
        # pdb.set_trace()
        return result  # (30,) =(10+10+10,)

    def count_single_member(self, X, A, y_fx, g1m_indices,
                            m1, m2, n_e, pool=None):
        X_nA_y = np.concatenate([  # X_nA_yfx
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        n_a = len(g1m_indices)
        for i in range(n_a):
            # self.subproc_bin(
            #     X_nA_y, A[:, i], g1m_indices[i][0], m1, m2, n_e)
            # self.subproc_nonbin(
            #     X_nA_y, A[:, i], g1m_indices[i], m1, m2, n_e, pool)
            pass
        pdb.set_trace()
        return


class RevCompZA_efficient(RevCompZ_setup):
    def schedule_content(self, X, A, y_fx, g1m_indices,
                         m1, m2, n_e, pool=None):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        return self.subproc_multivar(
            X_nA_y, A, g1m_indices, m1, m2, n_e, pool)

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 30)
        csv_row_2c = ['multivar'] + [''] * 9 + ['sen-att #1'] + [
            ''] * 9 + ['sen-att #2'] + [''] * 9
        csv_row_3c = ['tim', '', '', '', 'max', '', '', '', 'avg', '']
        csv_row_4c = ['Naive', 'EffHD', 'Direct', 'Extend',
                      'Naive', 'EffHD', 'Direct', 'Approx',
                      'Direct', 'Approx']  # 'ApproxDist'
        csv_row_3c = csv_row_3c * 3
        csv_row_4c = csv_row_4c * 3
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class RevCompZB_efficient(RevCompZ_setup):
    def schedule_content(self, X, A, y_fx, g1m_indices,
                         m1, m2, n_e, pool=None):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        result, n_a = [], len(g1m_indices)
        for i in range(n_a):
            tmp = self.subproc_nonbin(
                X_nA_y,  # A[:, i].copy(),
                A[:, i], g1m_indices[i], m1, m2, n_e, pool)
            result.extend(tmp)  # (31,)
        if n_a == 1:
            result.extend([''] * 31)
        del n_a, X_nA_y
        return result

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 31 * 2)
        csv_row_2c = ['sa#1 tim'] + [''] * 12 + ['max'] + [''] * 12 + [
            'avg'] + [''] * 4 + ['sa#2 tim'] + [''] * 12 + [
            'max'] + [''] * 12 + ['avg'] + [''] * 4  # (13+13+5)*2
        # csv_row_2c = ['sen-att #1'] + [''] * 30 + ['sen-att #2'] + [''] * 30
        # csv_row_3c = ['tim'] + [''] * 12 + [
        #     'max'] + [''] * 12 + ['avg'] + [''] * 4  # 13*2+5 =31
        csv_row_4c = ['Naive', '', '', 'EffHDD', '', '',
                      'DirectDist', '', '',
                      'ApproxDist_bin', '', 'DistApprox_nonbin', '']
        csv_row_5c = [  # 'bin', 'nonbin/bin', 'nonbin',
            'bin', 'nonbin (bi-val)', 'nonbin (multi-val)',
            'bin', 'nonbin (bi-val)', 'nonbin (multi-val)',
            'bin', 'nonbin (bi-val)', 'nonbin (multi-val)',
            'bin', 'bin_alter', 'nonbin (bi-val)', 'nonbin (multi-val)']
        csv_row_4c = csv_row_4c * 2 + [
            'DirectDist', '', '', 'DistApprox_nonbin', '']  # 13*2+5
        csv_row_5c = csv_row_5c * 2 + [
            'bin', 'nonbin (bi-val)', 'nonbin (multi-val)',
            'nonbin (bi-val)', 'nonbin (multi-val)']  # 13*2+5 =31
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


class RevCompZC_efficient(RevCompZ_setup):
    def schedule_content(self, X, A, y_fx, g1m_indices,
                         m1, m2, n_e, pool=None):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        result, n_a = [], len(g1m_indices)
        for i in range(n_a):
            tmp = self.subproc_bin(
                X_nA_y,  # A[:, i].copy(),
                A[:, i], g1m_indices[i][0], m1, m2, n_e)
            result.extend(tmp)  # (21,)
        if n_a == 1:
            result.extend([''] * 21)
        del n_a, X_nA_y
        return result

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 21 * 2)
        csv_row_2c = ['sa#1 tim'] + [''] * 8 + ['max'] + [''] * 8 + [
            'avg'] + [''] * 2 + ['sa#2 tim'] + [''] * 8 + [
            'max'] + [''] * 8 + ['avg'] + [''] * 2  # (9+9+3)*2 =21*2
        # csv_row_2c = ['sen-att #1'] + [''] * 20 + ['sen-att #2'] + [''] * 20
        # csv_row_3c = ['tim'] + [''] * 8 + ['max'] + [''] * 8 + ['avg', '', '']
        csv_row_4c = ['Naive', '', 'EffHDD', '', 'DirectDist', '',
                      'ApproxDist_bin', '', 'DistApprox_nonbin']
        csv_row_5c = ['bin', 'nonbin (bi-val)', 'bin', 'nonbin (bi-val)',
                      'bin', 'nonbin (bi-val)',
                      'bin', 'bin_alter', 'nonbin (bi-val)']
        csv_row_4c = csv_row_4c * 2 + [
            'DirectDist', '', 'DistApprox_nonbin']  # 9*2+3 =21
        csv_row_5c = csv_row_5c * 2 + [
            'bin', 'nonbin (bi-val)', 'nonbin (bi-val)']  # 9*2+3
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


# -------------------------------------
# MLP (neural network) computation

# https://www.zhihu.com/question/550458967
# https://stackoverflow.com/questions/75102134/mat1-and-mat2-must-have-the-same-dtype
# https://zhuanlan.zhihu.com/p/789486358

# https://www.zhihu.com/tardis/zm/art/35709485?source_id=1003


class My_MLP(nn.Module):
    def __init__(self, input_siz, hidden_layer=[256, 64],
                 output_siz=2):
        # super().__init__()
        super(My_MLP, self).__init__()
        '''
    self.fc1 = nn.Linear(input_siz, hidden_layer[0])
    self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
    self.fc3 = nn.Linear(hidden_layer[1], output_siz)
    '''
        self.classifier = nn.Sequential(
            nn.Linear(input_siz, hidden_layer[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_layer[0], hidden_layer[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_layer[1], output_siz, bias=True))

    def forward(self, x):
        '''
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        return z
        '''
        return self.classifier(x)


# class My_BaseNet(nn.Module):
#   def __init__(self, input_siz, hidden_siz, output_siz=2):
#     pass
#     nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True), nn.ReLU)


class RevCompY_setup:
    # def __init__(self, abbr_cls='MLP', *, omitted=True):
    def __init__(self, training_blk=128, training=20,
                 abbr_cls='MLP', *, omitted=True):
        self._training_blk = training_blk
        self._training = training
        self._omit = omitted
        # self._member = INDIVIDUALS[abbr_cls]

    def neural_network(self, X_wA, y, training, training_blk):
        since = time.time()
        model = My_MLP(X_wA.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # data = torch.tensor(X_wA).to(torch.float32)
        data = torch.tensor(X_wA, dtype=torch.float32)
        labels = torch.tensor(y)
        '''
    output = model(data)  # 前向传播
    loss = criterion(output, labels)  # 计算损失
    optimizer.zero_grad()  # 清零梯度
    loss.backward()   # 反向传播
    optimizer.step()  # 更新权重
    '''

        training_los = []
        chunk = X_wA.shape[0] // training_blk  # _item
        for i in range(training):
            for j in range(chunk):
                output = model(data[
                    training_blk * j: training_blk * (j + 1)])
                loss = criterion(output, labels[
                    training_blk * j: training_blk * (j + 1)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_los.append(loss.detach().numpy().tolist())
            output = model(data[training_blk * chunk:])
            loss = criterion(output, labels[training_blk * chunk:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_los.append(loss.detach().numpy().tolist())
        since = time.time() - since
        self._member_losses = training_los
        output = model(data)
        loss = criterion(output, labels)
        comp = nn.functional.softmax(output, dim=1)
        # comp = nn.functional.softmax(output)
        output = output.detach().numpy()
        loss = loss.detach().numpy().tolist()

        fx = comp.argmax(dim=1)
        fx = fx.detach().numpy()
        # fx = comp.detach().numpy().argmax(axis=1)
        comp = comp.detach().numpy()
        comp = np.log(comp[:, 1]) * y + np.log(comp[:, 0]) * (1 - y)
        comp = float(np.mean(-comp))  # approx. loss
        acc_score = float(np.mean(fx == y))
        self._member = model
        return output, fx, loss, comp, acc_score, since

    def count_single_member(self, X, A, y, X_wA, g1m_indices,
                            m1, m2, n_e, pool=None,
                            training=20, training_blk=128):
        '''
        self._member.fit(X_wA, y)
        fx = self._member.predict(X_wA)
        self._member.predict_proba(X_wA)
        '''

        if 2 in y:
            y[y == 2] = 0
        # embedding, fx, loss, comp, acc = self.neural_network(
        #     X_wA, y, training, training_blk)
        embedding, fx, loss, comp, acc, _ = self.neural_network(
            X_wA, y, training, training_blk)

        X_nA_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_fx = np.concatenate([
            fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        embed_fx = np.concatenate([
            fx.reshape(-1, 1).astype(DTY_FLT), embedding], axis=1)
        n_a = len(g1m_indices)

        '''
        result = self.subproc_bin(
            X_nA_y, X_nA_fx, embed_fx, A[:, 1], g1m_indices[1][0],
            m1, m2, n_e)
        result = self.subproc_nonbin(
            X_nA_y, X_nA_fx, embed_fx, A[:, 1], g1m_indices[1],
            m1, m2, n_e)
        result = self.subproc_multivar(
            X_nA_y, X_nA_fx, embed_fx, A, g1m_indices, m1, m2, n_e)
        '''
        return

    def subproc_bin(self, X_nA_y, X_nA_fx, embed_fx,
                    A_j, non_sa, m1, m2, n_e):
        (Ds, Ds_avg), t_Ds = DistDirect_bin(X_nA_y, non_sa)
        (Df, Df_avg), t_Df = DistDirect_bin(X_nA_fx, non_sa)
        (Dh, Dh_avg), t_Dh = DistDirect_bin(embed_fx, non_sa)
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
        dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
        result = [t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg]
        result.extend([dff_ecai, dff_nips, dfh_ecai, dfh_nips,
                       t_Ds + t_Df + ut_ddf_ecai,
                       t_Ds + t_Df + ut_ddf_nips,
                       t_Ds + t_Dh + ut_ddh_ecai,
                       t_Ds + t_Dh + ut_ddh_nips])  # 3*3+4*2 =9+8=17

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, [non_sa, ~non_sa])
        (Dh, Dh_avg), t_Dh = DistDirect_nonbin(embed_fx, [non_sa, ~non_sa])
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
        dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
        result.extend([
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg])
        result.extend([dff_ecai, dff_nips, dfh_ecai, dfh_nips,
                       t_Ds + t_Df + ut_ddf_ecai,
                       t_Ds + t_Df + ut_ddf_nips,
                       t_Ds + t_Dh + ut_ddh_ecai,
                       t_Ds + t_Dh + ut_ddh_nips])
        return result  # (34,) =(17*2,)

    def subproc_nonbin(self, X_nA_y, X_nA_fx, embed_fx,
                       A_j, g1m, m1, m2, n_e, pool=None):
        non_sa = g1m[0]  # non_sa, n_ai = g1m[0], len(g1m)
        result = self.subproc_bin(
            X_nA_y, X_nA_fx, embed_fx, A_j, non_sa, m1, m2, n_e)

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, g1m)
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, g1m)
        (Dh, Dh_avg), t_Dh = DistDirect_nonbin(embed_fx, g1m)
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
        dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
        result.extend([
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg])
        result.extend([dff_ecai, dff_nips, dfh_ecai, dfh_nips,
                       t_Ds + t_Df + ut_ddf_ecai,
                       t_Ds + t_Df + ut_ddf_nips,
                       t_Ds + t_Dh + ut_ddh_ecai,
                       t_Ds + t_Dh + ut_ddh_nips])
        return result  # (51,) =(17*3,)

    def subproc_multivar(self, X_nA_y, X_nA_fx, embed_fx,
                         A, g1m_ind, m1, m2, n_e):
        (Ds, Ds_avg, Ds_mid), t_Ds = DistDirect_multivar(X_nA_y, g1m_ind)
        (Df, Df_avg, Df_mid), t_Df = DistDirect_multivar(X_nA_fx, g1m_ind)
        (Dh, Dh_avg, Dh_mid), t_Dh = DistDirect_multivar(embed_fx, g1m_ind)
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
        dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
        result = [t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg]
        result.extend([dff_ecai, dff_nips, dfh_ecai, dfh_nips,
                       t_Ds + t_Df + ut_ddf_ecai,
                       t_Ds + t_Df + ut_ddf_nips,
                       t_Ds + t_Dh + ut_ddh_ecai,
                       t_Ds + t_Dh + ut_ddh_nips])  # 9+4+4 =17

        n_a = len(g1m_ind)
        for i in range(n_a):
            Ds, Ds_avg, t_Ds = Ds_mid[0][i], Ds_mid[1][i], Ds_mid[2][i]
            Df, Df_avg, t_Df = Df_mid[0][i], Df_mid[1][i], Df_mid[2][i]
            Dh, Dh_avg, t_Dh = Dh_mid[0][i], Dh_mid[1][i], Dh_mid[2][i]
            dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
            dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
            dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
            dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
            result.extend([
                t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg])
            result.extend([dff_ecai, dff_nips, dfh_ecai, dfh_nips,
                           t_Ds + t_Df + ut_ddf_ecai,
                           t_Ds + t_Df + ut_ddf_nips,
                           t_Ds + t_Dh + ut_ddh_ecai,
                           t_Ds + t_Dh + ut_ddh_nips])
        if n_a == 1:
            result.extend([''] * 17)
        return result  # (51,) =(17+17*2,)

    def count_sing_part1(self, y, y_hat, positive_label):
        tp, fp, fn, tn = contingency_tab(y, y_hat, positive_label)
        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        # res_indi.append(calc_recall(tp, fp, fn, tn))
        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tp)
        res_indi.extend([sen, spe])
        res_indi.append(calc_f1_score(tp, fp, fn, tn))
        res_indi.extend([imba_geometric_mean(sen, spe),
                         imba_discriminant_power(sen, spe)])
        return res_indi  # (7,) =(5+2,)

    def count_sing_part2(self, y, y_hat, non_sa,
                         positive_label=1):
        _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
            y, y_hat, positive_label, non_sa)  # g1_Cij, g0_Cij
        cmp_fair = []

        # tmp_0 = unpriv_unaware(g1_Cm, g0_Cm)
        tmp_1 = unpriv_group_one(g1_Cm, g0_Cm)
        tmp_2 = unpriv_group_two(g1_Cm, g0_Cm)
        tmp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
        # tmp_4 = unpriv_manual(g1_Cm, g0_Cm)
        cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))

        # cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        # cmp_fair.extend(hat_L_loss(y_hat, y))
        return cmp_fair  # (9,) =(6+3+2 -2,)

    def count_sing_part2p(self, y, y_hat, y_qtb):
        # return [hat_L_fair(y_hat, y_qtb), hat_L_loss(y_hat, y)]
        return [hat_L_loss(y_hat, y), hat_L_fair(y_hat, y_qtb)]


class RevCompYA_NN(RevCompY_setup):
    def schedule_content(self, X, A, y, X_wA, X_wAq, g1m_indices,
                         m1, m2, n_e, pool=None,
                         positive_label=1):
        embedding, y_hat, loss, comp, acc, ut = self.neural_network(
            X_wA, y, self._training, self._training_blk)
        X_nA_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        embed_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), embedding], axis=1)
        n_a = len(g1m_indices)
        result = [ut, loss, comp, acc]
        for i in range(n_a):
            tmp = self.subproc_nonbin(
                X_nA_y, X_nA_y_hat, embed_y_hat, A[:, i],  # .copy(),
                g1m_indices[i], m1, m2, n_e)
            result.extend(tmp)
        if n_a == 1:
            result.extend([''] * 51)
        # result .shape= (105,) =(3+51*2,)
        del ut, acc, comp, loss, embedding

        output = torch.tensor(X_wAq, dtype=torch.float32)
        output = self._member(output)  # model(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(output).argmax(dim=1)
        y_qtb = output.detach().numpy()
        # y_qtb = nn.functional.softmax(output).detach().numpy().argmax(axis=1)
        tmp_1 = self.count_sing_part1(y, y_hat, positive_label)
        tmp_2 = self.count_sing_part1(y, y_qtb, positive_label)
        tmp_3 = self.count_sing_part2p(y, y_hat, y_qtb)
        result.extend(tmp_1 + tmp_2 + tmp_3)  # 105+(7*2+2) =+16
        del tmp_1, tmp_2, tmp_3
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, g1m_indices[i][0], positive_label)
            result.extend(tmp)
        if n_a == 1:
            result.extend([''] * 9)   # 105+16+9*2 =139
        # pdb.set_trace()
        return result  # (1+139,) =(105+16+18,)

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 140)
        csv_row_2c = ['NN training'] + [''] * 3
        csv_row_3c = ['tim', 'loss', '', 'acc']
        csv_row_4c = ['', '', '(alter)', '']  # [''] * 4

        tmp_r3c = ['tim', '', '', 'max', '', '', 'avg', '', '',
                   'df', '', 'df(h)', '', 'T(df)', '', '', '']
        tmp_r4c = ['t_Ds', 't_Df', 't_Dh', 'Ds', 'Df', 'Dh',
                   'Ds_avg', 'Df_avg', 'Dh_avg',
                   'df.prev', 'df.', 'df(h).prev', 'df(h).',
                   'T df.prev', 'T df.', 'T df(h).prev', 'T df(h).']
        for i in range(1, 3):  # [1,2]:  # range(2):
            csv_row_2c.extend([f'sa#{i} DistDirect_bin'] + [''] * 16 + [
                'DistDirect_nonbin (bi-val)'] + [''] * 16 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 16)
            csv_row_3c.extend(tmp_r3c * 3)
            csv_row_4c.extend(tmp_r4c * 3)
        # del tmp_r3c, tmp_r4c  # 4+17*3 *2 =55+51=106

        csv_row_2c.extend(['Normal performance'] + [''] * 6 + [
            'Perturbing..'] + [''] * 6 + ['', 'DR'])
        csv_row_3c.extend([''] * 7 * 2 + ['DR_prime', 'DR'])
        tmp_r4c = ['Accuracy', 'Precision', 'Recall /Sensitivity',
                   'Specificity', 'f1_score', 'g_mean', 'DiscPower']
        csv_row_4c.extend(tmp_r4c * 2 + ['hatL_loss', 'hatL_fair'])
        # del tmp_r4c  # 106+ (7*2+2) =106+16 =71+51=122

        for i in range(2):  # n_a):
            csv_row_2c.extend([f'sen-att sa#{i+1}'] + [''] * 8)
            csv_row_3c.extend(['DP', '', 'EO', '', 'PQP', '',
                               'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)
        del tmp_r3c, tmp_r4c  # 122+ 9*2 =122+18 =89+51=140
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class RevCompYB_NN(RevCompY_setup):
    def curr_subset(self, X, A, y, X_wA, X_wAq, g1m,
                    embed, y_hat, y_qtb,
                    m1, m2, n_e, positive_label, pool=None):
        X_nA_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        embed_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), embed], axis=1)
        n_a = len(g1m)
        curr_res = []
        for i in range(n_a):
            tmp = self.subproc_nonbin(
                X_nA_y, X_nA_y_hat, embed_y_hat, A[:, i],  # .copy(),
                g1m[i], m1, m2, n_e, pool=pool)
            curr_res.extend(tmp)
        if n_a == 1:
            curr_res.extend([''] * len(tmp))  # 51)

        tmp_1 = self.count_sing_part1(y, y_hat, positive_label)
        tmp_2 = self.count_sing_part1(y, y_qtb, positive_label)
        tmp_3 = self.count_sing_part2p(y, y_hat, y_qtb)
        curr_res.extend(tmp_1 + tmp_2 + tmp_3)
        del tmp_1, tmp_2, tmp_3  # 51*2+(7*2+2) =102+16=118
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, g1m[i][0], positive_label)
            curr_res.extend(tmp)
        if n_a == 1:
            curr_res.extend([''] * 9)
        del tmp                  # 118+9*2 =118+18=136
        return curr_res

    def schedule_content(
            self,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            m1, m2, n_e, pool=None, positive_label=1):
        embed, y_insp, loss, comp, acc, ut = self.neural_network(
            X_wA_trn, y_trn, self._training, self._training_blk)
        result = [ut, loss, comp, acc]

        output = torch.tensor(X_wAq_trn, dtype=torch.float32)
        output = self._member(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(output).argmax(dim=1)
        yq_insp = output.detach().numpy()
        # yq_insp = nn.functional.softmax(output).detach().numpy().argmax(axis=1)
        tmp = self.curr_subset(
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            embed, y_insp, yq_insp, m1, m2, n_e, positive_label,
            pool=pool)
        result.extend(tmp)

        embed = torch.tensor(X_wA_tst, dtype=torch.float32)
        embed = self._member(embed)
        output = nn.functional.softmax(embed, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(embed).argmax(dim=1)
        y_pred = output.detach().numpy()
        # y_pred = nn.functional.softmax(embed).detach().numpy().argmax(axis=1)
        embed = embed.detach().numpy()

        output = torch.tensor(X_wAq_tst, dtype=torch.float32)
        output = self._member(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(output).argmax(dim=1)
        yq_pred = output.detach().numpy()
        # yq_pred = nn.functional.softmax(output).detach().numpy().argmax(axis=1)
        tmp = self.curr_subset(
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            embed, y_pred, yq_pred, m1, m2, n_e, positive_label,
            pool=pool)
        result.extend(tmp)

        del ut, loss, comp, acc, embed, output
        del y_insp, yq_insp, yq_pred, y_pred, tmp
        return result  # (276,) =(4+136*2,)

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 276)  # 176)
        csv_row_2c = ['NN training'] + [''] * 3
        csv_row_3c = ['tim', 'loss', '', 'acc']
        csv_row_4c = ['', '', '(alter)', '']

        tmp_r3c = ['tim', '', '', 'max', '', '', 'avg', '', '',
                   'df', '', 'df(h)', '', 'T(df)', '', '', '']
        tmp_r4c = ['t_Ds', 't_Df', 't_Dh', 'Ds', 'Df', 'Dh',
                   'Ds_avg', 'Df_avg', 'Dh_avg',
                   'df.prev', 'df.', 'df(h).prev', 'df(h).',
                   'T df.prev', 'T df.', 'T df(h).prev', 'T df(h).']
        for i in [1, 2]:
            csv_row_2c.extend([f'Training sa#{i}: DistDirect_bin'] + [
                ''] * 16 + ['DistDirect_nonbin (bi-val)'] + [''] * 16 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 16)
            csv_row_3c.extend(tmp_r3c * 3)
            csv_row_4c.extend(tmp_r4c * 3)
        tmp_r5c = ['Accuracy', 'Precision', 'Recall /Sensitivity',
                   'Specificity', 'f1_score', 'g_mean', 'DiscPower']
        csv_row_2c.extend(['Training performance: '] + [
            ''] * 13 + ['', 'DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR prime', 'DR'])
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])

        for i in [1, 2]:
            csv_row_2c.extend([f'sen-att #{i}'] + [''] * 8)
            csv_row_3c.extend([
                'DP', '', 'EO', '', 'PQP', '', 'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)

        for i in [1, 2]:
            csv_row_2c.extend([f'Test sa#{i}: DistDirect_bin'] + [
                ''] * 16 + ['DistDirect_nonbin (bi-val)'] + [''] * 16 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 16)
            csv_row_3c.extend(tmp_r3c * 3)
            csv_row_4c.extend(tmp_r4c * 3)
        csv_row_2c.extend(['Test performance:'] + [''] * 13 + ['', 'DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR prime', 'DR'])
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])
        for i in [1, 2]:
            csv_row_2c.extend([f'sen-att #{i}'] + [''] * 8)
            csv_row_3c.extend([
                'DP', '', 'EO', '', 'PQP', '', 'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)

        del tmp_r3c, tmp_r4c, tmp_r5c  # not 4+51*2+16+9*2 =20+60*2=140
        # del       # 4+(51*2+16+9*2) =4+136*2 =276
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class RevCompYC_NN(RevCompYB_NN):
    def subproc_df(self, t_Ds, t_Df, t_Dh, Ds, Df, Dh,
                   Ds_avg, Df_avg, Dh_avg):
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        if Ds_avg is None or Ds_avg == '':
            dff_nips_avg, ut_ddf_nips_avg = '', 0.
        else:
            dff_nips_avg, ut_ddf_nips_avg = fair_degree_v4(Ds_avg, Df_avg)
        dfh_ecai, ut_ddh_ecai = fair_degree_v3(Ds, Dh)
        dfh_nips, ut_ddh_nips = fair_degree_v4(Ds, Dh)
        if Ds_avg is None or Ds_avg == '':
            dfh_nips_avg, ut_dfh_nips_avg = '', 0.
        else:
            dfh_nips_avg, ut_dfh_nips_avg = fair_degree_v4(Ds_avg, Dh_avg)
        return [t_Ds, t_Df, t_Dh,
                Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg,
                dff_ecai, dff_nips, dff_nips_avg,
                dfh_ecai, dfh_nips, dfh_nips_avg,
                t_Ds + t_Df + ut_ddf_ecai,
                t_Ds + t_Df + ut_ddf_nips,
                '' if not Ds_avg else t_Ds + t_Df + ut_ddf_nips_avg,
                t_Ds + t_Dh + ut_ddh_ecai,
                t_Ds + t_Dh + ut_ddh_nips,
                '' if not Ds_avg else t_Ds + t_Dh + ut_dfh_nips_avg,
                ]  # 3+6+4*2+4 =21

    def subproc_bin(self, X_nA_y, X_nA_fx, embed_fx,
                    A_j, non_sa, m1, m2, n_e, pool):
        result = []
        (Ds, Ds_avg), t_Ds = DistDirect_bin(X_nA_y, non_sa)
        (Df, Df_avg), t_Df = DistDirect_bin(X_nA_fx, non_sa)
        (Dh, Dh_avg), t_Dh = DistDirect_bin(embed_fx, non_sa)
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg))

        # pm = {'idx_S0': ~non_sa, 'idx_S1': non_sa, 'm1': m1, 'm2': m2}
        pm = {'idx_S1': non_sa, 'm1': m1, 'm2': m2}  # pm.pop('idx_S0')  # 'non_sa'
        Ds, t_Ds = ApproxDist_bin(X_nA_y, A_j, **pm)
        Df, t_Df = ApproxDist_bin(X_nA_fx, A_j, **pm)
        Dh, t_Dh = ApproxDist_bin(embed_fx, A_j, **pm)
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, None, None, None))

        pm = {'idx_S1': non_sa, 'm1': m1, 'm2': m2}  # 'non_sa'
        # pm = {'non_sa': ~non_sa, 'm1': m1, 'm2': m2}
        Ds, t_Ds = ApproxDist_alter(X_nA_y, **pm)    # A_j,
        Df, t_Df = ApproxDist_alter(X_nA_fx, **pm)   # A_j,
        Dh, t_Dh = ApproxDist_alter(embed_fx, **pm)  # A_j,
        Ds, Df, Dh = Ds[0], Df[0], Dh[0]
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, None, None, None))

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, [non_sa, ~non_sa])
        (Dh, Dh_avg), t_Dh = DistDirect_nonbin(embed_fx, [non_sa, ~non_sa])
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg))

        del pm['idx_S1']  # del pm['non_sa']
        pm['n_e'] = n_e
        pm['pool'] = pool
        Aj_bin = non_sa.astype('int')
        (Ds, Ds_avg), t_Ds = DistApprox_nonbin(X_nA_y, Aj_bin, **pm)
        (Df, Df_avg), t_Df = DistApprox_nonbin(X_nA_fx, Aj_bin, **pm)
        (Dh, Dh_avg), t_Dh = DistApprox_nonbin(embed_fx, Aj_bin, **pm)
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg))
        return result  # (105,) =(84+21,) =(21*(4+1),)

    def subproc_nonbin(self, X_nA_y, X_nA_fx, embed_fx,
                       A_j, g1m, m1, m2, n_e, pool=None):
        non_sa, n_ai = g1m[0], len(g1m)
        result = self.subproc_bin(
            X_nA_y, X_nA_fx, embed_fx, A_j, non_sa, m1, m2, n_e, pool)

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, g1m)
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, g1m)
        (Dh, Dh_avg), t_Dh = DistDirect_nonbin(embed_fx, g1m)
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg))

        pm = {'m1': m1, 'm2': m2, 'n_e': n_e, 'pool': pool}
        (Ds, Ds_avg), t_Ds = DistApprox_nonbin(X_nA_y, A_j, **pm)
        (Df, Df_avg), t_Df = DistApprox_nonbin(X_nA_fx, A_j, **pm)
        (Dh, Dh_avg), t_Dh = DistApprox_nonbin(embed_fx, A_j, **pm)
        result.extend(self.subproc_df(
            t_Ds, t_Df, t_Dh, Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg))
        return result  # (147,) =(126+21,) =(21*(6+1),)

    def prepare_trial(self):  # 4+ 51|126 *2+16+9*2 =140|290
        # csv_row_1 = unique_column(10 + 366)  # 4+147*2+(16+18)*2 #332)
        csv_row_1 = unique_column(10 + 660)  # 147*2+16+18=?=328, 4+?*2
        csv_row_2c = ['NN training'] + [''] * 3
        csv_row_3c = ['tim', 'loss', '', 'acc']
        csv_row_4c = ['', '', '(alter)', '']  # [''] * 4

        tmp_r3c = ['tim', '', '', 'max', '', '', 'avg', '', '',
                   'df', '', '', 'df(h)', '', '',
                   'T(df)', '', '', '', '', '']  # =21
        tmp_r4c = ['t_Ds', 't_Df', 't_Dh', 'Ds', 'Df', 'Dh',
                   'Ds_avg', 'Df_avg', 'Dh_avg',
                   'df.prev', 'df.', 'df.avg',
                   'df(h).prev', 'df(h).', 'df(h).avg',
                   'T df.prev', 'T df.', 'T df.avg',
                   'T df(h).prev', 'T df(h).', 'D df(h).avg']  # =21
        tmp_r5c = ['Accuracy', 'Precision', 'Recall /Sensitivity',  # 7
                   'Specificity', 'f1_score', 'g_mean', 'DiscPower']

        for i in [1, 2]: 
            csv_row_2c.extend([f'Training sa#{i} DistDirect_bin'] + [
                ''] * 20 + ['ApproxDist_bin'] + [''] * 20 + [
                'ApproxDist_alter'] + [''] * 20 + [
                'DistDirect_nonbin (bi-val)'] + [''] * 20 + [
                'DistApprox_nonbin (bi-val)'] + [''] * 20 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 20 + [
                'DistApprox_nonbin (multi-val)'] + [''] * 20)
            csv_row_3c.extend(tmp_r3c * 7)  # 3)
            csv_row_4c.extend(tmp_r4c * 7)  # 3)
        csv_row_2c.extend(['Training performance'] + [''] * 14 + ['DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR_prime', 'DR'])  # 14+2=16
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])
        for i in range(2):  # n_a):
            csv_row_2c.extend([f'sen-att #{i+1}'] + [''] * 8)
            # csv_row_2c.extend([f'sen-att sa#{i+1}'] + [''] * 8)
            csv_row_3c.extend(['DP', '', 'EO', '', 'PQP', '',
                               'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)

        for i in [1, 2]:
            csv_row_2c.extend([f'Test sa#{i} DistDirect_bin'] + [
                ''] * 20 + ['ApproxDist_bin'] + [''] * 20 + [
                'ApproxDist_alter'] + [''] * 20 + [
                'DistDirect_nonbin (bi-val)'] + [''] * 20 + [
                'DistApprox_nonbin (bi-val)'] + [''] * 20 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 20 + [
                'DistApprox_nonbin (multi-val)'] + [''] * 20)
            # csv_row_2c.extend([f'Test sa#{i} DistDirect_bin'] + [
            #     ''] * 20 + ['DistDirect_nonbin (bi-val)'] + [''] * 20 + [
            #     'DistDirect_nonbin (multi-val)'] + [''] * 20)
            csv_row_3c.extend(tmp_r3c * 7)
            csv_row_4c.extend(tmp_r4c * 7)
        csv_row_2c.extend(['Test performance'] + [''] * 14 + ['DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR_prime', 'DR'])  # 14+2=16
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])
        for i in range(2):  # n_a):
            csv_row_2c.extend([f'sen-att #{i+1}'] + [''] * 8)
            csv_row_3c.extend(['DP', '', 'EO', '', 'PQP', '',
                               'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)

        del tmp_r3c, tmp_r4c, tmp_r5c  # 4+(21*7+16+9*2)*2 =4+97*2=366
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


# -------------------------------------
# Statistical disparity


# class RevCompX_setup(RevCompY_setup):
#   def __init__(self, training_blk=128, training=20,
#                abbr_cls='BaseNet', *, omitted=True):
#     super().__init__(training_blk, training, abbr_cls,
#                      omitted=omitted)
#     self._abbr_cls = abbr_cls  # 'NN'

class RevCompX_setup:
    # class RevCompX_setup(RevCompYC_NN):
    def neural_network(self, X_wA, y, training, training_blk):
        since = time.time()
        model = My_MLP(X_wA.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        data = torch.tensor(X_wA).to(torch.float32)
        # data = torch.tensor(X_wA, dtype=torch.float32)
        labels = torch.tensor(y)

        training_los = []
        chunk = X_wA.shape[0] // training_blk  # _item
        for i in range(training):
            for j in range(chunk):
                output = model(data[
                    training_blk * j: training_blk * (j + 1)])
                loss = criterion(output, labels[
                    training_blk * j: training_blk * (j + 1)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_los.append(loss.detach().numpy().tolist())
            output = model(data[training_blk * chunk:])
            loss = criterion(output, labels[training_blk * chunk:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_los.append(loss.detach().numpy().tolist())
        since = time.time() - since
        self._member = model
        self._member_losses = training_los

        output = model(data)
        loss = criterion(output, labels)
        comp = nn.functional.softmax(output, dim=1)
        # comp = nn.functional.softmax(output)
        output = output.detach().numpy()
        loss = loss.detach().numpy().tolist()
        y_hat = comp.argmax(dim=1)
        y_hat = y_hat.detach().numpy()
        comp = comp.detach().numpy()
        comp = np.log(comp[:, 1]) * y + np.log(comp[:, 0]) * (1 - y)
        comp = float(np.mean(-comp))  # approx. loss
        acc_score = float(np.mean(y_hat == y))
        return output, y_hat, loss, comp, acc_score, since

    def __init__(self, training_blk=128, training=20,
                 abbr_cls='BaseNet', *, omitted=True):
        self._training_blk = training_blk
        self._training = training
        self._omit = omitted
        self._abbr_cls = abbr_cls  # 'NN'

    def sklearn_regular(self, X_wA, y):
        since = time.time()
        clf = INDIVIDUALS[self._abbr_cls]
        clf.fit(X_wA, y)
        since = time.time() - since
        y_hat = clf.predict(X_wA)
        self._member = clf
        acc_score = np.mean(y_hat == y)
        return y_hat, float(acc_score), since

    def count_single_member(self, X, A, y, X_wA, X_wAq,
                            g1m_indices, m1, m2, n_e, pool=None):
        embedding, fx, loss, comp, acc, ut = self.neural_network(
            X_wA, y, self._training, self._training_blk)
        X_nA_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_fx = np.concatenate([
            fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        embed_fx = np.concatenate([
            fx.reshape(-1, 1).astype(DTY_FLT), embedding], axis=1)
        n_a = len(g1m_indices)
        '''
        fx, acc, ut = self.sklearn_regular(X_wA, y)
        X_nA_y = np.concatenate([
        y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_fx = np.concatenate([
        fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        '''

        positive_label = 1
        # self.count_sing_part3(y, fx, g1m_indices[1], positive_label)
        '''
        tmp = self.subproc_bin(X_nA_y, X_nA_fx, embed_fx, A[:, 1],
                           g1m_indices[1][0], m1, m2, n_e, pool)
        tmp = self.subproc_nonbin(X_nA_y, X_nA_fx, embed_fx, A[:, 1], 
                              g1m_indices[1], m1, m2, n_e, pool)
        '''
        return

    def count_sing_part3(self, y, y_hat, g1m, positive_label):
        cmp_fair = []  # res_tim, res_fair = [], []
        tmp_1, ut_1 = extGrp1_DP_sing(y, y_hat, g1m, positive_label)
        tmp_2, ut_2 = extGrp2_EO_sing(y, y_hat, g1m, positive_label)
        tmp_3, ut_3 = extGrp3_PQP_sing(y, y_hat, g1m, positive_label)
        # if np.isnan(tmp_2[1]):
        #   pdb.set_trace()
        # res_tim.extend([ut_1, ut_2, ut_3])
        cmp_fair.extend(tmp_1[:2] + tmp_2[:2] + tmp_3[:2])
        cmp_fair.extend(alterGrps_sing(tmp_1[-1], g1m)[0])
        cmp_fair.extend(alterGrps_sing(tmp_2[-1], g1m)[0])
        cmp_fair.extend(alterGrps_sing(tmp_3[-1], g1m)[0])
        del ut_1, ut_2, ut_3  # pdb.set_trace()
        return cmp_fair  # (12,) =(2*3+2*3,)

    # def subproc_nonbin(self, X_nA_y, X_nA_fx,
    #                    A_j, g1m, m1, m2, n_e, pool=None):
    #   pass


# class RevCompXB_NN(RevCompX_setup):  # , RevCompYB_NN):
class RevCompXB_NN(RevCompX_setup, RevCompYC_NN):
    def subproc_df(self, t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg):
        dff_ecai, ut_ddf_ecai = fair_degree_v3(Ds, Df)
        dff_nips, ut_ddf_nips = fair_degree_v4(Ds, Df)
        if Ds_avg is None or Ds_avg == '':
            dff_nips_avg, ut_ddf_nips_avg = '', 0.
        else:
            dff_nips_avg, ut_ddf_nips_avg = fair_degree_v4(Ds_avg, Df_avg)
        return [t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg,
                dff_ecai, dff_nips, dff_nips_avg,
                t_Ds + t_Df + ut_ddf_ecai,
                t_Ds + t_Df + ut_ddf_nips,
                '' if not Ds_avg else t_Ds + t_Df + ut_ddf_nips_avg,
                ]  # 2*3+3*2 =6+6=12

    def subproc_bin(self, X_nA_y, X_nA_fx, A_j, non_sa,
                    m1, m2, n_e, pool):
        result = []
        (Ds, Ds_avg), t_Ds = DistDirect_bin(X_nA_y, non_sa)
        (Df, Df_avg), t_Df = DistDirect_bin(X_nA_fx, non_sa)
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg))

        # pm = {'idx_S0': ~non_sa, 'idx_S1': non_sa, 'm1': m1, 'm2': m2}
        pm = {'idx_S1': non_sa, 'm1': m1, 'm2': m2}  # pm.pop('idx_S0')  # 'non_sa'
        Ds, t_Ds = ApproxDist_bin(X_nA_y, A_j, **pm)
        Df, t_Df = ApproxDist_bin(X_nA_fx, A_j, **pm)
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, None, None))
        # del pm['idx_S0']
        # del pm['idx_S1']
        pm['idx_S1'] = non_sa  # pm['non_sa'] = ~non_sa
        Ds, t_Ds = ApproxDist_alter(X_nA_y, **pm)   # A_j,
        Df, t_Df = ApproxDist_alter(X_nA_fx, **pm)  # A_j,
        Ds, Df = Ds[0], Df[0]
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, None, None))

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, [non_sa, ~non_sa])
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg))
        del pm['idx_S1']  # del pm['non_sa']
        pm['n_e'] = n_e
        pm['pool'] = pool
        Aj_bin = non_sa.astype('int')
        (Ds, Ds_avg), t_Ds = DistApprox_nonbin(X_nA_y, Aj_bin, **pm)
        (Df, Df_avg), t_Df = DistApprox_nonbin(X_nA_fx, Aj_bin, **pm)
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg))
        return result  # (60,) =(12*5,)

    def subproc_nonbin(self, X_nA_y, X_nA_fx, A_j, g1m,
                       m1, m2, n_e, pool):
        non_sa, n_ai = g1m[0], len(g1m)
        result = self.subproc_bin(
            X_nA_y, X_nA_fx, A_j, non_sa, m1, m2, n_e, pool)

        (Ds, Ds_avg), t_Ds = DistDirect_nonbin(X_nA_y, g1m)
        (Df, Df_avg), t_Df = DistDirect_nonbin(X_nA_fx, g1m)
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg))
        pm = {'m1': m1, 'm2': m2, 'n_e': n_e, 'pool': pool}
        (Ds, Ds_avg), t_Ds = DistApprox_nonbin(X_nA_y, A_j, **pm)
        (Df, Df_avg), t_Df = DistApprox_nonbin(X_nA_fx, A_j, **pm)
        result.extend(self.subproc_df(t_Ds, t_Df, Ds, Df, Ds_avg, Df_avg))
        return result  # (84,) =(12*7,)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 456)
        csv_row_2c = ['NN training'] + [''] * 3
        csv_row_3c = ['tim', 'loss', '', 'acc']
        csv_row_4c = ['', '', '(alter)', '']

        tmp_r3c = ['tim', '', 'max', '', 'avg', '',
                   'df', '', '', 'T(df)', '', '']     # 6+6=12
        tmp_r4c = ['t_Ds', 't_Df', 'Ds', 'Df', 'Ds_avg', 'Df_avg',
                   'df.prev', 'df.', 'df.avg',
                   'T df.prev', 'T df.', 'T df.avg']  # 6+6=12
        tmp_r5c = ['Accuracy', 'Precision', 'Recall /Sensitivity',
                   'Specificity', 'f1_score', 'g_mean', 'DiscPower']
        for i in range(1, 3):
            csv_row_2c.extend([f'Training sa#{i}: DistDirect_bin'] + [
                ''] * 11 + ['ApproxDist_bin'] + [''] * 11 + [
                'ApproxDist_alter'] + [''] * 11 + [
                'DistDirect_nonbin (bi-val)'] + [''] * 11 + [
                'DistApprox_nonbin (bi-val)'] + [''] * 11 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 11 + [
                'DistApprox_nonbin (multi-val)'] + [''] * 11)
            csv_row_3c.extend(tmp_r3c * 7)
            csv_row_4c.extend(tmp_r4c * 7)
        csv_row_2c.extend(['Training performance'] + [''] * 14 + ['DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR_prime', 'DR'])  # 7+7+2 =16
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])
        for i in range(2):
            csv_row_2c.extend([f'sen-att #{i+1}'] + [''] * 8)
            csv_row_3c.extend([
                'DP', '', 'EO', '', 'PQP', '', 'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)
            csv_row_2c.extend([f'sen-att #{i+1} ext.'] + [''] * 11)
            csv_row_3c.extend([
                'grp1 |SP', '', 'grp2 |SP', '', 'grp3 |SP', '',
                'alternative pairwise (grp1-3) |SP'] + [''] * 5)
            csv_row_4c.extend(['max', 'avg'] * 3 + ['max', 'avg'] * 3)

        for i in range(1, 3):
            csv_row_2c.extend([f'Test sa#{i}: DistDirect_bin'] + [
                ''] * 11 + ['ApproxDist_bin'] + [''] * 11 + [
                'ApproxDist_alter'] + [''] * 11 + [
                'DistDirect_nonbin (bi-val)'] + [''] * 11 + [
                'DistApprox_nonbin (bi-val)'] + [''] * 11 + [
                'DistDirect_nonbin (multi-val)'] + [''] * 11 + [
                'DistApprox_nonbin (multi-val)'] + [''] * 11)
            csv_row_3c.extend(tmp_r3c * 7)
            csv_row_4c.extend(tmp_r4c * 7)
        csv_row_2c.extend(['Test performance'] + [''] * 14 + ['DR'])
        csv_row_3c.extend(['Normal'] + [''] * 6 + ['Perturbing..'] + [
            ''] * 6 + ['DR_prime', 'DR'])  # 7+7+2 =16
        csv_row_4c.extend(tmp_r5c * 2 + ['hatL_loss', 'hatL_fair'])
        for i in range(2):
            csv_row_2c.extend([f'sen-att #{i+1}'] + [''] * 8)
            csv_row_3c.extend([
                'DP', '', 'EO', '', 'PQP', '', 'DP', 'EO', 'PQP'])
            csv_row_4c.extend(['g1', 'g0'] * 3 + ['abs'] * 3)
            csv_row_2c.extend([f'sen-att #{i+1} ext.'] + [''] * 11)
            csv_row_3c.extend([
                'grp1 |SP', '', 'grp2 |SP', '', 'grp3 |SP', '',
                'alternative pairwise (grp1-3) |SP'] + [''] * 5)
            csv_row_4c.extend(['max', 'avg'] * 3 + ['max', 'avg'] * 3)

        del tmp_r3c, tmp_r4c, tmp_r5c  # 4+(12*7*2+16+21*2)*2 !=288
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def curr_subset(self, X, A, y, X_wA, X_wAq, g1m,  # embed,
                    y_hat, y_qtb, m1, m2, n_e, pool, positive_label):
        X_nA_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_nA_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        # embed_y_hat = np.concatenate([
        #     y_hat.reshape(-1, 1).astype(DTY_FLT), embed], axis=1)
        n_a = len(g1m)
        curr_res = []
        for i in range(n_a):
            tmp = self.subproc_nonbin(
                X_nA_y, X_nA_y_hat,  # embed, A[:, i].copy()
                A[:, i], g1m[i], m1, m2, n_e, pool)
            curr_res.extend(tmp)
        if n_a == 1:
            curr_res.extend([''] * len(tmp))  # (84,)

        tmp_1 = self.count_sing_part1(y, y_hat, positive_label)
        tmp_2 = self.count_sing_part1(y, y_qtb, positive_label)
        tmp_3 = self.count_sing_part2p(y, y_hat, y_qtb)
        curr_res.extend(tmp_1 + tmp_2 + tmp_3)
        del tmp_1, tmp_2, tmp_3  # ?*2+ (7*2+2) =?*2+16
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, g1m[i][0], positive_label)
            curr_res.extend(tmp)
            tmp = self.count_sing_part3(y, y_hat, g1m[i], positive_label)
            curr_res.extend(tmp)
        if n_a == 1:
            curr_res.extend([''] * (9 + 12))
        del tmp                  # ?*2+16+ (9+12)*2 =?*2+58
        return curr_res  # 84*2+16+21*2 =226

    def schedule_content(
            self,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            m1, m2, n_e, pool=None, positive_label=1):
        embed, y_insp, loss, comp, acc, ut = self.neural_network(
            X_wA_trn, y_trn, self._training, self._training_blk)
        result = [ut, loss, comp, acc]

        output = torch.tensor(X_wAq_trn, dtype=torch.float32)
        output = self._member(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(output).argmax(dim=1)
        yq_insp = output.detach().numpy()
        # yq_insp = nn.functional.softmax(output).detach().numpy().argmax(axis=1)
        tmp = self.curr_subset(
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            # embed,
            y_insp, yq_insp, m1, m2, n_e, pool, positive_label)
        result.extend(tmp)

        embed = torch.tensor(X_wA_tst, dtype=torch.float32)
        embed = self._member(embed)
        output = nn.functional.softmax(embed, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(embed).argmax(dim=1)
        y_pred = output.detach().numpy()
        # y_pred = nn.functional.softmax(embed).detach().numpy().argmax(axis=1)
        embed = embed.detach().numpy()

        output = torch.tensor(X_wAq_tst, dtype=torch.float32)
        output = self._member(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        # output = nn.functional.softmax(output).argmax(dim=1)
        yq_pred = output.detach().numpy()
        # yq_pred = nn.functional.softmax(output).detach().numpy().argmax(axis=1)
        tmp = self.curr_subset(
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            # embed,
            y_pred, yq_pred, m1, m2, n_e,  # positive_label,
            pool, positive_label)  # pool=pool)
        result.extend(tmp)

        del ut, loss, comp, acc, embed, output
        del y_insp, yq_insp, yq_pred, y_pred, tmp
        return result  # (456,) =(4+226*2,)


class RevCompXD_NN(RevCompXB_NN):
    '''
    def curr_subset(self, X, A, y, X_wA, X_wAq, g1m,
                    y_hat, y_qtb, m1, m2, n_e, positive_label):
      X_nA_y = np.concatenate([
          y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
      X_nA_y_hat = np.concatenate([
          y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
      n_a = len(g1m)
      curr_res = []
      for i in range(n_a):
        tmp = self.subproc_nonbin(
            X_nA_y, X_nA_y_hat, )
      pdb.set_trace()
      return
    '''

    def schedule_content(
            self,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            # m1, m2, n_e, positive_label=1):
            m1, m2, n_e, pool=None, positive_label=1):
        '''
        _, y_insp, loss, comp, acc, ut = self.neural_network(
            X_wA_trn, y_trn, self._training, self._training_blk)
        result = [ut, loss, comp, acc]

        output = torch.tensor(X_wAq_trn, dtype=torch.float32)
        output = self._member(output)
        output = nn.functional.softmax(output, dim=1).argmax(dim=1)
        yq_insp = output.detach().numpy()
        tmp = self.curr_subset(
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            y_insp, yq_insp, m1, m2, n_e, positive_label)
        '''

        y_insp, acc, ut = self.sklearn_regular(X_wA_trn, y_trn)
        y_pred = self._member.predict(X_wA_tst)
        yq_insp = self._member.predict(X_wAq_trn)
        yq_pred = self._member.predict(X_wAq_tst)
        result = [ut, '', '', acc]

        tmp = self.curr_subset(
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            y_insp, yq_insp, m1, m2, n_e, pool, positive_label)
        result.extend(tmp)
        tmp = self.curr_subset(
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            y_pred, yq_pred, m1, m2, n_e, pool, positive_label)
        result.extend(tmp)
        return result


'''
class RevCompXE_learner(RevCompXB_NN):
  def schedule_content(
          self,
          X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
          X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
          m1, m2, n_e, pool=None, positive_label=1):
    self._abbr_cls_set = []
    return

  def schedule_content_sub(self):
    pass


class RevCompXF_ensemble(RevCompXB_NN):
  def schedule_content(
          self,
          X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
          X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
          m1, m2, n_e, pool=None, positive_label=1):
    self._abbr_cls_set = []
    return

  def schedule_content_sub(self):
    pass
'''


# class RevCompXE_ensemble(RevCompXD_NN):
class RevCompXE_ensemble(RevCompXB_NN):
    learners_inside = [
        # 'bagging', 'adaboost', 'lightgbm','fairgbm', 'adafair']
        'bagging', 'AdaBoost', 'LightGBM',  # lightGBM
        'FairGBM |FPR', 'FairGBM |FNR', 'FairGBM |FPR,FNR',
        'AdaFair']

    def schedule_content(
            self,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            m1, m2, n_e, pool=None, positive_label=1,
            nb_cls=7, saIndex=list(), saValue=list()):
        # self._abbr_cls_set = []
        total_result = []
        pms = {'m1': m1, 'm2': m2, 'n_e': n_e, 'pool': pool,
               'positive_label': positive_label}
        n_a = len(g1m_trn)  # A_trn.shape[1]
        for name_ens in self.learners_inside[:3]:
            clf, ut = self.learner_part_sub(X_wA_trn, y_trn, name_ens, nb_cls)
            total_result.append(self.learner_part_ans(
                clf, ut,
                X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst, pms))
        for i in range(n_a):
            for name_ens in self.learners_inside[3:6]:
                clf, ut = self.learner_part_sub(X_wA_trn, y_trn, name_ens, nb_cls,
                                                non_sa_trn=g1m_trn[i][0])
                total_result.append(self.learner_part_ans(
                    clf, ut,
                    X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                    X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst, pms))
            for name_ens in self.learners_inside[6:7]:
                clf, ut = self.learner_part_sub(X_wA_trn, y_trn, name_ens, nb_cls,
                                                sa_idx=saIndex[i], sa_val=saValue[i])
                total_result.append(self.learner_part_ans(
                    clf, ut,
                    X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                    X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst, pms))
        return total_result  # (3+(3+1)*n_a, 456)

    def learner_part_ans(self, clf, ut,
                         X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                         X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
                         pms):
        y_insp = clf.predict(X_wA_trn)
        y_pred = clf.predict(X_wA_tst)
        yq_insp = clf.predict(X_wAq_trn)
        yq_pred = clf.predict(X_wAq_tst)
        curr_res = [ut, '', '', float(np.mean(y_trn == y_insp))]
        tmp = self.curr_subset(
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            y_insp, yq_insp, **pms)
        curr_res.extend(tmp)
        tmp = self.curr_subset(
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            y_pred, yq_pred, **pms)
        curr_res.extend(tmp)
        del y_insp, y_pred, yq_insp, yq_pred, tmp
        return curr_res

    # def schedule_content_sub(self):
    #   pass
    def learner_part_sub(self, X_wA_trn, y_trn, name_ens,
                         nb_cls=7, non_sa_trn=None,
                         sa_idx=None, sa_val=None):
        since = time.time()
        if name_ens == 'bagging':
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)
        elif name_ens in ['AdaBoost', 'adaboost']:
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)
        elif name_ens in ['lightgbm', 'LightGBM', 'lightGBM']:
            clf = LGBMClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)

        elif name_ens[:7] in ['fairgbm', 'FairGBM', 'fairGBM']:
            clf = FairGBMClassifier(
                n_estimators=nb_cls,
                constraint_type=name_ens.split('|')[-1])
            clf.fit(X_wA_trn, y_trn, constraint_group=~non_sa_trn)
        elif name_ens in ['adafair', 'AdaFair']:
            clf = AdaFair(n_estimators=nb_cls, saIndex=sa_idx, saValue=sa_val)
            clf.fit(X_wA_trn, y_trn)
        since = time.time() - since
        return clf, since


# class RevCompXF_learner(RevCompXD_NN):
class RevCompXF_learner(RevCompXB_NN):
    learners_inside = [
        'DT', 'NB', 'SVM', 'linSVM', 'LR1',
        'LR2', 'LM1', 'LM2', 'kNNu', 'kNNd',
        'MLP']  # learner_inside

    def schedule_content(
            self,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            m1, m2, n_e, pool=None, positive_label=1):
        # self._abbr_cls_set = []
        total_result = []
        for abbr_cls in self.learners_inside:
            clf, ut = self.learner_part_sub(X_wA_trn, y_trn, abbr_cls)
            y_insp = clf.predict(X_wA_trn)
            y_pred = clf.predict(X_wA_tst)
            yq_insp = clf.predict(X_wAq_trn)
            yq_pred = clf.predict(X_wAq_tst)

            curr_res = [ut, '', '', float(np.mean(y_trn == y_insp))]
            tmp = self.curr_subset(
                X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                y_insp, yq_insp, m1, m2, n_e, pool, positive_label)
            curr_res.extend(tmp)
            tmp = self.curr_subset(
                X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
                y_pred, yq_pred, m1, m2, n_e, pool, positive_label)
            curr_res.extend(tmp)

            total_result.append(curr_res)
            del y_insp, y_pred, yq_insp, yq_pred, clf
            del ut, tmp, curr_res
        return total_result

    # def schedule_content_sub(self):
    #   pass
    # def learner_part_sub(self, X_trn, y_trn, abbr_cls):
    # def learner_part(self, abbr_cls, X_wA_trn, y_trn, X_wA_tst, y_tst):
    def learner_part_sub(self, X_wA_trn, y_trn, abbr_cls):
        since = time.time()
        clf = INDIVIDUALS[abbr_cls]
        clf.fit(X_wA_trn, y_trn)
        since = time.time() - since
        # y_insp = clf.predict(X_wA_trn)
        # y_pred = clf.predict(X_wA_tst)
        return clf, since


# -------------------------------------
# -------------------------------------
# Convergence
# refer to: mext_exp_mp.py


class ConvergeE_setup:
    def __init__(self):  # , *, omitted=True):
        self._metric_part1 = [
            'Accuracy', 'Precision', 'Recall/sensitivity',
            'specificity', 'f1_score', 'g_mean', 'dp', ]
        self._metric_part2 = ['Grp1', 'Grp2', 'Grp3',  # 'DR',
                              'hat_L(fair)', 'hat_L(loss)']
        # self._metric_part4 = []
        # self._metric_part3 = []

        # self._hfm_bin = [
        #     'T(Naive)', '', 'T(EarlyStop)', '',
        #     'T(Direct)', '', 'T(Approx)', '', '',
        #     'Naive', '', 'EarlyStop', '', 'Direct_bin', '',
        #     'Approx_bin', '', 'Approx_nonbin',
        #     'Direct_bin.avg', '', 'Approx_nonbin.avg']
        # self._hfm_nonbin = [
        #     'T(Naive)', '', '', 'T(EarlyStop)', '', '',
        #     'T(Direct)', '', '', 'T(Approx_bin)',
        #     'T(Approx_nonbin)', '', 'T(StratES)', '', ]
        # self._hfm_nonbin = [[
        #     'Naive', '', '', 'EarlyStop', '', '',
        #     'Direct_bin', 'Direct_nonbin', '', 'Approx_bin',
        #     'Approx_nonbin', '', 'Converge_nonbin', ''], [
        #     'Direct_bin', 'Direct_nonbin', '', 
        #     'Approx_nonbin', '', 'Converge_nonbin', '']]
        # self._hfm_bin = [[
        #     'Naive', '', 'EarlyStop', '', 'Direct', '',
        #     'Approx_bin', '', 'Approx_nonbin'], [
        #     'Direct_bin', 'Direct_nonbin', '']]
        self._hfm_bin = [['bin', '', '', '',
                          'nonbin(bin-val)', '', '', '', '',
                          'nonbin(multival)', ''], [
            'bin', 'nonbin(bin-val)', '', '', 'nonbin(multival)', ''], [
            'Naive', 'EarlyStop', 'Direct', 'Approx', 'Converge']]  # 'StratES'
        self._hfm_nonbin = [[
            'bin', '', '', '', 'nonbin(bin-val)', '', '', '', '',
            'nonbin(multival)', '', '', '', ''], [
            'bin', 'nonbin(bin-val)', '', '', 'nonbin(multival)', '', ''], [
            'Naive', 'EarlyStop', 'Direct', 'Approx', 'Converge']]  # 'StratES'

    def count_sing_part1(self, y, y_hat, pos_label=1):  # NB. ndarray
        tp, fp, fn, tn = contingency_tab(y, y_hat, pos_label)
        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        sen = calc_recall(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, calc_f1_score(tp, fp, fn, tn)])
        res_indi.append(imba_geometric_mean(sen, spe))
        res_indi.append(imba_discriminant_power(sen, spe))
        return res_indi  # shape=(7,)

    def count_sing_part2(self, y, y_hat, y_qtb, non_sa, pos_label):
        _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
            y, y_hat, pos_label, non_sa)
        cmp_fair = []
        tmp_1 = unpriv_group_one(g1_Cm, g0_Cm)
        tmp_2 = unpriv_group_two(g1_Cm, g0_Cm)
        tmp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
        cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)

        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[0]))
        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair  # shape=(11,) =(6+3+2,)

    def subproc_bin(self, X_nA_y, A_j, non_sa, m1, m2, n_e):
        Aj_bin = non_sa.astype('int')
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)  # /Aj_bin
        # app_1 = ApproxDist_alter(X_nA_y, non_sa, m1, m2)
        # app_2 = DistApprox_nonbin(X_nA_y, Aj_bin, m1, m2, n_e)
        app_2 = StratVacant(X_nA_y, Aj_bin, m1, m2, n_e)
        app_3 = StratVacant(X_nA_y, A_j, m1, m2, n_e)
        app_4 = StratEarlyStop(X_nA_y, Aj_bin, n_e)
        app_5 = StratEarlyStop(X_nA_y, A_j, n_e)

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])

        ans_tim = [hdd_1[1], eff_1[1], luo_1[1], app_0[1],
                   hdd_2[1], eff_2[1], luo_2[1], app_2[1], app_4[1],
                   app_3[1], app_5[1], ]
        ans_max = [hdd_1[0], eff_1[0], luo_1[0][0], app_0[0],  # app_1[0][0], 
                   hdd_2[0], eff_2[0], luo_2[0][0], app_2[0][0],
                   app_4[0][0], app_3[0][0], app_5[0][0], ]
        ans_avg = [luo_1[0][1], luo_2[0][1],  # app_1[0][1],
                   app_2[0][1], app_4[0][1], app_3[0][1], app_5[0][1], ]
        return ans_tim + ans_max + ans_avg  # 11*2+6=28 # (22,)=(9+9+4,)

    def subproc_nonbin(self, X_nA_y, A_j, g1m, m1, m2, n_e):  # ,pool):
        non_sa = g1m[0]
        Aj_bin = non_sa.astype('int')
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        luo_3 = DistDirect_nonbin(X_nA_y, g1m)

        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)  # HFM /Aj_bin
        app_2 = StratVacant(X_nA_y, Aj_bin, m1, m2, n_e)
        app_3 = StratVacant(X_nA_y, A_j, m1, m2, n_e)        # HFM ext
        app_4 = StratEarlyStop(X_nA_y, Aj_bin, n_e)
        app_5 = StratEarlyStop(X_nA_y, A_j, n_e)             # HFM cvg

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        hdd_3 = NaiveHD_nonbin(X_nA_y, g1m)
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_3 = EffHDD_nonbin(X_nA_y, g1m)

        ans_tim = [hdd_1[1], eff_1[1], luo_1[1], app_0[1],
                   hdd_2[1], eff_2[1], luo_2[1], app_2[1], app_4[1], 
                   hdd_3[1], eff_3[1], luo_3[1], app_3[1], app_5[1]]
        ans_max = [hdd_1[0], eff_1[0], luo_1[0][0], app_0[0],
                   hdd_2[0], eff_2[0], luo_2[0][0], app_2[0][0], app_4[0][0], 
                   hdd_3[0], eff_3[0], luo_3[0][0], app_3[0][0], app_5[0][0]]
        ans_avg = [luo_1[0][1], 
                   luo_2[0][1], app_2[0][1], app_4[0][1],
                   luo_3[0][1], app_3[0][1], app_5[0][1]]
        return ans_tim + ans_max + ans_avg  # (35,)=(14+14+7,)

    def subproc_multivar(self, X_nA_y, A, g1m_ind, m1, m2, n_e, pool):
        luo_4 = DistDirect_multivar(X_nA_y, g1m_ind)
        # app_4 = DistExtend_multivar(X_nA_y, A, m1, m2, n_e)
        app_4 = EffExact(X_nA_y, A, StratVacant, m1, m2, n_e, pool)
        app_5 = EffExact(X_nA_y, A, StratEarlyStop, m1, m2, n_e, pool)
        luo_mid, app_mid, cvg_mid = luo_4[0][2], app_4[0][2], app_5[0][2]
        luo_4 = (luo_4[0][: 2], luo_4[1])
        app_4 = (app_4[0][: 2], app_4[1])
        app_5 = (app_5[0][: 2], app_5[1])

        hdd_4 = NaiveHD_multivar(X_nA_y, g1m_ind)
        eff_4 = EffHDD_multivar(X_nA_y, g1m_ind)
        hdd_mid, eff_mid = hdd_4[0][1], eff_4[0][1]
        hdd_4 = (hdd_4[0][0], hdd_4[1])
        eff_4 = (eff_4[0][0], eff_4[1])

        ans_tim = [hdd_4[1], eff_4[1], luo_4[1], app_4[1], app_5[1]]
        ans_max = [hdd_4[0], eff_4[0], luo_4[0][0], app_4[0][0], app_5[0][0]]
        ans_avg = [luo_4[0][1], app_4[0][1], app_5[0][1]]
        result = ans_tim + ans_max + ans_avg  # (13,)=(5+5+3,)
        n_a = len(g1m_ind)
        for i in range(n_a):
            tmp_tim = [hdd_mid[1][i], eff_mid[1][i],
                       luo_mid[2][i], app_mid[2][i], cvg_mid[2][i]]
            tmp_max = [hdd_mid[0][i], eff_mid[0][i],
                       luo_mid[0][i], app_mid[0][i], cvg_mid[0][i]]
            tmp_avg = [luo_mid[1][i], app_mid[1][i], cvg_mid[1][i]]
            result.extend(tmp_tim + tmp_max + tmp_avg)  # 5+5+3=13
        if n_a == 1:
            result.extend([''] * 13)
        # pdb.set_trace()
        return result  # (39,)=(13+13*2,)

    # def count_sing_part3(self, X_y, X_y_hat, non_sa, A_j,
    #                      m1=20, m2=8, n_e=2, pool=None):
    #     cmp_fair = []
    #     ut_a = time.time()
    #     (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_y, non_sa)

    def count_single_member(self, X, A, y, y_hat, y_qtb,
                            g1m_indices, m1, m2, n_e,
                            pos_label, jt, pool=None):
        ut_c = time.time()
        res_indi = []
        ta_1 = self.count_sing_part1(y, y_hat, pos_label)
        ta_2 = self.count_sing_part1(y, y_qtb, pos_label)
        ta_3 = [abs(t1 - t2) for t1, t2 in zip(ta_1, ta_2)]
        res_indi.extend(ta_1)
        res_indi.extend(ta_2)
        res_indi.extend(ta_3)
        del ta_1, ta_2, ta_3  # res_indi: 7*3=21

        far_2, far_3 = [], []  # ,far_4=[]
        n_a = len(g1m_indices)
        X_and_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_and_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, y_qtb, g1m_indices[i][0], pos_label)
            far_2.extend(tmp)
        if n_a == 1:
            far_2.extend([''] * 11 * 3)
        else:
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[0], pos_label))  # 'and'&
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[1], pos_label))  # 'or' |
        ut_c = time.time() - ut_c

        A_i = [A[:, i].copy() for i in range(n_a)]
        if n_a > 1:
            A_0 = np.logical_and(A_i[0], A_i[1]).astype(DTY_FLT)
            A_1 = np.logical_or(A_i[0], A_i[1]).astype(DTY_FLT)
            # pdb.set_trace()  # SOMETHING WRONG??
        else:
            A_0, A_1 = None, None
        for i in range(n_a):
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, g1m_indices[i][0], A_i[i],
                m1, m2, n_e))
        if n_a == 1:
            far_3.extend([''] * 39 * 3)  # 35*3)
        else:
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[0], A_0, m1, m2, n_e))
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[1], A_1, m1, m2, n_e))

        del A_0, A_1, A_i
        far_4 = self.count_sing_part4(
            X_and_y, X_and_y_hat, g1m_indices, A, m1, m2, n_e, pool)
        far_4.append(ut_c)
        # return res_indi + far_2 + far_3 + far_4  # 21+44+140+81=286
        return res_indi + far_2 + far_3 + far_4  # 21+44+156+84+1=306

    def count_sing_part3(self, X_y, X_y_hat, non_sa, A_j,
                         m1=20, m2=8, n_e=2):  # , pool=None):
        cmp_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_y, non_sa)
        (Df_01, Df_avg), t_Df = DistDirect_bin(X_y_hat, non_sa)
        df_prev, ut_df_prev = fair_degree_v3(Ds_01, Df_01)
        df_max, ut_df_max = fair_degree_v4(Ds_01, Df_01)
        df_avg, ut_df_avg = fair_degree_v4(Ds_avg, Df_avg)
        ut_a = time.time() - ut_a
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df_max, df_avg, t_Ds, t_Df,
                         ut_df_prev, ut_df_max, ut_df_avg])
        ut_b = time.time()
        Ds_01, t_Ds = ApproxDist_bin(X_y, A_j, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist_bin(X_y_hat, A_j, non_sa, m1, m2)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        ut_b = time.time() - ut_b
        cmp_fair.extend([Ds_01, Df_01, df_prev, t_Ds, t_Df])

        B_j = non_sa.astype(DTY_INT)
        ut_c = time.time()
        (Ds_01, Ds_avg), t_Ds = StratVacant(X_y, B_j, m1, m2, n_e)
        (Df_01, Df_avg), t_Df = StratVacant(X_y_hat, B_j, m1, m2, n_e)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df_max, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_c = time.time() - ut_c
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df_max, df_avg, t_Ds, t_Df])
        ut_d = time.time()
        (Ds_01, Ds_avg), t_Ds = StratEarlyStop(X_y, B_j, n_e)
        (Df_01, Df_avg), t_Df = StratEarlyStop(X_y_hat, B_j, n_e)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df_max, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_d = time.time() - ut_d
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df_max, df_avg, t_Ds, t_Df])

        cmp_fair.extend([ut_a, ut_b, ut_c, ut_d])
        del Ds_01, Ds_avg, Df_01, Df_avg, df_prev, df_max, df_avg
        del t_Ds, t_Df, ut_df_prev, ut_df_max, ut_df_avg
        return cmp_fair  # shape=(39,) =(12+5+9+9+4,)

    def count_sing_part4(self, X_y, X_y_hat, g1m_indices, A,
                         m1=20, m2=8, n_e=2, pool=None):
        cmp_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(
            X_y, g1m_indices)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistDirect_multivar(
            X_y_hat, g1m_indices)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df_max, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df_max, df_avg, t_Ds, t_Df])
        n_a = len(g1m_indices)

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_a = time.time() - ut_a
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        ut_b = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = EffExact(
            X_y, A, StratVacant, m1, m2, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = EffExact(
            X_y_hat, A, StratVacant, m1, m2, n_e, pool)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_b = time.time() - ut_b
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        ut_d = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = EffExact(
            X_y, A, StratEarlyStop, None, None, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = EffExact(
            X_y_hat, A, StratEarlyStop, None, None, n_e, pool)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_d = time.time() - ut_d
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        cmp_fair.extend([ut_a, ut_b, ut_d])
        del Ds_01, Ds_avg, Df_01, Df_avg, df_prev, df, df_avg
        del t_Ds, t_Df, Ds_midtmp, Df_midtmp, ut_a, ut_b, ut_d
        return cmp_fair  # .shape=(81+3,) =(9*3*3+3,)

    def count_scores(self,
                     X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn,
                     X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst,
                     m1, m2, n_e, pos_label, pool,
                     jt_trn=None, jt_tst=None):
        ans_trn = self.count_single_member(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn,
            m1, m2, n_e, pos_label, jt_trn, pool)
        ans_tst = self.count_single_member(
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst,
            m1, m2, n_e, pos_label, jt_tst, pool)
        return ans_trn + ans_tst

    def prepare_trial(self, omitted=True):
        return [], [], [], []


class ConvergeE3_with(ConvergeE_setup):
    # refer to `ZB_efficient`
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3):  # , *, omitted=True):
        super().__init__()  # omitted=omitted)
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        self._n_e = n_e

    def schedule_content(self, X, A, y_fx,           # X_wA,
                         g1m_indices, m1, m2, n_e):  # , pool):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        result, n_a = [], len(g1m_indices)
        for i in range(n_a):
            tmp = self.subproc_bin(
                X_nA_y, A[:, i], g1m_indices[i][0], m1, m2, n_e)
            result.extend(tmp)
        if n_a == 1:
            result.extend([''] * 28)  # 21)
        # pdb.set_trace()
        del n_a, X_nA_y
        return result

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 28 * 2)
        csv_row_2c = ['sa#1 tim'] + [''] * 10 + ['max'] + [
            ''] * 10 + ['avg'] + [''] * 5 + ['sa#2 tim'] + [
            ''] * 10 + ['max'] + [''] * 10 + ['avg'] + [''] * 5
        csv_row_4c = self._hfm_bin[0] * 2 + self._hfm_bin[1]
        csv_row_5c = (['Naive', 'EarlyStop', 'Direct', 'Approx'] * 2 + [
            'StratES', 'Approx', 'StratES']) * 2 + [
            'Direct', 'Direct', 'Approx', 'StratES', 'Approx', 'StratES']
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


class ConvergeE4_with(ConvergeE3_with):
    def schedule_content(self, X, A, y_fx, g1m_indices,
                         m1, m2, n_e):  # , pool):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        result, n_a = [], len(g1m_indices)
        for i in range(n_a):
            tmp = self.subproc_nonbin(
                X_nA_y, A[:, i], g1m_indices[i], m1, m2, n_e)
            result.extend(tmp)
        if n_a == 1:
            result.extend([''] * 35)
        # pdb.set_trace()
        del n_a, X_nA_y
        return result

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 35 * 2)
        csv_row_2c = ['sa#1 tim'] + [''] * 13 + ['max'] + [
            ''] * 13 + ['avg'] + [''] * 6 + ['sa#2 tim'] + [
            ''] * 13 + ['max'] + [''] * 13 + ['avg'] + [''] * 6
        csv_row_4c = self._hfm_nonbin[0] * 2 + self._hfm_bin[1]
        csv_row_5c = ([
            'Naive', 'EarlyStop', 'Direct', 'Approx'] * 2 + [
            'StratES', 'Naive', 'EarlyStop', 'Direct', 'Approx',
            'StratES']) * 2 + ['Direct', 'Direct', 'Approx', 'StratES',
                               'Direct', 'Approx', 'StratES']  # 14*2+7
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


class ConvergeE5_with(ConvergeE3_with):
    def schedule_content(self, X, A, y_fx, g1m_indices,
                         m1, m2, n_e, pool=None):
        X_nA_y = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        return self.subproc_multivar(
            X_nA_y, A, g1m_indices, m1, m2, n_e, pool)

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 39)
        csv_row_2c = ['multivar'] + [''] * 12 + [
            'sen-att #1'] + [''] * 12 + ['sen-att #2'] + [''] * 12
        csv_row_3c = ['tim'] + [''] * 4 + ['max'] + [''] * 4 + ['avg', '', '']
        csv_row_4c = ['Naive', 'EarlyStop', 'Direct', 'Extend', 'EfficA',
                      'Naive', 'EarlyStop', 'Direct', 'Approx', 'StratES',
                      'Direct', 'Approx', 'StratES']
        return csv_row_1, csv_row_2c, csv_row_3c * 3, csv_row_4c * 3


class ConvergeE2_with(ConvergeE_setup):
    learners_inside = ['bagging', 'AdaBoost', 'LightGBM',
                       'FairGBM |FPR', 'FairGBM |FNR',
                       'FairGBM |FPR,FNR', 'AdaFair']

    def __init__(self, nb_cls=7, saIndex=list(), saValue=list(),
                 n_e=3):
        super().__init__()
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        self._n_e = n_e

    def schedule_content(
            self,  # logger, pool,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1ms_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1ms_tst,
            m1=20, m2=8, n_e=2, pos_label=1, pool=None,
            jt_trn=None, jt_tst=None):  # , *, omitted=True):
        res_iter = []
        tmp = self.subroute_one_norm_att(
            X_wA_trn, y_trn, X_wAq_trn, g1ms_trn,
            X_wA_tst, y_tst, X_wAq_tst, g1ms_tst,
            pos_label, m1, m2, pool,
            X_trn, A_trn, jt_trn, X_tst, A_tst, jt_tst)
        res_iter.extend(tmp)  # (3,1+306*2)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_wA_trn, y_trn, X_wAq_trn, g1ms_trn[0][0],
                X_wA_tst, y_tst, X_wAq_tst, g1ms_tst[0][0],
                self.saIndex[0], self.saValue[0], pos_label,
                m1, m2, pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst)
            res_iter.extend(tmp)
            return res_iter

        sa_len = len(g1ms_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_wA_trn, y_trn, X_wAq_trn, g1ms_trn[i][0],
                X_wA_tst, y_tst, X_wAq_tst, g1ms_tst[i][0],
                self.saIndex[i], self.saValue[i], pos_label,
                m1, m2, pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst)
            res_iter.extend(tmp)
        # pdb.set_trace()
        return res_iter  # shape=(3+4*i, 1+306*2)

    def subroute_one_sens_att(
            self,
            X_wA_trn, y_trn, X_wAq_trn, nsa_trn,
            X_wA_tst, y_tst, X_wAq_tst, nsa_tst,
            sa_idx, sa_val, pos_label=1, m1=20, m2=8, pool=None,
            X_trn=None, A_trn=None, g1m_trn=None, jt_trn=None,
            X_tst=None, A_tst=None, g1m_tst=None, jt_tst=None):
        res_att = []
        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            (y_insp, y_pred, yq_insp, yq_pred,
             ut) = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_wA_trn, y_trn, X_wAq_trn, nsa_trn,
                X_wA_tst, y_tst, X_wAq_tst, nsa_tst,
                constraint=constraint_type)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst,
                m1, m2, self._n_e, pos_label, pool, jt_trn, jt_tst)
            res_att.append([ut] + tmp)

        (y_insp, y_pred, yq_insp, yq_pred,
         ut) = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_wA_trn, y_trn, X_wAq_trn, nsa_trn,
            X_wA_tst, y_tst, X_wAq_tst, nsa_tst,
            sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst,
            m1, m2, self._n_e, pos_label, pool, jt_trn, jt_tst)
        res_att.append([ut] + tmp)

        return res_att  # shape=(4,1+306*@)  # g1ms_trn/tst

    def subroute_one_norm_att(self,
                              X_wA_trn, y_trn, X_wAq_trn, g1m_trn,
                              X_wA_tst, y_tst, X_wAq_tst, g1m_tst,
                              pos_label=1, m1=20, m2=8, pool=None,
                              X_trn=None, A_trn=None, jt_trn=None,
                              X_tst=None, A_tst=None, jt_tst=None
                              ):  # *, omitted=True):
        res_att = []

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_wA_trn, y_trn, X_wAq_trn, None,
            X_wA_tst, y_tst, X_wAq_tst, None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst,
            m1, m2, self._n_e, pos_label, pool, jt_trn, jt_tst)
        res_att.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_wA_trn, y_trn, X_wAq_trn, None,  # nsa_trn,
            X_wA_tst, y_tst, X_wAq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst,
            m1, m2, self._n_e, pos_label, pool, jt_trn, jt_tst)
        res_att.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_wA_trn, y_trn, X_wAq_trn, None,  # nsa_trn,
            X_wA_tst, y_tst, X_wAq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst,
            m1, m2, self._n_e, pos_label, pool, jt_trn, jt_tst)
        res_att.append([ut] + tmp)

        return res_att

    def subroute_one_fair_ens(self, name_ens, nb_cls,
                              X_wA_trn, y_trn, X_wAq_trn, nsa_trn,
                              X_wA_tst, y_tst, X_wAq_tst, nsa_tst,
                              constraint='FPR,FNR',
                              sa_idx=None, sa_val=None):
        ut = time.time()

        if name_ens == 'bagging':
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)
        elif name_ens == 'adaboost':
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)
        elif name_ens == 'lightgbm':
            # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
            clf = LGBMClassifier(n_estimators=nb_cls)
            clf.fit(X_wA_trn, y_trn)

        elif name_ens == 'fairgbm':
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint)
            clf.fit(X_wA_trn, y_trn, constraint_group=~nsa_trn)
        elif name_ens == 'adafair':
            clf = AdaFair(n_estimators=nb_cls,
                          saIndex=sa_idx, saValue=sa_val)
            clf.fit(X_wA_trn, y_trn)

        ut = time.time() - ut
        y_insp = clf.predict(X_wA_trn)
        y_pred = clf.predict(X_wA_tst)
        yq_insp = clf.predict(X_wAq_trn)
        yq_pred = clf.predict(X_wAq_tst)
        return y_insp, y_pred, yq_insp, yq_pred, ut

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 1 + 306 * 2)
        # csv_row_2c = ['', 'Ensem'] + ['Training set'] + [
        #     ''] * 305 + ['Test set'] + [''] * 305

        sub_pt1 = self._metric_part1 * 3
        sub_pt2 = (['g1', 'g0'] * 3 + self._metric_part2) * 4
        sub_pt3_a = ['Ds', 'Ds_avg', 'Df', 'Df_avg',
                     'df_prev', 'df', 'df_avg', 't_Ds', 't_Df',
                     'T(df_prev)', 'T(df)', 'T(df_avg)']
        sub_pt3_b = ['Ds', 'Df', 'df_prev', 't_Ds', 't_Df']
        sub_pt3_c = sub_pt3_a[: 9] * 2 + [
            # 'T(DistDirect_bin)', 'T(ApproxDist_bin)',
            # 'T(DistApprox_nonbin /StratVacant)', 'T(StratEarlyStop)']
            'T(Direct_bin)', 'T(ApproxDist_bin)',
            'T(Approx_nonbin /StratVacant)', 'T(StratES)']
        sub_pt3 = sub_pt3_a + sub_pt3_b + sub_pt3_c  # 12+5+22=39
        sub_pt4 = sub_pt3_a[: 9] * (1 + 2 + 1 + 2 + 1 + 2) + [
            # 'T(DistDirect_multivar)', 'T(EffExact-StratVacant)',
            # 'EffExact-StratES', 'T(comp.performance)']  # 9*9+3 +1=84+1
            'T(Direct_multivar)', 'T(Extend_multivar_mp /EffExact-StratVacant)',
            'T(EffExact-StratES)', 'T(comp.performance)']
        # self._metric_part2[:3] * 3 + self._metric_part2[-2:]
        # in total, 7*3+11*4+39*4+(84+1) = 305+1=306    # computing
        csv_row_4c = ['classifier', 'T(learning)'] + (
            sub_pt1 + sub_pt2 + sub_pt3 * 4 + sub_pt4) * 2
        del sub_pt3_a, sub_pt3_b, sub_pt3_c
        del sub_pt1, sub_pt2, sub_pt3, sub_pt4

        # sub_r3c_ab = ['Acc', '', '', '', 'f1', 'g_mean', 'dp'] * 3 + ([
        sub_r3c_ab = (['Normal'] + [''] * 6 + ['Adversarial'] + [''] * 6 + [
            r'$\Delta$(performance)'] + [''] * 6) * 1 + ([
                'Grp.intermediate'] + [''] * 5 + [
                'GrpFair: DP,EOpp,PP', '', '', 'DR', '']) * 4  # 21+44=65
        sub_r3c_c = ['Direct_bin'] + [''] * 11 + ['ApproxDist_bin'] + [
            ''] * 4 + ['Approx_nonbin /StratVacant'] + [''] * 8 + [
            'StratEarlyStop'] + [''] * 8  # 12+5+9+9=35, 35+4=39
        sub_r3c_c = (sub_r3c_c + [
            'T(Direct_bin)', 'T(ApproxDist_bin)',
            'T(Approx_nonbin)', 'T(StratES) converged']) * 4  # +39*4
        sub_r3c_d = ['Direct_multivar'] + [''] * 8 + (['Direct_nonbin'] + [
            ''] * 8) * 2 + ['Extend_multivar_mp /EffExact'] + [''] * 8 + ([
                'Approx_nonbin /StratVacant'] + [''] * 8) * 2 + [
            'EffExact'] + [''] * 8 + (['StratEarlyStop'] + [''] * 8) * 2
        sub_r3c_d += ['T(Direct_multivar)', 'T(EffExact .StratVacant)',
                      'T(EffExact .StratES)'] + ['T(computing perf.)']  # +85
        csv_row_3c = ['', ''] + (sub_r3c_ab + sub_r3c_c + sub_r3c_d) * 2
        del sub_r3c_c, sub_r3c_d, sub_r3c_ab  # sub_r3c_a, sub_r3c_b,

        sub_r2c_ab = [''] * 20 + ['fairness sa#1'] + [''] * 10 + [
            'fairness sa#2'] + [''] * 10 + ['Grp intersection'] + [
            ''] * 10 + ['Grp union'] + [''] * 10  # joint and|or
        sub_r2c_c = ['HFM sa#1'] + [''] * 38 + ['HFM sa#2'] + [''] * 38 + [
            'HFM intersection'] + [''] * 38 + ['HFM union'] + [''] * 38
        sub_r2c_d = ['HFM.ext w/converged'] + [''] * (9 * 9 - 1) + [
            'HFM.ext tim_elapsed', '', '', 'T(comp.performance)']
        # sub_r2c = sub_r2c_ab + sub_r2c_c + sub_r2c_d  # 20+44+156+85=305
        csv_row_2c = ['Ensem', ''] + ['Training set: performance'] + (
            sub_r2c_ab + sub_r2c_c + sub_r2c_d) + [
            'Test set: performance'] + (sub_r2c_ab + sub_r2c_c + sub_r2c_d)
        del sub_r2c_ab, sub_r2c_c, sub_r2c_d  # , sub_r2c

        # pdb.set_trace()  #

        # sub_r2c_a = ['Normal'] + [''] * 6 + ['Adversarial'] + [
        #     ''] * 6 + [r'$\Delta$(performance)'] + [''] * 6
        # sub_r2c_b = ['Fair sa#1'] + [''] * 10 + ['Fair sa#2'] + [
        #     ''] * 10 + ['Grp intersection'] + [''] * 10 + [
        #     'Grp union'] + [''] * 10  # joint and|or
        # sub_r2c_c = ['HFM sa#1'] + [''] * 38 + ['HFM sa#2'] + [''] * 38 + [
        #     'HFM intersection'] + [''] * 38 + ['HFM union'] + [''] * 38
        # sub_r2c_d = ['HFM.ext w/converged'] + [''] * (9 * 9 - 1) + [
        #     'HFM.ext tim_elapsed', '', '', 'T(comp.performance)']
        # csv_row_3c = ['classifier', 'T(learning)'] + (
        #     sub_r2c_a + sub_r2c_b + sub_r2c_c + sub_r2c_d)
        # del sub_r2c_a, sub_r2c_b, sub_r2c_c, sub_r2c_d

        # sub_r3c_a = ['Acc', 'P', 'R/sen', 'spe', 'f1', 'g_mean', 'dp'] * 3
        # sub_r3c_b = (['Grp.intermediate'] + [''] * 5 + [
        #     'Grp fairness', '', '', 'DR', '']) * 4
        # sub_r3c_c = ['DistDirect_bin'] + [''] * 11 + ['ApproxDist_bin'] + [
        #     ''] * 4 + ['DistApprox_nonbin /StratVacant'] + [
        #     ''] * 8 + ['StratEarlyStop'] + [''] * 8  # 12+5+9+9 =35
        # sub_r3c_c += ['T(Direct_bin)', 'T(ApproxDist_bin)',
        #               'T(Approx_nonbin)', 'T(StratES) converged']
        # sub_r3c_c = sub_r3c_c * 4
        # sub_r3c_d = ['DistDirect_multivar'] + [''] * 8 + ([
        #     'DistDirect_nonbin'] + [''] * 8) * 2 + [
        #     'DistExtend_multivar'] + [''] * 8 + (['DistApprox_nonbin'] + [
        #         ''] * 8) * 2 + ['EfficExact_multivar'] + [''] * 8 + ([
        #             'StratEarlyStop'] + [''] * 8) * 2
        # sub_r3c_d += ['T(Direct_multivar)', 'T(EffExact .StratVacant)',
        #               'T(EffExact .StratES)'] + ['T(computing perf.)']
        # csv_row_4c = ['', ''] + sub_r3c_a + sub_r3c_b + sub_r3c_c + sub_r3c_d
        # del sub_r3c_a, sub_r3c_b, sub_r3c_c, sub_r3c_d

        # csv_row_5c = ['classifier', 'T(learning)'] + (
        #     sub_pt1 + sub_pt2 + sub_pt3 + sub_pt4)
        # del sub_pt1, sub_pt2, sub_pt3, sub_pt4
        # return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c, csv_row_5c
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


# -------------------------------------
# Convergence


class ConvergeF_setup(ConvergeE_setup):
    def __init__(self):
        super().__init__()
        self._hfm_bin = [['bin', '', '', '',
                          'nonbin(bin-val)', '', '', '', '', '',
                          'nonbin(multival)', '', ''], [
            'bin', 'nonbin(bin-val)', '', '', '',
            'nonbin(multival)', '', ''], [
            'Naive', 'EarlyStop',
            'Direct', 'Approx', 'StratES', 'StratRA']]
        self._hfm_nonbin = [[
            'bin', '', '', '', 'nonbin(bin-val)', '', '', '', '',
            '', 'nonbin(multival)', '', '', '', '', ''], [
            'bin', 'nonbin(bin-val)', '', '', '',
            'nonbin(multival)', '', '', ''], [
            'Naive', 'EarlyStop',
            'Direct', 'Approx', 'StratES', 'StratRA']]

    def subproc_bin(self, X_nA_y, A_j, non_sa, m1, m2, n_e):
        Aj_bin = non_sa.astype(DTY_INT)  # 'int')
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)

        app_2 = StratVacant(X_nA_y, Aj_bin, m1, m2, n_e)
        app_3 = StratVacant(X_nA_y, A_j, m1, m2, n_e)
        app_4 = StratEarlyStop(X_nA_y, Aj_bin, n_e)
        app_5 = StratEarlyStop(X_nA_y, A_j, n_e)
        app_6 = StratRearrange(X_nA_y, Aj_bin, m1, m2, n_e)
        app_7 = StratRearrange(X_nA_y, A_j, m1, m2, n_e)

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])

        ans_tim = [hdd_1[1], eff_1[1], luo_1[1], app_0[1],
                   hdd_2[1], eff_2[1], luo_2[1],
                   app_2[1], app_4[1], app_6[1],
                   app_3[1], app_5[1], app_7[1], ]
        ans_max = [hdd_1[0], eff_1[0], luo_1[0][0], app_0[0],
                   hdd_2[0], eff_2[0], luo_2[0][0],
                   app_2[0][0], app_4[0][0], app_6[0][0],
                   app_3[0][0], app_5[0][0], app_7[0][0], ]
        ans_avg = [luo_1[0][1], luo_2[0][1],
                   app_2[0][1], app_4[0][1], app_6[0][1],
                   app_3[0][1], app_6[0][1], app_7[0][1], ]
        return ans_tim + ans_max + ans_avg  # 13*2+8=34

    def subproc_nonbin(self, X_nA_y, A_j, g1m, m1, m2, n_e):
        non_sa = g1m[0]
        Aj_bin = non_sa.astype(DTY_INT)  # 'int')
        luo_1 = DistDirect_bin(X_nA_y, non_sa)
        luo_2 = DistDirect_nonbin(X_nA_y, [non_sa, ~non_sa])
        luo_3 = DistDirect_nonbin(X_nA_y, g1m)

        app_0 = ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2)  # /Aj_bin
        app_2 = StratVacant(X_nA_y, Aj_bin, m1, m2, n_e)
        app_3 = StratVacant(X_nA_y, A_j, m1, m2, n_e)  # hfm extension
        app_4 = StratEarlyStop(X_nA_y, Aj_bin, n_e)
        app_5 = StratEarlyStop(X_nA_y, A_j, n_e)
        app_6 = StratRearrange(X_nA_y, Aj_bin, m1, m2, n_e)
        app_7 = StratRearrange(X_nA_y, A_j, m1, m2, n_e)

        hdd_1 = NaiveHD_bin(X_nA_y, non_sa)
        hdd_2 = NaiveHD_nonbin(X_nA_y, [non_sa, ~non_sa])
        hdd_3 = NaiveHD_nonbin(X_nA_y, g1m)
        eff_1 = EffHDD_bin(X_nA_y, non_sa)
        eff_2 = EffHDD_nonbin(X_nA_y, [non_sa, ~non_sa])
        eff_3 = EffHDD_nonbin(X_nA_y, g1m)

        ans_tim = [
            hdd_1[1], eff_1[1], luo_1[1], app_0[1],
            hdd_2[1], eff_2[1], luo_2[1], app_2[1], app_4[1], app_6[1],
            hdd_3[1], eff_3[1], luo_3[1], app_3[1], app_5[1], app_7[1]]
        ans_max = [hdd_1[0], eff_1[0], luo_1[0][0], app_0[0],
                   hdd_2[0], eff_2[0],
                   luo_2[0][0], app_2[0][0], app_4[0][0], app_6[0][0],
                   hdd_3[0], eff_3[0],
                   luo_3[0][0], app_3[0][0], app_5[0][0], app_7[0][0]]
        ans_avg = [luo_1[0][1],
                   luo_2[0][1], app_2[0][1], app_4[0][1], app_6[0][1],
                   luo_3[0][1], app_3[0][1], app_5[0][1], app_7[0][1]]
        return ans_tim + ans_max + ans_avg  # (41,)=(16+16+9,)

    def subproc_multivar(self, X_nA_y, A, g1m_ind, m1, m2, n_e, pool):
        luo_4 = DistDirect_multivar(X_nA_y, g1m_ind)
        app_4 = EffExact(X_nA_y, A, StratVacant, m1, m2, n_e, pool)
        app_5 = EffExact(X_nA_y, A, StratEarlyStop, m1, m2, n_e, pool)
        app_6 = EffExact(X_nA_y, A, StratRearrange, m1, m2, n_e, pool)
        luo_mid, app_mid, ces_mid, cra_mid = luo_4[
            0][2], app_4[0][2], app_5[0][2], app_6[0][2]
        luo_4 = (luo_4[0][: 2], luo_4[1])
        app_4 = (app_4[0][: 2], app_4[1])
        app_5 = (app_5[0][: 2], app_5[1])
        app_6 = (app_6[0][: 2], app_6[1])

        hdd_4 = NaiveHD_multivar(X_nA_y, g1m_ind)
        eff_4 = EffHDD_multivar(X_nA_y, g1m_ind)
        hdd_mid, eff_mid = hdd_4[0][1], eff_4[0][1]
        hdd_4 = (hdd_4[0][0], hdd_4[1])
        eff_4 = (eff_4[0][0], eff_4[1])

        ans_tim = [hdd_4[1], eff_4[1],
                   luo_4[1], app_4[1], app_5[1], app_6[1]]
        ans_max = [hdd_4[0], eff_4[0],
                   luo_4[0][0], app_4[0][0], app_5[0][0], app_6[0][0]]
        ans_avg = [luo_4[0][1], app_4[0][1], app_5[0][1], app_6[0][1]]
        result = ans_tim + ans_max + ans_avg  # (16,)=(6+6+4,)
        n_a = len(g1m_ind)
        # pdb.set_trace()
        for i in range(n_a):
            tmp_tim = [hdd_mid[1][i], eff_mid[1][i],
                       luo_mid[2][i],
                       app_mid[2][i], ces_mid[2][i], cra_mid[2][i]]
            tmp_max = [hdd_mid[0][i], eff_mid[0][i],
                       luo_mid[0][i],
                       app_mid[0][i], ces_mid[0][i], cra_mid[0][i]]
            tmp_avg = [luo_mid[1][i],
                       app_mid[1][i], ces_mid[1][i], cra_mid[1][i]]
            result.extend(tmp_tim + tmp_max + tmp_avg)  # 6+6+4=16
        if n_a == 1:
            result.extend([''] * 16)
        return result  # (48,)=(16+16*2,)
    # def count_single_member: 7*3+11*4+49*4+112+1 =65+196+113=374

    def count_single_member(self, X, A, y, y_hat, y_qtb,
                            g1m_indices, m1, m2, n_e,
                            pos_label, jt, pool=None):
        ut_c = time.time()
        res_indi = []
        ta_1 = self.count_sing_part1(y, y_hat, pos_label)
        ta_2 = self.count_sing_part1(y, y_qtb, pos_label)
        ta_3 = [abs(t1 - t2) for t1, t2 in zip(ta_1, ta_2)]
        res_indi.extend(ta_1)
        res_indi.extend(ta_2)
        res_indi.extend(ta_3)
        del ta_1, ta_2, ta_3  # res_indi: 7*3=21

        far_2, far_3 = [], []  # ,far_4=[]
        n_a = len(g1m_indices)
        X_and_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_and_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, y_qtb, g1m_indices[i][0], pos_label)
            far_2.extend(tmp)
        if n_a == 1:
            far_2.extend([''] * 11 * 3)
        else:
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[0], pos_label))  # 'and'&
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[1], pos_label))  # 'or' |
        ut_c = time.time() - ut_c

        A_i = [A[:, i].copy() for i in range(n_a)]
        if n_a > 1:
            A_0 = np.logical_and(A_i[0], A_i[1]).astype(DTY_FLT)
            A_1 = np.logical_or(A_i[0], A_i[1]).astype(DTY_FLT)
            # pdb.set_trace()  # SOMETHING WRONG?
        else:
            A_0, A_1 = None, None
        for i in range(n_a):
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, g1m_indices[i][0], A_i[i],
                m1, m2, n_e))
        if n_a == 1:
            far_3.extend([''] * 49 * 3)  # not 39 anymore
        else:
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[0], A_0, m1, m2, n_e))
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[1], A_1, m1, m2, n_e))

        del A_0, A_1, A_i
        far_4 = self.count_sing_part4(
            X_and_y, X_and_y_hat, g1m_indices, A, m1, m2, n_e, pool)
        far_4.append(ut_c)  # not 21+44+156+84+1=306 anymore
        return res_indi + far_2 + far_3 + far_4  # 65+49*4+113=374

    def count_sing_part3(self, X_y, X_y_hat, non_sa, A_j,
                         m1=20, m2=8, n_e=2):
        cmp_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_y, non_sa)
        (Df_01, Df_avg), t_Df = DistDirect_bin(X_y_hat, non_sa)
        df_prev, ut_df_prev = fair_degree_v3(Ds_01, Df_01)
        df, ut_df = fair_degree_v4(Ds_01, Df_01)
        df_avg, ut_df_avg = fair_degree_v4(Ds_avg, Df_avg)
        ut_a = time.time() - ut_a
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df,
                         ut_df_prev, ut_df, ut_df_avg])
        ut_b = time.time()
        Ds_01, t_Ds = ApproxDist_bin(X_y, A_j, non_sa, m1, m2)
        Df_01, t_Df = ApproxDist_bin(X_y_hat, A_j, non_sa, m1, m2)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        ut_b = time.time() - ut_b
        cmp_fair.extend([Ds_01, Df_01, df_prev, t_Ds, t_Df])

        B_j = non_sa.astype(DTY_INT)
        ut_c = time.time()
        (Ds_01, Ds_avg), t_Ds = StratVacant(X_y, B_j, m1, m2, n_e)
        (Df_01, Df_avg), t_Df = StratVacant(X_y_hat, B_j, m1, m2, n_e)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_c = time.time() - ut_c
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        ut_d = time.time()
        (Ds_01, Ds_avg), t_Ds = StratEarlyStop(X_y, B_j, n_e)
        (Df_01, Df_avg), t_Df = StratEarlyStop(X_y_hat, B_j, n_e)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_d = time.time() - ut_d
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])

        ut_e = time.time()
        (Ds_01, Ds_avg), t_Ds = StratRearrange(X_y, B_j, m1, m2, n_e)
        (Df_01, Df_avg), t_Df = StratRearrange(X_y_hat, B_j, m1, m2, n_e)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_e = time.time() - ut_e
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])

        cmp_fair.extend([ut_a, ut_b, ut_c, ut_d, ut_e])
        del Ds_01, Ds_avg, Df_01, Df_avg, df_prev, df, df_avg
        del t_Ds, t_Df, ut_df_prev, ut_df, ut_df_avg
        # return cmp_fair  # shape=(39,) =(12+5+9+9+4,)
        return cmp_fair  # shape=(49,) =(12+5+9+9+9+5,)

    def count_sing_part4(self, X_y, X_y_hat, g1m_indices, A,
                         m1=20, m2=8, n_e=2, pool=None):
        cmp_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(
            X_y, g1m_indices)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistDirect_multivar(
            X_y_hat, g1m_indices)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        n_a = len(g1m_indices)

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_a = time.time() - ut_a
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        ut_b = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = EffExact(
            X_y, A, StratVacant, m1, m2, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = EffExact(
            X_y_hat, A, StratVacant, m1, m2, n_e, pool)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_b = time.time() - ut_b
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        ut_d = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = EffExact(
            X_y, A, StratEarlyStop, None, None, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = EffExact(
            X_y_hat, A, StratEarlyStop, None, None, n_e, pool)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_d = time.time() - ut_d
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        ut_e = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = EffExact(
            X_y, A, StratRearrange, m1, m2, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = EffExact(
            X_y_hat, A, StratRearrange, m1, m2, n_e, pool)
        df_prev, _ = fair_degree_v3(Ds_01, Df_01)
        df, _ = fair_degree_v4(Ds_01, Df_01)
        df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, Df_01, Df_avg,
                         df_prev, df, df_avg, t_Ds, t_Df])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            df_prev, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            df, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([Ds_01[i], Ds_avg[i], Df_01[i], Df_avg[i],
                             df_prev, df, df_avg, t_Ds[i], t_Df[i]])
        ut_e = time.time() - ut_e
        if n_a == 1:
            cmp_fair.extend([''] * 9)

        cmp_fair.extend([ut_a, ut_b, ut_d, ut_e])
        del Ds_01, Ds_avg, Df_01, Df_avg, df_prev, df, df_avg
        del t_Ds, t_Df, Ds_midtmp, Df_midtmp, ut_a, ut_b, ut_d
        # return cmp_fair  # .shape=(81+3,) =(9*3*3+3,)
        return cmp_fair   # .shape=(108+4,) =(9*3*4+4,)


# class ConvergeF3_with(ConvergeF_setup, ConvergeE3_with):
class ConvergeF3_with(ConvergeE3_with, ConvergeF_setup):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3):
        # super(ConvergeE3_with, self).__init__(
        # ConvergeE3_with.__init__(
        super().__init__(nb_cls, saIndex, saValue, n_e)

    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 34 * 2)
        csv_row_2c = ['sa#1 tim'] + [''] * 12 + ['max'] + [
            ''] * 12 + ['avg'] + [''] * 7 + ['sa#2 tim'] + [
            ''] * 12 + ['max'] + [''] * 12 + ['avg'] + [''] * 7
        csv_row_4c = self._hfm_bin[0] * 2 + self._hfm_bin[1]
        # csv_row_5c = (self._hfm_bin[2]*2+)
        csv_row_5c = ([
            'Naive', 'EarlyStop', 'Direct', 'Approx'] * 2 + [
            'StratES', 'StratRA', 'Approx', 'StratES', 'StratRA'
        ]) * 2 + ['Direct', 'Direct', 'Approx', 'StratES',
                  'StratRA', 'Approx', 'StratES', 'StratRA']
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


class ConvergeF4_with(ConvergeE4_with, ConvergeF_setup):
    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 41 * 2)  # =35+6
        csv_row_2c = ['sa#1 tim'] + [''] * 15 + ['max'] + [
            ''] * 15 + ['avg'] + [''] * 8 + ['sa#2 tim'] + [
            ''] * 15 + ['max'] + [''] * 15 + ['avg'] + [''] * 8
        csv_row_4c = self._hfm_nonbin[0] * 2 + self._hfm_nonbin[1]
        # self._hfm_nonbin[1] is the same as self._hfm_bin[1]
        csv_row_5c = ([
            'Naive', 'EarlyStop', 'Direct', 'Approx'] * 2 + [
            'StratES', 'StratRA', 'Naive', 'EarlyStop',
            'Direct', 'Approx', 'StratES', 'StratRA']) * 2 + [
            'Direct', 'Direct', 'Approx', 'StratES', 'StratRA',
            'Direct', 'Approx', 'StratES', 'StratRA']  # 16*2+9=41
        return csv_row_1, csv_row_2c, csv_row_4c * 2, csv_row_5c * 2


class ConvergeF5_with(ConvergeE5_with, ConvergeF_setup):
    def prepare_trial(self):
        csv_row_1 = unique_column(10 + 48)  # 13|16*3
        csv_row_2c = ['multivar'] + [''] * 15 + ['sen-att #1'] + [
            ''] * 15 + ['sen-att #2'] + [''] * 15
        csv_row_3c = ['tim'] + [''] * 5 + ['max'] + [''] * 5 + [
            'avg', '', '', '']
        csv_row_4c = [
            'Naive', 'EarlyStop', 'Direct', 'Extend', 'EfficA', '',
            'Naive', 'EarlyStop', 'Direct', 'Approx', 'StratES', 'StratRA',
            'Direct', 'Approx', 'StratES', 'StratRA']
        return csv_row_1, csv_row_2c, csv_row_3c * 3, csv_row_4c * 3


class ConvergeF2_with(ConvergeE2_with, ConvergeF_setup):
    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 1 + 374 * 2)

        sub_pt1, sub_pt2, sub_pt3, sub_pt4 = self.trial_prep_pt4()
        csv_row_4c = ['classifier', 'T(learning)'] + (
            sub_pt1 + sub_pt2 + sub_pt3 * 4 + sub_pt4) * 2
        del sub_pt1, sub_pt2, sub_pt3, sub_pt4

        sub_r3c_c, sub_r3c_d, sub_r3c_ab = self.trial_prep_pt3()
        csv_row_3c = ['', ''] + (sub_r3c_ab + sub_r3c_c + sub_r3c_d) * 2
        del sub_r3c_c, sub_r3c_d, sub_r3c_ab  # 65+196+113 =374
        sub_r2c_ab, sub_r2c_c, sub_r2c_d = self.trial_prep_pt2()
        csv_row_2c = ['Ensem', ''] + ['Training set: performance'] + (
            sub_r2c_ab + sub_r2c_c + sub_r2c_d) + [
            'Test set: performance'] + (sub_r2c_ab + sub_r2c_c + sub_r2c_d)
        del sub_r2c_ab, sub_r2c_c, sub_r2c_d

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def trial_prep_pt4(self):
        sub_pt1 = self._metric_part1 * 3
        sub_pt2 = (['g1', 'g0'] * 3 + self._metric_part2) * 4  # +11*4
        sub_pt3_a = ['Ds', 'Ds_avg', 'Df', 'Df_avg',
                     'df_prev', 'df', 'df_avg', 't_Ds', 't_Df',
                     'T(df_prev)', 'T(df)', 'T(df_avg)']
        sub_pt3_b = ['Ds', 'Df', 'df_prev', 't_Ds', 't_Df']
        sub_pt3_c = sub_pt3_a[: 9] * 3 + [
            'T(Direct_bin)', 'T(ApproxDist_bin)',
            'T(Approx_nonbin /StratVacant)', 'T(StratES)', 'T(StratRA)']
        sub_pt3 = sub_pt3_a + sub_pt3_b + sub_pt3_c  # 12+5+32=49
        sub_pt4 = sub_pt3_a[: 9] * (1 + 2) * 4 + [
            'T(Direct_multivar)',
            'T(Extend_multivar_mp /EffExact-StratVacant)',
            'T(EffExact-StratES)', 'T(EffExact-StratRA)',
            'T(comp.performance)']  # 21+44+49*4+113 =374 in total
        del sub_pt3_a, sub_pt3_b, sub_pt3_c
        return sub_pt1, sub_pt2, sub_pt3, sub_pt4

    def trial_prep_pt3(self):
        sub_r3c_ab = (['Normal'] + [''] * 6 + ['Adversarial'] + [
            ''] * 6 + [r'$\Delta$(performance)'] + [''] * 6) * 1 + ([
                'Grp.intermediate'] + [''] * 5 + [
                'GrpFair: DP,EOpp,PP', '', '', 'DR', '']) * 4  # 21+44=65
        sub_r3c_c = ['Direct_bin'] + [''] * 11 + [
            'ApproxDist_bin'] + [''] * 4 + [
            'Approx_nonbin /StratVacant'] + [''] * 8 + [
            'StratES'] + [''] * 8 + ['StratRA'] + [
            ''] * 8  # 12+5+9*3=44, 44+5=49  # NB> not *9
        sub_r3c_c = (sub_r3c_c + [
            'T(Direct_bin)', 'T(ApproxDist_bin)',
            'T(Approx_nonbin)', 'T(StratES)', 'T(StratRA)']) * 4  # +49*4
        sub_r3c_d = ['Direct_multivar'] + [''] * 8 + ([
            'Direct_nonbin'] + [''] * 8) * 2 + [
            'Extend_multivar_mp /EffExact'] + [''] * 8 + ([
                'Approx_nonbin /StratVacant'] + [''] * 8) * 2 + [
            'EffExact -ES'] + [''] * 8 + (['StratES'] + [''] * 8) * 2 + [
            'EffExact -RA'] + [''] * 8 + (['StratRA'] + [''] * 8) * 2
        sub_r3c_d += ['T(Direct_multivar)', 'T(EffExact.StratVacant)',
                      'T(EffExact .StratES)', 'T(EffExact .StratRA)'
                      ] + ['T(computing perf.)']  # +9*3*(3+1)+4 +1=112+1
        return sub_r3c_c, sub_r3c_d, sub_r3c_ab

    def trial_prep_pt2(self):
        sub_r2c_ab = [''] * 20 + ['fairness sa#1'] + [''] * 10 + [
            'fairness sa#2'] + [''] * 10 + ['Grp intersection'] + [
            ''] * 10 + ['Grp union'] + [''] * 10  # joint and|or
        sub_r2c_c = ['HFM sa#1'] + [''] * 48 + ['HFM sa#2'] + [''] * 48 + [
            'HFM intersection'] + [''] * 48 + ['HFM union'] + [''] * 48
        sub_r2c_d = ['HFM.ext w/converged'] + [''] * (9 * 3 * 4 - 1) + [
            'HFM.ext tim_elapsed', '', '', '', 'T(comp. performance)']
        # # sub_r2c =ab+c+d  # 21+22*2 +49*4+9*12+5 =374
        return sub_r2c_ab, sub_r2c_c, sub_r2c_d


class ConvergeF7_with(ConvergeF2_with):
    def count_sing_part5(self, y, y_hat, g1m, pos_label):
        # _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
        #     y, y_hat, pos_label, non_sa)
        cmp_fair, ut_f = [], time.time()
        tmp_1, ut1 = extGrp1_DP_sing(y, y_hat, g1m, pos_label)
        tmp_2, ut2 = extGrp2_EO_sing(y, y_hat, g1m, pos_label)
        tmp_3, ut3 = extGrp3_PQP_sing(y, y_hat, g1m, pos_label)
        cmp_fair.extend(tmp_1[:2] + tmp_2[:2] + tmp_3[:2])
        cmp_fair.extend(alterGrps_sing(tmp_1[-1], g1m)[0])
        cmp_fair.extend(alterGrps_sing(tmp_2[-1], g1m)[0])
        cmp_fair.extend(alterGrps_sing(tmp_3[-1], g1m)[0])
        ut_f = time.time() - ut_f
        cmp_fair.extend([ut1, ut2, ut3, ut_f])
        return cmp_fair  # 6+6+4=16

    def count_single_member(self, X, A, y, y_hat, y_qtb,
                            g1m_indices, m1, m2, n_e,
                            pos_label, jt, pool=None):
        ut_c = time.time()
        res_indi = []
        ta_1 = self.count_sing_part1(y, y_hat, pos_label)
        ta_2 = self.count_sing_part1(y, y_qtb, pos_label)
        ta_3 = [abs(t1 - t2) for t1, t2 in zip(ta_1, ta_2)]
        res_indi.extend(ta_1 + ta_1 + ta_3)  # tuple(ta_3))
        del ta_1, ta_2, ta_3  # +7*3=21

        far_2, far_3, far_5 = [], [], []
        n_a = len(g1m_indices)
        for i in range(n_a):
            tmp = self.count_sing_part2(
                y, y_hat, y_qtb, g1m_indices[i][0], pos_label)
            far_2.extend(tmp)
        if n_a == 1:
            far_2.extend([''] * 11 * 3)
        else:
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[0], pos_label))  # & jt
            far_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[1], pos_label))  # | jt
        ut_c = time.time() - ut_c

        X_and_y = np.concatenate([
            y.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        X_and_y_hat = np.concatenate([
            y_hat.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        A_i = [A[:, i].copy() == 1 for i in range(n_a)]
        if n_a > 1:
            A_0 = np.logical_and(A_i[0], A_i[1]).astype(DTY_FLT)
            A_1 = np.logical_or(A_i[0], A_i[1].astype(DTY_FLT))
        else:
            A_0, A_1 = None, None
        for i in range(n_a):
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, g1m_indices[i][0],
                A_i[i], m1, m2, n_e))
        if n_a == 1:
            far_3.extend([''] * 49 * 3)
        else:
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[0], A_0, m1, m2, n_e))
            far_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[1], A_1, m1, m2, n_e))
        del A_0, A_1, A_i

        for i in range(n_a):
            far_5.extend(self.count_sing_part5(
                y, y_hat, g1m_indices[i], pos_label))
        if n_a == 1:
            far_5.extend([''] * 16 * 3)
        else:
            far_5.extend(self.count_sing_part5(
                y, y_hat, [jt[0], ~jt[0]], pos_label))
            far_5.extend(self.count_sing_part5(
                y, y_hat, [jt[1], ~jt[1]], pos_label))
            # NB. this is not precise

        far_4 = self.count_sing_part4(
            X_and_y, X_and_y_hat, g1m_indices, A, m1, m2,
            n_e, pool)
        far_4.append(ut_c)  # total 65+(16*4)+49*4+113 =438
        # return res_indi + far_2 + far_3 + far_5 + far_4
        return res_indi + far_2 + far_5 + far_3 + far_4

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 1 + 438 * 2)

        sub_pt1, sub_pt2, sub_pt3, sub_pt4 = self.trial_prep_pt4()
        sub_pt5 = (['max', 'avg', ] * 3 * 2 + [
            'T(extG1)', 'T(extG2)', 'T(extG3)',
            'T(extGrp* total)']) * 4  # StatsParity
        csv_row_4c = ['classifier', 'T(learning)'] + (
            sub_pt1 + sub_pt2 + sub_pt5 + sub_pt3 * 4 + sub_pt4) * 2
        del sub_pt1, sub_pt2, sub_pt5, sub_pt3, sub_pt4

        sub_r3c_c, sub_r3c_d, sub_r3c_ab = self.trial_prep_pt3()
        sub_r3c_f = (['extG1', '', 'extG2', '', 'extG3', '',
                      'altG1', '', 'altG2', '', 'altG3', '',
                      'Time cost', '', '', '']) * 4
        csv_row_3c = ['', ''] + (sub_r3c_ab + sub_r3c_f +
                                 sub_r3c_c + sub_r3c_d) * 2
        del sub_r3c_c, sub_r3c_d, sub_r3c_ab, sub_r3c_f

        sub_r2c_ab, sub_r2c_c, sub_r2c_d = self.trial_prep_pt2()
        sub_r2c_f = ['StatsParity sa#1'] + [''] * 15 + [
            'StatsParity sa#2'] + [''] * 15 + ['SP.ext joint&'] + [
            ''] * 15 + ['SP.ext joint|'] + [''] * 15
        csv_row_2c = ['Ensem', ''] + ['Training set: performance'] + (
            sub_r2c_ab + sub_r2c_f + sub_r2c_c + sub_r2c_d) + [
            'Test set: performance'] + (
                sub_r2c_ab + sub_r2c_f + sub_r2c_c + sub_r2c_d)
        del sub_r2c_ab, sub_r2c_c, sub_r2c_d, sub_r2c_f

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


# -------------------------------------

# -------------------------------------
# --------
# --------
# --------
