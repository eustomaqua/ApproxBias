# coding: utf-8
# Experiments


import time
import numpy as np

from hfm.utils.verifiers import unique_column, DTY_FLT, DTY_INT
from experiment.classifiers import (
    RelativeFairClsf, IndividualClsf, INDIVIDUALS)

from hfm.dist_drt import DirectDist_bin as DistDirect_bin
# from hfm.dist_drt import DirectDist_nonbin as DistDirect_nonbin
from hfm.dist_drt import DirectDist_multiver as DistDirect_multivar
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4
# from hfm.dist_est_nonbin import AcceleDist_nonbin as DistAccele
from hfm.dist_est_nonbin import ApproxDist_nonbin_mpver as DistApprox
from hfm.dist_est_nonbin import ExtendDist_multiver_mp as DistExtend
from hfm.dist_est_bin import ApproxDist_bin


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
# from hfm.discriminative_risk import (
#     E_rho_L_fair_f, hat_L_fair, E_rho_L_loss_f, hat_L_loss)
from hfm.discriminative_risk import hat_L_fair, hat_L_loss


from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier)
from experiment.classifiers import (
    LGBMClassifier, FairGBMClassifier, AdaFair)


# ===============================
# fairmanf


# -------------------------------
#


class ComparisonB_setup:
    def __init__(self, *, omitted=True):
        self._metric_part1 = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                              'fpr', 'fnr', 'sensitivity', 'specificity',
                              'g_mean', 'dp',  # 'G_mean', 'DPower',
                              'Matthew', 'Cohen', '(random_acc)']  # 13-5
        if omitted:
            self._metric_part1.remove('fpr')
            self._metric_part1.remove('fnr')
            self._metric_part1.remove('Matthew')
            self._metric_part1.remove('Cohen')
            self._metric_part1.remove('(random_acc)')

        self._metric_part2 = [
            'unaware', 'grp_one', 'grp_two', 'grp_thr', 'manual',
            'hat_L(fair)', 'hat_L(loss)']  # 5+2 or 3+2
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
            # return lightgbm.LGBMClassifier(n_estimators=nb_cls)
            return LGBMClassifier(n_estimators=nb_cls)
        elif name_ens == 'fairgbm':
            return FairGBMClassifier(n_estimators=nb_cls,
                                     constraint_type=constraint)
        elif name_ens == 'adafair':
            return AdaFair(n_estimators=nb_cls,
                           saIndex=saIndex, saValue=saValue)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    def count_single_member(X, A, y, y_hat, y_qtb,
                            g1m_indices, jt, positive_label=1,
                            m1=20, m2=8, n_e=3, *, omitted=True):
        # NB. must be np.ndarray
        raise NotImplementedError

    def count_scores(self,
                     X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn, jt_trn,
                     X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst, jt_tst,
                     positive_label=1, m1=20, m2=8, n_e=3, pool=None,
                     *, omitted=True):
        ans_trn = self.count_single_member(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1m_trn, jt_trn,
            positive_label, m1, m2, n_e, pool, omitted=omitted)
        ans_tst = self.count_single_member(
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1m_tst, jt_tst,
            positive_label, m1, m2, n_e, pool, omitted=omitted)
        return ans_trn + ans_tst

    def count_sing_part1(self, y, y_hat, positive_label,
                         *, omitted=True):
        # NB. must be np.ndarray
        tp, fp, fn, tn = contingency_table(y, y_hat, positive_label)

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

    def count_sing_part2(self, y, y_hat, y_qtb, non_sa, positive_label=1,
                         *, omitted=True):
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

    def count_sing_part3(self, X_y, X_y_hat, non_sa, A_j,
                         m1=20, m2=8, n_e=2, pool=None):
        cmp_fair = []

        ut_a = time.time()
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_y, non_sa)
        (Df_01, Df_avg), t_Df = DistDirect_bin(X_y_hat, non_sa)
        df_ecai, ut_ddf_ecai = fair_degree_v3(Ds_01, Df_01)
        df_nips, ut_ddf_nips = fair_degree_v4(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        ut_a = time.time() - ut_a
        cmp_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df])
        cmp_fair.extend([df_ecai, df_nips, v3_df_avg, v4_df_avg])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del df_ecai, ut_ddf_ecai, df_nips, ut_ddf_nips, v3_df_avg, v4_df_avg

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
        cmp_fair.extend([Ds_01, t_Ds, Df_01, t_Df, df_ecai, df_nips])
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
        cmp_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df])
        cmp_fair.extend([df_ecai, df_nips, v3_df_avg, v4_df_avg, ])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del df_ecai, df_nips, v3_df_avg, v4_df_avg

        cmp_fair.extend([ut_a, ut_b, ut_c])
        return cmp_fair    # shape= (29,) =(10+6+10+3,)

    def count_sing_part4(self, X_y, X_y_hat, g1m_indices, A,
                         m1=20, m2=8, n_e=2, pool=None):  # _4b
        cmp_fair = []
        ut_a = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(X_y, g1m_indices)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistDirect_multivar(X_y_hat, g1m_indices)
        v3_df_max, _ = fair_degree_v3(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_max, _ = fair_degree_v4(Ds_01, Df_01)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        n_a = len(g1m_indices)

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            cmp_fair.extend([
                Ds_01[i], Ds_avg[i], t_Ds[i], Df_01[i], Df_avg[i], t_Df[i]])
            v3_df_max, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            v3_df_avg, _ = fair_degree_v3(Ds_avg[i], Df_avg[i])
            v4_df_max, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            v4_df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        if n_a == 1:
            cmp_fair.extend([''] * 10)
        del v3_df_max, v3_df_avg, v4_df_max, v4_df_avg
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        del Ds_midtmp, Df_midtmp
        ut_a = time.time() - ut_a

        ut_b = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistExtend(X_y, A, m1, m2, n_e, pool)
        (Df_01, Df_avg, Df_midtmp), t_Df = DistExtend(X_y_hat, A, m1, m2, n_e, pool)
        v3_df_max, _ = fair_degree_v3(Ds_01, Df_01)
        v3_df_avg, _ = fair_degree_v3(Ds_avg, Df_avg)
        v4_df_max, _ = fair_degree_v4(Ds_01, Df_01)
        v4_df_avg, _ = fair_degree_v4(Ds_avg, Df_avg)
        cmp_fair.extend([Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df,
                         v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        del Ds_01, Ds_avg, t_Ds, Df_01, Df_avg, t_Df
        del v3_df_max, v4_df_max, v3_df_avg, v4_df_avg

        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        Df_01, Df_avg, t_Df = Df_midtmp
        for i in range(n_a):
            cmp_fair.extend([
                Ds_01[i], Ds_avg[i], t_Ds[i], Df_01[i], Df_avg[i], t_Df[i]])
            v3_df_max, _ = fair_degree_v3(Ds_01[i], Df_01[i])
            v3_df_avg, _ = fair_degree_v3(Ds_avg[i], Df_avg[i])
            v4_df_max, _ = fair_degree_v4(Ds_01[i], Df_01[i])
            v4_df_avg, _ = fair_degree_v4(Ds_avg[i], Df_avg[i])
            cmp_fair.extend([v3_df_max, v4_df_max, v3_df_avg, v4_df_avg])
        if n_a == 1:
            cmp_fair.extend([''] * 10)
        del v3_df_max, v3_df_avg, v4_df_max, v4_df_avg
        del t_Ds, t_Df, Ds_01, Ds_avg, Df_01, Df_avg
        del Ds_midtmp, Df_midtmp
        ut_b = time.time() - ut_b

        cmp_fair.extend([ut_a, ut_b])
        return cmp_fair  # .shape= (62,) =(10*3+10*3+2,)


class ComparisonB2_withDirectComput(ComparisonB_setup):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3, *, omitted=True):
        super().__init__(omitted=omitted)
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        self._n_e = n_e

    def count_sing_part2(self, y, y_hat, y_qtb, non_sa, positive_label=1,
                         *, omitted=True):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, y_hat, positive_label, non_sa)
        cmp_fair = []

        # tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        # tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)

        if not omitted:
            cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)

        # cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        # cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair  # shape= (5,) or (11,)

    def count_single_member(self, X, A, y, y_hat, y_qtb,
                            g1m_indices, jt, positive_label=1,
                            m1=20, m2=8, n_e=3, pool=None,
                            *, omitted=True):
        ut_c = time.time()
        res_indi = self.count_sing_part1(y, y_hat, positive_label,
                                         omitted=omitted)  # 13/8
        fair_2, fair_3 = [], []
        n_a = len(g1m_indices)
        X_and_y = np.concatenate([y.reshape(-1, 1), X], axis=1,
                                 dtype=DTY_FLT).copy()
        X_and_y_hat = np.concatenate([y_hat.reshape(-1, 1), X],
                                     axis=1, dtype=DTY_FLT).copy()
        for i in range(n_a):
            tmp = self.count_sing_part2(y, y_hat, y_qtb, g1m_indices[i][0],
                                        positive_label, omitted=omitted)
            fair_2.extend(tmp)  # +17/11 --> 11/5
        if n_a == 1:
            tmp = 5 if omitted else 11  # tmp = 11 if omitted else 17
            fair_2.extend([''] * tmp * 3)
        else:
            fair_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[0], positive_label, omitted=omitted))
            fair_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[1], positive_label, omitted=omitted))
        # fair_2.shape= (44/20,) =(11/5*4,)  # (68/44) =(17/11*4,)
        ut_c = time.time() - ut_c

        A_i = [A[:, i].copy() for i in range(n_a)]
        if n_a > 1:
            A_0 = np.logical_and(A_i[0], A_i[1]).astype(DTY_FLT)
            A_1 = np.logical_or(A_i[0], A_i[1]).astype(DTY_FLT)
        else:
            A_0, A_1 = None, None
        for i in range(n_a):
            fair_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, g1m_indices[i][0], A_i[i],
                m1, m2, n_e, pool))
        if n_a == 1:
            fair_3.extend([''] * 29 * 3)  # 12|8 * 3)
        else:
            fair_3.extend(self.count_sing_part3(X_and_y, X_and_y_hat, jt[0],
                                                A_j=A_0, m1=m1, m2=m2, n_e=n_e,
                                                pool=pool))
            fair_3.extend(self.count_sing_part3(X_and_y, X_and_y_hat, jt[1],
                                                A_j=A_1, m1=m1, m2=m2, n_e=n_e,
                                                pool=pool))
        del A_0, A_1, A_i
        #   fair_3.shape= (87,) =(29*3,)  # fair_4.shape= (63,) =(10*6+3,)

        fair_4 = self.count_sing_part4(X_and_y, X_and_y_hat, g1m_indices,
                                       A, m1, m2, n_e=n_e, pool=pool)
        fair_4.extend([ut_c])
        #   if not omitted: .shape= (236,) =(13+11*4+ 29*4+63,)  #233
        #   if     omitted: .shape= (207,) =( 8+ 5*4+ 29*4+63,)  #204
        return res_indi + fair_2 + fair_3 + fair_4

    def prepare_trial(self, omitted=True):
        tmp_p = 207 if omitted else 236
        csv_row_1 = unique_column(11 + 1 + tmp_p * 2)
        tmp_p = 5 if omitted else 11
        tmp_2 = ['fairvote'] + [''] * (tmp_p * 4 - 1) + ['fairmanf'] + [
            ''] * (29 * 4 - 1) + ['fairmanf_ext'] + [''] * (10 * 6 + 3 - 1)
        tmp_p = (8 if omitted else 13) - 1
        csv_row_2c = ['Ensem'] + ['Training set'] + [''] * tmp_p + tmp_2 + [
            'Test set'] + [''] * tmp_p + tmp_2
        del tmp_p, tmp_2  # 1+?*2  # ?= (8|13+5|11*4 +29*4+62+1)

        tmp_3_1 = ['Normal'] + [''] * (7 if omitted else 12)
        tmp_p = (5 if omitted else 11) - 1
        tmp_3_2 = [
            'sens_att#1'] + [''] * tmp_p + ['sens_att#2'] + [''] * tmp_p + [
            'joint_and'] + [''] * tmp_p + ['joint_or'] + [''] * tmp_p
        tmp_3_3 = ['sens_att#1'] + [''] * 28 + ['sens_att#2'] + [''] * 28 + [
            'att_#1&2'] + [''] * 28 + ['att_#1|2'] + [''] * 28    # *11
        tmp_3_4 = ['DistDirect multivar'] + [''] * 9 + [
            'sen_att_#1'] + [''] * 9 + ['sen_att_#2'] + [''] * 9  # *13
        tmp_3_5 = ['DistExtend'] + [''] * 9 + ['DistApprox sen_att_#1'] + [
            ''] * 9 + ['DistApprox sen_att_#2'] + [''] * 9 + [
            'TimeCost: DistDirect_multivar, DistExtend+ DistApprox *2',
            '', '']  # siz= 10*3*2+3-1+1 =62+1 =63
        tmp_3 = tmp_3_1 + tmp_3_2 + tmp_3_3 + tmp_3_4 + tmp_3_5
        #   tmp_3.siz= 8+5*4+ 29*4+10*3*2+2 =28+116+62+1 =207-1+1 =207
        csv_row_3c = ['Time Cost (sec)'] + tmp_3 + tmp_3  # 1+207*2
        del tmp_3_1, tmp_3_2, tmp_3_3, tmp_3_4, tmp_3, tmp_p, tmp_3_5

        tmp_4_2 = ['Group fairness: g1,g0 (Gfm one/two/thr)',
                   '', '', '', '', ''] if not omitted else []
        tmp_4_2.extend(['Group fairness: abs(g1-g0)', '', '',
                        'hat_L(fair)', 'hat_L(loss)'])
        tmp_4 = (self._metric_part1 + tmp_4_2 * 4 +
                 self._metric_part3 * 4 + self._metric_part4 * 6 + [
                     't(DistDirect_multivar)',
                     't(DistExtend) + t(DistApprox)',
                     't(baseline + fairvote)'])  # *(3-1)
        #   tmp_4.siz= 8+5*4+ 29*4+10*6+3 =28+116+(63-1+1) =207-1+1
        csv_row_4c = ['ut'] + tmp_4 + tmp_4  # 1+207*2
        del tmp_4_2, tmp_4
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        # raise NotImplementedError
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2, pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        # res_iter.shape= (3, 1+ 131|102 *2)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[0][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[0][0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
            return res_iter  # shape= (3+4*1, 1+?*2)

        sa_len = len(g1ms_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[i][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[i][0],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
        return res_iter  # shape= (3+4*2, 1+?*2)

    def subroute_one_sens_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst,
            sa_idx, sa_val, positive_label=1, m1=20, m2=8,
            pool=None,
            X_trn=None, A_trn=None, g1ms_trn=None, jt_trn=None,
            X_tst=None, A_tst=None, g1ms_tst=None, jt_tst=None,
            *, omitted=True):
        res_attr = []

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst, constraint=constraint_type)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
                positive_label, m1, m2, self._n_e, pool, omitted=omitted)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst, sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        # return res_attr  # shape= (4, 263|205)= (3+1, 1+ 131|102 *2)
        return res_attr    # shape= (4, 467|409)= (3+1, 1+ 233|204 *2)

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
            # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
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

    def subroute_one_norm_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            # positive_label=1, m1=20, m2=8,
            positive_label=1, m1=20, m2=8, pool=None,
            X_trn=None, A_trn=None, X_tst=None, A_tst=None,
            *, omitted=True):
        res_attr = []

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn=None,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        # return res_attr  # shape= (3, 263|205) =(3, 1+ 131|102 *2)
        return res_attr    # shape= (3, 467|409)= (3, 1+ 233|204 *2)


# -------------------------------
#


class ComparisonC2_withDirectComput(ComparisonB2_withDirectComput):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3, *, omitted=True):
        super().__init__(nb_cls, saIndex, saValue, n_e, omitted=omitted)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_two_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        # # res_iter.shape= (11+3, 1+ 131|102 *2)
        #   res_iter.shape= (11+3, 1+ 233|204 *2)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[0][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[0][0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
            return res_iter  # shape= (14+4*1, 1+?*2)

        sa_len = len(g1ms_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[i][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[i][0],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
        return res_iter  # shape= (14+4*2, 1+?*2)

    def subroute_two_norm_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            # positive_label=1, m1=20, m2=8,
            positive_label=1, m1=20, m2=8, pool=None,
            X_trn=None, A_trn=None, X_tst=None, A_tst=None,
            *, omitted=True):
        res_attr = []

        for abbr_cls in self._abbr_clfs:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_gene_clf(
                abbr_cls, X_A_trn, y_trn, X_Aq_trn, X_A_tst, y_tst, X_Aq_tst)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
                positive_label, m1, m2, self._n_e, pool, omitted=omitted)
            res_attr.append([ut] + tmp)

        # return res_attr  # shape= (11, 263|205) =(11, 1+ 131|102 *2)
        return res_attr    # shape= (11, 467|409)= (11, 1+ 233|204 *2)

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
    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_two_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)

        return res_iter  # .shape= (11+3, 1+ 131|102 *2)


# ===============================
# fairmanf_ext.


# -------------------------------
#


class ComparisonD2_withDirectComput(ComparisonB2_withDirectComput):
    def schedule_content(self, logger):
        pass


class ComparisonD3_withDirectComput(ComparisonC2_withDirectComput):
    def schedule_content(self, logger):
        pass


class ComparisonD4_withDirectComput(ComparisonC4_withDirectComput):
    def schedule_content(self, logger):
        pass


# -------------------------------
#


class ComparisonE_setup(ComparisonB_setup):
    def __init__(self, *, omitted=True):
        super().__init__(omitted=omitted)

    def count_sing_part2(self, y, y_hat, y_qtb, non_sa, positive_label=1,
                         *, omitted=True):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, y_hat, positive_label, non_sa)
        cmp_fair = []

        if not omitted:
            tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
            tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)

        if not omitted:
            cmp_fair.extend(tmp_0 + tmp_4)
        cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)

        if not omitted:
            cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
            cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))

        cmp_fair.append(hat_L_loss(y_hat, y))
        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        return cmp_fair  # shape= (17|11,) or (5|3*2+5|3+2,)

    def count_single_member(self, X, A, y, y_hat, y_qtb,
                            g1m_indices, jt, positive_label=1,
                            # m1=20, m2=8, n_e=3, *, omitted=True):
                            m1=20, m2=8, n_e=3, pool=None, *, omitted=True):
        ut_c = time.time()
        res_indi = []
        ta_1 = self.count_sing_part1(y, y_hat, positive_label, omitted=omitted)
        ta_2 = self.count_sing_part1(y, y_qtb, positive_label, omitted=omitted)
        ta_3 = [abs(t1 - t2) for t1, t2 in zip(ta_1, ta_2)]
        res_indi.extend(ta_1)  # 13/8
        res_indi.extend(ta_2)  # 13/8
        res_indi.extend(ta_3)
        del ta_1, ta_2, ta_3

        fair_2, fair_3 = [], []
        n_a = len(g1m_indices)
        X_and_y = np.concatenate([y.reshape(-1, 1), X], axis=1, dtype=DTY_FLT).copy()
        X_and_y_hat = np.concatenate([
            y_hat.reshape(-1, 1), X], axis=1, dtype=DTY_FLT).copy()
        for i in range(n_a):
            tmp = self.count_sing_part2(y, y_hat, y_qtb, g1m_indices[i][0],
                                        positive_label, omitted=omitted)
            fair_2.extend(tmp)  # +17/11 <-- 11/5
        if n_a == 1:
            tmp = 11 if omitted else 17  # 5 if omitted else 11
            fair_2.extend([''] * tmp * 3)
        else:
            fair_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[0], positive_label, omitted=omitted))
            fair_2.extend(self.count_sing_part2(
                y, y_hat, y_qtb, jt[1], positive_label, omitted=omitted))
        ut_c = time.time() - ut_c

        A_i = [A[:, i].copy() for i in range(n_a)]
        if n_a > 1:
            A_0 = np.logical_and(A_i[0], A_i[1]).astype(DTY_FLT)
            A_1 = np.logical_or(A_i[0], A_i[1]).astype(DTY_FLT)
        else:
            A_0, A_1 = None, None
        for i in range(n_a):
            fair_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, g1m_indices[i][0], A_i[i], m1, m2, n_e,
                pool))
        if n_a == 1:
            fair_3.extend([''] * 29 * 3)  # 12|8 * 3)
        else:
            fair_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[0], A_j=A_0, m1=m1, m2=m2, n_e=n_e,
                pool=pool))
            fair_3.extend(self.count_sing_part3(
                X_and_y, X_and_y_hat, jt[1], A_j=A_1, m1=m1, m2=m2, n_e=n_e,
                pool=pool))
        del A_0, A_1, A_i    # fair_3.shape= (87,) =(29*3,)

        fair_4 = self.count_sing_part4(
            X_and_y, X_and_y_hat, g1m_indices, A, m1, m2, n_e=n_e,
            pool=pool)
        fair_4.append(ut_c)  # fair_4.shape= (63,) =(10*6+2+1,)
        # if not omitted: .shape= (286,) =(13*3+17*4+ 29*4+63,)  # 236
        # if     omitted: .shape= (247,) =( 8*3+11*4+ 29*4+63,)  # 207
        return res_indi + fair_2 + fair_3 + fair_4

    def prepare_trial(self, omitted=True):
        tmp_p = 247 if omitted else 286  # 207 if omitted else 236
        csv_row_1 = unique_column(11 + 1 + tmp_p * 2)

        tmp_norm = 8 if omitted else 13  # ??
        # tmp_vote = 5 if omitted else 11  # ?*4
        tmp_vote = 11 if omitted else 17   # ?*4
        tmp_manf = 29      # 29*4   =116
        tmp_manf_ext = 10  # 10*6+3 =63
        norm_minus = tmp_norm - 1
        vote_minus = tmp_vote - 1

        tmp_2 = ['fairvote'] + [''] * (tmp_vote * 4 - 1) + ['fairmanf'] + [
            ''] * (29 * 4 - 1) + ['fairmanf_ext'] + [''] * (10 * 6 + 3 - 1)
        tmp_1 = [''] * (tmp_norm * 2 - 1) + ['delta='] + [''] * norm_minus
        csv_row_2c = ['Ensem'] + ['Training set'] + tmp_1 + tmp_2 + [
            'Test set'] + tmp_1 + tmp_2  # siz= 1+(norm*3+vote*4 +29*4+63)*2
        del tmp_1, tmp_2  # each: (norm*3+vote*4 +29*4+63)*2+1 =247|286 *2+1

        tmp_3_1 = ['Normal'] + [''] * norm_minus + ['Adversarial'] + [
            ''] * norm_minus + ['delta= abs(Normal-Adversary)'] + [''] * norm_minus
        tmp_3_2 = ['sen_att #1'] + [''] * vote_minus + ['sen_att #2'] + [
            ''] * vote_minus + ['joint_and'] + [''] * vote_minus + ['joint_or'] + [
            ''] * vote_minus  # vote*4  # ↑ norm*3
        tmp_3_3 = ['sen_att #1'] + [''] * 28 + ['sen_att #2'] + [''] * 28 + [
            'att_#1&2'] + [''] * 28 + ['att_#1|2'] + [''] * 28  # *11
        tmp_3_4 = ['DisDirect multivar'] + [''] * 9 + [
            'sen_att_#1'] + [''] * 9 + ['sen_att_#2'] + [''] * 9  # *13
        tmp_3_5 = ['DistExtend'] + [''] * 9 + ['DistApprox sen_att_#1'] + [
            ''] * 9 + ['DistApprox sen_att_#2'] + [''] * 9 + [
            'TimeCost: DistDirect_multivar, DistExtend+ DistApprox *2',
            '', '']  # siz= 10*3*2+3-1+1 =62+1 =63
        tmp_3 = tmp_3_1 + tmp_3_2 + tmp_3_3 + tmp_3_4 + tmp_3_5
        csv_row_3c = ['Time Cost (sec)'] + tmp_3 + tmp_3
        del tmp_3_1, tmp_3_2, tmp_3_3, tmp_3_4, tmp_3_5

        tmp_42_1 = ['Group fairness: g1,g0 (Gfm pre,manual)', '', '', ''
                    ] if not omitted else []
        tmp_42_2 = ['Group fairness: g1,g0 (Gfm one/two/thr)', '', '', '', '', '']
        tmp_42_3 = ['Gfm,abs: pre,manual', ''] if not omitted else []
        tmp_42_3.extend(['Group fairness: abs(g1-g0)', '', '',
                         'hat_L(loss)', 'hat_L(fair)'])
        tmp_4_2 = tmp_42_1 + tmp_42_2 + tmp_42_3  # 11|17 =0|4+6+0|2+3+2
        tmp_4 = (self._metric_part1 * 3 + tmp_4_2 * 4 + self._metric_part3 * 4 +
                 self._metric_part4 * 6 + [
                     't(DistDirect_multivar)', 't(DistExtend) + t(DistApprox)',
                     't(baseline + fairvote)'])  # *(3-1)
        csv_row_4c = ['ut'] + tmp_4 + tmp_4  # 1+(8|13*3 + 11|17*4 +29*4+63)*2
        del tmp_4_2, tmp_42_1, tmp_42_2, tmp_42_3
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class ComparisonE2_with(ComparisonE_setup):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3, *, omitted=True):
        super().__init__(omitted=omitted)
        self._nb_cls = nb_cls
        self.saIndex = saIndex
        self.saValue = saValue
        self._n_e = n_e

    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2, pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        # res_iter.shape= (3, 1+ 286|247 *2) =(3, 573|495)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[0][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[0][0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
            return res_iter  # shape= (3+4*1, 1+?*2)

        sa_len = len(g1ms_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[i][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[i][0],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
        return res_iter    # shape= (3+4*2, 1+?*2)

    def subroute_one_sens_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst,
            sa_idx, sa_val, positive_label=1, m1=20, m2=8,
            pool=None,
            X_trn=None, A_trn=None, g1ms_trn=None, jt_trn=None,
            X_tst=None, A_tst=None, g1ms_tst=None, jt_tst=None,
            *, omitted=True):
        res_attr = []

        for constraint_type in ['FPR', 'FNR', 'FPR,FNR']:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
                'fairgbm', self._nb_cls,
                X_A_trn, y_trn, X_Aq_trn, nsa_trn,
                X_A_tst, y_tst, X_Aq_tst, nsa_tst, constraint=constraint_type)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
                positive_label, m1, m2, self._n_e, pool, omitted=omitted)
            res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adafair', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, nsa_tst, sa_idx=sa_idx, sa_val=sa_val)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        return res_attr    # shape= (4, 573|495)= (3+1, 1+ 286|247 *2)

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
            # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
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

    def subroute_one_norm_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label=1, m1=20, m2=8,
            pool=None,
            X_trn=None, A_trn=None, X_tst=None, A_tst=None,
            *, omitted=True):
        res_attr = []

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'bagging', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn=None,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'adaboost', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_fair_ens(
            'lightgbm', self._nb_cls,
            X_A_trn, y_trn, X_Aq_trn, None,  # nsa_trn,
            X_A_tst, y_tst, X_Aq_tst, None)  # nsa_tst=None)
        tmp = self.count_scores(
            X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
            X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
            positive_label, m1, m2, self._n_e, pool, omitted=omitted)
        res_attr.append([ut] + tmp)

        # return res_attr  # shape= (3, 467|409)= (3  , 1+ 233|204 *2)
        return res_attr    # shape= (4, 573|495)= (3+1, 1+ 286|247 *2)


class ComparisonE3_with(ComparisonE2_with):
    def __init__(self, nb_cls=1, saIndex=list(), saValue=list(),
                 n_e=3, *, omitted=True):
        super().__init__(nb_cls, saIndex, saValue, n_e, omitted=omitted)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_two_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        #   res_iter.shape= (11+3, 1+ 233|204 *2)

        if len(jt_trn) == 0:
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[0][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[0][0],
                self.saIndex[0], self.saValue[0], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
            return res_iter  # shape= (14+4*1, 1+?*2)

        sa_len = len(g1ms_trn)
        for i in range(sa_len):
            tmp = self.subroute_one_sens_att(
                X_A_trn, y_trn, X_Aq_trn, g1ms_trn[i][0],
                X_A_tst, y_tst, X_Aq_tst, g1ms_tst[i][0],
                self.saIndex[i], self.saValue[i], positive_label, m1, m2,
                pool,
                X_trn, A_trn, g1ms_trn, jt_trn,
                X_tst, A_tst, g1ms_tst, jt_tst, omitted=omitted)
            res_iter.extend(tmp)
        return res_iter  # shape= (14+4*2, 1+?*2)

    def subroute_two_norm_att(
            self,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label=1, m1=20, m2=8,
            pool=None,
            X_trn=None, A_trn=None, X_tst=None, A_tst=None,
            *, omitted=True):
        res_attr = []

        for abbr_cls in self._abbr_clfs:
            y_insp, y_pred, yq_insp, yq_pred, ut = self.subroute_one_gene_clf(
                abbr_cls, X_A_trn, y_trn, X_Aq_trn, X_A_tst, y_tst, X_Aq_tst)
            tmp = self.count_scores(
                X_trn, A_trn, y_trn, y_insp, yq_insp, g1ms_trn, jt_trn,
                X_tst, A_tst, y_tst, y_pred, yq_pred, g1ms_tst, jt_tst,
                positive_label, m1, m2, self._n_e, pool, omitted=omitted)
            res_attr.append([ut] + tmp)

        # return res_attr  # shape= (11, 467|409)= (11, 1+ 233|204 *2)
        return res_attr    # shape= (11, 573|495)= (11, 1+ 286|247 *2)

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


class ComparisonE4_with(ComparisonE3_with):
    def schedule_content(self, logger, pool,
                         X_trn, A_trn, y_trn, Aq_trn, g1ms_trn, jt_trn,
                         X_tst, A_tst, y_tst, Aq_tst, g1ms_tst, jt_tst,
                         m1=20, m2=8, positive_label=None,
                         *, omitted=True):
        X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
        X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
        X_Aq_trn = np.concatenate([X_trn, Aq_trn], axis=1)
        X_Aq_tst = np.concatenate([X_tst, Aq_tst], axis=1)
        return self.schedule_content_prime(
            logger, pool,
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            m1, m2, positive_label, X_trn, A_trn, X_tst, A_tst, omitted=omitted)

    def schedule_content_prime(self, logger, pool,
                               X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
                               X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
                               m1=20, m2=8, positive_label=None,
                               X_trn=None, A_trn=None, X_tst=None, A_tst=None,
                               *, omitted=True):
        res_iter = []
        tmp = self.subroute_two_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)
        tmp = self.subroute_one_norm_att(
            X_A_trn, y_trn, X_Aq_trn, g1ms_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1ms_tst, jt_tst,
            positive_label, m1, m2,  # omitted=omitted)
            pool,
            X_trn, A_trn, X_tst, A_tst, omitted=omitted)
        res_iter.extend(tmp)

        # return res_iter  # .shape= (11+3, 1+ 131|102 *2)
        return res_iter    # .shape= (11+3, 1+ 286|247 *2)


# -------------------------------
#


# ===============================
# fairmanf_ext


# -------------------------------
# RQ5. Will the choice of hyperparameters (that is, m_1 and
#      m_2 in ApproxDist) affect the approximation results,
#      and if the answer is yes, how?
# Parameter Sensitivity


class ParameterE_setup(IndividualClsf):
    def __init__(self, abbr_cls, *, omitted=True):
        super().__init__(abbr_cls)
        self._omit = omitted


class HyperEA_analysis(ParameterE_setup):
    def __init__(self, abbr_cls, *, omitted=True):
        super().__init__(abbr_cls, omitted=omitted)
        self._m2_set = list(range(2, 23, 1))    # len=21
        if omitted:
            self._m2_set = list(range(2, 14, 1))  # len=12

    def schedule_content(self, X, A, y_fx, g1m_indices, m1, n_e,
                         pool=None,
                         alternative=False):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1), X], axis=1, dtype=DTY_FLT).copy()
        n_l = len(self._m2_set)  # n_l, n_ell, ell
        n_a = len(g1m_indices)
        if not alternative:
            curr_res = [self.sub_process_core(
                X_yfx, A[:, i], g1m_indices[i][0], m1, n_e,
                n_l, pool) for i in range(n_a)]
            # curr_res.shape= (n_a, 2,66)
            # i.e., DistDirect_bin, ApproxDist_bin|DistApprox(bin-val)
        else:
            curr_res = [self.subproc_core_alt(
                X_yfx, A[:, i], g1m_indices[i][0], m1, n_e,
                n_l, pool) for i in range(n_a)]
            # curr_res.shape= (n_a, 3,66)
        ans_bin = []
        for i in range(n_a):
            ans_bin.extend(curr_res[i])
        if n_a == 1:
            ans_bin.extend([[''] * (3 + n_l * 3), [''] * (3 + n_l * 3)])
            if alternative:
                ans_bin.append([''] * (3 + n_l * 3))
        # ans_bin.shape= (4,66)  # or alt(6,66)

        # fairmanf_ext, multival, direct calculation
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(X_yfx, g1m_indices)
        if n_a > 1:
            direct_1, direct_2 = zip(*Ds_midtmp)
            ans_direct = [[Ds_01, Ds_avg, t_Ds], list(direct_1), list(direct_2)]
            del direct_2  # del direct_1, direct_2
        elif n_a == 1:
            direct_1, = zip(*Ds_midtmp)
            ans_direct = [[Ds_01, Ds_avg, t_Ds], list(direct_1), [''] * 3]
        del direct_1, Ds_01, Ds_avg, Ds_midtmp, t_Ds

        ans_approx = [DistExtend(X_yfx, A, m1, m2, n_e, pool) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds_01, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx  # hat_Ds_01|avg,ans_ut .shape=(21,)
        # hat_Ds_midtmp.shape=(21, 3, 2=n_a), i.e.,half_max|avg|ut
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)
        ans_approx = [list(hat_Ds_01) + list(hat_Ds_avg) + list(ans_ut)]
        for i in range(n_a):
            tmp = hat_Ds_midtmp[:, 0, i].tolist() + hat_Ds_midtmp[
                :, 1, i].tolist() + hat_Ds_midtmp[:, 2, i].tolist()
            ans_approx.append(tmp)
        if n_a == 1:
            ans_approx.append([''] * n_l * 3)
        ans_multival = [t_dir + t_ap for t_dir, t_ap in zip(ans_direct, ans_approx)]
        del ans_direct, ans_approx  # , t_dir, t_ap  # t_app
        del hat_Ds_01, hat_Ds_avg, ans_ut, hat_Ds_midtmp
        # ans_multival.shape= (3,66) =(DistExtend|DistApprox(sa#?), 3+21*3)

        ans_bin.extend(ans_multival)
        del curr_res, ans_multival
        return ans_bin  # .shape= (7,66) =(n_a*2+3, 3+21*3)

    def subproc_core_alt(self, X_yfx, A_j, non_sa, m1, n_e, n_l,
                         pool=None):
        # fairmanf, bin-val, direct calculation
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, ~non_sa)
        curr_res = [Ds_01, Ds_avg, t_Ds] + [''] * n_l * 3
        curr_res = [curr_res]

        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, non_sa)
        # fairmanf, bin-val
        # ↓ one sensitive attribute, with binay value
        ans_approx = [ApproxDist_bin(
            # X_yfx, A_j, ~non_sa, non_sa, m1, m2) for m2 in self._m2_set]
            X_yfx, A_j, non_sa, m1, m2) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        curr_res.append([Ds_01, Ds_avg, t_Ds
                         ] + list(ans_approx) + [''] * n_l + list(ans_ut))

        # fairmanf_ext, bin-val
        # ↓ one sensitive attribute, with binary value
        B_j = non_sa.astype(DTY_INT)
        # ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e) for m2 in self._m2_set]
        ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e, pool) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds_01, hat_Ds_avg = zip(*ans_approx)
        curr_res.append([''] * 3 + 
                        list(hat_Ds_01) + list(hat_Ds_avg) + list(ans_ut))
        return curr_res  # .shape= (3,66) =(1+2, 3+21*3)

    def sub_process_core(self, X_yfx, A_j, non_sa, m1, n_e, n_l=21,
                         pool=None):
        # fairmanf, bin-val, direct calculation
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, non_sa)
        # fairmanf, bin-val
        # ↓ one sensitive attribute, with binay value
        ans_approx = [ApproxDist_bin(
            # X_yfx, A_j, ~non_sa, non_sa, m1, m2) for m2 in self._m2_set]
            X_yfx, A_j, non_sa, m1, m2) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        curr_res = [Ds_01, Ds_avg, t_Ds] + list(
            ans_approx) + [''] * n_l + list(ans_ut)

        # fairmanf_ext, bin-val
        # ↓ one sensitive attribute, with binary value
        B_j = non_sa.astype(DTY_INT)
        # ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e) for m2 in self._m2_set]
        ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e, pool) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        Ds_01, Ds_avg = zip(*ans_approx)
        curr_res = [curr_res]
        # curr_res.append([''] * 3 + Ds_01 + Ds_avg + ans_ut)
        curr_res.append([''] * 3 + list(Ds_01) + list(Ds_avg) + list(ans_ut))

        '''
        # fairmanf_ext, multival
        # ↓ one sensitive attribute, with multiple values
        ans_approx = [DistApprox(X_yfx, A_j, m1, m2, n_e) for m2 in self._m2_set]
        ans_approx, ans_ut = zip(*ans_approx)
        Ds_01, Ds_avg = zip(*ans_approx)
        curr_res.append([''] * 3 + Ds_01 + Ds_avg + ans_ut)
        '''
        return curr_res  # shape= (2,66) =(|3,3+21*3)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 66)    # =3+63
        if self._omit:
            csv_row_1 = unique_column(11 + 39)  # =3+12*3

        n_l = len(self._m2_set) - 1  # =20
        csv_row_2c = ['direct_01', '(^avg)', ''] + ['approx_01'] + [
            ''] * n_l + ['approx_01^{avg}'] + [''] * n_l + [
            'approx time_cost'] + [''] * n_l
        csv_row_3c = ['', '', 'ut'] + [
            "m2= {}".format(i) for i in self._m2_set] * 3
        return csv_row_1, csv_row_2c, csv_row_3c


class HyperEB_analysis(ParameterE_setup):
    def __init__(self, abbr_cls, *, omitted=True):
        super().__init__(abbr_cls, omitted=omitted)
        self._m1_set = list(range(3, 50, 2))    # len=24
        if omitted:
            self._m1_set = list(range(3, 34, 2))  # len=16

    def schedule_content(self, X, A, y_fx, g1m_indices, m2, n_e,
                         pool=None,
                         alternative=False):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1), X], axis=1, dtype=DTY_FLT).copy()
        n_l = len(self._m1_set)  # n_ell
        n_a = len(g1m_indices)
        if not alternative:
            curr_res = [self.sub_process_core(
                X_yfx, A[:, i], g1m_indices[i][0], m2, n_e,
                n_l, pool) for i in range(n_a)]
            # curr_res.shape= (n_a, 2,75)
            # i.e., DistDirect_bin, ApproxDist_bin|DistApprox(bin-val)
        else:
            curr_res = [self.subproc_core_alt(
                X_yfx, A[:, i], g1m_indices[i][0], m2, n_e,
                n_l, pool) for i in range(n_a)]
            # curr_res.shape= (n_a, 3,75)
        ans_bin = []
        for i in range(n_a):
            ans_bin.extend(curr_res[i])
        if n_a == 1:
            ans_bin.extend([[''] * (3 + n_l * 3), [''] * (3 + n_l * 3)])
            if alternative:
                ans_bin.append([''] * (3 + n_l * 3))
        # ans_bin.shape= (4,75)  # or alt(6,75)

        # fairmanf_ext, multival, direct computation
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(X_yfx, g1m_indices)
        if n_a > 1:
            direct_1, direct_2 = zip(*Ds_midtmp)
            ans_direct = [[Ds_01, Ds_avg, t_Ds], list(direct_1), list(direct_2)]
            del direct_2
        elif n_a == 1:
            direct_1, = zip(*Ds_midtmp)
            ans_direct = [[Ds_01, Ds_avg, t_Ds], list(direct_1), [''] * 3]
        del direct_1, Ds_01, Ds_avg, t_Ds, Ds_midtmp

        ans_approx = [DistExtend(X_yfx, A, m1, m2, n_e, pool) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds_01, hat_Ds_avg, hat_Ds_midtmp = zip(*ans_approx)
        del ans_approx  # hat_Ds_01|avg,ans_ut .shape=(24,)
        # hat_Ds_midtmp.shape=(24, 3, 2=n_a), i.e.,half_max|avg|ut
        hat_Ds_midtmp = np.array(hat_Ds_midtmp)
        ans_approx = [list(hat_Ds_01) + list(hat_Ds_avg) + list(ans_ut)]
        for i in range(n_a):
            tmp = hat_Ds_midtmp[:, 0, i].tolist() + hat_Ds_midtmp[
                :, 1, i].tolist() + hat_Ds_midtmp[:, 2, i].tolist()
            ans_approx.append(tmp)
        if n_a == 1:
            ans_approx.append([''] * n_l * 3)
        ans_multival = [t_dir + t_ap for t_dir, t_ap in zip(ans_direct, ans_approx)]
        del ans_direct, ans_approx
        del hat_Ds_01, hat_Ds_avg, ans_ut, hat_Ds_midtmp
        # ans_multival.shape= (3,75) =(DistExtend|DistApprox(sa#?), 3+24*3)

        ans_bin.extend(ans_multival)
        del curr_res, ans_multival
        return ans_bin  # .shape= (7,75) =(n_a*2+3, 3+24*#)

    def subproc_core_alt(self, X_yfx, A_j, non_sa, m2, n_e, n_l,
                         pool=None):
        # fairmanf, bin-val, direct computation
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, ~non_sa)
        curr_res = [Ds_01, Ds_avg, t_Ds] + [''] * n_l * 3
        curr_res = [curr_res]

        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, non_sa)
        # fairmanf, bin-val
        # ↓ one sensitive attribute, with binary value
        ans_approx = [ApproxDist_bin(
            # X_yfx, A_j, ~non_sa, non_sa, m1, m2) for m1 in self._m1_set]
            X_yfx, A_j, non_sa, m1, m2) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        curr_res.append([Ds_01, Ds_avg, t_Ds
                         ] + list(ans_approx) + [''] * n_l + list(ans_ut))

        # fairmanf_ext, bin-val
        # ↓ one sensitive attribute, with binary value
        B_j = non_sa.astype(DTY_INT)
        ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e, pool) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        hat_Ds_01, hat_Ds_avg = zip(*ans_approx)
        curr_res.append([''] * 3 + 
                        list(hat_Ds_01) + list(hat_Ds_avg) + list(ans_ut))
        return curr_res  # .shape= (3,75) =(1+2, 3+24*3)

    def sub_process_core(self, X_yfx, A_j, non_sa, m2, n_e, n_l=24,
                         pool=None):
        # fairmanf, bin-val, direct computation
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, non_sa)
        # fairmanf, bin-val
        # ↓ one sensitive attribute, with binary value
        ans_approx = [ApproxDist_bin(
            # X_yfx, A_j, ~non_sa, non_sa, m1, m2) for m1 in self._m1_set]
            X_yfx, A_j, non_sa, m1, m2) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        curr_res = [Ds_01, Ds_avg, t_Ds] + list(
            ans_approx) + [''] * n_l + list(ans_ut)

        # fairmanf_ext, bin-val
        # ↓ one sensitive attribute, with binary value
        B_j = non_sa.astype(DTY_INT)
        # ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e) for m1 in self._m1_set]
        ans_approx = [DistApprox(X_yfx, B_j, m1, m2, n_e, pool) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        Ds_01, Ds_avg = zip(*ans_approx)
        curr_res = [curr_res]
        curr_res.append([''] * 3 + list(Ds_01) + list(Ds_avg) + list(ans_ut))

        '''
        # fairmanf_ext, multival
        # ↓ one sensitive attribute, with multiple values
        ans_approx = [DistApprox(X_yfx, A_j, m1, m2, n_e) for m1 in self._m1_set]
        ans_approx, ans_ut = zip(*ans_approx)
        Ds_01, Ds_avg = zip(*ans_approx)
        curr_res.append([''] * 3 + Ds_01 + Ds_avg + ans_ut)
        '''
        return curr_res  # shape= (2,75) =(|3,3+24*3)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 75)    # =3+72
        if self._omit:
            csv_row_1 = unique_column(11 + 51)  # =3+16*3

        n_l = len(self._m1_set) - 1  # 23
        csv_row_2c = ['direct_01', '(^avg)', ''] + ['approx_01'] + [
            ''] * n_l + ['approx_01^{avg}'] + [''] * n_l + [
            'approx time_cost'] + [''] * n_l
        csv_row_3c = ['', '', 'ut'] + [
            "m1= {}".format(i) for i in self._m1_set] * 3
        return csv_row_1, csv_row_2c, csv_row_3c


# -------------------------------
# RQ5.


class ParameterF_setup(RelativeFairClsf, ParameterE_setup):
    def __init__(self, abbr_cls, nb_cls=1, constraint_type='FPR,FNR',
                 saIndex=list(), saValue=list(), *, omitted=True):
        super().__init__(abbr_cls, nb_cls,
                         constraint_type, saIndex, saValue)
        self._raw_abbr_cls = abbr_cls
        self._omit = omitted

    @property
    def raw_abbr_cls(self):
        return self._raw_abbr_cls

    def subproc_calc_df_ver(self, curr_Ds, curr_Df):
        # curr_Ds|Df .shape= (7|9, 3+ 21|24 *3) =(9, 66|75)
        nb_row, nb_col = np.shape(curr_Ds)  # curr_Df
        df_ver3 = np.zeros_like(curr_Ds).tolist()
        df_ver4 = df_ver3.copy()
        for i in range(nb_row):
            for j in range(nb_col):
                if (curr_Ds[i][j] == '') or (curr_Df[i][j] == ''):
                    df_ver3[i][j] = ''
                    df_ver4[i][j] = ''
                    continue
                df_ver3[i][j] = fair_degree_v3(curr_Ds[i][j], curr_Df[i][j])[0]
                df_ver4[i][j] = fair_degree_v4(curr_Ds[i][j], curr_Df[i][j])[0]
        return df_ver3, df_ver4  # .shape= (9, 3+ 63|72)

    def subproc_adversarial(self, y, y_hat, y_qtb, positive_label,
                            g1m_indices, idx_jt, n_l):
        # each part in training | test: 3+n_l*3, i.e., D_max,D_avg,ut
        curr_res = []
        n_ell = 3 + n_l * 3

        tmp_norm = self.sub_sub_proc_part1(y, y_hat, positive_label)
        curr_res.append(tmp_norm + [''] * (n_ell - 8))
        tmp_advs = self.sub_sub_proc_part1(y, y_qtb, positive_label)
        curr_res.append(tmp_advs + [''] * (n_ell - 8))
        tmp = [abs(t1 - t2) for t1, t2 in zip(tmp_norm, tmp_advs)]
        curr_res.append(tmp + [''] * (n_ell - 8))
        del tmp_norm, tmp_advs, tmp

        n_a = len(g1m_indices)
        for i in range(n_a):
            tmp = self.sub_sub_proc_part2(
                y, y_hat, y_qtb, g1m_indices[i][0], positive_label)
            curr_res.append(tmp + [''] * (n_ell - 17))
        if n_a == 1:
            curr_res.append([''] * n_ell)
            curr_res.append([''] * n_ell)
            curr_res.append([''] * n_ell)
            return curr_res
        tmp = self.sub_sub_proc_part2(y, y_hat, y_qtb, idx_jt[0], positive_label)
        curr_res.append(tmp + [''] * (n_ell - 17))
        tmp = self.sub_sub_proc_part2(y, y_hat, y_qtb, idx_jt[1], positive_label)
        curr_res.append(tmp + [''] * (n_ell - 17))
        return curr_res  # .shape= (7, n_ell) =(3+2+2, 3+n_l*3)

    def sub_sub_proc_part1(self, y, y_hat, positive_label):
        tp, fp, fn, tn = contingency_table(y, y_hat, positive_label)
        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        res_indi.append(calc_recall(tp, fp, fn, tn))
        res_indi.append(calc_f1_score(tp, fp, fn, tn))
        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, imba_geometric_mean(sen, spe)])
        res_indi.append(imba_discriminant_power(sen, spe))
        return res_indi  # .shape= (8,) =(4+3+1,)

    def sub_sub_proc_part2(self, y, y_hat, y_qtb, non_sa, positive_label):
        _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(y, y_hat,
                                                       positive_label, non_sa)
        cmp_fair = []
        tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)
        cmp_fair.extend(tmp_0)
        cmp_fair.extend(tmp_1 + tmp_2 + tmp_3)
        cmp_fair.extend(tmp_4)
        cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))
        cmp_fair.append(hat_L_loss(y_hat, y))
        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        return cmp_fair  # .shape= (17,) =(5*2+5+2,)


class HyperFA_analysis(ParameterF_setup, HyperEA_analysis):
    def __init__(self, abbr_cls, nb_cls, constraint_type, saIndex, saValue,
                 *, omitted=True):
        super().__init__(abbr_cls, nb_cls, constraint_type, saIndex, saValue,
                         omitted=omitted)
        self._m2_set = list(range(2, 23, 1))  # siz=21
        if omitted:
            self._m2_set = list(range(2, 14, 1))  # = 12

    def schedule_content_wcf(self,
                             X_trn, A_trn, y_trn, y_insp, g1ms_trn,
                             X_tst, A_tst, y_tst, y_pred, g1ms_tst,
                             m1, n_e, pool=None):
        '''
        X_y_trn = np.concatenate([y_trn.reshape(-1, 1), X_trn], axis=1).copy()
        X_y_tst = np.concatenate([y_tst.reshape(-1, 1), X_tst], axis=1).copy()
        X_yhat_trn = np.concatenate([y_insp.reshape(-1, 1), X_trn],
                                    axis=1).copy()
        X_yhat_tst = np.concatenate([y_pred.reshape(-1, 1), X_tst],
                                    axis=1).copy()
        n_l, n_a = len(self._m2_set), len(g1ms_trn)
        '''
        alternative = True
        gather_trn, gather_tst = [], []

        # TRAINING SET
        curr_ans_Ds = self.schedule_content(
            # X_trn, A_trn, y_trn, g1ms_trn, m1, n_e, alternative)   # (9,66)
            X_trn, A_trn, y_trn, g1ms_trn, m1, n_e, pool, alternative)
        curr_ans_Df = self.schedule_content(
            # X_trn, A_trn, y_insp, g1ms_trn, m1, n_e, alternative)  # (9,66)
            X_trn, A_trn, y_insp, g1ms_trn, m1, n_e, pool, alternative)
        curr_ver3, curr_ver4 = self.subproc_calc_df_ver(curr_ans_Ds, curr_ans_Df)
        gather_trn.extend(curr_ans_Ds)
        gather_trn.extend(curr_ans_Df)
        gather_trn.extend(curr_ver3)
        gather_trn.extend(curr_ver4)
        del curr_ans_Ds, curr_ans_Df, curr_ver3, curr_ver4  # (9*4, 66)

        # TEST SET
        curr_ans_Ds = self.schedule_content(
            X_tst, A_tst, y_tst, g1ms_tst, m1, n_e, pool, alternative)
        curr_ans_Df = self.schedule_content(
            X_tst, A_tst, y_pred, g1ms_tst, m1, n_e, pool, alternative)
        curr_ver3, curr_ver4 = self.subproc_calc_df_ver(curr_ans_Ds, curr_ans_Df)
        gather_tst.extend(curr_ans_Ds)
        gather_tst.extend(curr_ans_Df)
        gather_tst.extend(curr_ver3)
        gather_tst.extend(curr_ver4)
        del curr_ans_Ds, curr_ans_Df, curr_ver3, curr_ver4  # (9*4, 66)

        gather_ans = []
        for curr_trn, curr_tst in zip(gather_trn, gather_tst):
            gather_ans.append(curr_trn + curr_tst)
        del gather_trn, gather_tst, alternative
        # return gather_ans  # .shape= (9*4, 66+66)
        return [gather_ans[:9], gather_ans[9: 18],
                gather_ans[18: 27], gather_ans[27:]]  # .shape= (4, 9, 66*2)

    def prepare_trial(self):
        n_l = 3 * len(self._m2_set) + 3 - 1  # 66|39-1 if omitted
        csv_row_1 = unique_column(12 + (n_l + 1) * 2)
        csv_row_2c = ['Training set'] + [''] * n_l + ['Test set'] + [''] * n_l

        # csv_row_1 = unique_column(12 + 66 * 2)
        n_l = len(self._m2_set) - 1  # =21-1
        # csv_row_2c = ['Training set'] + [''] * 65 + ['Test set'] + [''] * 65

        tmp_3_1 = ['direct_01', '(^avg)', ''] + ['approx_01'] + [''] * n_l + [
            'approx_01^{avg}'] + [''] * n_l + ['approx tim_cost'] + [''] * n_l
        csv_row_3c = tmp_3_1 + tmp_3_1
        tmp_4_1 = ['', '', 'ut'] + ['m2={}'.format(i) for i in self._m2_set] * 3
        csv_row_4c = tmp_4_1 + tmp_4_1
        del tmp_3_1, tmp_4_1, n_l

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


class HyperFB_analysis(ParameterF_setup, HyperEB_analysis):
    def __init__(self, abbr_cls, nb_cls, constraint_type, saIndex, saValue,
                 *, omitted=True):
        super().__init__(abbr_cls, nb_cls, constraint_type, saIndex, saValue,
                         omitted=omitted)
        self._m1_set = list(range(3, 50, 2))  # siz=24
        if omitted:
            self._m1_set = list(range(3, 34, 2))  # = 16

    def schedule_content_wcf(self,
                             X_trn, A_trn, y_trn, y_insp, g1ms_trn,
                             X_tst, A_tst, y_tst, y_pred, g1ms_tst,
                             m2, n_e, pool=None):
        '''
        X_y_trn = np.concatenate([y_trn.reshape(-1, 1), X_trn], axis=1).copy()
        X_y_tst = np.concatenate([y_tst.reshape(-1, 1), X_tst], axis=1).copy()
        X_yhat_trn = np.concatenate([y_insp.reshape(-1, 1), X_trn],
                                    axis=1).copy()
        X_yhat_tst = np.concatenate([y_pred.reshape(-1, 1), X_tst],
                                    axis=1).copy()
        n_l, n_a = len(self._m1_set), len(g1ms_trn)
        '''
        alternative = True
        gather_trn, gather_tst = [], []

        # TRAINING SET
        curr_ans_Ds = self.schedule_content(
            # X_trn, A_trn, y_trn, g1ms_trn, m2, n_e, alternative)   # (9,75)
            X_trn, A_trn, y_trn, g1ms_trn, m2, n_e, pool, alternative)
        curr_ans_Df = self.schedule_content(
            # X_trn, A_trn, y_insp, g1ms_trn, m2, n_e, alternative)  # (9,75)
            X_trn, A_trn, y_insp, g1ms_trn, m2, n_e, pool, alternative)
        curr_ver3, curr_ver4 = self.subproc_calc_df_ver(curr_ans_Ds, curr_ans_Df)
        gather_trn.extend(curr_ans_Ds)
        gather_trn.extend(curr_ans_Df)
        gather_trn.extend(curr_ver3)
        gather_trn.extend(curr_ver4)
        del curr_ans_Ds, curr_ans_Df, curr_ver3, curr_ver4  # (9*4, 75)

        # TEST SET
        curr_ans_Ds = self.schedule_content(
            X_tst, A_tst, y_tst, g1ms_tst, m2, n_e, pool, alternative)
        curr_ans_Df = self.schedule_content(
            X_tst, A_tst, y_pred, g1ms_tst, m2, n_e, pool, alternative)
        curr_ver3, curr_ver4 = self.subproc_calc_df_ver(curr_ans_Ds, curr_ans_Df)
        gather_tst.extend(curr_ans_Ds)
        gather_tst.extend(curr_ans_Df)
        gather_tst.extend(curr_ver3)
        gather_tst.extend(curr_ver4)
        del curr_ans_Ds, curr_ans_Df, curr_ver3, curr_ver4  # (9*4, 75)

        gather_ans = []
        for curr_trn, curr_tst in zip(gather_trn, gather_tst):
            gather_ans.append(curr_trn + curr_tst)
        del gather_trn, gather_tst, alternative
        # return gather_ans  # .shape= (9*4, 75+75)
        return [gather_ans[:9], gather_ans[9: 18],
                gather_ans[18: 27], gather_ans[27:]]  # .shape= (4, 9, 75*2)

    def prepare_trial(self):
        n_l = 3 * len(self._m1_set) + 3 - 1  # 75|51-1 if omitted
        csv_row_1 = unique_column(12 + (n_l + 1) * 2)
        csv_row_2c = ['Training set'] + [''] * n_l + ['Test set'] + [''] * n_l

        # csv_row_1 = unique_column(12 + 75 * 2)
        n_l = len(self._m1_set) - 1  # =24-1
        # csv_row_2c = ['Training set'] + [''] * 74 + ['Test set'] + [''] * 74

        tmp_3_1 = ['direct_01', '(^avg)', ''] + ['approx_01'] + [''] * n_l + [
            'approx_01^{avg}'] + [''] * n_l + ['approx tim_cost'] + [''] * n_l
        csv_row_3c = tmp_3_1 + tmp_3_1
        tmp_4_1 = ['', '', 'ut'] + ['m1={}'.format(i) for i in self._m1_set] * 3
        csv_row_4c = tmp_4_1 + tmp_4_1
        del tmp_3_1, tmp_4_1, n_l

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


# -------------------------------
# RQ 5'.
#   adversarial accuracy, based on F*
# 考虑补充这个实验，要不要和其他 fairness measure 比较呢
# 那就得另列单独一行了
#
# 已经写好了，还是exp6开头，但是注意加上 --rep，就可以视为exp7
#


# -------------------------------
# RQ 7.
#   compared without pool.map, how efficient it is with it
#
# refer to `ParameterE_setup`
#


class ParallelComputing_G_setup(IndividualClsf):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)

    def compare_mp_sing_part3(self, X_yfx, non_sa, A_j, m1, m2, n_e,
                              pool=None):
        curr_ans = []
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, ~non_sa)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])

        # ut_a = time.time()
        (Ds_01, Ds_avg), t_Ds = DistDirect_bin(X_yfx, non_sa)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])
        # idx_sa = ~non_sa
        # Ds_01, t_Ds = ApproxDist_bin(X_yfx, A_j, ~non_sa, non_sa, m1, m2)
        Ds_01, t_Ds = ApproxDist_bin(X_yfx, A_j, non_sa, m1, m2)
        curr_ans.extend([Ds_01, t_Ds])

        B_j = non_sa.astype(DTY_INT)
        (Ds_01, Ds_avg), t_Ds = DistApprox(X_yfx, B_j, m1, m2, n_e, pool)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])
        (Ds_01, Ds_avg), t_Ds = DistApprox(X_yfx, B_j, m1, m2, n_e, None)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])

        return curr_ans  # .shape= (3+11,) =(3+2+3*2,)

    def compare_mp_sing_part4(self, X_yfx, g1m_indices, A, m1, m2,
                              n_e, pool=None):
        curr_ans = []

        ut_a = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistDirect_multivar(X_yfx, g1m_indices)
        n_a = len(g1m_indices)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        for i in range(n_a):
            curr_ans.extend([Ds_01[i], Ds_avg[i], t_Ds[i]])
        if n_a == 1:
            curr_ans.extend(['', ] * 3)
        ut_a = time.time() - ut_a
        # curr_ans.shape= up to now (9,) = (3+3*2,)

        ut_b = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistExtend(X_yfx, A, m1, m2, n_e, pool)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        for i in range(n_a):
            curr_ans.extend([Ds_01[i], Ds_avg[i], t_Ds[i]])
        if n_a == 1:
            curr_ans.extend(['', ] * 3)
        ut_b = time.time() - ut_b
        # curr_ans.shape= up to now (18,)

        ut_c = time.time()
        (Ds_01, Ds_avg, Ds_midtmp), t_Ds = DistExtend(X_yfx, A, m1, m2, n_e, None)
        curr_ans.extend([Ds_01, Ds_avg, t_Ds])
        Ds_01, Ds_avg, t_Ds = Ds_midtmp
        for i in range(n_a):
            curr_ans.extend([Ds_01[i], Ds_avg[i], t_Ds[i]])
        if n_a == 1:
            curr_ans.extend(['', ] * 3)
        ut_c = time.time() - ut_c
        # curr_ans.shape= up to now (27,)

        # 就不管 DistApprox 的加速了，这篇文章里也不这么写了，暂时就先管这个
        # 而且 part3 里算是一半证明了 DistApprox 加速后有多有用了
        curr_ans.extend([ut_a, ut_b, ut_c])
        return curr_ans  # .shape= (30,) =(9*3+3,)


class Multiprocess_GA_comparison(ParallelComputing_G_setup):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)

    def schedule_content(self, X, A, y_fx, g1m_indices, m1, m2,
                         n_e, pool=None):
        X_yfx = np.concatenate([y_fx.reshape(-1, 1),
                                X], axis=1, dtype=DTY_FLT).copy()
        n_a = len(g1m_indices)
        curr_res = []

        for i in range(n_a):
            non_sa = g1m_indices[i][0]
            A_j = A[:, i].copy()
            tmp = self.compare_mp_sing_part3(X_yfx, non_sa, A_j, m1, m2,
                                             n_e, pool)
            curr_res.extend(tmp)
            del A_j, non_sa
        if n_a == 1:
            # curr_ans.extend(['', '', ''])
            curr_res.extend(['', ] * 14)

        tmp = self.compare_mp_sing_part4(X_yfx, g1m_indices, A, m1, m2,
                                         n_e, pool)
        curr_res.extend(tmp)
        # return curr_res  # .shape= (52,) =(11*2+30,)
        return curr_res    # .shape= (58,) =(14*2+30,)

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 52 + 6)

        cr_2_1 = ['fairmanf ECAI implementation (sa#1)', ] + [''] * 13 + [
            'fairmanf ECAI implementation (sa#2)', ] + [''] * 13  # 14*2
        cr_2_3 = ['DistDirect_multivar', ] + [''] * 8 + [
            'DistExtend (.pool =3)'] + [''] * 8 + ['DistExtend w/o'] + [
            ''] * 8 + ['TimeCost (sec)', '', '']  # 9*3+3 =30
        csv_row_2c = cr_2_1 + cr_2_3  # =28+30
        del cr_2_1, cr_2_3

        cr_3_1 = ['DistDirect_bin \'_10', '', ''] + [
            'DistDirect_bin', '', '', 'ApproxDist_bin', ''] + [
            'DistApprox (.pool =3)', '', '', 'DistApprox w/o', '', '']  # 14
        cr_3_3 = ['both', '', '', 'sa#1', '', '', 'sa#2', '', '']       # 9
        csv_row_3c = cr_3_1 * 2 + cr_3_3 * 3 + ['ut', '', '']  # =14*2+27+3
        del cr_3_1, cr_3_3

        # cr_4_1 = ['Ds', 'avg', 't(Ds)', 'Ds', 't(Ds)'] + ['Ds', 'avg', 't(Ds)'] * 2
        cr_4_1 = ['D', '', 't'] + [
            'Ds', 'avg', 't(Ds)', 'Ds', 't(Ds)'] + ['Ds', 'avg', 't(Ds)'] * 2
        cr_4_3 = ['Ds', 'avg', 't(Ds)'] * 3 * 3 + [  # t(DistExtend .None)
            't(DistDirect_multivar)', 't(DistExtend .pool)', 't(DistExtend w/o)']
        csv_row_4c = cr_4_1 * 2 + cr_4_3  # =(3+11)*2+(27+3) =14*2+30=58
        del cr_4_1, cr_4_3

        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c


# -------------------------------
#
