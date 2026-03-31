# coding: utf-8
# Experiments


import time
import numpy as np

from hfm.dist_drt import DirectDist_bin as DirectDist
from hfm.dist_est_bin import ApproxDist_bin as ApproxDist
# from hfm.dist_est_bin import AcceleDist_bin as AcceleDist
from hfm.hfm_df import bias_degree as fair_degree

from hfm.utils.verifiers import unique_column, DTY_FLT
from experiment.utils_learner import IndividualClsf, RelativeFairClsf
from hfm.earlybreak import EffHD_bin


# -------------------------------
# RQ5. Will the choice of hyperparameters (that is, m_1 and
#      m_2 in ApproxDist) affect the approximation results,
#      and if the answer is yes, how?
# Parameter Sensitivity


class PartE_ParaSen(IndividualClsf):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)

    def schedule_subrout(self, X_yfx, idx_S0, idx_S1):
        ans = []  # tuple()
        # '''
        # dist_10, tim_elapsed = DirectDist(X_yfx, idx_S1, idx_S0)
        # ans.extend([dist_10, tim_elapsed])
        # dist_01, tim_elapsed = DirectDist(X_yfx, idx_S0, idx_S1)
        # ans.extend([dist_01, tim_elapsed])
        # '''
        dist_10, tim_elapsed = DirectDist(X_yfx, idx_S0)
        ans.extend([dist_10, tim_elapsed])
        dist_01, tim_elapsed = DirectDist(X_yfx, idx_S1)
        ans.extend([dist_01, tim_elapsed])
        return ans  # shape= (4,)

    # def prepare_trial(self):
    #   raise NotImplementedError


class PartE1_ParaSenAnalysis(PartE_ParaSen):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)
        self._m2_set = list(range(2, 23, 1))  # len=21

    def schedule_content(self, X, A, y_fx, idx_S0, idx_S1, m1):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        # X_yfx = np.concatenate([y_fx.reshape(-1, 1), X], axis=1,
        #                        dtype=DTY_FLT)
        # ans_app = [ApproxDist(
        #     X_yfx, A, idx_S0, idx_S1, m1, m2) for m2 in self._m2_set]
        ans_app = [ApproxDist(
            X_yfx, A, idx_S1, m1, m2) for m2 in self._m2_set]
        ans_app, ans_ut = zip(*ans_app)
        ans_dis = self.schedule_subrout(X_yfx, idx_S0, idx_S1)
        return ans_dis + list(ans_app) + list(ans_ut)  # 4+21*2 =46

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 3 + 46)
        csv_row_2c = ['direct_s10', '', 'direct_01', ''] + [
            'approx_01'] + [''] * 20 + ['approx time_cost'] + [''] * 20
        # csv_row_3c = ['', 'ut'] * 2 + self._m2_set * 2
        csv_row_3c = ['', 'ut'] * 2 + [
            "m2= {}".format(i) for i in self._m2_set] * 2
        return csv_row_1, csv_row_2c, csv_row_3c


class PartE2_ParaSenAnalysis(PartE_ParaSen):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)
        self._m1_set = list(range(3, 50, 2))  # len=24

    def schedule_content(self, X, A, y_fx, idx_S0, idx_S1, m2):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        # X_yfx = np.concatenate([y_fx.reshape(-1, 1), X], axis=1,
        #                        dtype=DTY_FLT)
        ans_app = [ApproxDist(
            X_yfx, A, idx_S1, m1, m2) for m1 in self._m1_set]
        ans_app, ans_ut = zip(*ans_app)
        ans_dis = self.schedule_subrout(X_yfx, idx_S0, idx_S1)
        return ans_dis + list(ans_app) + list(ans_ut)  # 4+24*2 =52

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 3 + 52)
        csv_row_2c = ['direct_s10', '', 'direct_01', ''] + [
            'approx_01'] + [''] * 23 + ['approx time_cost'] + [''] * 23
        # csv_row_3c = ['', 'ut'] * 2 + self._m1_set * 2
        csv_row_3c = ['', 'ut'] * 2 + [
            "m1= {}".format(i) for i in self._m1_set] * 2
        return csv_row_1, csv_row_2c, csv_row_3c


class PartE3_ParamsSen(PartE_ParaSen):
    def __init__(self, abbr_cls):
        super().__init__(abbr_cls)
        self._m2_set = list(range(2, 12, 1))  # 2, 23, 1)) # siz 10
        self._m1_set = list(range(3, 23, 2))  # 3, 50, 2)) # siz 10

    def schedule_content(self, X, A, y_fx, idx_S0, idx_S1):
        X_yfx = np.concatenate([
            y_fx.reshape(-1, 1).astype(DTY_FLT), X], axis=1)
        # X_yfx = np.concatenate([y_fx.reshape(-1, 1), X], axis=1,
        #                        dtype=DTY_FLT)
        ans_app_m1, ans_ut = [], []
        for m1 in self._m1_set:
            ans_app_m2 = [ApproxDist(
                X_yfx, A, idx_S1, m1, m2) for m2 in self._m2_set]
            ans_app_m2, tmp_ut = zip(*ans_app_m2)
            ans_app_m1.append(list(ans_app_m2))
            ans_ut.append(list(tmp_ut))
            del ans_app_m2, tmp_ut
        ans_dis = self.schedule_subrout(X_yfx, idx_S0, idx_S1)
        return [ans_dis, ans_app_m1, ans_ut]

    def prepare_trial(self):
        csv_row_1 = unique_column(11 + 24)  # 4+10*2 =24
        csv_row_2c = ['direct_s1s0', '', 'direct_01', ''] + [
            'approx_01'] + [''] * 9 + ['approx time_cost'] + [''] * 9
        csv_row_3c = ['', 'ut'] * 2 + [
            'm2= {}'.format(i) for i in self._m2_set] * 2
        return csv_row_1, csv_row_2c, csv_row_3c


# -------------------------------
# RQ5-->RQ6. Parameter Sensitivity


class PartF_ParaSen(RelativeFairClsf, PartE_ParaSen):
    def __init__(self, abbr_cls, nb_cls=1,
                 constraint_type='FPR,FNR',
                 # saIndex=list(), saValues=list()):
                 saIndex=tuple(), saValues=tuple()):
        super().__init__(
            abbr_cls, nb_cls, constraint_type, saIndex, saValues)

    def schedule_subrout(self,
                         X_and_y_trn, X_and_y_insp, s0_trn, s1_trn,
                         X_and_y_tst, X_and_y_pred, s0_tst, s1_tst):
        since = time.time()
        ans = []

        Ds_01, _ = DirectDist(X_and_y_trn, s1_trn)   # s0_trn,
        Df_01, _ = DirectDist(X_and_y_insp, s1_trn)  # s0_trn,
        Ds_01, Df_01 = Ds_01[0], Df_01[0]
        tmp, _ = fair_degree(Ds_01, Df_01)
        ans.extend((Ds_01, Df_01) + tmp)  # df.v3.v4  # siz=4

        Ds_01, _ = DirectDist(X_and_y_tst, s1_tst)   # s0_tst,
        Df_01, _ = DirectDist(X_and_y_pred, s1_tst)  # s0_tst,
        Ds_01, Df_01 = Ds_01[0], Df_01[0]
        tmp, _ = fair_degree(Ds_01, Df_01)
        ans.extend((Ds_01, Df_01) + tmp)  # df.v3.v4  # siz=4

        tim_elapsed = time.time() - since
        ans.append(tim_elapsed)
        return ans  # shape= (9,)


class PartF1_ParaSenAnalysis(PartF_ParaSen):
    def schedule_content(self,
                         X_trn, A_trn, y_trn, y_insp, s0_trn, s1_trn,
                         X_tst, A_tst, y_tst, y_pred, s0_tst, s1_tst,
                         m1):
        X_and_y_trn = np.concatenate([
            y_trn.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        X_and_y_tst = np.concatenate([
            y_tst.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        X_and_y_insp = np.concatenate([
            y_insp.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        X_and_y_pred = np.concatenate([
            y_pred.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        # X_and_y_trn = np.concatenate([
        #     y_trn.reshape(-1, 1), X_trn], axis=1, dtype=DTY_FLT)
        # X_and_y_tst = np.concatenate([
        #     y_tst.reshape(-1, 1), X_tst], axis=1, dtype=DTY_FLT)
        # X_and_y_insp = np.concatenate([
        #     y_insp.reshape(-1, 1), X_trn], axis=1, dtype=DTY_FLT)
        # X_and_y_pred = np.concatenate([
        #     y_pred.reshape(-1, 1), X_tst], axis=1, dtype=DTY_FLT)
        ans_app = []

        for m2 in self._m2_set:
            ans = []
            since = time.time()

            Ds_01, _ = ApproxDist(X_and_y_trn, A_trn, s1_trn,
                                  m1, m2)  # s0_trn,
            Df_01, _ = ApproxDist(X_and_y_insp, A_trn, s1_trn,
                                  m1, m2)  # s0_trn,
            tmp, _ = fair_degree(Ds_01, Df_01)
            ans.extend((Ds_01, Df_01) + tmp)

            Ds_01, _ = ApproxDist(X_and_y_tst, A_tst, s1_tst,
                                  m1, m2)  # s0_tst,
            Df_01, _ = ApproxDist(X_and_y_pred, A_tst, s1_tst,
                                  m1, m2)  # s0_tst,
            tmp, _ = fair_degree(Ds_01, Df_01)
            ans.extend((Ds_01, Df_01) + tmp)

            tim_elapsed = time.time() - since
            ans.append(tim_elapsed)
            ans_app.append(ans)

        ans_dis = self.schedule_subrout(
            X_and_y_trn, X_and_y_insp, s0_trn, s1_trn,
            X_and_y_tst, X_and_y_pred, s0_tst, s1_tst)
        return ans_dis, ans_app  # shape= (1+21, 9 =4*2+1)

    def prepare_trial(self):
        self._m2_set = list(range(2, 23, 1))  # siz=21
        # csv_row_1 = unique_column(11 + 154)  # (1+21)*7
        csv_row_1 = unique_column(11 + 198)    # (1+21)*9
        csv_row_2c = ['DirectDist'] + [''] * 8
        csv_row_3c = ['Ds:trn', 'Df:trn', 'df:trn', '',
                      'Ds:tst', 'Df:tst', 'df:tst', '', 'time_cost']
        for m2 in self._m2_set:
            csv_row_2c.extend(['Approx(m2={})'.format(m2)] + [''] * 8)
            csv_row_3c.extend(['trn', '', '', '',
                               'tst', '', '', '', 'ut'])
        return csv_row_1, csv_row_2c, csv_row_3c


class PartF2_ParaSenAnalysis(PartF_ParaSen):
    """
    def schedule_content(self,  # m2,
                         X_trn, A_trn, y_trn, y_insp, s0_trn, s1_trn,
                         X_tst, A_tst, y_tst, y_pred, s0_tst, s1_tst,
                         m2):
        ans_app = []

        for m1 in self._m1_set:
            ans = []
            since = time.time()

            Ds_01 = ApproxDist(X_trn, A_trn, y_trn, s1_trn,
                                  m1, m2)  # s0_trn,
            Df_01 = ApproxDist(X_trn, A_trn, y_insp, s1_trn,
                                  m1, m2)  # s0_trn,
            tmp = fair_degree_v2(Ds_01, Df_01)
            ans.extend([Ds_01, Df_01, tmp])

            Ds_01 = ApproxDist(X_tst, A_tst, y_tst, s1_tst,
                                  m1, m2)  # s0_tst,
            Df_01 = ApproxDist(X_tst, A_tst, y_pred, s1_tst,
                                  m1, m2)  # s0_tst,
            tmp = fair_degree_v2(Ds_01, Df_01)
            ans.extend([Ds_01, Df_01, tmp])

            tim_elapsed = time.time() - since
            ans.append(tim_elapsed)
            ans_app.append(ans)

        ans_dis = self.schedule_subroutine(
            X_trn, A_trn, y_trn, y_insp, s0_trn, s1_trn,
            X_tst, A_tst, y_tst, y_pred, s0_tst, s1_tst)
        return ans_dis, ans_app  # siz=(24+1,7)
    """

    def schedule_content(self,
                         X_trn, A_trn, y_trn, y_insp, s0_trn, s1_trn,
                         X_tst, A_tst, y_tst, y_pred, s0_tst, s1_tst,
                         m2):
        X_and_y_trn = np.concatenate([
            y_trn.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        X_and_y_tst = np.concatenate([
            y_tst.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        X_and_y_insp = np.concatenate([
            y_insp.reshape(-1, 1).astype(DTY_FLT), X_trn], axis=1)
        X_and_y_pred = np.concatenate([
            y_pred.reshape(-1, 1).astype(DTY_FLT), X_tst], axis=1)
        # X_and_y_trn = np.concatenate([
        #     y_trn.reshape(-1, 1), X_trn], axis=1, dtype=DTY_FLT)
        # X_and_y_tst = np.concatenate([
        #     y_tst.reshape(-1, 1), X_tst], axis=1, dtype=DTY_FLT)
        # X_and_y_insp = np.concatenate([
        #     y_insp.reshape(-1, 1), X_trn], axis=1, dtype=DTY_FLT)
        # X_and_y_pred = np.concatenate([
        #     y_pred.reshape(-1, 1), X_tst], axis=1, dtype=DTY_FLT)
        ans_app = []

        for m1 in self._m1_set:
            ans = []
            since = time.time()

            Ds_01, _ = ApproxDist(X_and_y_trn, A_trn, s1_trn,
                                  m1, m2)  # s0_trn,
            Df_01, _ = ApproxDist(X_and_y_insp, A_trn, s1_trn,
                                  m1, m2)  # s0_trn,
            tmp, _ = fair_degree(Ds_01, Df_01)
            ans.extend((Ds_01, Df_01) + tmp)

            Ds_01, _ = ApproxDist(X_and_y_tst, A_tst, s1_tst,
                                  m1, m2)  # s0_tst,
            Df_01, _ = ApproxDist(X_and_y_pred, A_tst, s1_tst,
                                  m1, m2)  # s0_tst,
            tmp, _ = fair_degree(Ds_01, Df_01)
            ans.extend((Ds_01, Df_01) + tmp)

            tim_elapsed = time.time() - since
            ans.append(tim_elapsed)
            ans_app.append(ans)

        # pdb.set_trace()
        ans_dis = self.schedule_subrout(
            X_and_y_trn, X_and_y_insp, s0_trn, s1_trn,
            X_and_y_tst, X_and_y_pred, s0_tst, s1_tst)
        return ans_dis, ans_app  # shape= (1+24, 9 =4*2+1)

    def prepare_trial(self):
        self._m1_set = list(range(3, 50, 2))  # siz=24
        # csv_row_1 = unique_column(11 + 175)  # (1+24)*7
        csv_row_1 = unique_column(11 + 225)    # (1+24)*9
        csv_row_2c = ['DirectDist'] + [''] * 8
        csv_row_3c = ['Ds:trn', 'Df:trn', 'df:trn', '',
                      'Ds:tst', 'Df:tst', 'df:tst', '', 'time_cost']
        for m1 in self._m1_set:
            csv_row_2c.extend(['Approx(m1={})'.format(m1)] + [''] * 8)
            csv_row_3c.extend(['trn', '', '', '',
                               'tst', '', '', '', 'ut'])
        return csv_row_1, csv_row_2c, csv_row_3c


# -------------------------------
# RQ1. Can ApproxDist approximate the direct computation of
#      distances in Eq.(5) precisely?


# -------------------------------
# RQ2. How efficient is ApproxDist compared with the direct
#      computation of distances in Eq.(5)?


# -------------------------------
# RQ3. Considering the comparison with the state-of-the-art
#      (SOTA) baseline fairness measures, does the proposed
#      HFM capture the discriminative degree of one classi-
#      fier effectively?


# -------------------------------
# RQ4.


# -------------------------------
#
