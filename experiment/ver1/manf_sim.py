# coding: utf-8


from copy import deepcopy
import csv
import json
import os
import sys
import time
import numpy as np
import pandas as pd

from hfm.utils.decorators import (elegant_dated, elegant_durat,
                                  elegant_durat_core)
from hfm.utils.recorders import elegant_print, get_elogger
from experiment.generic import DataSetup
from experiment.classifiers import AVAILABLE_CLFS, AVAILABLE_ENSF
from experiment.datasets import preprocess, adversarial, transform_X_and_y

from experiment.utils.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    manual_repetitive, scale_normalize_helper)
from experiment.ver1.manf_data import (
    binarized_data_set, transform_X_A_and_y, transform_unpriv_tag,
    transform_disturb_prime, normalise_disturb_whole)
from experiment.ver2.mext_data import (
    renewed_transform_disturb, renewed_normalise_disturb,
    renewed_normalise_separate)

from experiment.ver1.manf_hpm import (
    PartE1_ParaSenAnalysis, PartE2_ParaSenAnalysis, PartE3_ParamsSen,
    PartF1_ParaSenAnalysis, PartF2_ParaSenAnalysis)
from experiment.ver1.manf_exp import (
    ComparisonB1_withDirectComput, ComparisonB2_withDirectComput,
    ComparisonC2_withDirectComput,
    ComparisonC4_withDirectComput, ComparisonC5_withDirectComput)


# ===============================
# Empirical


# -------------------------------
# Maniford empirical


class ManfEmpirical(DataSetup):
    def __init__(self, trial_type, data_type,
                 abbr_cls='DT', nb_iter=5, gen=False, rep=False,
                 m1=30, m2=10, nb_cls=1,
                 ratio=.5, constraint_type='FPR,FNR',
                 prep=False, screen=True, logged=False):
        super().__init__(data_type)
        self._ratio = ratio
        self.preparing_iterator(trial_type, abbr_cls,
                                nb_iter, gen, rep, m1, m2,
                                nb_cls, constraint_type,
                                prep, screen, logged)

    def preparing_iterator(self, trial_type, abbr_cls,
                           nb_iter, gen, rep,
                           m1, m2, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        self._trial_type = trial_type
        self._abbr_cls = abbr_cls
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep     # cv-split
        self._prep = prep        # pre-processing data

        self._log_document = "_".join([
            trial_type,
            "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
            self._prep, abbr_cls, self._log_document, 'pms'])
        self._screen, self._logged = screen, logged
        self._m1, self._m2 = m1, m2
        # self.prepare_iterator(trial_type, m1, m2)

        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        self.saIndex = [-1] if self._data_type == 'ricci' else [-2, -1]

        if trial_type.endswith('expt5a'):
            self._log_document += '_ma{}'.format(m1)
            self._iterator = PartE1_ParaSenAnalysis(abbr_cls)
        elif trial_type.endswith('expt5b'):
            self._log_document += '_mb{}'.format(m2)
            self._iterator = PartE2_ParaSenAnalysis(abbr_cls)
        elif trial_type.endswith('expt5c'):
            self._iterator = PartE3_ParamsSen(abbr_cls)
        elif trial_type.endswith('expt6a'):
            self._iterator = PartF1_ParaSenAnalysis(
                abbr_cls, nb_cls,
                constraint_type, self.saIndex, self.saValue)
        elif trial_type.endswith('expt6b'):
            self._iterator = PartF2_ParaSenAnalysis(
                abbr_cls, nb_cls,
                constraint_type, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2a'):
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

        if trial_type[-6:] in ['expt5a', 'expt5b', 'expt5c']:
            self._log_document = self._log_document.replace(
                '_{}'.format(self._abbr_cls), '')
        elif trial_type[-6:] in ['expt2a', 'expt2b', 'expt2c',
                                 'expt2d', 'expt2e']:
            self._log_document = self._data_type  # .copy()
            self._log_document = self._log_document.replace(
                'propublica-violent-recidivism', 'ppvr')
            self._log_document = self._log_document.replace(
                'propublica-recidivism', 'ppr')
            self._log_document = "_".join([
                trial_type,
                'iter{}'.format(nb_iter) if nb_iter > 0 else 'sing',
                'cls{}'.format(nb_cls), prep, self._log_document,
                'pms', 'ratio{}'.format(int(self._ratio * 100)), ])
            #
        if abbr_cls in AVAILABLE_ENSF:
            self._log_document += '_cls{}'.format(nb_cls)
        if abbr_cls == 'FairGBM':
            self._log_document += '_{}'.format(constraint_type)

        # self._log_document += ('_gen' if gen else '')
        # self._log_document += ('_rep' if rep else '')
        self._log_document += ('_gen' * gen + '_rep' * rep)
        return

    def trial_one_process(self, mode="w"):
        since = time.time()
        csv_t = open(self._log_document + ".csv", mode)
        csv_w = csv.writer(csv_t)
        if mode == "a":
            csv_w.writerows([[''], [''], [''], ['']])

        # if (not self._screen) and (not self._logged):
        if not (self._screen or self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + ".log", "w")
            sys.stdout = fsock
        if self._logged:
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            # logger, formatter, file_handler = get_elogger(
            #     "fairmanf", self._log_document + ".txt")
            logger = get_elogger(
                "fairmanf", self._log_document + ".txt")
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
            "\t  constr = {}".format(self._constraint_type),
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
        # if self._logged:
        #     rm_ehandler(logger, formatter, file_handler)
        # else:
        #     del logger
        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    # EACH SUBROUTE

    def coding_per_procedure(self, csv_w, logger):
        csv_row_2a = ['data_name', 'binary', 'abbr_cls', 'nb_iter',
                      'gen', 'rep/cvs', 'm1', 'm2']
        csv_row_2b = ['#sens_attr', '#iter', '#eval']
        if not 'expt2' in self._trial_type:
            csv_row_1, csv_r2c, csv_r3c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            csv_w.writerows([csv_row_1, csv_row_2, [''] * 11 + csv_r3c])

        else:
            csv_row_2b = ['#sens_att', 'fair_ens', '#iter']
            csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            csv_w.writerows([csv_row_1,
                             csv_row_2,
                             [''] * 11 + csv_r3c, [''] * 11 + csv_r4c])

            del csv_r4c
        del csv_r3c, csv_r2c, csv_row_2, csv_row_1, csv_row_2b, csv_row_2a

        # START

        res_data, res_aux = self.coding_per_dataset(logger)
        json_saver = json.dumps({
            "res_aux": res_aux, "res_data": res_data})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w  # shape (1|nb_iter, 1|4, ?)

        csv_w.writerow(res_aux[0])
        if self._trial_type[-6:] in ['expt5a', 'expt5b']:
            for t, tmp in enumerate(res_aux[3]):
                k = 0
                csv_w.writerow([''] * 8 + [tmp, k, ''] + res_data[k][t])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 9 + [k, ''] + res_data[k][t])
        elif self._trial_type.endswith('expt5c'):
            len_m1, len_m2 = len(res_aux[-2]), len(res_aux[-1])
            for t, tmp in enumerate(res_aux[3]):
                for k in range(1 if self._nb_iter <= 0 else self._nb_iter):
                    csv_w.writerow([''] * 8 + [tmp, k, ''] + res_data[k][t][0])
                    for tk in range(len_m1):
                        csv_w.writerow(  # [''] * 11
                            [''] * 6 + [res_aux[-2][tk], ''] + [''] * 7 +
                            res_data[k][t][1][tk] + res_data[k][t][2][tk])
            del len_m1, len_m2
        elif self._trial_type[-6:] in ['expt6a', 'expt6b']:
            # siz= (5, 1|4, 1+21|24, 7)
            for t, tmp in enumerate(res_aux[3]):
                k = 0
                csv_w.writerow([''] * 8 + [tmp, k, ''] + np.array(
                    res_data[k][t]).reshape(-1).tolist())
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 9 + [k, ''] + np.array(
                        res_data[k][t]).reshape(-1).tolist())

        elif self._trial_type[-6:] in ['expt2a', 'expt2b']:
            # elif self._trial_type.endswith('expt2'):  # siz=(5,2,7,77)
            sens_att = res_aux[3][: 2]
            fair_ens = res_aux[-1]  # i.e., res_aux[4]
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens):
                    k = 0
                    csv_w.writerow([''] * 8 + [tmp_a, tmp_b, k] + res_data[k][t_a][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_a][t_b])
            del sens_att, fair_ens

        elif self._trial_type[-6:] in ['expt2c']:
            sens_att = res_aux[3][: 2]
            # siz= (5, 14+4*?, 199)= (5, 11+3+4+4, 199)
            fair_ens = res_aux[-2]  # i.e. res_aux[4]; len=3+4
            abbr_clf = res_aux[-1]  # i.e. res_aux[5]; len=11
            for t_b, tmp_b in enumerate(abbr_clf):
                csv_w.writerow([''] * 8 + ['---', tmp_b, 0] + res_data[0][t_b])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b])
            for t_b, tmp_b in enumerate(fair_ens[: 3]):
                csv_w.writerow([''] * 8 + ['', tmp_b, 0] + res_data[0][t_b + 11])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b + 11])
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens[3:]):
                    csv_w.writerow([''] * 8 + [
                        tmp_a, tmp_b, 0] + res_data[0][t_b + 14 + t_a * 4])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([
                            ''] * 8 + ['', '', k] + res_data[k][t_b + 14 + t_a * 4])
            del sens_att, fair_ens, abbr_clf

        elif self._trial_type[-6:] in ('expt2d', 'expt2e'):
            sens_att = res_aux[3][: 2]
            abbr_clf = res_aux[-1]
            if self._trial_type.endswith('expt2d'):
                for t_b, tmp_b in enumerate(abbr_clf):
                    csv_w.writerow([''] * 8 + ['---', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b])
            elif self._trial_type.endswith('expt2e'):
                for t_b, tmp_b in enumerate(abbr_clf[: 3]):
                    csv_w.writerow([''] * 8 + ['', tmp_b, 0] + res_data[0][t_b])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b])
                for t_a, tmp_a in enumerate(sens_att):
                    for t_b, tmp_b in enumerate(abbr_clf[3:]):
                        csv_w.writerow([''] * 8 + [
                            tmp_a, tmp_b, 0] + res_data[0][t_b + 3 + t_a * 4])
                        for k in range(1, self._nb_iter):
                            csv_w.writerow([''] * 8 + [
                                '', '', k] + res_data[k][t_b + 3 + t_a * 4])
            del sens_att, abbr_clf

        else:
            pass
        return

    def coding_per_dataset(self, logger):
        X, A, y, idx_g1, idx_jt, new_attr, \
            res_aux, Aq = self.preparing_current_data(logger)

        if self._trial_type[-6:] in ('expt5a', 'expt6a'):
            res_aux.append(self._iterator._m2_set)
        elif self._trial_type[-6:] in ('expt5b', 'expt6b'):
            res_aux.append(self._iterator._m1_set)
        elif self._trial_type.endswith('expt5c'):
            res_aux.append(self._iterator._m1_set)
            res_aux.append(self._iterator._m2_set)
        elif 'expt2' in self._trial_type:  # .endswith():
            res_aux.append(['bagging', 'adaboost', 'lightgbm',
                            'fairgbm : FPR', 'fairgbm : FNR',
                            'fairgbm : FPR,FNR', 'adafair'])
            if 'expt2c' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs)
            elif 'expt2d' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs + [
                    'bagging', 'adaboost'])
            elif 'expt2e' in self._trial_type:
                res_aux.append(['bagging', 'adaboost', 'lightgbm'] +
                               ['fairgbm : fpr', 'fairgbm : fnr',
                                'fairgbm : fpr,fnr', 'adafair'])  # *2)

        if self._nb_iter <= 0:  # not cv_split
            elegant_print("Running /executing as a whole", logger)
            if self._prep != 'none':
                scaler = scale_normalize_helper(self._prep)
                scaler, Xp, Ap = normalise_disturb_whole(scaler, X, A)
            else:
                Xp, Ap = X, A
            res_data = self.coding_per_iteration_as_whole(
                logger, Xp, Ap, y, idx_g1, idx_jt)
            return [res_data], res_aux

        elegant_print("Not-repetitively, via cv_split?: {}".format(
            'Yes' if self._rep_iter else 'No'), logger)  # cv_split
        if not self._rep_iter:
            elegant_print("nb_iter={}, repetitive".format(
                self._nb_iter), logger)
            split_idx = manual_repetitive(self._nb_iter, y,
                                          self._gen_iter)
            res_ans = []
            for k, idx in enumerate(split_idx):
                elegant_print("Iteration {}-th starts.".format(
                    k + 1), logger)
                if self._prep != 'none':
                    scaler = scale_normalize_helper(self._prep)
                    scaler, Xp, Ap = normalise_disturb_whole(
                        scaler, X, A)
                else:
                    Xp, Ap = X, A

                ptb = [] if new_attr is None else [
                    tmp[idx] for tmp in idx_jt]
                res_iter = self.coding_per_iteration_as_whole(
                    logger, Xp[idx], Ap[idx], y[idx], [
                        tmp[idx] for tmp in idx_g1], ptb)
                res_ans.append(res_iter)
                elegant_print("Iteration {}-th done.".format(
                    k + 1), logger)
            return res_ans, res_aux

        elegant_print(
            "nb_iter={}, cv_split / cross_valid".format(self._nb_iter), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
            elegant_print("\t CrossValid  {}\n".format('mCV'), logger)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
            elegant_print("\t CrossValid  {}\n".format('KFS'), logger)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
            elegant_print("\t CrossValid  {}\n".format('KF'), logger)
        else:
            raise ValueError("No proper CV (cross-validation).")

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            X_trn, A_trn, y_trn, g1_trn, jt_trn = transform_disturb_prime(
                X, A, y, i_trn, idx_g1, idx_jt)  # belongs_priv, ptb_with_joint)
            X_tst, A_tst, y_tst, g1_tst, jt_tst = transform_disturb_prime(
                X, A, y, i_tst, idx_g1, idx_jt)  # belongs_priv, ptb_with_joint)

            if isinstance(X_trn, pd.DataFrame):
                Aq_trn = Aq.iloc[i_trn]
                Aq_tst = Aq.iloc[i_tst]
            elif isinstance(X_trn, np.ndarray):
                Aq_trn = Aq[i_trn]
                Aq_tst = Aq[i_tst]

            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                scaler, X_trn, A_trn, _, _, X_tst, A_tst = normalise_disturb_prime(
                    scaler, X_trn, A_trn, [], [], X_tst, A_tst)

            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.coding_per_iteration_cv_split(
                logger, k,
                X_trn, A_trn, y_trn, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, g1_tst, jt_tst,
                Aq_trn, Aq_tst)
            res_ans.append(res_iter)  # siz=(5,1|4,1+21|24,7)
            # for `_expt2`: siz= (5, 2, 7, 77)
            del X_trn, A_trn, y_trn, g1_trn, jt_trn
            del X_tst, A_tst, y_tst, g1_tst, jt_tst
        del X, A, y, idx_g1, idx_jt
        return res_ans, res_aux

    def coding_per_iteration_as_whole(self, logger,
                                      X, A, y, idx_g1, idx_jt):
        since = time.time()
        res_iter = []

        if self._trial_type[-6:] in ['expt5a', 'expt5b', 'expt5c']:
            pm_m = {}
            if self._trial_type.endswith('5a'):
                pm_m['m1'] = self._m1
            elif self._trial_type.endswith('5b'):
                pm_m['m2'] = self._m2
            # pm_m = self._m1 if self._trial_type.endswith('5a') else self._m2
            tmp = self._iterator.schedule_content(
                X, A, y, ~idx_g1[0], idx_g1[0], **pm_m)  # ,pm_m)
            res_iter.append(tmp)
            if len(idx_g1) == 1:
                tim_elapsed = time.time() - since
                elegant_print("As a whole, consumed {}".format(
                    elegant_durat_core(tim_elapsed, True)), logger)
                return res_iter
            tmp = self._iterator.schedule_content(
                X, A, y, ~idx_g1[1], idx_g1[1], **pm_m)
            res_iter.append(tmp)
            for jt in idx_jt:
                tmp = self._iterator.schedule_content(
                    X, A, y, ~jt, jt, **pm_m)
                res_iter.append(tmp)
            # res_iter: list, shape=(1,?) or (4,?), each with 46/52 elements

            # elif self._trial_type[-6:] in ['expt5c']:
            # tmp = self._iterator.schedule_content(X, A, y, ~idx_g1[0], idx_g1[0])
            # res_iter: list with size 1|4, each size of 3: [4,]+[10,10]*2

        else:
            pass

        tim_elapsed = time.time() - since
        elegant_print("As a whole, consumed {}".format(
            elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_trn, A_trn, y_trn, g1_trn, jt_trn,
            X_tst, A_tst, y_tst, g1_tst, jt_tst,
            Aq_trn=None, Aq_tst=None):
        since = time.time()
        res_iter = []

        if self._trial_type[-6:] in ['expt6a', 'expt6b', 'expt6c']:
            pm_m = {}
            if self._trial_type.endswith('6a'):
                pm_m['m1'] = self._m1
            elif self._trial_type.endswith('6b'):
                pm_m['m2'] = self._m2

            X_A_trn = np.concatenate([X_trn, A_trn], axis=1)
            X_A_tst = np.concatenate([X_tst, A_tst], axis=1)
            clf = self._iterator.member
            if self._abbr_cls == 'FairGBM':
                clf.fit(X_A_trn, y_trn, constraint_group=~jt_trn[1])
            else:
                clf.fit(X_A_trn, y_trn)
            y_insp = clf.predict(X_A_trn)  # fx_trn
            y_pred = clf.predict(X_A_tst)  # fx_tst

            tmp_dis, tmp_app = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, y_insp, ~g1_trn[0], g1_trn[0],
                X_tst, A_tst, y_tst, y_pred, ~g1_tst[0], g1_tst[0],
                **pm_m)
            tmp_dis = [tmp_dis]
            tmp_dis.extend(tmp_app)  # siz=(1+21|24,7)
            res_iter.append(tmp_dis)

            if len(g1_trn) == 1 or len(g1_tst) == 1:
                tim_elapsed = time.time() - since
                elegant_print("CV iteration {}-th, consumed {}".format(
                    k, elegant_durat_core(tim_elapsed, True)), logger)
                return res_iter

            tmp_dis, tmp_app = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, y_insp, ~g1_trn[1], g1_trn[1],
                X_tst, A_tst, y_tst, y_pred, ~g1_tst[1], g1_tst[1], **pm_m)
            tmp_dis = [tmp_dis]
            tmp_dis.extend(tmp_app)  # siz=(1+21|24,7)
            res_iter.append(tmp_dis)

            for jt_a, jt_b in zip(jt_trn, jt_tst):
                tmp_dis, tmp_app = self._iterator.schedule_content(
                    X_trn, A_trn, y_trn, y_insp, ~jt_a, jt_a,
                    X_tst, A_tst, y_tst, y_pred, ~jt_b, jt_b, **pm_m)
                tmp_dis = [tmp_dis]
                tmp_dis.extend(tmp_app)  # siz=(1+21|24,7)
                res_iter.append(tmp_dis)

            # res_iter, siz=(1|4, 1+21|24, 7)
        elif 'expt2' in self._trial_type:  # .endswith('expt2'):
            pm_m = {'m1': self._m1, 'm2': self._m2}
            positive_label = self._dataset.get_positive_class_val(
                'numerical-binsensitive')

            res_iter = self._iterator.schedule_content(
                logger,
                X_trn, A_trn, y_trn, Aq_trn, g1_trn, jt_trn,
                X_tst, A_tst, y_tst, Aq_tst, g1_tst, jt_tst,
                # positive_label=positive_label, **pm_m)
                self._m1, self._m2, positive_label=positive_label)
            # expt2a: res_iter, siz= (1|2, 7,  77)
            # expt2b: res_iter, siz= (1|2, 7, 147)
            # expt2c: res_iter, siz= (14+4*{1|2}, 199)
        else:
            pass

        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter

    # PREPARATION

    def preparing_current_data(self, logger=None):
        processed_data = preprocess(self._dataset, self._data_frame, logger)
        disturbed_data = adversarial(
            self._dataset, self._data_frame, self._ratio, logger)

        processed_Xy = processed_data['numerical-binsensitive']
        disturbed_Xy = disturbed_data['numerical-binsensitive']
        binarized_Xy = binarized_data_set(processed_Xy)

        X, A, y, new_attr = transform_X_A_and_y(self._dataset,
                                                binarized_Xy)
        _, Aq, _, _ = transform_X_A_and_y(
            self._dataset, binarized_data_set(disturbed_Xy))
        # NB. X, A, y: all pd.DataFrame

        self._m2 = np.ceil(2 * np.log10(len(y)))
        self._m2 = int(self._m2)
        elegant_print("Due to #inst = {}".format(len(y)), logger)
        elegant_print("self._m2 = {}".format(self._m2), logger)
        elegant_print("self._m1 = {}".format(self._m1), logger)

        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'], 'both')

        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values))], logger)
        del tmp

        sens_attr = deepcopy(self._dataset.sensitive_attrs)
        tmp_cls = self._abbr_cls if not self._trial_type[-6:] in [
            'expt6a', 'expt6b'] else self._iterator._abbr_cls
        if 'expt2' in self._trial_type:
            tmp_cls = self._nb_cls
        res_aux = [[
            self._dataset.dataset_name, len(set(y.values)),
            tmp_cls,
            self._nb_iter, self._gen_iter, self._rep_iter,
            self._m1, self._m2, len(sens_attr), '', ''],
            self._dataset.sensitive_attrs,
            self._dataset.privileged_vals, ]
        if len(sens_attr) == 1:
            res_aux.append(sens_attr)
        else:
            res_aux.append(sens_attr + [new_attr] + ['joint_and,or'])

        # return X, A, y, new_attr, belongs_priv, ptb_with_joint
        return (X.values, A.values, y.values, belongs_priv,
                ptb_with_joint, new_attr, res_aux, Aq.values)


# -------------------------------
#


class ManfPrime_Empirical(ManfEmpirical):
    def preparing_current_data(self, logger=None):
        processed_data = preprocess(self._dataset, self._data_frame, logger)
        disturbed_data = adversarial(
            self._dataset, self._data_frame, self._ratio, logger)
        processed_Xy = processed_data['numerical-binsensitive']
        disturbed_Xy = disturbed_data['numerical-binsensitive']
        binarized_Xy = binarized_data_set(processed_Xy)
        disturbed_Xy = binarized_data_set(disturbed_Xy)

        X_A, y = transform_X_and_y(self._dataset, binarized_Xy)
        X_Aq, y = transform_X_and_y(self._dataset, disturbed_Xy)
        # NB. X, A, Aq, y: all pd.DataFrame
        X, A, _, new_attr = transform_X_A_and_y(self._dataset, binarized_Xy)
        _, Aq, _, _ = transform_X_A_and_y(self._dataset, disturbed_Xy)

        self._m2 = np.ceil(2 * np.log10(len(y)))
        self._m2 = int(self._m2)
        elegant_print("Due to #inst = {}".format(len(y)), logger)
        elegant_print("self._m2 = {}".format(self._m2), logger)
        elegant_print("self._m1 = {}".format(self._m1), logger)
        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values))], logger)
        del tmp

        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'], 'both')
        sens_attr = deepcopy(self._dataset.sensitive_attrs)
        tmp_cls = self._abbr_cls if not self._trial_type[-6:] in [
            'expt6a', 'expt6b'] else self._iterator._abbr_cls
        if ('expt3' in self._trial_type) or ('expt2' in self._trial_type):
            tmp_cls = self._nb_cls
        elif ('expt5' in self._trial_type):
            tmp_cls = ''  # None
        res_aux = [[
            self._dataset.dataset_name, len(set(y.values)), tmp_cls,
            self._nb_iter, self._gen_iter, self._rep_iter,
            self._m1, self._m2, len(sens_attr), '', ''],
            self._dataset.sensitive_attrs,
            self._dataset.privileged_vals, ]
        if len(sens_attr) == 1:
            res_aux.append(sens_attr)
        else:
            res_aux.append(sens_attr + [new_attr] + ['joint_and,or'])

        # return X, A, y, new_attr, belongs_priv, ptb_with_joint
        return (X_A.values, y.values, X_Aq.values,
                belongs_priv, ptb_with_joint, new_attr, res_aux,
                X.values, A.values, Aq.values)

    def preparing_iterator(self, trial_type, abbr_cls, nb_iter, gen, rep,
                           m1, m2, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep     # cv-split
        self._prep = prep        # pre-processing data
        self._m1, self._m2 = m1, m2

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        self._screen, self._logged = screen, logged
        self._abbreviation_dat_name = self._log_document

        formatted_log = "_".join([trial_type, "iter{}".format(
            nb_iter) if nb_iter > 0 else 'sing',
            self._prep, abbr_cls, self._log_document, 'pms'])

        if trial_type.endswith('expt5a'):
            self._iterator = PartE1_ParaSenAnalysis(abbr_cls)
            formatted_log += '_ma{}'.format(m1)
        elif trial_type.endswith('expt5b'):
            self._iterator = PartE2_ParaSenAnalysis(abbr_cls)
            formatted_log += '_mb{}'.format(m2)
        elif trial_type.endswith('expt5c'):
            self._iterator = PartE3_ParamsSen(abbr_cls)
        elif trial_type.endswith('expt6a'):
            self._iterator = PartF1_ParaSenAnalysis(
                abbr_cls, nb_cls,
                constraint_type, self.saIndex, self.saValue)
        elif trial_type.endswith('expt6b'):
            self._iterator = PartF2_ParaSenAnalysis(
                abbr_cls, nb_cls,
                constraint_type, self.saIndex, self.saValue)
        elif trial_type.endswith('expt2a'):
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

        if 'expt5' in trial_type:
            formatted_log = formatted_log.replace(
                '_{}'.format(abbr_cls), '')
        elif 'expt2' in trial_type:
            formatted_log = "_".join([
                trial_type, 'iter{}'.format(nb_iter) if nb_iter > 0 else 'sing',
                'cls{}'.format(nb_cls), prep, self._log_document,
                'pms', 'ratio{}'.format(int(self._ratio * 100)), ])

        if abbr_cls in AVAILABLE_ENSF:
            formatted_log += '_cls{}'.format(nb_cls)
        if abbr_cls == 'FairGBM':
            formatted_log += '_{}'.format(constraint_type)
        self._log_document = formatted_log + (
            '_gen' * gen + '_rep' * rep)
        return

    def coding_per_dataset(self, logger):
        (X_and_A, y, X_and_Aq, idx_g1, idx_jt, new_attr,
         res_aux, X, A, Aq) = self.preparing_current_data(logger)
        del X, A, Aq  # no use if self.prep
        n_a = len(idx_g1)
        g1m_indices = [[idx_g1[i]] for i in range(n_a)]  # non sa_indices

        if self._trial_type[-6:] in ('expt5a', 'expt6a'):
            res_aux.append(self._iterator._m2_set)
        elif self._trial_type[-6:] in ('expt5b', 'expt6b'):
            res_aux.append(self._iterator._m1_set)
        elif self._trial_type.endswith('expt5c'):
            res_aux.append(self._iterator._m1_set)
            res_aux.append(self._iterator._m2_set)
        elif 'expt2' in self._trial_type:  # .endswith():
            res_aux.append(['bagging', 'adaboost', 'lightgbm',
                            'fairgbm : FPR', 'fairgbm : FNR',
                            'fairgbm : FPR,FNR', 'adafair'])
            if 'expt2c' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs)
            elif 'expt2d' in self._trial_type:
                res_aux.append(self._iterator._abbr_clfs + [
                    'bagging', 'adaboost'])
            elif 'expt2e' in self._trial_type:
                res_aux.append(['bagging', 'adaboost', 'lightgbm'] +
                               ['fairgbm : fpr', 'fairgbm : fnr',
                                'fairgbm : fpr,fnr', 'adafair'])  # *2)

        if self._nb_iter <= 0:
            elegant_print('Running /executing as a whole', logger)
            if self._prep != 'none':
                scaler = scale_normalize_helper(self._prep)
                scaler, XA_p, _, _, Xp, Ap, _, _, _, _ = renewed_normalise_disturb(
                    scaler, X_and_A, [], X_and_A, self.saIndex)
            else:
                Xp, Ap, _, _, _, _ = renewed_normalise_separate(
                    X_and_A, [], X_and_A, self.saIndex)
            res_data = self.coding_per_iteration_as_whole(logger, Xp, Ap, y, idx_g1, idx_jt)
            return [res_data], res_aux

        # NON-CROSS VALIDATION, JUST SIMPLE REPEAT
        elegant_print('Not-repetitively, via cv_split?: {}'.format(
            'Yes' if self._rep_iter else 'No'), logger)  # cv_split
        if not self._rep_iter:
            elegant_print("nb_iter={}, repetitive".format(
                self._nb_iter), logger)
            split_idx = manual_repetitive(self._nb_iter, y,
                                          self._gen_iter)
            res_ans = []
            for k, idx in enumerate(split_idx):
                elegant_print("Iteration {}-th starts.".format(
                    k + 1), logger)
                XA_p, _, yp, XAq_p, g1_p, jt_p = renewed_transform_disturb(
                    X_and_A, None, y, X_and_Aq, idx, g1m_indices, idx_jt)
                g1_p = [t[0] for t in g1_p]
                if self._prep in ['standard', 'min_max', 'normalize']:
                    scaler = scale_normalize_helper(self._prep)
                    scaler, XA_p, _, _, Xp, Ap, _, _, _, _ = renewed_normalise_disturb(
                        scaler, XA_p, [], XA_p, self.saIndex)
                else:
                    Xp, Ap, _, _, _, _ = renewed_normalise_separate(
                        XA_p, [], XA_p, self.saIndex)
                res_iter = self.coding_per_iteration_as_whole(
                    logger, Xp, Ap, yp, g1_p, jt_p)
                res_ans.append(res_iter)
                elegant_print("Iteration {}-th done.".format(k + 1), logger)
            return res_ans, res_aux

        # CROSS VALIDATION, usually 5-fold cross validation
        elegant_print('nb_iter={}, cross_valid'.format(self._nb_iter), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif "KF" in self._trial_type:  # 'Kcv'
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print("\t CrossValid  {}\n".format(
            self._trial_type[: 3]), logger)
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
        return res_ans, res_aux

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
            X_trn, A_trn, X_tst, A_tst):
        since = time.time()
        res_iter = []

        if 'expt2' in self._trial_type:
            pm_m = {'m1': self._m1, 'm2': self._m2}
            positive_label = self._dataset.get_positive_class_val('numerical-binsensitive')
            res_iter = self._iterator.schedule_content_prime(
                logger,
                X_A_trn, y_trn, X_Aq_trn, g1_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, g1_tst, jt_tst,
                X_trn, A_trn, X_tst, A_tst, self._m1, self._m2, positive_label)
            # expt2a: res_iter, siz= (1|2, 7, 1+ 46*2)
            # expt2b: res_iter, siz= (1|2, 7, 1+105*2)
            # expt2c: res_iter, siz= (14+4*{1|2}, 1+131*2)

        elif 'expt6' in self._trial_type:  # [-6:] in ['expt6a','expt6b','expt6c']
            pm_m = {}
            if self._trial_type.endswith('6a'):
                pm_m['m1'] = self._m1
            elif self._trial_type.endswith('6b'):
                pm_m['m2'] = self._m2
            clf = self._iterator.member
            if self._abbr_cls == 'FairGBM':
                # clf.fit(X_A_trn, y_trn, constraint_group=~jt_trn[1])
                non_sa_idx = g1_trn[0] if len(g1_trn) == 1 else jt_trn[1]
                non_sa_idx = ~non_sa_idx
                clf.fit(X_A_trn, y_trn, constraint_group=non_sa_idx)
                del non_sa_idx
            else:
                clf.fit(X_A_trn, y_trn)
            y_insp = clf.predict(X_A_trn)  # fx_trn
            y_pred = clf.predict(X_A_tst)  # fx_tst

            tmp_dis, tmp_app = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, y_insp, ~g1_trn[0], g1_trn[0],
                X_tst, A_tst, y_tst, y_pred, ~g1_tst[0], g1_tst[0],
                **pm_m)
            tmp_dis = [tmp_dis]
            tmp_dis.extend(tmp_app)  # siz= (1+21|24, 9 =7+2)
            res_iter.append(tmp_dis)
            if len(g1_trn) == 1 or len(g1_tst) == 1:
                tim_elapsed = time.time() - since
                elegant_print("CV iteration {}-th, consumed {}".format(
                    k, elegant_durat_core(tim_elapsed, True)), logger)
                return res_iter

            tmp_dis, tmp_app = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, y_insp, ~g1_trn[1], g1_trn[1],
                X_tst, A_tst, y_tst, y_pred, ~g1_tst[1], g1_tst[1],
                **pm_m)
            tmp_dis = [tmp_dis]
            tmp_dis.extend(tmp_app)  # siz= (1+21|24, 9 =7+2)
            res_iter.append(tmp_dis)
            for jt_a, jt_b in zip(jt_trn, jt_tst):
                tmp_dis, tmp_app = self._iterator.schedule_content(
                    X_trn, A_trn, y_trn, y_insp, ~jt_a, jt_a,
                    X_tst, A_tst, y_tst, y_pred, ~jt_b, jt_b, **pm_m)
                tmp_dis = [tmp_dis]
                tmp_dis.extend(tmp_app)  # siz= (1+21|24, 9 =7+2)
                res_iter.append(tmp_dis)
            # res_iter.shape= (1|4, 1+21|24, 9 =7+2)

        else:
            pass
        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter


# -------------------------------
# Manifold simulation


class ManfSimulative(ManfEmpirical):
    def __init__(self, trial_type, data_type="simulative",
                 abbr_cls='DT', nb_iter=0, gen=False, rep=False,
                 m1=30, m2=10, nb_cls=1, ratio=.5,
                 constraint_type='FPR,FNR',
                 prep=False, screen=True, logged=False):
        self._data_type = data_type
        self._log_document = data_type  # tmp_simulative
        self.saIndex = [-2, -1]
        self.saValue = [1, 1]
        self._ratio = ratio
        self.preparing_iterator(
            trial_type, abbr_cls, nb_iter, gen, rep,
            m1, m2, nb_cls, constraint_type,
            prep, screen, logged)

    def preparing_current_data(self, logger=None):
        nb_inst, nb_feat = 110, 4

        X = np.random.rand(nb_inst, nb_feat)
        A = np.random.randint(3, size=(nb_inst, 2))
        y = np.random.randint(2, size=nb_inst)

        # self._m2 = 2 * np.ceil(np.log10(nb_inst))  # len(y)=
        self._m2 = np.ceil(2 * np.log10(nb_inst))
        elegant_print("Due to #inst = {}".format(nb_inst), logger)
        elegant_print("self._m2 = {}".format(self._m2), logger)
        elegant_print("self._m1 = {}".format(self._m1), logger)

        idx_g1 = [A[:, i] == 1 for i in [0, 1]]
        idx_jt = [np.logical_and(idx_g1[0], idx_g1[1]),
                  np.logical_or(idx_g1[0], idx_g1[1])]

        res_aux = [
            ['tmp_simulative', 2, None, self._nb_iter,
             self._gen_iter, self._rep_iter, self._m1, self._m2,
             '#sens= 2'], [], [],
            ['sens#1', 'sens#2', 'jt_and', 'jt_or']]

        Ap = np.random.randint(3, size=(nb_inst, 2))
        return X, A, y, idx_g1, idx_jt, 'sens#1-2', res_aux, Ap
