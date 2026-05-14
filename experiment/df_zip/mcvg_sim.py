# coding: utf-8


import csv
import json
import os
import sys
import time
import pdb
import numpy as np


from hfm.utils.recorders import get_elogger, elegant_print
from hfm.utils.decorators import (
    elegant_dated, fantasy_durat, fantasy_durat_major)
from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    manual_repetitive, scale_normalize_helper)

from experiment.utils_empirical import DataSetup
from experiment.datasets import (
    transform_X_and_y, transform_unpriv_tag)
from experiment.preprocessing_bin import normalise_disturb_prime
from experiment.preprocessing_nonbin import (
    renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
    check_marginalised_indices, renewed_transform_disturb,
    renewed_normalise_disturb, renewed_normalise_separate)

from experiment.df_zip.mcvg_exp import (
    cvgExp1A_anal, cvgExp1B_anal, cvgExp1C_take)
from hfm.utils.verifiers import DTY_INT


# =====================================
# fairmanf_ext
#


# -------------------------------------
# Manifold ext. cvg. empirical


class ManfCvgEmpir(DataSetup):
    def __init__(self, trial_type, data_type, prep=False, nb_cv=5,
                 m1=20, m2=8, n_e=2, n_p=3, m2_fixed=False, ratio=.97,
                 screen=True, logged=False):
        super().__init__(data_type)
        self._ratio = ratio  # discriminative risk
        self._m2_fixed = m2_fixed  # hfm
        self._m1, self._m2, self._n_e = m1, m2, n_e
        self._n_p = n_p       # Minkowski distance

        self._trial_type = trial_type
        self._nb_cv = nb_cv
        self._prep = prep
        self._screen, self._logged = screen, logged

        self.preparing_iterator(trial_type)
        return

    def preparing_iterator(self, trial_type):
        self.saIndex = [-1] if self._data_type == 'ricci' else [-2, 1]
        self._subcore_iterator(trial_type)
        return

    def _subcore_iterator(self, trial_type):
        priv_val = self._dataset.get_privileged_group_with_joint('')[:2]
        if trial_type.endswith('cvg1c'):
            self._iterator = cvgExp1C_take(priv_val)
        elif trial_type.endswith('cvg1a'):
            self._iterator = cvgExp1A_anal(priv_val, omitted=True)
        elif trial_type.endswith('cvg1b'):
            self._iterator = cvgExp1B_anal(priv_val, omitted=True)
        return

    def trial_one_process(self, mode="w"):
        since = time.time()
        csv_t = open(self._log_document + ".csv", mode=mode)
        csv_w = csv.writer(csv_t)
        if mode == "a":
            csv_w.writerows([['']] * 4)

        if not (self._screen or self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + ".log", "w")
            sys.stdout = fsock
        if self._logged:
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            logger = get_elogger("fairmanf_cvg.", self._log_document + ".txt")
        else:
            logger = None

        elegant_print([
            "[BEGAN AT {}]".format(elegant_dated(since)),
            "EXPERIMENT",
            "\t binary? = {}".format(not self._trial_type.startswith('mu')),
            "\t   trail = {}".format(self._trial_type),
            "\t dataset = {}".format(self._data_type),
            "\t nb_iter = {}, gen {}, rep/cv(s) {}".format(
                self._nb_iter,
                str(self._gen_iter)[0], str(self._rep_iter)[0]),
            "\tdata prep= {}".format(self._prep),
            "\t w/o omit= {}".format(self._omit),
            "PARAMETERS",
            "\t  m1, m2 = {}, {}".format(self._m1, self._m2),
            "\t  n_e    = {}    ".format(self._n_e),
            "\t  nb_cls = {}".format(self._nb_cls),
            "\t  constr = {}".format(self._constraint_type),
            "HYPER-PARAMS", ""], logger)

        # START
        # END

        tim_elapsed = time.time() - since
        tim_consumed = fantasy_durat(tim_elapsed, False, True)
        elegant_print([
            "",
            "Duration /TimeCost: {}".format(tim_consumed),
            "Alternative         {}".format(
                tim_consumed.replace('\n\t i.e.,', '')),
            "[ENDED AT {:s}]".format(elegant_dated(time.time()))], logger)
        del tim_consumed

        del logger
        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    def coding_per_procedure(self, csv_w, logger):
        csv_row_2a = ['dat_name', 'binary', '#sen-att', 'pre', '#cv',
                      'm1', 'm2', 'n_e', 'k?']
        # csv_row_1, cr2c, cr3c, cr4c = self._iterator.prepare_trial()

        # START
        res_data, res_aux = self.coding_per_dataset(logger)
        json_saver = json.dumps({"res_aux": res_aux, "res_data": res_data})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w
        # END
        return

    # EACH SUB-ROUTE

    def coding_per_iteration_as_whole(
            self, logger,  # k,
            X, A, y, X_wA, g1m_indices, idx_jt=None, Aq=None, X_wAq=None):
        res_iter = None  # []
        if self._trial_type.endswith('cvg1c'):
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices, self._m1, self._m2, self._n_e, self._n_p)
        elif self._trial_type[-5:] in ('cvg1a', 'cvg1b'):
            curr_m = self._m1 if self._trial_type.endswith('a') else self._m2
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices, curr_m, self._n_e, self._n_p)
        return res_iter

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
            X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst):
        return

    def coding_per_dataset(self, logger):
        X, A, y, Aq, adjunctive, res_aux = self.preparing_curr_dat(logger)
        marginalised_grp, g1m_indices, new_attr, idx_g1, idx_jt = adjunctive

        if self._nb_cv <= 1:
            elegant_print("Running /executing as a whole", logger)
            res_iter = self.coding_per_iteration_as_whole(
                logger, X, A, y, None, g1m_indices, idx_jt, Aq, None)
            pdb.set_trace()
            return [res_iter], res_aux

        elegant_print(f"nb_cv={self._nb_cv}, cross_valid", logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_cv, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_cv, y, X)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_cv, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print(f"\t CrossValid  {self._trial_type[:3]}\n", logger)

        res_aux = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X, A, y, Aq, i_trn, g1m_indices, idx_jt)
            (X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X, A, y, Aq, i_tst, g1m_indices, idx_jt)
            if self._prep in ['standard', 'min_max', 'normalize', 'min_abs']:
                scaler = scale_normalize_helper(self._prep)
                # (scaler, X_trn, A_trn, _, _, X_tst, A_tst
                #  ) = normalise_disturb_prime(
                #     scaler, X_trn, A_trn, [], [], X_tst, A_tst)
                (scaler, X_trn, _, X_tst, _, _, _, _, _, _
                 ) = renewed_normalise_disturb(
                    scaler, X_trn, [], X_tst, saIndex=[])  # self.saIndex)
            # else:
            #     pass

            # i-th K-Fold  # TODO
            elegant_print("Iteration {}-th".format(k + 1), logger)
            pdb.set_trace()
            res_iter = self.coding_per_iteration_cv_split()
        return

    # def _subcore_perset(self):
    #     return

    def preparing_curr_dat(self, logger):
        (origin_dat, processed_dat, process_mult, disturbed_dat,
         disturb_mult) = renewed_prep_and_adversarial(
            self._dataset, self._data_frame, self._ratio, logger=logger)
        processed_Xy = process_mult['numerical-multisen']
        disturbed_Xy = disturb_mult['numerical-multisen']
        X, A, y, _ = renewed_transform_X_A_and_y(
            self._dataset, processed_Xy, with_joint=False)
        _, Aq, _, _ = renewed_transform_X_A_and_y(
            self._dataset, disturbed_Xy, with_joint=False)

        if not self._m2_fixed:
            self._m2 = np.ceil(2 * np.log10(len(y)))
            self._m2 = int(self._m2)
        elegant_print(["Due to #inst = {}".format(len(y)),
                       "self._m2 = {}".format(self._m2),
                       "self._m1 = {}".format(self._m1)], logger)
        tmp = processed_dat['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values)),
            ""], logger)
        for tt in range(A.shape[1]):
            tmp = set(A.values[:, tt])
            elegant_print(
                "\t sen-att-# {}: #val= {}, opt. value set= {}".format(
                    tt + 1, len(tmp), tmp), logger)
        del tmp
        elegant_print(["\t X .shape {}".format(X.shape),
                       "\t A .shape {}".format(A.shape),
                       "\t Aq.shape {}".format(Aq.shape),
                       "\t y .shape {}".format(y.shape), ""], logger)

        adjunctive, res_aux = self._subcore_currdat(origin_dat, processed_dat, y)
        # pdb.set_trace()
        return X.values, A.values, y.values, Aq.values, adjunctive, res_aux

    def _subcore_currdat(self, origin_dat, processed_dat, y):
        sens_att = self._dataset.get_sensitive_attrs_with_joint()[:2]
        priv_val = self._dataset.get_privileged_group_with_joint('')[:2]
        marginalised_grp = origin_dat['marginalised_groups']
        self.saValue = [(np.arange(
            len(sa_val)) + 2).tolist() for sa_val in marginalised_grp]
        marginal_indices = check_marginalised_indices(
            processed_dat['original'], sens_att, priv_val, marginalised_grp)
        new_attr = '-'.join(sens_att) if len(sens_att) > 1 else None
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_dat['original'], 'both')

        res_aux = [[self._dataset.dataset_name, len(set(y.values)),
                    len(sens_att), self._prep, self._nb_cv,  # self._nb_iter,
                    self._m1, self._m2, self._n_e, ''],
                   sens_att, priv_val, marginalised_grp, ]
        return (marginalised_grp, marginal_indices, new_attr,
                belongs_priv, ptb_with_joint), res_aux


class ManfCvgPrime(ManfCvgEmpir):
    def preparing_iterator(self, trial_type):
        self._subcore_iterator(trial_type)  # pass

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            jt_trn=None, jt_tst=None):
        return

    def coding_per_dataset(self, logger):
        X_and_A, y, X_and_Aq, adjunctive, res_aux = self.preparing_curr_dat(logger)
        marginalised_grp, g1m_indices, new_attr, idx_g1, idx_jt = adjunctive

        if self._nb_cv <= 1:
            elegant_print("Running /executing as a whole", logger)
            if self._prep != 'none':
                scaler = scale_normalize_helper(self._prep)
                (scaler, X_and_A, _, X_and_Aq, X, A, _, _, Xq, Aq
                 ) = renewed_normalise_disturb(
                    scaler, X_and_A, [], X_and_Aq, self.saIndex)
                del scaler
            else:
                X, A, _, _, Xq, Aq = renewed_normalise_separate(
                    X_and_A, [], X_and_Aq, self.saIndex)
            A, Aq = A.astype(DTY_INT), Aq.astype(DTY_INT)
            res_iter = self.coding_per_iteration_as_whole(
                logger, X, A, y, X_and_A, g1m_indices, idx_jt, Aq, X_and_Aq)
            return [res_iter], res_aux

        elegant_print(f"nb_cv={self._nb_cv}, cross_valid", logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_cv, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_cv, y, X)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_cv, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print(f"\t CrossValid  {self._trial_type[:3]}\n", logger)

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (X_A_trn, _, y_trn, X_Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_trn, g1m_indices, idx_jt)
            (X_A_tst, _, y_tst, X_Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_tst, g1m_indices, idx_jt)
            if self._prep not in ['none', None]:
                scaler = scale_normalize_helper(self._prep)
                (scaler, X_A_trn, _, X_A_tst, X_trn, A_trn, _, _,
                    X_tst, A_tst) = renewed_normalise_disturb(
                    scaler, X_A_trn, [], X_A_tst, self.saIndex)
                curr_non_sa = list(range(X_Aq_trn.shape[1]))
                for curr_i in self.saIndex:
                    curr_non_sa.remove(curr_i)
                for curr_i in curr_non_sa:
                    X_Aq_trn[:, curr_i] = X_A_trn[:, curr_i]
                    X_Aq_tst[:, curr_i] = X_A_tst[:, curr_i]
            else:
                X_trn, A_trn, _, _, X_tst, A_tst = renewed_normalise_separate(
                    X_A_trn, [], X_A_tst, self.saIndex)

            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            pdb.set_trace()
        return

    def preparing_curr_dat(self, logger):
        (origin_dat, processed_dat, process_mult, disturbed_dat,
         disturb_mult) = renewed_prep_and_adversarial(
            self._dataset, self._data_frame, self._ratio, logger=logger)
        processed_Xy = process_mult['numerical-multisen']
        disturbed_Xy = disturb_mult['numerical-multisen']
        X_A, y = transform_X_and_y(self._dataset, processed_Xy)
        X_Aq, y = transform_X_and_y(self._dataset, disturbed_Xy)
        # NB. X_A, X_Aq, y: all pd.DataFrame

        if not self._m2_fixed:
            self._m2 = np.ceil(2 * np.log10(len(y)))
            self._m2 = int(self._m2)
        elegant_print(["Due to #inst = {}".format(len(y)),
                       "self._m2 = {}".format(self._m2),
                       "self._m1 = {}".format(self._m1)], logger)
        tmp = processed_dat['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values)),
            ""], logger)
        del tmp
        elegant_print(["\t X_A .shape {}".format(X_A.shape),
                       "\t X_Aq.shape {}".format(X_Aq.shape),
                       "\t y   .shape {}".format(y.shape), ""], logger)

        adjunctive, res_aux = self._subcore_currdat(origin_dat, processed_dat, y)
        # pdb.set_trace()
        return X_A.values, y.values, X_Aq.values, adjunctive, res_aux


# -------------------------------------
#
