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


# =====================================
# fairmanf_ext
#


# -------------------------------------
# Manifold ext. cvg. empirical


class ManfCvgEmpir(DataSetup):
    def __init__(self, trial_type, data_type, prep=False, nb_iter=5,
                 m1=20, m2=8, n_e=2, m2_fixed=False, ratio=.97,
                 screen=True, logged=False):
        super().__init__(data_type)
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
        csv_row_2a = ['dat_name', 'binary', '#sen-att', '#cv',
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

    def coding_per_dataset(self, logger):
        return

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
                    len(sens_att), self.nb_iter, self._m1, self._m2, self._n_e, '']]

        pdb.set_trace()
        return


class ManfCvgPrime(ManfCvgEmpir):
    def preparing_iterator(self):
        pass

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

        pdb.set_trace()
        return


# -------------------------------------
#
