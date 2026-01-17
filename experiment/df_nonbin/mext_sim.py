# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds w. extension
#


import csv
import json
import os
import sys
import time

import numpy as np
# import pandas as pd
from pathos import multiprocessing as pp


from hfm.utils.recorders import get_elogger, elegant_print
from hfm.utils.decorators import (
    elegant_dated, fantasy_durat, fantasy_durat_major)
from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    manual_repetitive, scale_normalize_helper)

from experiment.datasets import (
    transform_X_and_y, DATASET_NAMES, DATASETS,
    transform_unpriv_tag)
from experiment.preprocessing_bin import normalise_disturb_prime
from experiment.preprocessing_nonbin import (
    renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
    check_marginalised_indices, renewed_transform_disturb,
    renewed_normalise_disturb, renewed_normalise_separate)

from experiment.df_nonbin.mext_exp_mp import (
    ComparisonC2_withDirectComput, ComparisonB2_withDirectComput,
    ComparisonC4_withDirectComput,
    ComparisonD2_withDirectComput, ComparisonD3_withDirectComput,
    ComparisonD4_withDirectComput,
    ComparisonE2_with, ComparisonE3_with, ComparisonE4_with,
    HyperEA_analysis, HyperEB_analysis, HyperFA_analysis, HyperFB_analysis,
    Multiprocess_GA_comparison)


# ===============================
# Empirical


# -------------------------------
# Maniford empirical


# from experiment.classifiers import INDIVIDUALS
# AVAILABLE_CLFS = list(INDIVIDUALS.keys())
# AVAILABLE_ENSF = [
#     'bagging', 'AdaBoost',  # 'Bagging@SK', 'AdaBoost@SK',
#     'LightGBM', 'FairGBM', 'AdaFair',
#     # 'lightGBM', 'fairGBM', 'AdaFair',  # LightGBM, FairGBM
# ]


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type

        if data_type == 'ppr':
            self._data_type = DATASET_NAMES[-2]
        elif data_type == 'ppvr':
            self._data_type = DATASET_NAMES[-1]
        elif data_type not in ['ricci', 'german', 'adult']:
            raise ValueError("Wrong dataset `{}`".format(data_type))

        # ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        idx = DATASET_NAMES.index(self._data_type)
        self._dataset = DATASETS[idx]
        self._data_frame = self._dataset.load_raw_dataset()

        if data_type == "ricci":
            self.saIndex = [2]     # 'Race' -2
        elif data_type == "german":
            self.saIndex = [3, 5]  # ['sex', 'age'] [,12]
        elif data_type == "adult":
            self.saIndex = [2, 3]  # ['race','sex'] [7,8]
        elif data_type == "ppr":
            self.saIndex = [0, 2]  # ['sex','race'] [0,3]
        elif data_type == "ppvr":
            self.saIndex = [0, 2]  # ['sex','race'] [0,3]

        self.saValue = self._dataset.get_privileged_group(
            'numerical-binsensitive')
        self.saValue = [0 for sa in self.saValue if sa == 1]

    @property
    def data_type(self):
        return self._data_type

    @property
    def log_document(self):
        return self._log_document

    # # ----------- mu -----------
    # def prepare_mu_datasets(self, ratio=.5, logger=None):
    #   pass
    # # ----------- tr -----------
    # # ----------- bi -----------
    # def prepare_bi_datasets(self, ratio=.5, logger=None):
    #   pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def trial_type(self):
        return self._trial_type


# -------------------------------
# Maniford ext. empirical


class ManfExtEmpirical(DataSetup):
    def __init__(self, trial_type, data_type,
                 # abbr_cls='DT', nb_iter=5, gen=False, rep=False,
                 # m1=25, m2=11, nb_cls=1,
                 # ratio=.7, constraint_type='FPR,FNR',
                 # prep=False, screen=True, logged=False):
                 # nb_iter=5, m1=25, m2=11, ratio=.7, prep=False,
                 nb_iter=5, m1=25, m2=11, n_e=3, ratio=.7, prep=False,
                 gen=False, rep=False, nb_cls=1, abbr_cls='DT',
                 constraint_type='FPR,FNR', omitted=True,
                 alternative=True, solo_cv_siz=.8, mp_cores=3,
                 m2_fixed=False,
                 screen=True, logged=False):
        super().__init__(data_type)
        self._ratio = ratio
        self._omit = omitted
        self._alternative = alternative  # for 'expt5*'
        self._solo_cv_siz = solo_cv_siz  # for 'expt6*'  # solo_cvs_size
        self._mp_cores = mp_cores  # number of cores for multiprocessing
        self._m2_fixed = m2_fixed
        self.preparing_iterator(trial_type, abbr_cls,
                                nb_iter, gen, rep, m1, m2, n_e,
                                nb_cls, constraint_type,
                                prep, screen, logged)

    def preparing_iterator(self, trial_type, abbr_cls, nb_iter, gen, rep,
                           m1, m2, n_e, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep  # ._cvs_iter
        self._prep = prep  # preprocess data, pre-processing
        self._m1, self._m2 = m1, m2
        self._n_e = n_e

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        self.saIndex = [-1] if self._data_type == 'ricci' else [-2, -1]
        self._screen, self._logged = screen, logged

        self._abbreviation_dat_name = self._log_document
        # formatted_log = "_".join([
        #     trial_type,
        #     "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
        # ])  # format_log  # self._prep,

        if trial_type.endswith('expt2b'):
            self._iterator = ComparisonB2_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt2c'):
            self._iterator = ComparisonC2_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt2d'):
            self._iterator = ComparisonC4_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)

        if 'expt2' in trial_type:
            formatted_log = "_".join([
                trial_type,
                'iter{}'.format(nb_iter) if nb_iter > 0 else 'sing',
                'cls{}'.format(nb_cls), prep, self._log_document,
                'pms', 'ratio{}'.format(int(self._ratio * 100)), ])

        # self._log_document = "_".join([
        #     # formatted_log, 'rep' * rep, 'gen' * gen])
        #     formatted_log, '_rep' * rep + '_gen' * gen])
        self._log_document = formatted_log
        self._log_document += ('_gen' * gen + '_rep' * rep)
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
            # logger, formatter, fil_handler = get_elogger(
            logger = get_elogger(
                "fairmanf_ext.", self._log_document + ".txt")
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
            "PARAMETERS",
            "\t  m1, m2 = {}, {}".format(self._m1, self._m2),
            "\t  n_e    = {}    ".format(self._n_e),
            "\t  nb_cls = {}".format(self._nb_cls),
            "\t  constr = {}".format(self._constraint_type),
            "HYPER-PARAMS", ""], logger)

        # START
        self.coding_per_procedure(csv_w, logger)
        # END

        tim_elapsed = time.time() - since
        elegant_print([
            "",
            "Duration /TimeCost: {}".format(fantasy_durat(tim_elapsed,
                                                          False, True)),
            "[ENDED AT {:s}]".format(elegant_dated(time.time()))], logger)

        # if self._logged:
        #     rm_ehandler(logger, formatter, fil_handler)
        # else:
        del logger
        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    # EACH SUBROUTE

    def coding_per_procedure(self, csv_w, logger):
        csv_row_2a = ['data_name', 'binary', 'abbr nb_cls',
                      'nb_iter', 'gen', 'rep/cv', 'm1', 'm2']
        # if 'expt2' in self._trial_type:
        # if ('expt2' in self._trial_type) or ('expt3' in self._trial_type):
        if ('expt2' in self._trial_type) or ('expt3' in self._trial_type) or (
                'expt4' in self._trial_type):
            csv_row_2b = ['#sens_att', 'fair_ens', '#iter']
            csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            # csv_w.writerows([csv_row_1, csv_row_2,
            #                  [''] * 11 + csv_r3c, [''] * 11 + csv_r4c])
            # csv_r5a = ['', '', self._prep, ] + [''] * 8
            # csv_r5a = ['', '', self._prep, 'n_e={}'.format(self._n_e)] + [''] * 7
            csv_r5a = ['', '', self._prep, 'n_e={}'.format(self._n_e)] + [
                ''] * 5 + ['omitted' if self._omit else 'verbose', '']
            csv_w.writerows([csv_row_1, csv_row_2,
                             csv_r5a + csv_r3c, csv_r5a + csv_r4c])
            del csv_r4c, csv_r5a
        elif 'expt5' in self._trial_type:
            csv_row_2b = ['#sen_att', '#eval', '#iter']
            csv_row_1, csv_r2c, csv_r3c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            # csv_w.writerows([csv_row_1, csv_row_2, [''] * 11 + csv_r3c])
            # csv_r5a = ['', '', self._prep, 'n_e= {}'.format(self._n_e)] + [''] * 7
            csv_r5a = ['', '', self._prep, 'n_e= {}'.format(self._n_e)] + [
                ''] * 5 + ['omitted' if self._omit else 'verbose', '']
            csv_w.writerows([csv_row_1, csv_row_2, csv_r5a + csv_r3c])
            del csv_r5a  # del csv_r2c, csv_r3c, csv_r5a
        elif 'expt6' in self._trial_type:
            csv_row_2b = ['curr measure', '#sen_att', '#eval', '#iter']
            csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            # csv_r5a = ['', '', self._prep, 'n_e={}'.format(self._n_e)] + [''] * 8
            csv_r5a = ['', '', self._prep, 'n_e={}'.format(self._n_e)] + [''] * 4 + [
                'omitted' if self._omit else 'verbose /non-omit'] + [''] * 3
            csv_w.writerows([csv_row_1, csv_row_2,
                             csv_r5a + csv_r3c, csv_r5a + csv_r4c])
            del csv_r4c, csv_r5a
        elif 'expt7' in self._trial_type:
            csv_row_2b = ['#sen_att', '#eval', '#iter']
            csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
            csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
            # csv_r5a = ['', '', self._prep, 'n_e= {}'.format(self._n_e),
            #            'fixed' if self._m2_fixed else 'floating'] + [''] * 6  # 流动的
            csv_r5a = ['', '', self._prep, 'n_e= {}'.format(self._n_e)] + [
                ''] * 3 + ['fixed' if self._m2_fixed else 'floating'] + [''] * 3
            csv_w.writerows([csv_row_1, csv_row_2,
                             csv_r5a + csv_r3c, csv_r5a + csv_r4c])
            del csv_r5a, csv_r4c
        else:
            csv_row_2b = ['#sens_attr', '#iter', '#eval']
        del csv_r2c, csv_r3c, csv_row_2
        del csv_row_2a, csv_row_2b, csv_row_1

        # START

        res_data, res_aux = self.coding_per_dataset(logger)
        json_saver = json.dumps({
            "res_aux": res_aux, "res_data": res_data})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w

        csv_w.writerow(res_aux[0])
        if self._trial_type[-6:] in ['expt2b', 'expt3b', 'expt4b']:
            sens_att = res_aux[4][: 2]
            fair_ens = res_aux[5]
            for t_b, tmp_b in enumerate(fair_ens[: 3]):
                csv_w.writerow([''] * 9 + [tmp_b, 0] + res_data[0][t_b])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 9 + ['', k] + res_data[k][t_b])
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens[3:]):
                    csv_w.writerow([''] * 8 + [
                        tmp_a, tmp_b, 0] + res_data[k][t_b + 3 + t_a * 4])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 8 + [
                            '', '', k] + res_data[k][t_b + 3 + t_a * 4])
            del sens_att, fair_ens, t_b, tmp_b, t_a, tmp_a
        elif self._trial_type[-6:] in ['expt2c', 'expt3c', 'expt4c']:
            sens_att = res_aux[4][: 2]
            fair_ens = res_aux[5]
            abbr_clf = res_aux[6]
            for t_b, tmp_b in enumerate(abbr_clf):
                csv_w.writerow([''] * 8 + ['---', tmp_b, 0] + res_data[0][t_b])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b])
            for t_b, tmp_b in enumerate(fair_ens[: 3]):
                csv_w.writerow([''] * 9 + [tmp_b, 0] + res_data[0][t_b + 11])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 9 + ['', k] + res_data[k][t_b + 11])
            for t_a, tmp_a in enumerate(sens_att):
                for t_b, tmp_b in enumerate(fair_ens[3:]):
                    csv_w.writerow([''] * 8 + [
                        tmp_a, tmp_b, 0] + res_data[k][t_b + 14 + t_a * 4])
                    for k in range(1, self._nb_iter):
                        csv_w.writerow([''] * 8 + [
                            '', '', k] + res_data[k][t_b + 14 + t_a * 4])
            del sens_att, fair_ens, abbr_clf, t_b, tmp_b, t_a, tmp_a
        elif self._trial_type[-6:] in ['expt2d', 'expt3d', 'expt4d']:
            sens_att = res_aux[4][: 2]
            fair_ens = res_aux[5][: 3]
            abbr_clf = res_aux[6]
            for t_b, tmp_b in enumerate(abbr_clf):
                csv_w.writerow([''] * 8 + ['---', tmp_b, 0] + res_data[0][t_b])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 8 + ['', '', k] + res_data[k][t_b])
            for t_b, tmp_b in enumerate(fair_ens):
                csv_w.writerow([''] * 9 + [tmp_b, 0] + res_data[0][t_b + 11])
                for k in range(1, self._nb_iter):
                    csv_w.writerow([''] * 9 + ['', k] + res_data[k][t_b + 11])
            del sens_att, fair_ens, abbr_clf, t_b, tmp_b

        elif self._trial_type[-6:] in ['expt5a', 'expt5b']:
            sens_att, priv_val, marginal_grp = res_aux[1: 4]  # sa_vals
            optional_ms = res_aux[5]  # sen_att_w_jt = res_aux[4]
            n_a = len(sens_att)
            n_k = 1 if self._nb_iter <= 0 else self._nb_iter
            row_tmp = [''] * 8
            if n_a == 1:
                sens_att.append('~BLANK~')
            if not self._alternative:
                for i, sa_val in enumerate(sens_att):  # first row: 'DistDirect_bin'
                    csv_w.writerow(row_tmp + [sa_val, 'ApproxDist_bin', 0] + res_data[0][2 * i])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][2 * i])
                    csv_w.writerow(row_tmp + ['', 'DistApprox (bin-val)', 0] + res_data[0][2 * i + 1])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][2 * i + 1])
                # ↑ bin-val ↓ multival
                csv_w.writerow(row_tmp + ['*JOINT*', 'DistExtend (multival)', 0] + res_data[0][4])
                for i_k in range(1, n_k):
                    csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][4])
                for i, sa_val in enumerate(sens_att):
                    csv_w.writerow(row_tmp + [sa_val, 'DistApprox (multival)', 0] + res_data[0][i + 5])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][i + 5])
                # ↑ multival
            else:
                for i, sa_val in enumerate(sens_att):
                    # csv_w.writerow(row_tmp + ['sa_val', 'DistDirect_bin (_10)', 0] + res_data[0][3 * i])
                    csv_w.writerow(row_tmp + [sa_val, 'DistDirect_bin \'_10', 0] + res_data[0][3 * i])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][3 * i])
                    csv_w.writerow(row_tmp + ['', 'ApproxDist_bin', 0] + res_data[0][3 * i + 1])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][3 * i + 1])
                    csv_w.writerow(row_tmp + ['', 'DistApprox (bin-val)', 0] + res_data[0][3 * i + 2])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][3 * i + 2])
                csv_w.writerow(row_tmp + ['*JOINT*', 'DistExtend (multival)', 0] + res_data[0][6])
                for i_k in range(1, n_k):
                    csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][i + 6])
                for i, sa_val in enumerate(sens_att):
                    csv_w.writerow(row_tmp + [sa_val, 'DistApprox (multival)', 0] + res_data[0][i + 7])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + ['', '', i_k] + res_data[i_k][i + 7])
            del sens_att, priv_val, marginal_grp, n_a, n_k, row_tmp

        elif self._trial_type[-6:] in ['expt6a', 'expt6b']:
            if self._rep_iter:  # .shape= (5, 2), first
                res_advs = [res_data[i_k][1] for i_k in range(self._nb_iter)]
                res_data = [res_data[i_k][0] for i_k in range(self._nb_iter)]
            # IF or IF-NOT CROSS-VALIDATION in 'expt6*'
            csv_w.writerow(res_aux[0][: 8] + [''] + res_aux[0][8:])
            sens_att, priv_val, marginal_grp = res_aux[1: 4]  # sa_vals
            optional_ms = res_aux[5]  # sen_att_w_jt = res_aux[4]
            n_a = len(sens_att)
            n_k = 1 if self._nb_iter <= 0 else self._nb_iter
            row_tmp = [''] * 8  # next column: 'current measure'
            if n_a == 1:
                sens_att.append('~BLANK~')
            # res_data= (#iter, 7|9 *4, (3+ 21|24 *3) *2) =(#iter, 36, 66|75 *2)
            # res_data= (#iter, 4, 9, 66|75 *2)    # change to this one!
            # for i_c, curr_m in enumerate(['Ds', 'Df', 'df_ver3', 'df_ver4']):
            for i_c, curr_m in enumerate(['Ds', 'Df', 'df_ecai', 'df_neurips']):
                for i, sa_val in enumerate(sens_att):
                    curr_s = curr_m if i == 0 else ''  # curr_show
                    csv_w.writerow(row_tmp + [curr_s, sa_val, 'DistDirect_bin \'_10',
                                              0] + res_data[0][i_c][3 * i])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + [''] * 3 + [i_k] + res_data[i_k][i_c][3 * i])
                    csv_w.writerow(row_tmp + ['', '', 'ApproxDist_bin',
                                              0] + res_data[0][i_c][3 * i + 1])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + [''] * 3 + [i_k] + res_data[i_k][i_c][3 * i + 1])
                    csv_w.writerow(row_tmp + ['', '', 'DistApprox (bin-val)',
                                              0] + res_data[0][i_c][3 * i + 2])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + [''] * 3 + [i_k] + res_data[i_k][i_c][3 * i + 2])
                    del curr_s
                # JOINT BOTH|EITHER (AND|OR) sensitive attributes
                csv_w.writerow(row_tmp + ['', '*JOINT*', 'DistExtend (multival)',
                                          0] + res_data[0][i_c][6])
                for i_k in range(1, n_k):
                    csv_w.writerow(row_tmp + [''] * 3 + [i_k] + res_data[i_k][i_c][6])
                for i, sa_val in enumerate(sens_att):
                    csv_w.writerow(row_tmp + ['', sa_val, 'DistApprox (multival)',
                                              0] + res_data[0][i_c][i + 7])
                    for i_k in range(1, n_k):
                        csv_w.writerow(row_tmp + [''] * 3 + [i_k] + res_data[i_k][i_c][i + 7])
                # END MULTIVAR in fairmanf_ext
            # IF CROSS-VALIDATION in 'expt6*'
            if self._rep_iter:  # res_advs.shape= (7, (3+n_l*3) *2)
                for i_c, curr_m in enumerate(['Normal', 'Adversarial', 'delta=abs()',
                                              'Fairness sa#1', 'Fairness sa#2',
                                              'Fairness jt 1&2', 'Fairness jt 1|2']):
                    csv_w.writerow(row_tmp + [curr_m, '', '', 0] + res_advs[0][i_c])
                    for i_k in range(1, self._nb_iter):  # n_k
                        csv_w.writerow(row_tmp + ['', '', '', i_k] + res_advs[i_k][i_c])
                del res_advs
            del sens_att, priv_val, marginal_grp, n_a, n_k, row_tmp

        elif self._trial_type[-6:] in ['expt7a']:
            sens_att, priv_val, marginal_grp = res_aux[1: 4]
            sa_with_joint = res_aux[4]  # and this is /that's the end
            n_k = 1 if self._nb_iter <= 0 else self._nb_iter
            row_tmp = [''] * 8 + ['', '', ]
            csv_w.writerow(row_tmp + [0] + res_data[0])
            for i_k in range(1, n_k):
                csv_w.writerow(row_tmp + [i_k] + res_data[i_k])

        else:
            pass

        # END
        return

    def coding_per_dataset(self, logger):
        # X, A, y, Aq, marginalised_group, margin_indices, new_attr, \
        #     res_aux, belongs_priv = self.preparing_current_data(logger)
        X, A, y, Aq, marginalised_group, g1m_indices, new_attr, \
            res_aux, idx_g1, idx_jt = self.preparing_current_data(logger)

        if 'expt2' in self._trial_type:
            res_aux.append(['bagging', 'adaboost', 'lightgbm',
                            'fairgbm : FPR', 'fairgbm : FNR',
                            'fairgbm : FPR,FNR', 'adafair'])
            # if 'expt2c' in self._trial_type:
            if self._trial_type[-2:] in ['2c', '2d']:
                res_aux.append(self._iterator._abbr_clfs)

        # NON- k-FOLD CROSS VALIDATION

        '''
    if self._nb_iter <= 0:
      elegant_print("Running /executing as a whole", logger)
    elegant_print("Not-repetitively, via cv_split?: {}".format(
        'Yes' if self._rep_iter else 'No'), logger)  # cv_split
    if not self._rep_iter:
      elegant_print("nb_iter={}, repeat".format(self._nb_iter), logger)
    '''

        # CROSS VALIDATION, usually 5-fold cross validation

        elegant_print("nb_iter={}, cross_valid".format(self._nb_iter), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif "KF" in self._trial_type:  # 'Kcv'
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print(
            "\t CrossValid  {}\n".format(self._trial_type[: 3]), logger)

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            # X_trn, A_trn, y_trn, g1_trn, jt_trn = transform_disturb_prime(
            #     X, A, y, i_trn, idx_g1, idx_jt)

            (X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X, A, y, Aq, i_trn, g1m_indices,
                                           idx_jt)
            (X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X, A, y, Aq, i_tst, g1m_indices,
                                           idx_jt)

            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                scaler, X_trn, A_trn, _, _, X_tst, A_tst = normalise_disturb_prime(
                    scaler, X_trn, A_trn, [], [], X_tst, A_tst)
            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.coding_per_iteration_cv_split(
                logger, k,
                X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
                X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst)
            res_ans.append(res_iter)  # siz=(5, 3|14 +4* 1|2, 1+ 131|102 *2)
            del X_trn, A_trn, y_trn, g1m_trn, jt_trn
            del X_tst, A_tst, y_tst, g1m_tst, jt_tst
        del X, A, y, Aq, idx_g1, idx_jt
        return res_ans, res_aux

    # PREPARATION (preparation)

    def preparing_current_data(self, logger=None):
        '''
        processed_data = process_above(
            self._dataset, self._data_frame, logger=logger)
        addtl_proc_dat = process_addtl(self._dataset, processed_data)
        addtl_mult_dat = process_addtl_multivalue(
            self._dataset, addtl_proc_dat['original'])  # addtl_binz_dat

        origin_dat, prep_bin, prep_mu, disturb_bin, disturb_mu = \
            renewed_prep_and_adversarial(
                self._dataset, self._data_frame, self._ratio, logger=logger)
        '''

        origin_dat, processed_data, process_mult,\
            disturbed_data, disturb_mult = renewed_prep_and_adversarial(
                self._dataset, self._data_frame, self._ratio, logger=logger)
        processed_Xy = process_mult['numerical-multisen']
        disturbed_Xy = disturb_mult['numerical-multisen']
        X, A, y, _ = renewed_transform_X_A_and_y(
            self._dataset, processed_Xy, with_joint=False)
        _, Aq, _, _ = renewed_transform_X_A_and_y(
            self._dataset, disturbed_Xy, with_joint=False)
        # NB. X, A, Aq, y: all pd.DataFrame

        self._m2 = np.ceil(2 * np.log10(len(y)))
        self._m2 = int(self._m2)
        elegant_print(["Due to #inst = {}".format(len(y)),
                       "self._m2 = {}".format(self._m2),
                       "self._m1 = {}".format(self._m1)], logger)
        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values))], logger)
        del tmp

        sens_att = self._dataset.get_sensitive_attrs_with_joint()[: 2]
        priv_val = self._dataset.get_privileged_group_with_joint('')[: 2]
        marginalised_group = origin_dat['marginalised_groups']
        self.saValue = [(np.arange(
            len(sa_val)) + 2).tolist() for sa_val in marginalised_group]
        # self.saValue = [(
        #   np.arange(len(sa_val)) + 2).tolist() for sa_val in margin_indices]
        margin_indices = check_marginalised_indices(
            processed_data['original'], sens_att, priv_val,
            marginalised_group)
        # new_attr_name = '-'.join(sensitive_attrs)
        new_attr = '-'.join(sens_att) if len(sens_att) > 1 else None
        # belongs_priv = origin_dat['belongs_priv']
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'], 'both')

        tmp_cls = ''  # None
        if 'expt2' in self._trial_type:
            tmp_cls = self._nb_cls
        res_aux = [[
            self._dataset.dataset_name, len(set(y.values)), tmp_cls,
            self._nb_iter, self._gen_iter, self._rep_iter,
            self._m1, self._m2, len(sens_att), '', ''],
            # self._dataset.sensitive_attrs,
            # self._dataset.privileged_vals,
            sens_att, priv_val, marginalised_group, ]  # margin_indices, ]
        if len(sens_att) == 1:
            res_aux.append(sens_att)
        else:
            res_aux.append(sens_att + [new_attr] + ['joint_and,or'])

        return (X.values, A.values, y.values, Aq.values,
                marginalised_group, margin_indices, new_attr, res_aux,
                belongs_priv, ptb_with_joint)

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
            X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst):
        since = time.time()
        res_iter = []

        if 'expt2' in self._trial_type:
            # pm_m = {'m1': self._m1, 'm2': self._m2}
            positive_label = self._dataset.get_positive_class_val(
                'numerical-binsensitive')
            res_iter = self._iterator.schedule_content(
                logger,
                X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
                X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst,
                self._m1, self._m2, positive_label, omitted=self._omit)
            # # expt2b: siz= ( 3+4* 1|2, 1+ 131|102 *2)
            # # expt2c: siz= (14+4* 1|2, 1+ 131|102 *2)

        else:
            pass
        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, fantasy_durat_major(tim_elapsed, True, False)), logger)
        return res_iter


# -------------------------------
# Maniford ext. empirical


class ManfExtPrime_Empirical(ManfExtEmpirical):
    def preparing_current_data(self, logger=None):
        origin_dat, processed_data, process_mult, \
            disturbed_data, disturb_mult = renewed_prep_and_adversarial(
                self._dataset, self._data_frame, self._ratio, logger=logger)
        processed_Xy = process_mult['numerical-multisen']
        disturbed_Xy = disturb_mult['numerical-multisen']
        # X, A, y, _, X_and_A = renewed_transform_X_A_and_y(
        #     self._dataset, processed_Xy, with_joint=False)
        # _, Aq, _, _, X_and_Aq = renewed_transform_X_A_and_y(
        #     self._dataset, disturbed_Xy, with_joint=False)
        # X_A, y, new_attr = renewed_transform_X_and_y(
        #     self._dataset, processed_Xy, with_joint=False)
        X_A, y = transform_X_and_y(self._dataset, processed_Xy)
        X_Aq, y = transform_X_and_y(self._dataset, disturbed_Xy)
        # NB. X, A, Aq, y: all pd.DataFrame

        if not self._m2_fixed:
            self._m2 = np.ceil(2 * np.log10(len(y)))
            self._m2 = int(self._m2)
        elegant_print(["Due to #inst = {}".format(len(y)),
                       "self._m2 = {}".format(self._m2),
                       "self._m1 = {}".format(self._m1)], logger)
        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values))], logger)
        del tmp

        sens_att = self._dataset.get_sensitive_attrs_with_joint()[: 2]
        priv_val = self._dataset.get_privileged_group_with_joint('')[: 2]
        marginalised_group = origin_dat['marginalised_groups']
        self.saValue = [(np.arange(
            len(sa_val)) + 2).tolist() for sa_val in marginalised_group]
        margin_indices = check_marginalised_indices(
            processed_data['original'], sens_att, priv_val,
            marginalised_group)
        new_attr = '-'.join(sens_att) if len(sens_att) > 1 else None
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'], 'both')

        tmp_cls = ''  # None
        # if 'expt3' in self._trial_type:
        if ('expt3' in self._trial_type) or ('expt4' in self._trial_type):
            tmp_cls = self._nb_cls
        tmp_ens = ''
        if ('expt6' in self._trial_type):
            tmp_cls = '#{}'.format(self._nb_cls)  # '{} ({}) #{}'.format()
            tmp_ens = self._abbr_cls
            if self._abbr_cls == 'FairGBM':
                tmp_ens += ' ({})'.format(self._constraint_type)
            # tmp_ens = '#{} {}'.format(, self._abbr_cls)
            # if self._abbr_cls == 'FairGBM':
            #   tmp _cls = '{} ({})'.format(tmp_cls, self._constraint_type)
        res_aux = [[
            self._dataset.dataset_name, len(set(y.values)), tmp_cls,
            self._nb_iter, self._gen_iter, self._rep_iter,
            # self._m1, self._m2, len(sens_att), '', ''],
            self._m1, self._m2, len(sens_att), '', tmp_ens],
            # self._dataset.sensitive_attrs,
            # self._dataset.privileged_vals,
            sens_att, priv_val, marginalised_group, ]  # margin_indices, ]
        if len(sens_att) == 1:
            res_aux.append(sens_att)
        else:
            res_aux.append(sens_att + [new_attr] + ['joint_and,or'])

        # return (X.values, A.values, y.values, Aq.values,
        #         marginalised_group, margin_indices, new_attr, res_aux,
        #         belongs_priv, ptb_with_joint,
        #         X_and_A.values, X_and_Aq.values)  # X_and_Ap.values)
        del tmp_cls, tmp_ens
        return (X_A.values, y.values, X_Aq.values,
                marginalised_group, margin_indices, new_attr, res_aux,
                belongs_priv, ptb_with_joint)

    def preparing_iterator(self, trial_type, abbr_cls, nb_iter, gen, rep,
                           m1, m2, n_e, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep  # ._cvs_iter
        self._prep = prep  # preprocess data, pre-processing
        self._m1, self._m2 = m1, m2
        self._n_e = n_e

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        # self.saIndex = [-1] if self._data_type == 'ricci' else [-2, -1]
        self._screen, self._logged = screen, logged

        self._abbreviation_dat_name = self._log_document

        # pdb.set_trace()
        # expt2* 是修改了 saIndex 的位置
        #   expt3* 未修改 saIndex 的位置，
        #   expt4* 是重写了 part2 (group fairness)，使输出信息更多
        #          不是信息输出更多，是改变了L_loss和L_fair的位置，以及其他GroupFairness的相对位置
        #          确实输出信息更多了，因为我 part1 多加了一些东西（所以有实验四，就不需要三了）
        #   expt5* 是参数变化，参数敏感性分析；expt6* 也是，只不过多了分类器结果
        #   expt5* 要重复多次；expt6* 要 mCV 交叉验证的，就不需要重复多次了
        if trial_type.endswith('expt3b'):
            self._iterator = ComparisonD2_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt3c'):
            self._iterator = ComparisonD3_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt3d'):
            self._iterator = ComparisonD4_withDirectComput(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt4b'):
            self._iterator = ComparisonE2_with(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt4c'):
            self._iterator = ComparisonE3_with(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        elif trial_type.endswith('expt4d'):
            self._iterator = ComparisonE4_with(
                nb_cls, self.saIndex, self.saValue, n_e, omitted=self._omit)
        # elif trial_type.endswith('expt5a'):
        #   self._iterator = HyperEA_analysis(abbr_cls)
        # elif trial_type.endswith('expt5b'):
        #   self._iterator = HyperEB_analysis(abbr_cls)
        # elif trial_type.endswith('expt6a'):
        #   self._iterator = HyperFA_analysis(
        #       abbr_cls, nb_cls, constraint_type, self.saIndex, self.saValue)
        # elif trial_type.endswith('expt6b'):
        #   self._iterator = HyperFB_analysis(
        #       abbr_cls, nb_cls, constraint_type, self.saIndex, self.saValue)
        elif trial_type.endswith('expt5a'):
            self._iterator = HyperEA_analysis(abbr_cls, omitted=self._omit)
        elif trial_type.endswith('expt5b'):
            self._iterator = HyperEB_analysis(abbr_cls, omitted=self._omit)
        elif trial_type.endswith('expt6a'):
            self._iterator = HyperFA_analysis(
                abbr_cls, nb_cls, constraint_type, self.saIndex, self.saValue,
                omitted=self._omit)
        elif trial_type.endswith('expt6b'):
            self._iterator = HyperFB_analysis(
                abbr_cls, nb_cls, constraint_type, self.saIndex, self.saValue,
                omitted=self._omit)
        elif trial_type.endswith('expt7a'):
            self._iterator = Multiprocess_GA_comparison(abbr_cls)

        # if 'expt3' in trial_type:
        '''
    if ('expt3' in trial_type) or ('expt4' in trial_type):
      formatted_log = "_".join([
          trial_type,
          'iter{}'.format(nb_iter) if nb_iter > 0 else 'sing',
          'cls{}'.format(nb_cls), prep, self._log_document,
          'pms', 'ratio{}'.format(int(self._ratio * 100)), ])
    '''
        formatted_log = "_".join([
            trial_type,
            'iter{}'.format(nb_iter) if nb_iter > 0 else 'sing',
            ''])
        if ('expt3' in trial_type) or ('expt4' in trial_type):
            formatted_log += '_'.join([
                'cls{}'.format(nb_cls), prep, self._log_document,
                'pms', 'ratio{}'.format(int(self._ratio * 100)), ])
        elif ('expt5' in trial_type) or ('expt6' in trial_type):
            formatted_log += '_'.join([prep, self._log_document, 'pms'])
            if trial_type[-6:] in ['expt5a', 'expt6a']:
                formatted_log = '{}_ma{}'.format(formatted_log, self._m1)
            elif trial_type[-6:] in ['expt5b', 'expt6b']:
                formatted_log = '{}_mb{}'.format(formatted_log, self._m2)
            formatted_log += ('_alt' * self._alternative)
            if trial_type[-6:] in ['expt6a', 'expt6b']:
                formatted_log = '{}_cvs{}_{}'.format(
                    formatted_log, int(self._solo_cv_siz * 100), abbr_cls)
                if abbr_cls in ['FairGBM', 'fairGBM']:
                    formatted_log += '_{}'.format(constraint_type)
        elif ('expt7' in trial_type):
            formatted_log += '_'.join([prep, self._log_document, 'pms'])
            formatted_log = '{}_ma{}'.format(formatted_log, self._m1)
            formatted_log = '{}_mb{}'.format(formatted_log, self._m2)
            formatted_log += ('_m2fixed' * self._m2_fixed)

        self._log_document = formatted_log
        self._log_document += ('_rep' * rep + '_gen' * gen)
        return

    def coding_per_dataset(self, logger):
        # (X, A, y, Aq, marginalised_group, g1m_indices, new_attr, res_aux,
        #     idx_g1, idx_jt, X_and_A, X_and_Aq) = self.preparing_current_data(
        #         logger)
        (X_and_A, y, X_and_Aq, marginalised_group, g1m_indices, new_attr,
         res_aux, idx_g1, idx_jt) = self.preparing_current_data(logger)
        pool = pp.ProcessingPool(nodes = self._mp_cores)  # i=3)
        # or None if self._mp_cores==0

        # if 'expt3' in self._trial_type:
        if ('expt3' in self._trial_type) or ('expt4' in self._trial_type):
            res_aux.append(['bagging', 'adaboost', 'lightgbm',
                            'fairgbm : FPR', 'fairgbm : FNR',
                            'fairgbm : FPR,FNR', 'adafair'])
            # if self._trial_type[-2:] in ['3c', '3d']:
            if self._trial_type[-2:] in ['3c', '3d', '4c', '4d']:
                res_aux.append(self._iterator._abbr_clfs)

        elif ('expt5' in self._trial_type) or ('expt6' in self._trial_type):
            if self._trial_type[-6:] in ['expt5a', 'expt6a']:
                res_aux.append(self._iterator._m2_set)
            elif self._trial_type[-6:] in ['expt5b', 'expt6b']:
                res_aux.append(self._iterator._m1_set)

        # NON- k-FOLD CROSS VALIDATION

        # #
        # # START OF non- cross-validation

        pm_k = {}  # pm_s
        if 'expt6' in self._trial_type:
            solo_cv_idx = list(range(len(y)))
            np.random.shuffle(solo_cv_idx)
            solo_cv_len = int(len(y) * self._solo_cv_siz)
            pm_k['i_trn'] = solo_cv_idx[: -solo_cv_len]
            pm_k['i_tst'] = solo_cv_idx[-solo_cv_len:]
            del solo_cv_idx, solo_cv_len
            # pm_k['idx_jt'] = idx_jt
        if self._nb_iter <= 0:
            elegant_print("Running /executing as a whole", logger)
            if self._prep != 'none':
                scaler = scale_normalize_helper(self._prep)
                # scaler, X_scaled, A_scaled, _, _ = normalise_disturb_prime(
                #   scaler, X_and_A, )
                (scaler, X_and_A, _, X_and_Aq, X, A, _, _, Xq, Aq
                 ) = renewed_normalise_disturb(
                    scaler, X_and_A, [], X_and_Aq, self.saIndex)
                del scaler
            else:
                X, A, _, _, Xq, Aq = renewed_normalise_separate(
                    X_and_A, [], X_and_Aq, self.saIndex)
            # res_data = self.coding_per_iteration_as_whole(X, A, y, g1m_indices)
            pm_k['idx_jt'] = idx_jt
            pm_k['X_and_A'] = X_and_A
            # res_data = self.coding_per_iteration_as_whole(logger, X, A, y,
            #                                               g1m_indices, **pm_k)
            res_data = self.coding_per_iteration_as_whole(
                logger, pool, X, A, y, g1m_indices, **pm_k)
            # currently res_data.shape =(7, 3+ 63|72)
            del X, A, Xq, Aq, X_and_A, X_and_Aq
            return [res_data], res_aux

        #   return
        # elegant_print("Not-repetitively, via cv_split?: {}".format(
        #     'Yes' if self._rep_iter else 'No'), logger)  # cv_split
        elegant_print("No solo-execution, via cv_split (/ non-repeatition)?"
                      "".format('Yes' if self._rep_iter else 'No'), logger)
        if not self._rep_iter:
            # elegant_print("nb_iter={}, repeat".format(self._nb_iter), logger)
            elegant_print("Repetitive {} times".format(self._nb_iter), logger)
            split_idx = manual_repetitive(self._nb_iter, y, self._gen_iter)
            res_data = []  # res_ans = []
            for k, idx in enumerate(split_idx):
                elegant_print("Iteration {}-th starts.".format(k + 1), logger)
                (prim_X_A, _, prim_y, prim_X_Aq, prim_g1m,
                 prim_jt) = renewed_transform_disturb(
                    X_and_A, None, y, X_and_Aq, idx, g1m_indices, idx_jt)
                pm_k['idx_jt'] = prim_jt

                if self._prep in ['standard', 'min_max', 'normalize']:
                    scaler = scale_normalize_helper(self._prep)
                    (scaler, prim_X_A, _, prim_X_Aq, X, A, _, _, Xq, Aq
                     ) = renewed_normalise_disturb(
                        scaler, prim_X_A, [], prim_X_Aq, self.saIndex)
                    del scaler
                else:
                    X, A, _, _, Xq, Aq = renewed_normalise_separate(
                        prim_X_A, [], prim_X_Aq, self.saIndex)
                pm_k['X_and_A'] = prim_X_A
                # res_iter = self.coding_per_iteration_as_whole(logger, X, A, y,
                #                                               prim_g1m, **pm_k)
                res_iter = self.coding_per_iteration_as_whole(
                    logger, pool, X, A, y, prim_g1m, **pm_k)
                # BUG? prim_y ABOVE
                res_data.append(res_iter)  # .shape= (7, 3+ 21|24 *3)
                del X, A, Xq, Aq  # , prim_X_A, prim_X_Aq
                del prim_X_A, prim_X_Aq, prim_y, prim_g1m, prim_jt
            return res_data, res_aux
        del pm_k

        # # END OF NON- CROSS-VALIDATION
        # #

        # CROSS VALIDATION, usually 5-fold cross validation

        elegant_print("nb_iter={}, cross_valid".format(self._nb_iter), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif "KF" in self._trial_type:  # 'Kcv'
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print(
            "\t CrossValid  {}\n".format(self._trial_type[: 3]), logger)

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            '''
            (X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X, A, y, Aq, i_trn, g1m_indices,
                                           idx_jt)
            (X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X, A, y, Aq, i_tst, g1m_indices,
                                           idx_jt)
            if isinstance(X_trn, pd.DataFrame):
              X_and_A_trn = X_and_A.iloc[i_trn]
              X_and_A_tst = X_and_A.iloc[i_tst]
              X_and_Aq_trn = X_and_Aq.iloc[i_trn]
              X_and_Aq_tst = X_and_Aq.iloc[i_tst]
            elif isinstance(X_trn, np.ndarray):
              X_and_A_trn = X_and_A[i_trn]
              X_and_A_tst = X_and_A[i_tst]
              X_and_Aq_trn = X_and_Aq[i_trn]
              X_and_Aq_tst = X_and_Aq[i_tst]
            '''

            (X_A_trn, _, y_trn, X_Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X_and_A, None, y, X_and_Aq, i_trn,
                                           g1m_indices, idx_jt)
            (X_A_tst, _, y_tst, X_Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X_and_A, None, y, X_and_Aq, i_tst,
                                           g1m_indices, idx_jt)
            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                # scaler, X_trn, A_trn, _, _, X_tst, A_tst = normalise_disturb_prime(
                #     scaler, X_trn, A_trn, X_val, A_val, X_tst, A_tst)
                (scaler, X_A_trn, _, X_A_tst,
                 X_trn, A_trn, _, _, X_tst, A_tst) = renewed_normalise_disturb(
                    scaler, X_A_trn, [], X_A_tst, self.saIndex)
            else:
                X_trn, A_trn, _, _, X_tst, A_tst = renewed_normalise_separate(
                    X_A_trn, [], X_A_tst, self.saIndex)
            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.coding_per_iteration_cv_split(
                logger, pool, k,
                X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
                # X_trn, X_tst)
                X_trn, X_tst, A_trn, A_tst)
            res_ans.append(res_iter)  # siz=(5, 3|14 +4* 1|2, 1+ 131|102 *2)
            del X_trn, A_trn, y_trn, g1m_trn, jt_trn, X_A_trn, X_Aq_trn
            del X_tst, A_tst, y_tst, g1m_tst, jt_tst, X_A_tst, X_Aq_tst
        del X_and_A, y, X_and_Aq, idx_g1, idx_jt
        del pool
        return res_ans, res_aux

    def coding_per_iteration_cv_split(
            # self, logger, k,
            self, logger, pool, k,
            X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
            X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
            # X_trn, X_tst):
            X_trn, X_tst, A_trn, A_tst):
        since = time.time()
        res_iter = []

        # if 'expt3' in self._trial_type:
        if ('expt3' in self._trial_type) or ('expt4' in self._trial_type):
            pm_m = {'m1': self._m1, 'm2': self._m2}
            positive_label = self._dataset.get_positive_class_val(
                'numerical-binsensitive')
            '''
      res_iter = self._iterator.schedule_content(
          logger,
          X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
          X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
          self._m1, self._m2, positive_label,
          X_trn, X_tst, omitted=self._omit)
      # expt3b: siz= ( 3+4* 1|2, 1+ 131|102 *2)
      # expt3c: siz= (14+4* 1|2, 1+ 131|102 *2)
      # expt3d: siz= (11+3+0   , 1+ 131|102 *2)
      '''
            res_iter = self._iterator.schedule_content_prime(
                logger, pool,
                X_A_trn, y_trn, X_Aq_trn, g1m_trn, jt_trn,
                X_A_tst, y_tst, X_Aq_tst, g1m_tst, jt_tst,
                self._m1, self._m2, positive_label,
                X_trn, A_trn, X_tst, A_tst, omitted=self._omit)

        elif 'expt6' in self._trial_type:
            self._iterator.initialize_clf(self._abbr_cls)
            clf = self._iterator.member
            if self._abbr_cls in ['FairGBM', 'fairGBM']:
                non_sa_idx = g1m_trn[0] if len(g1m_trn) == 0 else jt_trn[1]
                clf.fit(X_A_trn, y_trn, constraint_group=~non_sa_idx)
                del non_sa_idx
            else:
                clf.fit(X_A_trn, y_trn)
            y_insp = clf.predict(X_A_trn)
            y_pred = clf.predict(X_A_tst)
            hat_yq_insp = clf.predict(X_Aq_trn)
            hat_yq_pred = clf.predict(X_Aq_tst)

            pm_m = {'n_e': self._n_e}
            if self._trial_type[-2:] in ['6a']:
                pm_m['m1'] = self._m1
                n_l = len(self._iterator._m2_set)
            elif self._trial_type[-2:] in ['6b']:
                pm_m['m2'] = self._m2
                n_l = len(self._iterator._m1_set)
            pm_m['pool'] = pool
            res_iter = self._iterator.schedule_content_wcf(
                X_trn, A_trn, y_trn, y_insp, g1m_trn,
                X_tst, A_tst, y_tst, y_pred, g1m_tst, **pm_m)
            # res_iter.shape= (4, 9, 66|75 *2)
            positive_label = self._dataset.get_positive_class_val(
                'numerical-binsensitive')  # each gather.shape= (7,3+n_l*3)
            gather_trn = self._iterator.subproc_adversarial(
                y_trn, y_insp, hat_yq_insp, positive_label, g1m_trn, jt_trn, n_l)
            gather_tst = self._iterator.subproc_adversarial(
                y_tst, y_pred, hat_yq_pred, positive_label, g1m_tst, jt_tst, n_l)
            ans_gather = [t_trn + t_tst for t_trn, t_tst in zip(gather_trn, gather_tst)]
            del gather_trn, gather_tst  # ans_gather.shape= (7, (3+n_l*3) *2)
            res_iter = [res_iter, ans_gather]

        else:
            pass
        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, fantasy_durat_major(tim_elapsed, True, False)), logger)
        return res_iter

    def coding_per_iteration_as_whole(self, logger, pool,
                                      X, A, y, g1m_indices, idx_jt=None,
                                      i_trn=None, i_tst=None, X_and_A=None):
        since = time.time()
        res_iter = []

        if 'expt5' in self._trial_type:  # ['expt5a', 'expt5b']
            pm_m = {'n_e': self._n_e}
            pm_m['alternative'] = self._alternative
            if self._trial_type.endswith('5a'):
                pm_m['m1'] = self._m1
            elif self._trial_type.endswith('5b'):
                pm_m['m2'] = self._m2
            # tmp = self._iterator.schedule_content(X, A, y, g1m_indices, **pm_m)
            # res_iter.append(tmp)  # tmp.shape= (7,66|75) =(n_a*2+3, 3+21|24*3)
            pm_m['pool'] = pool
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices, **pm_m)  # .shape= (7, 3+ 21|24 *3)

        elif 'expt6' in self._trial_type:  # ['expt6a', 'expt6b']
            Aq = None  # Aq_trn, Aq_tst,
            (X_trn, A_trn, y_trn, _, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X, A, y, Aq, i_trn, g1m_indices, idx_jt)
            (X_tst, A_tst, y_tst, _, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X, A, y, Aq, i_tst, g1m_indices, idx_jt)
            X_A_trn, _, _, _, _, _ = renewed_transform_disturb(
                X_and_A, Aq, y, Aq, i_trn, g1m_indices, idx_jt)
            X_A_tst, _, _, _, _, _ = renewed_transform_disturb(
                X_and_A, Aq, y, Aq, i_tst, g1m_indices, idx_jt)
            clf = self._iterator.member
            if self._abbr_cls in ['FairGBM', 'fairGBM']:
                non_sa_idx = g1m_trn[0] if len(g1m_trn) == 1 else jt_trn[1]
                # non_sa_idx = g1m_trn[0] if len(g1m_trn) == 1 else jt_trn[1]  # jt_|
                # non_sa_idx = g1m_trn[0] if len(g1m_trn) == 1 else jt_trn[0]  # jt_&
                clf.fit(X_A_trn, y_trn, constraint_group=~non_sa_idx)
                del non_sa_idx
            else:
                clf.fit(X_A_trn, y_trn)
            y_insp = clf.predict(X_A_trn)  # fx_trn
            y_pred = clf.predict(X_A_tst)  # fx_tst
            del X_A_trn, X_A_tst, jt_trn, jt_tst, Aq

            pm_m = {'n_e': self._n_e}
            if self._trial_type.endswith('6a'):
                pm_m['m1'] = self._m1
            elif self._trial_type.endswith('6b'):
                pm_m['m2'] = self._m2
            pm_m['pool'] = pool
            res_iter = self._iterator.schedule_content_wcf(
                X_trn, A_trn, y_trn, y_insp, g1m_trn,
                X_tst, A_tst, y_tst, y_pred, g1m_tst, **pm_m)
            # res_iter.shape= (7|9 *4, (3+ 21|24 *3) *2) =(36, 66|75 *2)
            # res_iter.shape= (4, 9, 66|75 *2)

        elif 'expt7' in self._trial_type:  # ['expt7a',]
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices, self._m1, self._m2, self._n_e, pool)
            # res_iter.shape= (58,)

        tim_elapsed = time.time() - since
        elegant_print("As a whole, consumed {}".format(
            fantasy_durat_major(tim_elapsed, True, False)), logger)
        return res_iter
