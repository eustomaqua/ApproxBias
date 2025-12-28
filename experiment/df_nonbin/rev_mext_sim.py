# coding: utf-8

import csv
import json
import os
import sys
import time

import numpy as np
from pathos import multiprocessing as pp


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

from experiment.df_nonbin.rev_mext_exp_mp import (
    RevCompZA_efficient, RevCompZB_efficient, RevCompZC_efficient,
    RevCompYA_NN, RevCompYB_NN, RevCompYC_NN, RevCompXB_NN, RevCompXD_NN,
    RevCompXE_ensemble, RevCompXF_learner)
from experiment.df_nonbin.rev_mext_exp_mp import (
    ConvergeE2_with, ConvergeE3_with, ConvergeE4_with, ConvergeE5_with,
    ConvergeF2_with, ConvergeF3_with, ConvergeF4_with, ConvergeF5_with,
    ConvergeF7_with, ConvergeF8_with)
import pdb


# =====================================
# fairmanf_ext
# mext_sim.py


# AVAILABLE_CLFS = list(INDIVIDUALS.keys())
# AVAILABLE_ENSF = [
#     'bagging', 'AdaBoost',
#     # 'LightGBM', 'FairGBM', 'AdaFair',
#     'lightGBM', 'fairGBM', 'AdaFair', ]


# -------------------------------------
# Manifold ext. empirical


class Rev_ManfExtEmpir(DataSetup):
    def __init__(self, trial_type, data_type, nb_iter=5, m1=25, m2=11,
                 n_e=3, ratio=.7, prep=False, gen=False, rep=False,
                 nb_cls=1, abbr_cls='DT', constraint_type='FPR,FNR',
                 omitted=True, alternative=True, solo_cv_siz=.8,
                 mp_cores=3, m2_fixed=False, screen=True, logged=False):
        super().__init__(data_type)
        self._ratio = ratio
        self._omit = omitted
        self._alternative = alternative  # for 'expt5*'
        self._solo_cv_siz = solo_cv_siz  # for 'expt6*'
        self._mp_cores = mp_cores  # number of cores for multiprocessing
        self._m2_fixed = m2_fixed

        self._trial_type = trial_type
        self._nb_iter = nb_iter  # default:0
        self._gen_iter = gen
        self._rep_iter = rep  # ._cvs_iter (_cv_split)
        self._prep = prep  # preprocess data, pre-processing
        self._m1, self._m2, self._n_e = m1, m2, n_e
        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        self._screen, self._logged = screen, logged
        self._abbreviation_dat_name = self._log_document
        self.preparing_iterator(
            trial_type, abbr_cls, nb_iter, gen, rep, m1, m2, n_e,
            nb_cls, constraint_type, prep, screen, logged)

    def preparing_iterator(self, trial_type, abbr_cls, nb_iter, gen, rep,
                           m1, m2, n_e, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        self.saIndex = [-1] if self._data_type == 'ricci' else [-2, -1]
        # expt2* 是修改了 saIndex 的位置
        # expt3* 未修改 saIndex 的位置，expt4* 在此基础上重写，使输出信息更多
        #      不是信息输出更多，是改变L_loss和L_fair的位置，及其他GroupFairness的相对位置
        #      确实输出信息更多了，因为我 part1 多加了一些东西（所以有实验四，就不需要三了）
        # expt5* 是参数变化，参数敏感性分析；expt6* 也是，只不过多了分类器结果
        # expt5* 要重复多次；expt6* 要 mCV 交叉验证的，就不需要重复多次了
        if trial_type.endswith('expt2b'):
            pass
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
                "fairmanf_ext.Rev.", self._log_document + ".txt")
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
        tim_consumed = fantasy_durat(tim_elapsed, False, True)
        # tim_consumed = tim_consumed.replace('\n\t i.e.,', '')
        elegant_print([
            "",
            "Duration /TimeCost: {}".format(tim_consumed),
            "Alternative         {}".format(
                tim_consumed.replace('\n\t i.e.,', '')),
            "[ENDED AT {:s}]".format(elegant_dated(time.time()))], logger)
        del tim_consumed
        # elegant_print([
        #     "",
        #     "Duration /TimeCost: {}".format(fantasy_durat(tim_elapsed,
        #                                                   False, True)),
        #     "[ENDED AT {:s}]".format(elegant_dated(time.time()))], logger)

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

    # EACH SUB-ROUTE

    def coding_per_procedure(self, csv_w, logger):
        # csv_row_2a = ['data_name', 'binary', 'abbr nb_cls',
        #               'nb_iter', 'gen', 'rep/cv', 'm1', 'm2',
        #               '#sen-att', '', '']
        csv_row_2a = ['data_name', 'binary', '#sen-att',
                      '#cv', 'gen', 'rep', 'm1', 'm2',
                      'n_e', 'k?']  # 'k']  # '#k,nb_iter'
        csv_row_1, cr2c, cr3c, cr4c = self._iterator.prepare_trial()
        # if 'rexp1' in self._trial_type:
        if self._trial_type[-6: -1] in ('rexp1', 'rexp2'):
            # csv_row_1, cr2c, cr3c, cr4c = self._iterator.prepare_trial()
            cr2c = csv_row_2a + cr2c
            cr3c = [''] * 10 + cr3c
            cr4c = [''] * 9 + [self._prep] + cr4c
            # csv_w.writerows([csv_row_1, cr2c, cr3c, cr4c])
            # del csv_row_1, cr2c, cr3c, cr4c
        elif self._trial_type[-6: -1] in ('rexp3',):  # 'rexp4b',
            cr2c = csv_row_2a + ['abbr_cls'] + cr2c
            # cr2c = csv_row_2a[:-1] + ['abbr_cls', 'k?']
            cr3c = [''] * 11 + cr3c
            cr4c = [''] * 9 + [self._prep, ''] + cr4c
        elif self._trial_type[-6: -1] in (  # 'rexp5b', 'rexp4b',
                'rexp9b', 'rexp8b', 'rexp9f', 'rexp9g'):
            cr2c = csv_row_2a + ['abbr_cls'] + cr2c
            cr3c = [''] * 11 + cr3c
            cr4c = ['', '', '', self._prep] + [''] * 7 + cr4c

        elif self._trial_type[-6: -1] in (
                'rexp8', 'rexp9',):  # 'rexp4', 'rexp5',
            cr2c = csv_row_2a + cr2c
            cr3c = [''] * 10 + cr3c
            # cr4c = [''] * 9 + [self._prep] + cr4c
            cr4c = ['', '', '', self._prep] + [''] * 6 + cr4c
        csv_w.writerows([csv_row_1, cr2c, cr3c, cr4c])
        del csv_row_1, cr2c, cr3c, cr4c

        # START

        res_data, res_aux = self.coding_per_dataset(logger)
        # json_saver = json.dumps({
        #     'res_aux': res_aux, 'res_data': res_data})
        json_saver = json.dumps({
            "res_aux": res_aux, "res_data": res_data})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w

        # END

        csv_w.writerow(res_aux[0])
        sens_att, priv_val, mrg_grp = res_aux[1:4]
        nk = 1 if self._nb_iter <= 0 else self._nb_iter
        # if 'rexp1' in self._trial_type:
        if self._trial_type[-6: -1] in ('rexp1', 'rexp2'):
            for k in range(nk):
                csv_w.writerow([''] * 9 + [k] + res_data[k])
        elif self._trial_type[-6:] in ('rexp3c', 'rexp3d'):
            # elif self._trial_type[-6: -1] in ('rexp3'):
            # for k in range(nk):
            #   csv_w.writerow([''] * 10 + [k] + res_data[k])
            k = self._iterator._abbr_cls  # k = 0
            csv_w.writerow([''] * 9 + [0, k] + res_data[0])
            for k in range(1, nk):
                csv_w.writerow([''] * 9 + [k, ''] + res_data[k])

        elif self._trial_type[-6:] in ('rexp3f',):
            for j, jk in enumerate(self._iterator.learners_inside):
                csv_w.writerow([''] * 9 + [0, jk] + res_data[0][j])
                for k in range(1, nk):
                    csv_w.writerow([''] * 9 + [k, ''] + res_data[k][j])
        elif self._trial_type[-6:] in ('rexp3e',):
            for j, jk in enumerate(self._iterator.learners_inside[:3]):
                csv_w.writerow([''] * 9 + [0, jk] + res_data[0][j])
                for k in range(1, nk):
                    csv_w.writerow([''] * 9 + [k, ''] + res_data[k][j])
            for i, sa in enumerate(sens_att):
                for j, jk in enumerate(self._iterator.learners_inside[3:]):
                    csv_w.writerow([''] * 7 + [
                        f'sa #{i+1}', sa, 0, jk] + res_data[0][i * 4 + j + 3])
                    for k in range(1, nk):
                        csv_w.writerow([''] * 9 + [k, ''] + res_data[k][i * 4 + j + 3])
            # pass

        elif self._trial_type[-6:] in (
            # 'rexp4c', 'rexp4d', 'rexp4e',
            # 'rexp5c', 'rexp5d', 'rexp5e',
                'rexp8c', 'rexp8d', 'rexp8e',
                'rexp9c', 'rexp9d', 'rexp9e'):
            for k in range(nk):
                csv_w.writerow([''] * 9 + [k] + res_data[k])
        elif self._trial_type[-6:] in (  # 'rexp4b','rexp5b',
                'rexp8b', 'rexp9b', 'rexp9f', 'rexp9g'):
            # for j, jk in enumerate(self._iterator.learners_inside):
            #     csv_w.writerow([''] * 9 + [0, jk] + res_data[0][j])
            #     for k in range(1, nk):
            #         csv_w.writerow([''] * 9 + [k, ''] + res_data[k][j])
            #
            for j, jk in enumerate(self._iterator.learners_inside[:3]):
                csv_w.writerow([''] * 9 + [0, jk] + res_data[0][j])
                for k in range(1, nk):
                    csv_w.writerow([''] * 9 + [k, ''] + res_data[k][j])
            for t, sa in enumerate(res_aux[1]):
                for j, jk in enumerate(self._iterator.learners_inside[3:]):
                    csv_w.writerow([''] * 6 + [
                        sa, '', '', 0, jk] + res_data[0][3 + 4 * t + j])
                    for k in range(1, nk):
                        csv_w.writerow([''] * 9 + [
                            k, ''] + res_data[k][3 + 4 * t + j])
        # pdb.set_trace()

        del nk, sens_att, priv_val, mrg_grp
        del res_data, res_aux, csv_row_2a
        return

    def coding_per_dataset(self, logger):
        X, A, y, Aq, marginalised_group, g1m_indices, new_attr, \
            res_aux, idx_g1, idx_jt = self.preparing_current_data(logger)

        elegant_print("nb_iter={}, cross_valid".format(self._nb_iter), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")
        elegant_print("\t CrossValid  {}\n".format(self._trial_type[:3]), logger)

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(X, A, y, Aq, i_trn, g1m_indices, idx_jt)
            (X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(X, A, y, Aq, i_tst, g1m_indices, idx_jt)
            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                scaler, X_trn, A_trn, _, _, X_tst, A_tst = normalise_disturb_prime(
                    scaler, X_trn, A_trn, [], [], X_tst, A_tst)
            # i-th K-Fold  # TODO
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

    def coding_per_iteration_cv_split(
            self, logger, k,
            X_trn, A_trn, y_trn, Aq_trn, g1m_trn, jt_trn,
            X_tst, A_tst, y_tst, Aq_tst, g1m_tst, jt_tst):
        return

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

        origin_dat, processed_data, process_mult, disturbed_data, \
            disturb_mult = renewed_prep_and_adversarial(
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

        sens_att = self._dataset.get_sensitive_attrs_with_joint()[: 2]
        priv_val = self._dataset.get_privileged_group_with_joint('')[: 2]
        marginalised_group = origin_dat['marginalised_groups']
        self.saValue = [(np.arange(
            len(sa_val)) + 2).tolist() for sa_val in marginalised_group]
        mrg_indices = check_marginalised_indices(
            processed_data['original'], sens_att, priv_val, marginalised_group)
        new_attr = '-'.join(sens_att) if len(sens_att) > 1 else None
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'], 'both')

        tmp_cls = ''
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
                marginalised_group, mrg_indices, new_attr, res_aux,
                belongs_priv, ptb_with_joint)


# -------------------------------------
# Manifold ext. empirical


class Rev_ManfExtPrime_Empir(Rev_ManfExtEmpir):
    def preparing_iterator(self, trial_type, abbr_cls, nb_iter,
                           gen, rep,
                           m1, m2, n_e, nb_cls, constraint_type,
                           prep=False, screen=True, logged=False):
        # rexp1: 对比 an efficient method for Hausdorff
        #   rexp1a: multivar (incl. nonbin x n_a) 对比结果
        #   rexp1b: nonbin ~singularly~ 单独计算
        #   rexp1c: bin 单独计算，把 multi-val 当成 bi-val 计算
        # rexp2: 齐琪，考虑 embedding +y_hat 计算 df
        #   rexp2a: 不是交叉验证
        #   rexp2b: 交叉验证，用BaseNet,
        #   rexp2c: 交叉验证，用BaseNet, 比 rexp2b 信息更多
        #   # TODO: 其实少了 multivar 的计算，算了懒得算了
        # rexp3: 对比 statistical parity
        #   rexp3c/3b: 交叉验证，不要 embedding，直接算 statistical parity; 用BaseNet
        #   rexp3d: 交叉验证，用其他普通的 sklearn 的学习器，其他跟 rexp3c 一模一样
        #   rexp3e|3f: 交叉验证，用一些分类器，结果都放在一起，基于 rexp3d|3c
        if trial_type.endswith('rexp1a'):
            self._iterator = RevCompZA_efficient(omitted=self._omit)
        elif trial_type.endswith('rexp1b'):
            self._iterator = RevCompZB_efficient(omitted=self._omit)
        elif trial_type.endswith('rexp1c'):
            self._iterator = RevCompZC_efficient(omitted=self._omit)
        elif trial_type.endswith('rexp2a'):
            self._iterator = RevCompYA_NN()
        elif trial_type.endswith('rexp2b'):
            self._iterator = RevCompYB_NN()
        elif trial_type.endswith('rexp2c'):
            self._iterator = RevCompYC_NN()
        elif trial_type[-6:] in ('rexp3b', 'rexp3c'):
            self._iterator = RevCompXB_NN(abbr_cls=abbr_cls)
        elif trial_type.endswith('rexp3d'):
            self._iterator = RevCompXD_NN(abbr_cls=abbr_cls)
        elif trial_type.endswith('rexp3e'):
            self._iterator = RevCompXE_ensemble(abbr_cls='')
        elif trial_type.endswith('rexp3f'):
            self._iterator = RevCompXF_learner(abbr_cls='')

        elif trial_type.endswith('rexp8b'):  # 'rexp4b'):
            self._iterator = ConvergeE2_with(  # nb_cls=7,
                nb_cls, self.saIndex, self.saValue, n_e
            )  # , omitted=self._omit)
        elif trial_type.endswith('rexp8c'):  # 'rexp4c'):
            self._iterator = ConvergeE3_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp8d'):  # 'rexp4d'):
            self._iterator = ConvergeE4_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp8e'):  # 'rexp4e'):
            self._iterator = ConvergeE5_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp9b'):  # 'rexp5b'):
            self._iterator = ConvergeF2_with(  # nb_cls=7,
                nb_cls, self.saIndex, self.saValue, n_e
            )  # , omitted=self._omit)
        elif trial_type.endswith('rexp9f'):
            self._iterator = ConvergeF7_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp9g'):
            self._iterator = ConvergeF8_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp9c'):  # 'rexp5c'):
            self._iterator = ConvergeF3_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp9d'):  # 'rexp5d'):
            self._iterator = ConvergeF4_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        elif trial_type.endswith('rexp9e'):  # 'rexp5e'):
            self._iterator = ConvergeF5_with(
                nb_cls, self.saIndex, self.saValue, n_e)
        # rexp4b: refer to expt4b
        # rexp5b: refer to rexp4b (add another converged algo)

        nk = f'nk{self._nb_iter}' if self._nb_iter > 0 else 'sing'
        formatted = '_'.join([
            trial_type, nk, prep.replace('_', ''), self._log_document,
            'r{}'.format(int(self._ratio * 100)), 'pms', ])
        # if trial_type.endswith('rexp4b') or .endswith('rexp5b'):
        if trial_type[-6:] in ('rexp8b', 'rexp9b', 'rexp9f', 'rexp9g'):
            # formatted = formatted[:-3] + '_'.join([
            #     '', f'cls{nb_cls}', 'pms', ])
            formatted += f'_cf{nb_cls}'  # f'_cls{nb_cls}'
        self._log_document = formatted + ('_gen' * gen + '_rep' * rep)
        del nk
        if trial_type.endswith('rexp3d'):
            self._log_document += '_{}'.format(abbr_cls)  # formatted
        elif trial_type.endswith('rexp3e'):
            self._log_document += '_cf{}'.format(nb_cls)
        return

    def preparing_current_data(self, logger=None):
        origin_dat, processed_data, process_mult, disturbed_data, \
            disturb_mult = renewed_prep_and_adversarial(
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

        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y= {}".format(set(y.values)),
            "\t ori.label= {}".format(set(tmp.values)),
            ""], logger)
        del tmp
        elegant_print(["\t X_A .shape {}".format(X_A.shape),
                       "\t X_Aq.shape {}".format(X_Aq.shape),
                       "\t y   .shape {}".format(y.shape), ""], logger)

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

        tmp_cls = ''
        tmp_ens = ''
        res_aux = [[self._dataset.dataset_name, len(set(y.values)),
                    # tmp_cls, self._nb_iter,
                    # self._gen_iter, self._rep_iter,
                    # self._m1, self._m2, len(sens_att), '', tmp_ens],
                    len(sens_att), self._nb_iter,
                    self._gen_iter, self._rep_iter,
                    self._m1, self._m2, self._n_e, ''],
                   sens_att, priv_val, marginalised_group]  # = ''
        # del tmp_cls, tmp_ens
        return (X_A.values, y.values, X_Aq.values,
                marginalised_group, margin_indices, new_attr, res_aux,
                belongs_priv, ptb_with_joint)

    def coding_per_dataset(self, logger):
        (X_and_A, y, X_and_Aq, marginalised_group, g1m_indices, new_attr,
         res_aux, idx_g1, idx_jt) = self.preparing_current_data(logger)
        pool = pp.ProcessingPool(nodes = self._mp_cores)
        if 2 in y:
            y[y == 2] = 0

        pm_k = {}
        if self._nb_iter <= 0:
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
            # pm_k['idx_jt'] = idx_jt
            # pm_k['X_and_A'] = X_and_A
            pm_k['X_wAq'] = X_and_Aq  # pm_k['X_wA'] = X_and_A
            res_data = self.coding_per_iteration_as_whole(
                logger, pool, X, A, y, X_and_A, g1m_indices, **pm_k)
            return [res_data], res_aux
        # "No solo-execution, via cv_split (/ non-repeatition)?"
        elegant_print("No solo-executiton, non-repeatition? {}".format(
            'Yes' if self._rep_iter else 'No'), logger)
        if not self._rep_iter:
            elegant_print("Repetitive {} times".format(self._nb_iter), logger)
            split_idx = manual_repetitive(self._nb_iter, y, self._gen_iter)
            res_data = []
            for k, idx in enumerate(split_idx):
                elegant_print("Iteration {}-th starts.".format(k + 1), logger)
                (prim_X_A, _, prim_y, prim_X_Aq, prim_g1m,
                 prim_jt) = renewed_transform_disturb(
                    X_and_A, None, y, X_and_Aq, idx, g1m_indices, idx_jt)
                # pm_k['idx_jt'] = prim_jt
                if self._prep in ['standard', 'min_max', 'normalize']:
                    scaler = scale_normalize_helper(self._prep)
                    (scaler, prim_X_A, _, prim_X_Aq, X, A, _, _, Xq, Aq
                     ) = renewed_normalise_disturb(
                        scaler, prim_X_A, [], prim_X_Aq, self.saIndex)
                    del scaler
                else:
                    X, A, _, _, Xq, Aq = renewed_normalise_separate(
                        prim_X_A, [], prim_X_Aq, self.saIndex)
                # pm_k['X_and_A'] = prim_X_A
                pm_k['X_wAq'] = prim_X_Aq  # pm_k['X_wA'] = prim_X_A
                res_iter = self.coding_per_iteration_as_whole(
                    logger, pool, X, A, prim_y, prim_X_A, prim_g1m, **pm_k)
                res_data.append(res_iter)
                del X, A, Xq, Aq, prim_X_A, prim_X_Aq, prim_y
                del prim_g1m, prim_jt
            return res_data, res_aux
        del pm_k

        elegant_print("nb_iter={}, cross_valid".format(self._nb_iter), logger)
        if 'mCV' in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif 'KFS' in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif 'KF' in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError('No proper CV (cross-validation).')
        elegant_print('\t CrossValid  {}\n'.format(self._trial_type[:3]), logger)
        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (X_A_trn, _, y_trn, X_Aq_trn, g1m_trn, jt_trn
             ) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_trn, g1m_indices, idx_jt)
            (X_A_tst, _, y_tst, X_Aq_tst, g1m_tst, jt_tst
             ) = renewed_transform_disturb(
                X_and_A, None, y, X_and_Aq, i_tst, g1m_indices, idx_jt)
            if self._prep in ['standard', 'min_max', 'normalize']:
                scaler = scale_normalize_helper(self._prep)
                (scaler, X_A_trn, _, X_A_tst, X_trn, A_trn, _, _,
                 X_tst, A_tst) = renewed_normalise_disturb(
                    scaler, X_A_trn, [], X_A_tst, self.saIndex)
            else:
                X_trn, A_trn, _, _, X_tst, A_tst = renewed_normalise_separate(
                    X_A_trn, [], X_A_tst, self.saIndex)
            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            # kw = {} if not self._trial_type.endswith(
            #     'rexp4b') else {'jt_trn': jt_trn, 'jt_tst': jt_tst}
            kw = {}
            if self._trial_type[-6:] in (  # 'rexp4b','rexp5b',
                    'rexp8b', 'rexp9b', 'rexp9f', 'rexp9g'):
                # kw = {'jt_trn': jt_trn, 'jt_tst': jt_tst}
                kw['jt_trn'] = jt_trn
                kw['jt_tst'] = jt_tst
            res_iter = self.coding_per_iteration_cv_split(
                logger, pool, k,
                X_trn, A_trn, y_trn, X_A_trn, X_Aq_trn, g1m_trn,
                X_tst, A_tst, y_tst, X_A_tst, X_Aq_tst, g1m_tst,
                **kw)
            res_ans.append(res_iter)
            del X_trn, A_trn, y_trn, g1m_trn, jt_trn, X_A_trn, X_Aq_trn
            del X_tst, A_tst, y_tst, g1m_tst, jt_tst, X_A_tst, X_Aq_tst
        del X_and_A, y, X_and_Aq, idx_g1, idx_jt
        del pool
        return res_ans, res_aux

    def coding_per_iteration_cv_split(
            self, logger, pool, k,
            X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
            X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
            jt_trn=None, jt_tst=None):
        since = time.time()
        res_iter = []
        positive_label = self._dataset.get_positive_class_val(
            'numerical-binsensitive')

        if ('rexp2b' in self._trial_type) or (
                'rexp2c' in self._trial_type):
            res_iter = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
                self._m1, self._m2, self._n_e, pool, positive_label)
        elif ('rexp3b' in self._trial_type) or (
                'rexp3c' in self._trial_type) or (
                self._trial_type[-6:] in ['rexp3d', 'rexp3e', 'rexp3f']):
            # pm = {'nb_cls': self._nb_cls} if self._trial_type.endswith('3e') else {}
            pms = {}
            if self._trial_type.endswith('3e'):
                pms['nb_cls'] = self._nb_cls
                pms['saIndex'] = self.saIndex
                pms['saValue'] = self.saValue
            res_iter = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
                # self._m1, self._m2, self._n_e, positive_label)
                self._m1, self._m2, self._n_e, pool, positive_label, **pms)

        elif self._trial_type[-6:] in (
                'rexp8b', 'rexp9b', 'rexp9f', 'rexp9g'):
            # elif ('rexp4b' in self._trial_type) or (
            #     'rexp5b' in self._trial_type) or (
            #     'rexp8b' in self._trial_type) or (
            #     'rexp9b' in self._trial_type):
            # pms = {}  # pdb.set_trace()
            res_iter = self._iterator.schedule_content(
                X_trn, A_trn, y_trn, X_wA_trn, X_wAq_trn, g1m_trn,
                X_tst, A_tst, y_tst, X_wA_tst, X_wAq_tst, g1m_tst,
                self._m1, self._m2, self._n_e, positive_label, pool,
                jt_trn=jt_trn, jt_tst=jt_tst)

        del positive_label
        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, fantasy_durat_major(tim_elapsed, True, False)), logger)
        return res_iter

    def coding_per_iteration_as_whole(
            self, logger, pool,
            X, A, y, X_wA, g1m_indices, idx_jt=None,
            Aq=None, X_wAq=None):  # X_wA=None,
        # since = time.time()
        res_iter = []

        if 'rexp1' in self._trial_type:
            res_iter = self._iterator.schedule_content(
                X, A, y,
                g1m_indices, self._m1, self._m2, self._n_e, pool)
        elif 'rexp2' in self._trial_type:
            res_iter = self._iterator.schedule_content(
                X, A, y, X_wA, X_wAq,
                g1m_indices, self._m1, self._m2, self._n_e, pool)

        elif self._trial_type[-6:] in ('rexp8c', 'rexp8d',
                                       'rexp9c', 'rexp9d',):
            # elif [-6:] in ('rexp4c', 'rexp4d',  # 'rexp4e',
            #                            'rexp5c', 'rexp5d',):
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices,           # X_wA,
                self._m1, self._m2, self._n_e)  # , pool)
        elif self._trial_type[-6:] in (
                'rexp8e', 'rexp9e',):  # 'rexp4e', 'rexp5e',
            # kw = {'pool': pool} / {}  # ,**kw)
            res_iter = self._iterator.schedule_content(
                X, A, y, g1m_indices,
                self._m1, self._m2, self._n_e, pool)
        return res_iter


# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------

# =====================================

# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------

# -------------------------------------
