# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#


import numpy as np
import pandas as pd
# import pdb

from hfm.utils.verifiers import unique_column, check_zero, DTY_FLT
from experiment.generic import GraphSetupVer1 as GraphSetup
from experiment.generic import DAT_EXPT_ORG

from experiment.facil.draw_addtl import (
    scatter_k_cv_with_real, approximated_dist_comparison,
    multiple_scatter_comparison,
    scatter_with_marginal_distrib, lineplot_with_uncertainty,
    line_reg_with_marginal_distr, single_line_reg_with_distr)
from experiment.facil.draw_chart import analogous_confusion_extended


# ===============================
# Benchmarks


# -------------------------------
# RQ5.
# --> RQ6. hyper-parameters
#
# RQ3. effect of hyperparameters
#


class Plot5_hyperpm(GraphSetup):
    def __init__(self, nb_iter, gen, rep,
                 m1=30, m2=20, figname='exp5_'):
        self._nb_iter = nb_iter
        super().__init__(gen, rep, m1, m2, figname)

    def prepare_graph(self):
        raise NotImplementedError


def _diff_between(approx, direct):
    # difference: abs(approx - direct) / direct
    # Yss.shape= (#att_sen, nb_iter, #num)
    # zs .shape= (#att_sen, nb_iter)
    nb_att, nb_iter, _ = approx.shape  # ,num
    diff = np.zeros_like(approx) - 1.
    for j in range(nb_att):
        for i in range(nb_iter):
            tmp = np.abs(approx[j][i] - direct[j][i])
            diff[j][i] = tmp / check_zero(direct[j][i])
    return diff  # .shape= (#att_sens, nb_iter, #num)


class Plot5A_hyperpm(GraphSetup):
    def __init__(self, nb_iter, gen, rep, m1, m2,
                 figname='exp5a_'):
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter
        super().__init__(gen, rep, m1, m2, figname)
        self._m2_set = list(range(2, 23, 1))  # len=21

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 46)

        params = csv_row_1[: 11]
        direct = csv_row_1[11: 11 + 4]
        approx = csv_row_1[15: 15 + 21]
        app_ut = csv_row_1[-21:]
        return params, direct, approx, app_ut

    def schedule_mspaint(self, raw_dframe):
        nb_set, Ys_dir, Ys_app, Ys_ut, \
            picked_keys = self.painting_prep(raw_dframe)
        # X = self._m2_set.copy()
        '''
        self.painting_fig2(nb_set, X, Ys_dir, Ys_app, Ys_ut, picked_keys)
        self.painting_fig1(nb_set, X, Ys_dir, Ys_app, Ys_ut, picked_keys)
        '''

        nb_set, id_set = self.recap_sub_data(raw_dframe, nb_row=3)
        # nb_set, id_set = nb_set - 1, id_set[:-1]
        _, tag_direct, tag_approx, tag_ut = self.prepare_graph()  # tag_pm,
        picked_set = [0, 1, 2, 3, 4, 5]
        picked_set.remove(5)  # simulative data
        picked_m2 = [3, 7, 11, 15, 19]  # [3, 6, 9, 12, 15, 18]  # self._m2_set
        picked_m2 = [5, 9, 13, 17, 21]  # [4, 5, 9, 13, 17, 21]
        self.drawing_fig3_alt(
            raw_dframe, nb_set, id_set, tag_direct, tag_approx,
            'ua', picked_m2, picked_set, joint='none')
        self.drawing_fig3_alt(
            raw_dframe, nb_set, id_set, tag_direct, tag_ut,
            'ut', picked_m2, picked_set, joint='none')

    def drawing_fig3_alt(self, dframe, nb_set, id_set,
                         tag_direct, tag_approx, ind='ua',
                         picked_m2=[2, 3, 4, 5],
                         picked_set=[0, 1, 2, 3, 4, ],
                         joint='and|or', distrib=False):
        if ind == 'ua':
            curr_tag_dir = tag_direct[-2]
        elif ind == 'ut':
            curr_tag_dir = tag_direct[-1]
        curr_tag_app = [
            tag_approx[self._m2_set.index(i)] for i in picked_m2]
        index_jt, suffix = self.draw_sub2_jt(joint)

        index_wa, i = [], 0  # whole, overall
        for i in picked_set:
            if i == 0:
                index_wa += list(range(id_set[i] + 1, id_set[i + 1]))
                continue
            # else:
            #   pass
            curr_loc = id_set[i] + 1
            index_wa += list(range(curr_loc,
                                   curr_loc + 2 * self._nb_iter))
            for j in index_jt[2:]:
                curr_loc = id_set[i] + 1 + j * self._nb_iter
                index_wa += list(
                    range(curr_loc, curr_loc + self._nb_iter))

        df_raw = dframe[[curr_tag_dir] + curr_tag_app].iloc[index_wa]
        col_Y = 'Approximation'  # col_X, = curr_tag_dir,
        annotX = 'Distance via direct computation'
        annotY = 'Distance via approximation'
        picked_keys = [r'$m_2={:2d}$ '.format(i) for i in picked_m2]
        kws = {'snspec': 'sty5b', 'cmap_name': self._cmap_name}
        suff_6 = 'exp5a_minmax_{}_m1_{}'.format(suffix, ind)
        if ind == 'ua':
            annotX = r'$\mathbf{D}$'
            annotY = r'$\hat{\mathbf{D}}$'
            identity = r'$\hat{\mathbf{D}}=\mathbf{D}$'
            identity = '{:3s}'.format('') + identity
        elif ind == 'ut':
            annotX = r'$T_{\mathbf{D}}$ (sec)'
            annotY = r'$T_{\hat{\mathbf{D}}}$  (sec)'
            identity = r'$T_{\hat{\mathbf{D}}}=T_{\mathbf{D}}$'
            kws['snspec'] = 'sty4a'
            identity = '{:2s}'.format('') + identity
        kws['distrib'] = distrib  # False
        line_reg_with_marginal_distr(
            df_raw, curr_tag_dir, col_Y, curr_tag_app, picked_keys, 
            annotX, annotY, figname=suff_6 + '_x', invt_a=False,
            identity=identity, **kws)

        if ind == 'ua':
            df_tmp = df_raw.astype(DTY_FLT)  # df_raw.copy()
            for j in curr_tag_app:
                temp = df_tmp[j] - df_tmp[curr_tag_dir]
                temp = temp.abs() / df_tmp[curr_tag_dir]
                df_tmp[j] = temp
            annotY = r'$\frac{abs(\hat{\mathbf{D}}-\mathbf{D})}{\mathbf{D}}$'
            kws['snspec'] = 'sty4b'
            line_reg_with_marginal_distr(
                df_tmp, curr_tag_dir, col_Y, curr_tag_app,
                picked_keys, annotX, annotY,
                figname=suff_6 + '_xp', invt_a=False,
                identity=None, **kws)
            del df_tmp
        return

    def painting_prep(self, raw_dframe):
        nb_set, id_set = self.recap_sub_data(raw_dframe, nb_row=3)
        _, tag_dir, tag_app, tag_ut = self.prepare_graph()  # tag_pm,
        # X = self._m2_set.copy()
        Ys_dir, Ys_app, Ys_ut, picked_keys = [], [], [], []

        for i in range(nb_set):
            nb_att = (id_set[i + 1] - id_set[i] - 1) // self._nb_iter
            # curr_dat = raw_dframe['A'].iloc[id_set[i]]
            tmp_dir, tmp_app, tmp_ut, tmp_key = [], [], [], []

            for j in range(nb_att):
                curr_loc = id_set[i] + 1 + j * self._nb_iter
                curr_att = raw_dframe['I'].iloc[curr_loc]
                if nb_att > 1 and j == nb_att - 1:
                    curr_att = raw_dframe[
                        'I'].iloc[curr_loc - self._nb_iter]

                curr_row = list(range(curr_loc,
                                      curr_loc + self._nb_iter))
                direct = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_dir[-2:])
                approx = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_app)
                app_ut = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_ut)

                tmp_dir.append(direct)
                tmp_app.append(approx)
                tmp_ut.append(app_ut)
                tmp_key.append('{}: {}'.format(DAT_EXPT_ORG[i],
                                               curr_att))
            Ys_dir.append(tmp_dir)
            Ys_app.append(tmp_app)
            Ys_ut.append(tmp_ut)
            picked_keys.append(tmp_key)
        return nb_set, Ys_dir, Ys_app, Ys_ut, picked_keys

    def painting_fig2(self, nb_set, X, Ys_dir, Ys_app, Ys_ut, picked_keys):
        for i in range(nb_set):
            for j, curr_dat_or_att in enumerate(picked_keys[i]):
                suffix = 'expt5a_iter' + str(self._nb_iter)

                suffix += '_set{}_att{}'.format(i + 1, j + 1)
                if j == 2:
                    suffix += 'ad'
                elif j == 3:
                    suffix += 'or'
                kws = {'annotX': r'$m_2$',
                       'annotY': 'Approximated value'}

                direct = Ys_dir[i][j]
                approx = Ys_app[i][j]
                scatter_k_cv_with_real(X, approx, direct[:, 0],
                                       figname='pic2_' + suffix,
                                       **kws)

    def painting_fig1(self, nb_set, X, Ys_dir, Ys_app, Ys_ut, picked_keys):
        kws = {'annotX': r'$m_2$'}  # , 'annotY': 'Approximated value'}
        for i in range(nb_set):
            suffix = 'expt5a_iter' + str(self._nb_iter)
            suffix += '_set{}'.format(i + 1)

            direct = np.array(Ys_dir[i])  # shape= (1|4, 5, 2)
            approx = np.array(Ys_app[i])  # shape= (1|4, 5, 21)
            diff = _diff_between(approx, direct[:, :, 0])
            approximated_dist_comparison(
                X, diff, picked_keys[i],
                figname='pic1_' + suffix, **kws)

            approx = np.array(Ys_ut[i])  # shape= (1|4, 5, 21)
            multiple_scatter_comparison(
                X, approx, direct[:, :, 1], picked_keys[i],
                annotY='Time Cost (sec)',
                figname='pic3_' + suffix, **kws)

        # new_X, new_Ys_dir, new_Ys_app = [], [], []
        suffix = 'expt5a_merged_set'  # expt5a_merge_sets
        new_dim = 4 * self._nb_iter
        new_Ys_dir = [np.array(Ys_dir[0] * 4).reshape(new_dim, -1)]
        new_Ys_app = [np.array(Ys_app[0] * 4).reshape(new_dim, -1)]
        new_Ys_ut = [np.array(Ys_ut[0] * 4).reshape(new_dim, -1)]
        for i in range(1, nb_set):
            new_Ys_dir.append(
                np.array(Ys_dir[i]).reshape(new_dim, -1))
            new_Ys_app.append(
                np.array(Ys_app[i]).reshape(new_dim, -1))
            new_Ys_ut.append(np.array(Ys_ut[i]).reshape(new_dim, -1))
        new_Ys_dir = np.array(new_Ys_dir)  # (set=6, 4*5, 2)
        new_Ys_app = np.array(new_Ys_app)  # (set=6, 4*5, num=21)
        new_Ys_ut = np.array(new_Ys_ut)    # (set=6, 4*5, num=21)
        diff = _diff_between(new_Ys_app, new_Ys_dir[:, :, 0])
        approximated_dist_comparison(
            X, diff[:5], DAT_EXPT_ORG[:5],
            figname='pic1_' + suffix, **kws)
        multiple_scatter_comparison(
            X, new_Ys_ut[:5], new_Ys_dir[:, :, 1], DAT_EXPT_ORG[:5],
            annotY='Time Cost (sec)', figname='pic3_' + suffix, **kws)


class Plot5B_hyperpm(GraphSetup):
    def __init__(self, nb_iter, gen, rep, m1, m2,
                 figname='exp5a_'):
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter
        super().__init__(gen, rep, m1, m2, figname)
        self._m1_set = list(range(3, 50, 2))  # len=24

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 52)

        params = csv_row_1[: 11]
        direct = csv_row_1[11: 11 + 4]
        approx = csv_row_1[15: 15 + 24]
        app_ut = csv_row_1[-24:]
        return params, direct, approx, app_ut

    def schedule_mspaint(self, raw_dframe):
        nb_set, Ys_dir, Ys_app, Ys_ut, \
            picked_keys = self.painting_prep(raw_dframe)
        # X = self._m1_set.copy()

        nb_set, id_set = self.recap_sub_data(raw_dframe, nb_row=3)
        tag_pm, tag_direct, tag_approx, tag_ut = self.prepare_graph()
        picked_set = [0, 1, 2, 3, 4, ]  # without simulation data
        picked_m1 = [3, 11, 17, 23, 29, 35, ]
        picked_m1 = [3, 9, 15, 21, 27, ]
        self.drawing_fig3_alt(
            raw_dframe, nb_set, id_set, tag_direct, tag_approx,
            'ua', picked_m1, picked_set, joint='none')
        self.drawing_fig3_alt(
            raw_dframe, nb_set, id_set, tag_direct, tag_ut,
            'ut', picked_m1, picked_set, joint='none')

    def drawing_fig3_alt(self, dframe, nb_set, id_set,
                         tag_direct, tag_approx, ind='ua',
                         picked_m1=[3, 5, 7, 9],
                         picked_set=[0, 1, 2, 3, 4, ],
                         joint='and|or', distrib=False):
        if ind == 'ua':
            curr_tag_dir = tag_direct[-2]
        elif ind == 'ut':
            curr_tag_dir = tag_direct[-1]
        curr_tag_app = [tag_approx[
            self._m1_set.index(i)] for i in picked_m1]
        index_jt, suffix = self.draw_sub2_jt(joint)

        index_wa, i = [], 0  # whole, overall
        for i in picked_set:
            if i == 0:
                index_wa += list(range(id_set[i] + 1, id_set[i + 1]))
                continue
            curr_loc = id_set[i] + 1
            index_wa += list(range(curr_loc,
                                   curr_loc + 2 * self._nb_iter))
            for j in index_jt[2:]:
                curr_loc = id_set[i] + 1 + j * self._nb_iter
                index_wa += list(range(curr_loc,
                                       curr_loc + self._nb_iter))

        df_raw = dframe[[curr_tag_dir] + curr_tag_app].iloc[index_wa]
        col_Y = 'Approximation'  # col_X,=curr_tag_dir
        picked_keys = [r'$m_1={:2d}$'.format(i) for i in picked_m1]
        kws = {'snspec': 'sty5b', 'cmap_name': self._cmap_name}
        suff_6 = 'exp5b_minmax_{}_m2_{}'.format(suffix, ind)  # fixed m2
        if ind == 'ua':
            annotX = r'$\mathbf{D}$'
            annotY = r'$\hat{\mathbf{D}}$'
            identity = r'$\hat{\mathbf{D}}=\mathbf{D}$'
            identity = '{:3s}'.format('') + identity
        elif ind == 'ut':
            annotX = r'$T_{\mathbf{D}}$ (sec)'
            annotY = r'$T_{\hat{\mathbf{D}}}$  (sec)'
            identity = r'$T_{\hat{\mathbf{D}}}=T_{\mathbf{D}}$'
            identity = '{:2s}'.format('') + identity
            kws['snspec'] = 'sty4a'
        kws['distrib'] = distrib
        line_reg_with_marginal_distr(
            df_raw, curr_tag_dir, col_Y, curr_tag_app, picked_keys,
            annotX, annotY, figname=suff_6 + '_x', invt_a=False,
            identity=identity, **kws)

        if ind == 'ua':
            df_tmp = df_raw.astype(DTY_FLT)  # .copy().astype(DTY_FLT)
            for j in curr_tag_app:
                temp = df_tmp[j] - df_tmp[curr_tag_dir]
                temp = temp.abs() / df_tmp[curr_tag_dir]
                df_tmp[j] = temp
            annotY = r'$\frac{abs(\hat{\mathbf{D}}-\mathbf{D})}{\mathbf{D}}$'
            kws['snspec'] = 'sty4b'
            line_reg_with_marginal_distr(
                df_tmp, curr_tag_dir, col_Y, curr_tag_app,
                picked_keys, annotX, annotY,
                figname=suff_6 + '_xp', invt_a=False,
                identity=None, **kws)
            del df_tmp
        return

    def painting_prep(self, raw_dframe):
        nb_set, id_set = self.recap_sub_data(raw_dframe, nb_row=3)
        _, tag_dir, tag_app, tag_ut = self.prepare_graph()  # tag_pm,
        Ys_dir, Ys_app, Ys_ut, picked_keys = [], [], [], []

        for i in range(nb_set):
            nb_att = (id_set[i + 1] - id_set[i] - 1) // self._nb_iter
            # curr_dat = raw_dframe['A'].iloc[id_set[i]]
            tmp_dir, tmp_app, tmp_ut, tmp_key = [], [], [], []

            for j in range(nb_att):
                curr_loc = id_set[i] + 1 + j * self._nb_iter
                curr_att = raw_dframe['I'].iloc[curr_loc]
                if nb_att > 1 and j == nb_att - 1:
                    curr_att = raw_dframe[
                        'I'].iloc[curr_loc - self._nb_iter]

                curr_row = list(range(curr_loc,
                                      curr_loc + self._nb_iter))
                direct = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_dir[-2:])
                approx = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_app)
                app_ut = self.fetch_sub_data(raw_dframe, curr_row,
                                             tag_ut)

                tmp_dir.append(direct)
                tmp_app.append(approx)
                tmp_ut.append(app_ut)
                tmp_key.append('{}: {}'.format(DAT_EXPT_ORG[i],
                                               curr_att))
            Ys_dir.append(tmp_dir)
            Ys_app.append(tmp_app)
            Ys_ut.append(tmp_ut)
            picked_keys.append(tmp_key)
        return nb_set, Ys_dir, Ys_app, Ys_ut, picked_keys

    def painting_fig2(self, nb_set, X, Ys_dir, Ys_app, Ys_ut, picked_keys):
        for i in range(nb_set):
            for j, curr_dat_or_att in enumerate(picked_keys[i]):
                suffix = 'expt5b_iter' + str(self._nb_iter)

                suffix += '_set{}_att{}'.format(i + 1, j + 1)
                if j == 2:
                    suffix += 'ad'
                elif j == 3:
                    suffix += 'or'
                kws = {'annotX': r'$m_1$',
                       'annotY': 'Approximated value'}

                direct = Ys_dir[i][j]
                approx = Ys_app[i][j]
                scatter_k_cv_with_real(X, approx, direct[:, 0],
                                       figname='pic2_' + suffix,
                                       **kws)

    def painting_fig1(self, nb_set, X, Ys_dir, Ys_app, Ys_ut,
                      picked_keys):
        kws = {'annotX': r'$m_1$'}
        for i in range(nb_set):
            suffix = 'expt5b_iter' + str(self._nb_iter)
            suffix += '_set{}'.format(i + 1)

            direct = np.array(Ys_dir[i])  # shape= (1|4, 5, 2)
            approx = np.array(Ys_app[i])  # shape= (1|4, 5, 24)
            diff = _diff_between(approx, direct[:, :, 0])
            approximated_dist_comparison(
                X, diff, picked_keys[i], figname='pic1_' + suffix,
                **kws)

            approx = np.array(Ys_ut[i])  # shape= (1|4, 5, 24)
            multiple_scatter_comparison(
                X, approx, direct[:, :, 1], picked_keys[i],
                annotY='Time Cost (sec)', figname='pic3_' + suffix,
                **kws)

        suffix = 'expt5b_merged_set'
        new_dim = 4 * self._nb_iter
        new_Ys_dir = [np.array(Ys_dir[0] * 4).reshape(new_dim, -1)]
        new_Ys_app = [np.array(Ys_app[0] * 4).reshape(new_dim, -1)]
        new_Ys_ut = [np.array(Ys_ut[0] * 4).reshape(new_dim, -1)]
        for i in range(1, nb_set):
            new_Ys_dir.append(
                np.array(Ys_dir[i]).reshape(new_dim, -1))
            new_Ys_app.append(
                np.array(Ys_app[i]).reshape(new_dim, -1))
            new_Ys_ut.append(np.array(Ys_ut[i]).reshape(new_dim, -1))
        new_Ys_dir = np.array(new_Ys_dir)  # (set=6, 4*5, 2)
        new_Ys_app = np.array(new_Ys_app)  # (set=6, 4*5, num=21)
        new_Ys_ut = np.array(new_Ys_ut)    # (set=6, 4*5, num=21)
        diff = _diff_between(new_Ys_app, new_Ys_dir[:, :, 0])
        approximated_dist_comparison(
            X, diff[:5], DAT_EXPT_ORG[:5], figname='pic1_' + suffix,
            **kws)
        multiple_scatter_comparison(
            X, new_Ys_ut[:5], new_Ys_dir[:, :, 1], DAT_EXPT_ORG[:5],
            annotY='Time Cost (sec)', figname='pic3_' + suffix, **kws)


# -------------------------------
# RQ1. compared with sota fairness
#


class Plot2_comparison(GraphSetup):
    def __init__(self, nb_iter, nb_cls,  # gen, rep,
                 m1=30, m2=16, figname='exp2_'):
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter
        self._nb_cls = nb_cls
        gen, rep = None, True
        super().__init__(gen, rep, m1, m2, figname)
        self._picked_keys = ['DP', 'EO', 'PQP'] + [
            'DR', r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        self._pick_metric = [
            'Accuracy', 'Precision', 'Recall', r'$f_1$ score',
            'FPR rate', 'FNR rate', 'Sensitivity', 'Specificity']
        self._cmap_name = 'bright'

    def prepare_graph(self):
        raise NotImplementedError

    def recap_sub_data(self, dframe, nb_row=4):
        each_att = 7 * self._nb_iter
        each_set = each_att * 2 + 1

        nb_set = len(dframe) - nb_row + 1
        # set_p1 = each_att + 1
        # set_p2 = (nb_set - set_p1) // (each_att * 2 + 1)
        # nb_set = 1 + set_p2  # first set, two attr
        nb_set = (nb_set - (each_att + 1)) // each_set + 1

        id_set = [0] + [
            i * each_set + (each_att + 1) for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]

        id_att = [id_set[0] + 1]
        for i in id_set[1: -1]:
            id_att.extend([i + 1, i + 1 + each_att])
        return nb_set, id_set, id_att, each_att

    def painting_prep(self, raw_dframe):
        pass

    def painting_fig1(self):
        raise NotImplementedError

    def painting_fig2(self):
        raise NotImplementedError

    def drawing_fig1_alt(self):
        raise NotImplementedError

    def draw_sub2_dat1(self, dframe, nb_set, id_set,
                       tmp_f_vm, tmp_jt):
        i, j = 0, 0
        df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            j = 0
            df_tmp = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp_jt[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                                  id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)
        return df_raw

    def draw_sub2_dat2(self, dframe, nb_set, id_set, each_gen,
                       each_att, tmp_f_vm):  # , tmp_jt):
        i = 0  # i, k = 0, 0
        df_raw = dframe[tmp_f_vm[0]].iloc[
            id_set[i] + 1: id_set[i + 1]]
        for i in range(1, nb_set):
            curr_set = id_set[i] + 1

            curr_loc = list(range(curr_set,
                                  curr_set + each_gen + each_att))
            df_tmp = dframe[tmp_f_vm[0]].iloc[curr_loc]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

            curr_loc = list(range(
                curr_set, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_tmp = dframe[tmp_f_vm[1]].iloc[curr_loc]
            columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                tmp_f_vm[1])}
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        df_raw = df_raw.reset_index(drop=True)
        return df_raw


def _sub_depict_sep_alt(df_raw, tYs, suff, fig='_Ds', diff=True,
                        double_sep=True):
    '''
    # pdb.set_trace()  # 4,6,5,7
    scat_X = (df_raw[tYs[6]].values.astype(DTY_FLT) +
              df_raw[tYs[8]].values.astype(DTY_FLT))
    scat_Y = (df_raw[tYs[7]].values.astype(DTY_FLT) +
              df_raw[tYs[9]].values.astype(DTY_FLT))
    annotX = r'T_{\mathbf{D}}+T_{\mathbf{D}_f}'
    annotY = r'T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f}'
    annots = ['${}$ (sec)'.format(annotX), '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws = {'linreg': True, 'snspec': 'sty4'}
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + '_as4', **kws)  # +fig
    scat_Z = np.log10(scat_Y / scat_X)
    annots[1] = r'\lg(\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} })'
    annots[1] = '${}$'.format(annots[1])
    kws['snspec'] = 'sty6'
    single_line_reg_with_distr(scat_X, scat_Z, annots,
                               suff + '_as6', **kws)  # +fig

    scat_X = np.concatenate([
        df_raw[tYs[6]].values.astype(DTY_FLT),
        df_raw[tYs[8]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[7]].values.astype(DTY_FLT),
        df_raw[tYs[9]].values.astype(DTY_FLT)], axis=0)
    annotX = r'T_{\mathbf{D}_\cdot}'
    annotY = r'T_{\hat{\mathbf{D}}_\cdot}'
    annots = ['${}$ (sec)'.format(annotX), '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws['snspec'] = 'sty4'
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + '_cs4', **kws)
    scat_Z = np.log10(scat_Y / scat_X)
    annots[1] = r'\lg(\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} })'
    annots[1] = '${}$'.format(annots[1])
    kws['snspec'] = 'sty6'
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_cs6', **kws)
    '''
    if not diff:
        return

    if fig.endswith('TDs'):
        col_X, col_Y = tYs[6], tYs[7]
        annotX = r'T_{\mathbf{D}}'
        annotY = r'T_{\hat{\mathbf{D}}}'
        annotZ = [
            r'\frac{ T{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} } -1',
            r'\lg(\frac{ T{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} })']
    elif fig.endswith('TDf'):
        col_X, col_Y = tYs[8], tYs[9]  # tYs[6], tYs[7]
        annotX = r'T_{\mathbf{D}_f}'
        annotY = r'T_{\hat{\mathbf{D}}_f}'
        annotZ = [
            r'\frac{ T{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} } -1',
            r'\lg(\frac{ T{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} })']

    scat_X = df_raw[col_X].values.astype(DTY_FLT)
    scat_Y = df_raw[col_Y].values.astype(DTY_FLT)
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + fig, linreg=True,
        snspec='sty4')

    scat_Z = scat_Y / scat_X - 1.
    if double_sep:
        annots = ['${}$  (sec)'.format(annotX),
                  '${}$'.format(annotZ[0]),
                  '${}={}$'.format(annotY, annotX)]
        single_line_reg_with_distr(
            scat_X, scat_Z, annots, suff + fig + '_s6a',
            linreg=True, snspec='sty6')
    scat_Z = np.log10(scat_Z + 1)
    annots = ['${}$  (sec)'.format(annotX), '${}$'.format(
        annotZ[1]), '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Z, annots, suff + fig + '_s6b',
        linreg=True, snspec='sty6')
    return


def _sub_depict_sep(df_raw, tYs, suff, fig='_Ds', diff=True):
    if fig.endswith('Ds'):
        col_X, col_Y = tYs[0], tYs[1]
        annotX = r'\mathbf{D}'
        annotY = r'\hat{\mathbf{D}}'
        annotZ = r'\frac{abs(\hat{\mathbf{D}}-\mathbf{D})}{\mathbf{D}}'
    elif fig.endswith('Df'):
        col_X, col_Y = tYs[2], tYs[3]
        annotX = r'\mathbf{D}_f'
        annotY = r'\hat{\mathbf{D}}_f'
        annotZ = r'\frac{abs(\hat{\mathbf{D}}_f-\mathbf{D}_f)}{\mathbf{D}_f}'

    scat_X = df_raw[col_X].values.astype(DTY_FLT)
    scat_Y = df_raw[col_Y].values.astype(DTY_FLT)
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + fig, linreg=True,
        snspec='sty3b')
    if not diff:
        return

    scat_Z = np.abs(scat_Y - scat_X) / scat_X
    annots = ['${}$'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Z, annots, suff + '_diff' + fig,
        linreg=True, snspec='sty6')
    return


def _sub_depict_scat(df_raw, tYs, suff, diff=False):
    scat_X = np.concatenate([
        df_raw[tYs[0]].values.astype(DTY_FLT),
        df_raw[tYs[2]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[1]].values.astype(DTY_FLT),
        df_raw[tYs[3]].values.astype(DTY_FLT)], axis=0)
    annots = ['Distance via direct computation',
              'Distance via approximation']
    annotX = r'\mathbf{D}_\cdot'
    annotY = r'\hat{\mathbf{D}}_\cdot'
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Y, annots,
        suff + '_sty3', linreg=True, snspec='sty3b')

    if not diff:
        return
    scat_Z = np.zeros_like(scat_Y) - 1.
    for i in range(len(scat_Y)):
        tmp = np.abs(scat_Y[i] - scat_X[i])
        scat_Z[i] = tmp / check_zero(scat_X[i])
    annotZ = r'\frac{abs(\hat{\mathbf{D}}_\cdot-\mathbf{D}_\cdot)}{\mathbf{D}_\cdot}'
    annots = ['${}$'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Z, annots,
        suff + '_sty6', linreg=True, snspec='sty6')
    return


def _sub_depict_tim(df_raw, tYs, suff, diff=False,
                    log_taken=False):
    scat_X = df_raw[tYs[4]].values.astype(DTY_FLT)  # direct,ut
    scat_Y = df_raw[tYs[5]].values.astype(DTY_FLT)  # approx,ut

    ant_xs = r'\mathbf{D}'
    ant_ys = r'\hat{\mathbf{D}}'
    ant_xf = r'\mathbf{D}_f'
    ant_yf = r'\hat{\mathbf{D}}_f'
    annotX = r'T_{\mathbf{D}}+T_{\mathbf{D}_f}'
    annotY = r'T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f}'
    del ant_xs, ant_ys, ant_xf, ant_yf
    annots = ['${}$ (sec)'.format(annotX),
              '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws = {'linreg': True, 'snspec': 'sty4'}
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + '_sty4', **kws)

    if not diff:
        return
    scat_Z = scat_Y / scat_X - 1.
    annotZ = r'\frac{T_{\hat{\mathbf{D}}_\cdot}}{T_{\mathbf{D}_\cdot}}-1'
    annotZ = r'\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} }-1'
    annots = ['${}$  (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Z, annots,
        suff + '_sty6a', linreg=True, snspec='sty6')

    if not log_taken:
        return
    scat_Z = np.log10(scat_Z + 1)
    annotZ = r'\lg(\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} })'
    annots = ['${}$ (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Z, annots,
        suff + '_sty6b', linreg=True, snspec='sty6')
    return


class Plot2A_comparison(Plot2_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2a_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 46 * 2 + 1)
        # normal 13, group fairness: 5*2+5 +2 fairvote, fairmanf 14+2|8

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + 46]
        tag_tst = csv_row_1[58: 58 + 46]
        return params, tag_trn, tag_tst

    def painting_fig1(self, dframe, tag, index, ind=0, split=False):
        _, _, tag_manf, tag_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # tag_acc,tag_fair,
        cmap_name, col_Y, df_raw = 'Accent', 'Fairness', dframe.iloc[index]

        annotX = 'Performance ({})'.format(self._pick_metric[ind])
        annotY = 'Fairness measure'
        suffix_1 = self._figname + '_pc1_mat{}'.format(ind)  # _pcurve1
        suffix_2 = self._figname + '_pc2_mat{}'.format(ind)  # _pcurve2
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY, figname=suffix_1, cmap_name=cmap_name)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            figname=suffix_2, cmap_name=cmap_name)

        if split:
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-1],
                self._picked_keys[:-1], annotX, annotY,
                figname=suffix_1 + '_df')  # '_direct'
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-2] + [tag_Ys[-1]],
                self._picked_keys[:-2] + [self._picked_keys[-1]],
                annotX, annotY, figname=suffix_1 + '_hat')  # '_approx'

    def painting_fig2(self, dframe, nb_set, tag, id_att,
                      each_att, ind=0):
        _, tag_fair, tag_manf, tag_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # tag_acc,
        col_Y = 'Fairness'

        annotX = '{} performance'.format(self._pick_metric[ind])
        annotY = 'Fairness measure'
        prefix_1 = '{}_pc1_mat{}'.format(self._figname, ind)
        prefix_2 = '{}_pc2_mat{}'.format(self._figname, ind)

        i = 0
        df_raw = dframe.iloc[id_att[i]: id_att[i] + each_att]
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY,
            figname='{}_set{}_att{}'.format(prefix_1, i, 1))
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            figname='{}_set{}_att{}'.format(prefix_2, i, 1))

        for i in range(nb_set - 1):
            j = i * 2 + 1
            df_raw = dframe.iloc[id_att[j]: id_att[j] + each_att]
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY,
                figname='{}_set{}_att{}'.format(prefix_1, i + 1, 1))
            lineplot_with_uncertainty(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                figname='{}_set{}_att{}'.format(prefix_2, i + 1, 1))

            j = i * 2 + 2
            df_raw = dframe.iloc[id_att[j]: id_att[j] + each_att]
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY,
                figname='{}_set{}_att{}'.format(prefix_1, i + 1, 2))
            lineplot_with_uncertainty(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                figname='{}_set{}_att{}'.format(prefix_2, i + 1, 2))

    def drawing_fig1_alt(self, dframe, tag, index, ind=0,
                         fig='tst', split=False, linreg=False):
        _, _, tag_f_man, tag_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # _:tag_acc,tag_f_vot,
        col_Y = 'Fairness'

        annotX = 'Performance ({})'.format(self._pick_metric[ind])
        annotY = 'Fairness measure'
        suffix_1 = self._figname + '_{}_pc1_mat{}'.format(fig, ind)
        suffix_2 = self._figname + '_{}_pc2_mat{}'.format(fig, ind)

        df_raw = dframe.iloc[index]
        kwargs = {'snspec': 'sty5', 'cmap_name': self._cmap_name}
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY, figname=suffix_1 + '_s',
            cmap_name=self._cmap_name)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            figname=suffix_2, cmap_name=self._cmap_name)
        if linreg:
            line_reg_with_marginal_distr(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY, figname=suffix_1 + '_x',
                invt_a=False, **kwargs)
            line_reg_with_marginal_distr(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY, figname=suffix_1 + '_y',
                invt_a=True, **kwargs)
        kwargs.pop('snspec')
        if split:
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-1],
                self._picked_keys[:-1],
                annotX, annotY, figname=suffix_1 + '_df', **kwargs)
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-2] + tag_Ys[-1:],
                self._picked_keys[:-2] + self._picked_keys[-1:],
                annotX, annotY, figname=suffix_1 + '_hat', **kwargs)

    def schedule_mspaint(self, raw_dframe):
        nb_set, id_set, id_att, \
            each_att = self.recap_sub_data(raw_dframe, nb_row=4)
        each_gen = 0  # each generic / non-sensitive attribute?
        _, _, tag_tst = self.prepare_graph()  # tag_pm,tag_trn,
        index_overall = []
        for i in id_att:
            index_overall += list(range(i, i + each_att))

        kws = {'joint': 'none', 'fig': 'tst', 'split': False}
        kws['corrected'] = True
        self.drawing_fig2_alt(raw_dframe, tag_tst, nb_set, id_set, 
                              each_gen, each_att, **kws)

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[: 12]
        tag_f_vot = tag[23: 23 + 7]   # tag_fair_vote
        tag_f_man = tag[30: 30 + 16]  # tag_fair_manf
        tag_Ys = tag_f_vot[1: 1 + 3] + [
            tag_f_vot[-2], tag_f_man[2], tag_f_man[2 + 7]]
        col_X = tag_acc[ind]  # ↑ aka. tmp_f_vm
        return tag_acc, tag_f_vot, tag_f_man, tag_Ys, col_X

    def drawing_fig2_alt(self, dframe, tag, nb_set, id_set, 
                         each_gen, each_att, joint='none', fig='tst',
                         split=False, corrected=False):
        _, _, tag_f_man, _, _ = self.picking_fig_tags(
            tag, ind=0)  # _,tag_f_vot, # , corrected=corrected)
        # tYs_k2 = [0, 3, 1, 4, 6, 7]  # Direct vs Approx .Dist
        # tmp_f_vm = [tag_f_man[k] for k in [0, 3, 1, 4, 6, 7]]
        if corrected:
            tmp_f_vm = [tag_f_man[
                k] for k in [0, 7, 1, 8, 14, 15, 4, 11, 5, 12]]
        else:
            tmp_f_vm = [tag_f_man[k] for k in [0, 7, 1, 8, 14, 15]]
        _, suffix = self.draw_sub2_jt(joint)  # tmp,
        suff_4 = 'exp3a_minmax_{}_{}_scat'.format(suffix, fig)
        suff_5 = 'exp3a_minmax_{}_{}_tim'.format(suffix, fig)
        suff_6 = 'exp3a_minmax_{}_{}_sep'.format(suffix, fig)

        i = 0  # i, k = 0, 0
        df_raw = dframe[tmp_f_vm].iloc[id_set[i] + 1: id_set[i + 1]]
        if split:
            _sub_depict_scat(
                df_raw, tmp_f_vm, suff_4 + '_aset{}'.format(i))
            _sub_depict_tim(
                df_raw, tmp_f_vm, suff_5 + '_aset{}'.format(i))
        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(curr_set, curr_set + each_gen + each_att))
            df_tmp = dframe[tmp_f_vm].iloc[curr_loc]
            if split:
                _sub_depict_scat(df_tmp, tmp_f_vm,
                                 suff_4 + '_set{}_att0'.format(i))
                _sub_depict_tim(df_tmp, tmp_f_vm,
                                suff_5 + '_set{}_att0'.format(i))
            del df_tmp, curr_loc
            curr_loc = list(range(
                curr_set, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_tmp = dframe[tmp_f_vm].iloc[curr_loc]
            if split:
                _sub_depict_scat(df_tmp, tmp_f_vm,
                                 suff_4 + '_set{}_att1'.format(i))
                _sub_depict_tim(df_tmp, tmp_f_vm,
                                suff_5 + '_set{}_att1'.format(i))
            del df_tmp, curr_loc

            df_tmp = dframe[tmp_f_vm].iloc[id_set[i] + 1:
                                           id_set[i + 1]]
            if split:
                _sub_depict_scat(df_tmp, tmp_f_vm,
                                 suff_4 + '_aset{}'.format(i))
                _sub_depict_tim(df_tmp, tmp_f_vm,
                                suff_5 + '_aset{}'.format(i))
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            del df_tmp, curr_set

        df_raw = df_raw.reset_index(drop=True)
        _sub_depict_scat(df_raw, tmp_f_vm, suff_4, diff=True)
        _sub_depict_tim(df_raw, tmp_f_vm, suff_5, diff=True,
                        log_taken=True)
        _sub_depict_sep(df_raw, tmp_f_vm, suff_6, '_Ds')
        _sub_depict_sep(df_raw, tmp_f_vm, suff_6, '_Df')
        if corrected:
            _sub_depict_sep_alt(df_raw, tmp_f_vm,
                                suff_6 + 'alt', '_TDs',
                                True, False)
            _sub_depict_sep_alt(df_raw, tmp_f_vm,
                                suff_6 + 'alt', '_TDf',
                                True, False)
        return


class Plot2B_comparison(Plot2_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 211)

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + 105]
        tag_tst = csv_row_1[117: 117 + 105]
        return params, tag_trn, tag_tst

    def painting_fig1(self, dframe, tag, nb_set, id_set,
                      ind=0, joint='and|or', fig='tst'):
        _, tag_fair, tag_manf, tmp_fm, col_X = self.picking_fig_tags(
            tag, ind)  # tag_acc,
        tmp, _ = self.draw_sub2_jt(joint)
        col_Y, annotY = 'Fairness', 'Fairness measure'
        annotX = 'Performance ({})'.format(self._pick_metric[ind])
        suffix_1 = self._figname + '_{}_pc1_mat{}'.format(fig, ind)
        suffix_2 = self._figname + '_{}_pc2_mat{}'.format(fig, ind)

        i, j = 0, 0
        df_raw = dframe[tmp_fm[j]].iloc[id_set[i] + 1:
                                        id_set[i + 1]]
        for i in range(1, nb_set):
            j = 0
            df_tmp = dframe[tmp_fm[j]].iloc[id_set[i] + 1:
                                            id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_fm[j]].iloc[id_set[i] + 1:
                                                id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_fm[0],
                                                    tmp_fm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)

        cmap_name = self._cmap_name  # 'Accent'
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tmp_fm[0][1:], self._picked_keys,
            annotX, annotY, figname=suffix_1, cmap_name=cmap_name)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tmp_fm[0][1:], self._picked_keys,
            figname=suffix_2 + '_b4', cmap_name=cmap_name)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tmp_fm[0][1:], self._picked_keys,
            figname=suffix_2 + '_af', cmap_name=cmap_name, alpha_loc='af')

    def painting_fig2(self, dframe, nb_set, tag, id_att, each_att,
                      ind=0, joint='and|or'):
        tag_acc = tag[: 12]
        tag_fair = tag[13: 13 + 7 * 4]
        tag_manf = tag[41: 41 + 16 * 4]  # 41 + 8 * 4]

        col_X, col_Y = tag_acc[ind], 'Fairness'
        if joint == 'and':
            tag_fair = tag_fair[: 7 * 3]
            tag_manf = tag_manf[: 16 * 3]  # 8 * 3]
            tmp = [0, 1, 2, ]
        elif joint == 'or':
            tag_fair = tag_fair[: 7 * 2] + tag_fair[-7:]
            tag_manf = tag_manf[: 16 * 3] + tag_manf[-16:]
            tmp = [0, 1, 3, ]
        else:
            tmp = list(range(4))
        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9]  # [2, 5]
        annotX = '{} performance'.format(self._pick_metric[ind])
        annotY = 'Fairness measure'
        prefix_1 = '{}_pc1_mat{}'.format(self._figname, ind)
        prefix_2 = '{}_pc2_mat{}'.format(self._figname, ind)

        i = 0
        df_raw = dframe.iloc[id_att[i]: id_att[i] + each_att]
        j = 0
        tag_Ys = [tag_fair[k] for k in tYs_k1] + [tag_manf[k] for k in tYs_k2]
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys, annotX, annotY,
            figname='{}_set{}_att{}_on{}'.format(prefix_1, i, 0, j))
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            figname='{}_set{}_att{}_on{}'.format(prefix_2, i, 0, j))

        for i in range(1, nb_set):
            p = (i - 1) * 2 + 1
            df_raw = dframe.iloc[id_att[p]: id_att[p] + each_att]
            for j in tmp:
                tag_Ys = [tag_fair[k + 7 * j] for k in tYs_k1] + [
                    tag_manf[k + 16 * j] for k in tYs_k2]
                scatter_with_marginal_distrib(
                    df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                    annotX, annotY,
                    figname='{}_set{}_att{}_on{}'.format(
                        prefix_1, i, 0, j))
                lineplot_with_uncertainty(
                    df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                    figname='{}_set{}_att{}_on{}'.format(
                        prefix_2, i, 0, j))

            p = (i - 1) * 2 + 2
            df_raw = dframe.iloc[id_att[p]: id_att[p] + each_att]
            for j in tmp:
                tag_Ys = [tag_fair[k + 7 * j] for k in tYs_k1] + [
                    tag_manf[k + 16 * j] for k in tYs_k2]
                scatter_with_marginal_distrib(
                    df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                    annotX, annotY,
                    figname='{}_set{}_att{}_on{}'.format(
                        prefix_1, i, 0, j))
                lineplot_with_uncertainty(
                    df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                    figname='{}_set{}_att{}_on{}'.format(
                        prefix_2, i, 0, j))

    def drawing_fig1_alt(self, dframe, tag, nb_set, id_set, ind=0,
                         joint='and|or', dist='direct', fig='tst',
                         linreg=True, split=True):
        _, tag_f_vot, tag_f_man, tmp_f_vm, col_X = self.picking_fig_tags(
            tag, ind)  # tag_acc,
        tmp, suffix = self.draw_sub2_jt(joint)
        col_Y, annotY = 'Fairness', 'Fairness measure'
        annotX = 'Performance ({})'.format(self._pick_metric[ind])
        tag_Ys = tmp_f_vm[0][1:]
        suffix_1 = self._figname + '_{}_pc1_mat{}_{}'.format(suffix, ind, fig)
        suffix_2 = self._figname + '_{}_pc2_mat{}_{}'.format(suffix, ind, fig)

        i, j = 0, 0
        df_raw = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            j = 0
            df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                                  id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)

        kwargs = {'cmap_name': self._cmap_name}
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY, figname=suffix_1 + '_s', **kwargs)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            alpha_rev=True, alpha_loc='b4',
            figname=suffix_2 + '_b4', cmap_name='coolwarm_r')

        if split:
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-1],
                self._picked_keys[:-1],
                annotX, annotY, figname=suffix_1 + '_df', **kwargs)
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-2] + tag_Ys[-1:],
                self._picked_keys[:-2] + self._picked_keys[-1:],
                annotX, annotY, figname=suffix_1 + '_hat', **kwargs)
        kwargs['snspec'] = 'sty5'
        if linreg:
            line_reg_with_marginal_distr(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY, figname=suffix_1 + '_x',
                invt_a=False, **kwargs)
            line_reg_with_marginal_distr(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY, figname=suffix_1 + '_y',
                invt_a=True, **kwargs)

    def schedule_mspaint(self, raw_dframe):
        nb_set, id_set, _, each_att = self.recap_sub_data(
            raw_dframe, nb_row=4)  # _:id_att
        each_gen = 0
        _, tag_trn, tag_tst = self.prepare_graph()  # tag_pm,

        self.drawing_fig2_alt(raw_dframe, tag_tst, nb_set, id_set,
                              each_gen, each_att)

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[: 12]
        tag_f_vot = tag[13: 13 + 7 * 4]
        tag_f_man = tag[41: 41 + 16 * 4]
        col_X = tag_acc[ind]  # , 'Fairness'

        tag_f_vot = [tag_f_vot[: 7], tag_f_vot[7: 14],
                     tag_f_vot[14: 21], tag_f_vot[21:]]
        tag_f_man = [tag_f_man[: 16], tag_f_man[16: 32],
                     tag_f_man[32: 48], tag_f_man[48:]]

        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9],  # [2, 5, ]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tmp_f_vm = [[
            col_X] + t1 + t2 for t1, t2 in zip(tmp_f_vot, tmp_f_man)]

        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2
        return tag_acc, tag_f_vot, tag_f_man, tmp_f_vm, col_X

    def drawing_fig2_alt(self, dframe, tag, nb_set, id_set,
                         each_gen, each_att, joint='none', fig='tst'):
        # _, tag_f_vot, tag_f_man, _, _ = self.picking_fig_tags(tag, ind=0)
        _, _, tag_f_man, _, _ = self.picking_fig_tags(tag, ind=0)
        # tYs_k2 = [0, 3, 1, 4, 6, 7]  # DirectDist vs ApproxDist
        tYs_k2 = [0, 7, 1, 8, 14, 15, ] + [4, 11, 5, 12, ]
        tmp_f_vm = [[t[k] for k in tYs_k2] for t in tag_f_man]
        del tYs_k2
        tmp, suffix = self.draw_sub2_jt(joint)
        suff_4 = 'exp3b_minmax_{}_{}_scat'.format(suffix, fig)
        suff_5 = 'exp3b_minmax_{}_{}_tim'.format(suffix, fig)

        df_raw = self.draw_sub2_dat2(
            dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)
        tYs = tmp_f_vm[0]

        _sub_depict_scat(df_raw, tYs, suff_4)
        _sub_depict_tim(df_raw, tYs, suff_5)
        return


class Plot2C_comparison(Plot2B_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2c_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def recap_sub_data(self, dframe, nb_row=4):
        # 11 indiv, 3 gen ensem, 4* #att={1|2}
        each_gen = (11 + 3) * self._nb_iter
        each_att = 4 * self._nb_iter
        each_set = each_gen + each_att * 2 + 1
        temp_set = each_gen + each_att + 1

        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set - temp_set) // each_set + 1
        id_set = [0] + [
            i * each_set + temp_set for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]

        return nb_set, id_set, each_gen, each_att

    def schedule_mspaint(self, raw_dframe):
        nb_set, id_set, each_gen, \
            each_att = self.recap_sub_data(raw_dframe, nb_row=4)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()

        '''
        # for ind in [None, 0, 3, 1, 2, 7]:
        for ind in [0, 3, 1, 2, 7]:
            self.drawing_fig2_alt(
                raw_dframe, tag_tst, nb_set, id_set, each_gen, each_att,
                joint='none', fig='tst', ind=ind)
        '''
        self.drawing_fig2_alt(
            raw_dframe, tag_tst, nb_set, id_set, each_gen, each_att,
            joint='none', fig='tst', ind=None)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 263)  # + 199)

        params = csv_row_1[: 11 + 1]  # last: indiv/Ensem's ut
        tag_trn = csv_row_1[12: 12 + 131]
        tag_tst = csv_row_1[143: 143 + 131]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[13 * 2: 13 * 3 - 1]
        tag_f_vot = tag[39: 39 + 7 * 4]
        tag_f_man = tag[67: 67 + 16 * 4]
        col_X = tag_acc[ind]

        tag_f_vot = [tag_f_vot[: 7], tag_f_vot[7: 14],
                     tag_f_vot[14: 21], tag_f_vot[21:]]
        tag_f_man = [tag_f_man[: 16], tag_f_man[16: 32],
                     tag_f_man[32: 48], tag_f_man[48:]]
        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9, ]  # [2, 5, ]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tmp_f_vm = [[tag[ind], col_X] + t1 + t2 for t1, t2 in zip(
            tmp_f_vot, tmp_f_man)]
        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2

        return tag_acc, tag_f_vot, tag_f_man, tmp_f_vm, col_X

    def drawing_fig1_alt(self, dframe, tag, nb_set, id_set, ind=0,
                         joint='and|or', dist='approx', fig='tst',
                         linreg=True, split=True):
        col_Y = 'Fairness'
        annotX = r'$\Delta$ Performance ({})'.format(self._pick_metric[ind])
        annotY = 'Fairness measure'
        _, _, _, tmp_f_vm, col_X = self.picking_fig_tags(tag, ind=ind)
        tmp, suffix = self.draw_sub2_jt(joint)

        suffix_1 = self._figname + '_{}_pc1_{}_mat{}'.format(
            suffix, fig, ind)
        suffix_2 = self._figname + '_{}_pc2_{}_mat{}'.format(
            suffix, fig, ind)
        suffix_3 = suffix_1.replace('exp2c_', 'exp2b_')
        tag_Ys = tmp_f_vm[0][2:]

        i, j = 0, 0
        df_raw = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            j = 0
            df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                                  id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)

        kwargs = {'cmap_name': self._cmap_name}
        annotZ = (
            ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        )
        annotXpz = 'Performance ({})'.format(self._pick_metric[ind])
        annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(
            self._pick_metric[ind])

        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY, figname=suffix_1 + '_s', **kwargs)
        scatter_with_marginal_distrib(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            annotXpz, annotY, figname=suffix_3 + '_s', **kwargs)

        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            figname=suffix_2 + '_b4', alpha_loc='b4', alpha_rev=True,
            annotY=annotZ, cmap_name='coolwarm_r')

        kwargs['snspec'] = 'sty5'
        if linreg:
            line_reg_with_marginal_distr(
                df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
                annotX, annotY, figname=suffix_1 + '_x',
                invt_a=False, **kwargs)

        kwargs.pop('snspec')
        if split:
            scatter_with_marginal_distrib(
                df_raw, col_X, col_Y, tag_Ys[:-1],
                self._picked_keys[:-1],
                annotX, annotY, figname=suffix_1 + '_df', **kwargs)
        return

    def drawing_fig2_alt(self, dframe, tag, nb_set, id_set,
                         each_gen, each_att, joint='none', fig='tst',
                         ind=0, linreg=False, split=False):
        _, _, tag_f_man, _, _ = self.picking_fig_tags(
            tag, ind=0)  # ,tag_f_vot,
        # tYs_k2 = [0, 3, 1, 4, 6, 7]  # direct,approx,direct,approx,d_ut,a_ut
        tYs_k2 = [0, 7, 1, 8, 14, 15, ] + [4, 11, 5, 12, ]
        tmp_f_vm = [[t[k] for k in tYs_k2] for t in tag_f_man]
        del tYs_k2
        tmp, suffix = self.draw_sub2_jt(joint)

        suff_4 = 'exp3c_minmax_{}_{}_scat'.format(suffix, fig)
        suff_5 = 'exp3c_minmax_{}_{}_tim'.format(suffix, fig)

        if (ind is None) and split:
            tYs = tmp_f_vm[0]
            i = 0  # i, k = 0, 0
            df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            _sub_depict_scat(df_raw, tYs, suff_4 + '_aset{}'.format(i))
            _sub_depict_tim(df_raw, tYs, suff_5 + '_aset{}'.format(i))
            for i in range(1, nb_set):
                curr_set = id_set[i] + 1
                curr_loc = list(range(curr_set,
                                      curr_set + each_gen + each_att))
                df_raw = dframe[tmp_f_vm[0]].iloc[curr_loc]
                _sub_depict_scat(df_raw, tYs,
                                 suff_4 + '_set{}_att0'.format(i))
                _sub_depict_tim(df_raw, tYs,
                                suff_5 + '_set{}_att0'.format(i))
                curr_loc = list(range(
                    curr_set, curr_set + each_gen)) + list(range(
                        curr_set + each_gen + each_att,
                        curr_set + each_gen + each_att * 2))
                df_tmp = dframe[tmp_f_vm[1]].iloc[curr_loc]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[1])}
                df_tmp = df_tmp.rename(columns=columns)
                _sub_depict_scat(df_tmp, tYs,
                                 suff_4 + '_set{}_att1'.format(i))
                _sub_depict_tim(df_tmp, tYs,
                                suff_5 + '_set{}_att1'.format(i))
                df_raw = pd.concat([df_raw, df_tmp], axis=0).reset_index(drop=True)
                _sub_depict_scat(df_raw, tYs,
                                 suff_4 + '_aset{}'.format(i))
                _sub_depict_tim(df_raw, tYs,
                                suff_5 + '_aset{}'.format(i))
            return

        elif ind is None:
            df_raw = self.draw_sub2_dat2(
                dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)
            tYs = tmp_f_vm[0]

            _sub_depict_scat(df_raw, tYs, suff_4, diff=True)
            _sub_depict_tim(df_raw, tYs, suff_5, diff=True, log_taken=True)
            suff_6 = 'exp3c_minmax_{}_{}_sep'.format(suffix, fig)
            suff_7 = 'exp3c_minmax_{}_{}_altsep'.format(suffix, fig)
            _sub_depict_sep(df_raw, tYs, suff_6, '_Ds')
            _sub_depict_sep(df_raw, tYs, suff_6, '_Df')
            _sub_depict_sep_alt(df_raw, tYs, suff_7, '_TDs', double_sep=False)
            _sub_depict_sep_alt(df_raw, tYs, suff_7, '_TDf', double_sep=False)
            return
        #
        # RE DO PLOTTING
        #             #

        fig_nm = self._figname.replace('iter5_cls7_', '')
        suff_1 = fig_nm + '_{}_pc1_{}_mat{}'.format(suffix, fig, ind)
        suff_2 = fig_nm + '_{}_pc2_{}_mat{}'.format(suffix, fig, ind)
        suff_3 = suff_1.replace('exp2c_', 'exp2b_')
        del fig_nm

        col_Y, annotY = 'Fairness', 'Fairness measure'  # ' Measure'
        annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(
            self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        annotXpz = 'Performance ({})'.format(self._pick_metric[ind])
        _, _, _, tmp_f_vm, col_X = self.picking_fig_tags(tag, ind=ind)
        tag_Ys = tmp_f_vm[0][2:]
        df_raw = self.draw_sub2_dat2(
            dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)

        kws = {'cmap_name': self._cmap_name}  # kwargs
        scatter_with_marginal_distrib(
            df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
            annotX, annotY, figname=suff_1 + '_s', **kws)
        scatter_with_marginal_distrib(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            annotXpz, annotY, figname=suff_3 + '_s', **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            figname=suff_2 + '_b4', alpha_loc='b4', alpha_rev=True,
            annotY=annotZ, cmap_name='coolwarm_r', alpha_clarity=.15)
        return


# ===============================
# Replotting


# -------------------------------
#


class RePlot2_comparison(GraphSetup):
    def __init__(self, nb_iter, nb_cls,
                 m1=30, m2=16, figname='exp2_'):
        gen, rep = None, True
        super().__init__(gen, rep, m1, m2, figname)
        self._cmap_name = 'coolwarm_r'
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter
        self._nb_cls = nb_cls
        self._picked_keys = ['DP', 'EO', 'PQP'] + [
            'DR', r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        self._pick_metric = [
            'Accuracy', 'Precision', 'Recall', r'$f_1$ score',
            'FPR rate', 'FNR rate', 'Sensitivity', 'Specificity']

    def recap_sub_data(self, dframe, nb_row=4):
        each_att = 7 * self._nb_iter
        each_set = each_att * 2 + 1

        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set - (each_att + 1)) // each_set + 1

        id_set = [0] + [i * each_set + (
            each_att + 1) for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]

        id_att = [id_set[0] + 1]
        for i in id_set[1: -1]:
            id_att.extend([i + 1, i + 1 + each_att])
        return nb_set, id_set, id_att, each_att

    def painting_prep(self, raw_dframe):
        pass

    def painting_fig1(self):
        raise NotImplementedError

    def painting_fig2(self):
        raise NotImplementedError

    def drawing_fig1_alt(self):
        raise NotImplementedError

    def drawing_redo_fig6_inner(self):
        pass

    def draw_sub_manf_ext_dat1(self, dframe, nb_set, id_set, tmp_f_vm,
                               tmp):  # or tmp_jt
        i = j = 0
        df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            df_tmp = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                                  id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # 18+(18+4)*4 =106  → 18+(18+4)*2*4 =194
        return df_raw.reset_index(drop=True)

    def draw_sub_manf_ext_dat2(self, dframe, nb_set, id_set,
                               each_gen, each_att, tmp_f_vm):
        i = 0  # i = k = 0
        df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            curr_set = id_set[i] + 1

            curr_loc = list(range(curr_set,
                                  curr_set + each_gen + each_att))
            df_tmp = dframe[tmp_f_vm[0]].iloc[curr_loc]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

            curr_loc = list(range(
                curr_set, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_tmp = dframe[tmp_f_vm[1]].iloc[curr_loc]
            columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                tmp_f_vm[1])}
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        # 18+(18+ 14+4)*4 =162
        return df_raw.reset_index(drop=True)

    def draw_sub_manf_dat3(self, dframe, nb_set, id_set, each_att,
                           tmp_f_vm=None):
        id_att = [id_set[0] + 1]
        for i in id_set[1: -1]:
            id_att.extend([i + 1, i + 1 + each_att])
        index_overall = []
        for i in id_att:
            index_overall += list(range(i, i + each_att))
        if tmp_f_vm is None:
            return dframe.iloc[index_overall]
        return dframe[tmp_f_vm].iloc[index_overall]

    def draw_sub_manf_dat4_alt1(self,
                                dframe, nb_set, id_set, drop_gen,
                                tmp_f_vm, tmp):  # drop_gen= 11*5
        i = j = 0  # i = j = k = 0
        curr_set = id_set[i] + 1 + drop_gen
        df_raw = dframe[tmp_f_vm[0]].iloc[curr_set: id_set[i + 1]]
        for i in range(1, nb_set):
            curr_set = id_set[i] + 1 + drop_gen
            df_tmp = dframe[tmp_f_vm[0]].iloc[curr_set: id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[curr_set: id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0],
                                                    tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
            # curr_loc = list(range())   # 3+4+(3+4*2)*4 =51
        # → 3+4+(3+4*2)*2*4 =7+11*8 =95
        return df_raw.reset_index(drop=True)

    def draw_sub_manf_dat4_alt2(self,
                                dframe, nb_set, id_set, drop_gen,
                                each_gen, each_att, tmp_f_vm):
        i = 0  # i = j = k = 0
        curr_set = id_set[i] + 1 + drop_gen
        df_raw = dframe[tmp_f_vm[0]].iloc[curr_set: id_set[i + 1]]
        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(curr_set + drop_gen,
                                  curr_set + each_gen + each_att))
            df_tmp = dframe[tmp_f_vm[0]].iloc[curr_loc]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

            curr_loc = list(range(
                curr_set + drop_gen, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_tmp = dframe[tmp_f_vm[1]].iloc[curr_loc]
            columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0], tmp_f_vm[1])}
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # 7+(7*2)*4 =63
        return df_raw.reset_index(drop=True)


# RQ1. compared with sota fairness cont.

class Replot2A_comparison(RePlot2_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2a_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 46 * 2 + 1)
        # # normal 13, group fairness: 5*2+5 +2 fairvote, fairmanf 14+2|8
        # normal 13, group fairness: 5*2+5 +2 fairvote, fairmanf 7+7+2
        # that is, 13+17+16 =46

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + 46]
        tag_tst = csv_row_1[58: 58 + 46]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[: 12]
        tag_f_vot = tag[23: 23 + 7]   # tag_fair_vote
        tag_f_man = tag[30: 30 + 16]  # tag_fair_manf
        tag_Ys = tag_f_vot[1: 1 + 3] + [
            tag_f_vot[-2], tag_f_man[2], tag_f_man[2 + 7]]
        col_X = tag_acc[ind]  # ↑ aka. tmp_f_vm
        return tag_acc, tag_f_vot, tag_f_man, tag_Ys, col_X

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, each_att = self.recap_sub_data(
            raw_dframe, nb_row=4)  # id_att,
        # each_gen = 0  # each generic / non-sensitive attribute?
        tag_pm, tag_trn, tag_tst = self.prepare_graph()

        # kws = {'joint': 'none', 'fig': 'tst', 'split': False,
        #        'corrected': True, 'pre': pre}
        kws = {'fig': 'tst', 'pre': pre}
        for ind in [0, 3, 1, 2, 7]:
            self.drawing_fig4_fig1_alt(
                raw_dframe, tag_tst, ind, nb_set, id_set,
                each_att, **kws)
        return

    def drawing_fig4_fig1_alt(self, dframe, tag, ind, nb_set, id_set,
                              each_att, fig='tst', pre='minmax'):
        tag_acc, _, tag_fm, col_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # ,tag_fv,
        col_Y_alt = 'Fairness'  # ,annotY='Fairness measure'
        # annotX = 'Performance ({})'.format(self._pick_metric[ind])
        '''
        suff_1 = self._figname + '{}_{}_pc1_mat{}'.format(pre, fig, ind)
        '''
        suff_2 = self._figname + '{}_{}_lc2_mat{}'.format(pre, fig, ind)

        tmp_f_vm = [tag_acc[k] for k in [0, 3, 1, 2, 7, ]] + col_Ys
        key_A = [self._pick_metric[k] for k in [0, 3, 1, 2, 7]]  # annot_acc
        df_raw = self.draw_sub_manf_dat3(
            dframe, nb_set, id_set, each_att, tmp_f_vm)
        annotZ = ' error rate' if ind == 0 else '$($1- performance$)$'
        kwargs = {'alpha_loc': 'b4', 'alpha_rev': True, 'annotY': annotZ}
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y_alt, col_Ys, self._picked_keys,
            figname=suff_2, cmap_name=self._cmap_name, **kwargs)

        if ind != 0:
            return
        Mat_A = df_raw[tmp_f_vm[:5]].values.astype(DTY_FLT)
        Mat_B = df_raw[col_Ys].values.astype(DTY_FLT)
        suff_3 = self._figname + '{}_{}_each_confusion'.format(pre, fig)
        analogous_confusion_extended(
            Mat_A.T, Mat_B.T, key_A, self._picked_keys,
            figname=suff_3, cmap_name='Oranges', rotate=0)  # 'PuBu',Yl/OrRd
        return


class Replot2B_comparison(RePlot2_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 211)
        # normal 13, group fairness: (5+2 fairvote)*4, fairmanf (7+7+2)*4
        # that is, 13+7*4+16*4 =105

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + 105]
        tag_tst = csv_row_1[117: 117 + 105]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[: 12]
        tag_f_vot = tag[13: 13 + 7 * 4]
        tag_f_man = tag[41: 41 + 16 * 4]
        col_X = tag_acc[ind]  # , 'Fairness'

        tag_f_vot = [tag_f_vot[: 7], tag_f_vot[7: 14],
                     tag_f_vot[14: 21], tag_f_vot[21:]]
        tag_f_man = [tag_f_man[: 16], tag_f_man[16: 32],
                     tag_f_man[32: 48], tag_f_man[48:]]

        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tmp_f_vm = [[
            col_X] + t1 + t2 for t1, t2 in zip(tmp_f_vot, tmp_f_man)]

        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2
        return tag_acc, tag_f_vot, tag_f_man, tmp_f_vm, col_X

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, each_att = self.recap_sub_data(
            raw_dframe, nb_row=4)  # ,id_att,
        each_gen = 0
        tag_pm, tag_trn, tag_tst = self.prepare_graph()
        kws = {'joint': 'none', 'fig': 'tst', 'pre': pre}  # ={}
        for ind in [0, 3]:
            self.drawing_fig4_fig1_alt(
                raw_dframe, tag_tst, ind, nb_set, id_set, each_gen,
                each_att, **kws)
        return

    def drawing_fig4_fig1_alt(self, dframe, tag, ind, nb_set, id_set,
                              each_gen, each_att, joint='none',
                              fig='tst', pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        tag_acc, _, tag_fm, col_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # ,tag_fv,
        col_Y_alt = 'Fairness'  # ,annotY= 'Fairness measure'
        # annotX = 'Performance ({})'.format(self._pick_metric[ind])
        # tag_Ys_alt = tag_fm[0][1:]
        suff_2 = self._figname + '{}_{}_lc2_mat{}'.format(
            pre, fig, ind)
        suff_3 = self._figname + '{}_{}_each_confusion'.format(
            pre, fig)

        df_raw = self.draw_sub_manf_ext_dat1(dframe, nb_set, id_set,
                                             col_Ys, tmp)
        annotZ = ' error rate' if ind == 0 else '$($1- performance$)$'
        kwargs = {'alpha_loc': 'b4', 'alpha_rev': True,
                  'annotY': annotZ}
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y_alt, col_Ys[0][1:],
            self._picked_keys, figname=suff_2,
            cmap_name=self._cmap_name, **kwargs)

        if ind != 0:
            return
        key_A = [self._pick_metric[k] for k in [0, 3, 1, 2, 7, ]]
        tmp_f_vm = [[tag_acc[k] for k in [
            0, 3, 1, 2, 7]] + tY[1:] for tY in col_Ys]
        df_tmp = self.draw_sub_manf_ext_dat1(dframe, nb_set, id_set,
                                             tmp_f_vm, tmp)
        Mat_A = df_tmp[tmp_f_vm[0][:5]].values.astype(DTY_FLT)
        Mat_B = df_tmp[tmp_f_vm[0][5:]].values.astype(DTY_FLT)
        analogous_confusion_extended(
            Mat_A.T, Mat_B.T, key_A, self._picked_keys,
            figname=suff_3, cmap_name='Oranges', rotate=0)
        return


class Replot2C_comparison(Replot2B_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2c_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 263)
        # normal 13*3, group fair (5+2 fairvote)*4, fairmanf 16*4
        # that is, 13*3+28+(7+7+2)*4 =131

        params = csv_row_1[: 11 + 1]  # last: indiv/Ensem's ut
        tag_trn = csv_row_1[12: 12 + 131]
        tag_tst = csv_row_1[143: 143 + 131]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0):
        tag_acc = tag[13 * 2: 13 * 3 - 1]
        tag_f_vot = tag[39: 39 + 7 * 4]
        tag_f_man = tag[67: 67 + 16 * 4]
        col_X = tag_acc[ind]

        tag_f_vot = [tag_f_vot[: 7], tag_f_vot[7: 14],
                     tag_f_vot[14: 21], tag_f_vot[21:]]
        tag_f_man = [tag_f_man[: 16], tag_f_man[16: 32],
                     tag_f_man[32: 48], tag_f_man[48:]]
        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9, ]

        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tmp_f_vm = [[tag[ind], col_X] + t1 + t2 for t1, t2 in zip(
            tmp_f_vot, tmp_f_man)]
        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2

        return tag_acc, tag_f_vot, tag_f_man, tmp_f_vm, col_X

    def recap_sub_data(self, dframe, nb_row=4):
        # 11 indiv, 3 gen ensem, 4* #att= {1|2}
        each_gen = (11 + 3) * self._nb_iter
        each_att = 4 * self._nb_iter
        each_set = each_gen + each_att * 2 + 1
        temp_set = each_gen + each_att + 1

        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set - temp_set) // each_set + 1
        id_set = [0] + [i * each_set + temp_set for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]

        return nb_set, id_set, each_gen, each_att

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_gen, each_att = self.recap_sub_data(
            raw_dframe, nb_row=4)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()
        kws = {'joint': 'none', 'fig': 'tst', 'pre': pre}
        drop_gen = 11 * self._nb_iter  # 11*5
        for ind in [0, 3]:
            self.drawing_fig4_fig1_alt(
                raw_dframe, tag_tst, ind, nb_set, id_set,
                drop_gen, each_gen, each_att, **kws)
        return

    def drawing_fig4_fig1_alt(self, dframe, tag, ind, nb_set, id_set,
                              drop_gen, each_gen, each_att,
                              joint='none', fig='tst',
                              pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        tag_acc, _, _, col_Ys, col_X = self.picking_fig_tags(
            tag, ind)  # ,tag_fv,tag_fm,
        tag_non = tag[: 13 - 1]  # non_adversarial_acc
        col_Y_alt = 'Fairness'  # ,annotY='Fairness measure'
        # annotX = 'Performance ({})'.format(self._pick_metric[ind])
        suff_2 = self._figname + '{}_{}_lc2_mat{}'.format(
            pre, fig, ind)
        suff_3 = self._figname + '{}_{}_each_confusion'.format(
            pre, fig)
        suff_4 = self._figname + '{}_{}_ealt_confusion'.format(
            pre, fig)

        df_raw = self.draw_sub_manf_ext_dat1(dframe, nb_set, id_set,
                                             col_Ys, tmp)
        annotZ = ' error rate' if ind == 0 else '$($1- performance$)$'
        kwargs = {'alpha_loc': 'b4', 'alpha_rev': True, 'annotY': annotZ}
        lineplot_with_uncertainty(
            df_raw, col_Ys[0][0], col_Y_alt, col_Ys[0][2:],
            self._picked_keys, figname=suff_2,
            cmap_name=self._cmap_name, **kwargs)

        if ind != 0:
            return
        idx_A_C = [0, 3, 1, 2, 7, ]
        tmp_A_C = [tag_non[k] for k in idx_A_C] + [
            tag_acc[k] for k in idx_A_C]
        key_A = [self._pick_metric[k] for k in idx_A_C]
        key_C = [r'$\Delta${}'.format(
            self._pick_metric[k]) for k in idx_A_C]
        tmp_f_vm = [tmp_A_C + tY[2:] for tY in col_Ys]
        df_tmp = self.draw_sub_manf_dat4_alt1(
            dframe, nb_set, id_set, drop_gen, tmp_f_vm, tmp)
        Mat_A = df_tmp[tmp_f_vm[0][: 5]].values.astype(DTY_FLT).T
        Mat_C = df_tmp[tmp_f_vm[0][5:10]].values.astype(DTY_FLT).T
        Mat_B = df_tmp[tmp_f_vm[0][10:]].values.astype(DTY_FLT).T
        kwargs = {'cmap_name': 'Oranges', 'rotate': 0}
        analogous_confusion_extended(
            Mat_A, Mat_B, key_A, self._picked_keys,
            figname=suff_3 + '_norm', **kwargs)
        analogous_confusion_extended(
            Mat_C, Mat_B, key_C, self._picked_keys,
            figname=suff_3 + '_advr', **kwargs)

        df_tmp = self.draw_sub_manf_dat4_alt2(
            dframe, nb_set, id_set, drop_gen, each_gen, each_att, tmp_f_vm)
        Mat_A = df_tmp[tmp_f_vm[0][:5]].values.astype(DTY_FLT).T
        Mat_C = df_tmp[tmp_f_vm[0][5:10]].values.astype(DTY_FLT).T
        Mat_B = df_tmp[tmp_f_vm[0][10:]].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            Mat_A, Mat_B, key_A, self._picked_keys,
            figname=suff_4 + '_norm', **kwargs)
        analogous_confusion_extended(
            Mat_C, Mat_B, key_C, self._picked_keys,
            figname=suff_4 + '_advr', **kwargs)
        return


# ===============================
#


# -------------------------------
#

# -------------------------------
#
