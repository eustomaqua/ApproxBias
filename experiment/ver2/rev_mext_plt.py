# coding: utf-8


import pdb
import pandas as pd
import numpy as np

from experiment.facil.draw_addtl import (
    multi_lin_reg_without_distr, lineplot_with_uncertainty)
from experiment.facil.draw_graph import scatter_id_chart
from experiment.facil.draw_chart import analogous_confusion_extended
from experiment.generic import GraphSetupVer2

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4


# ==============================
# Benchmarks


# ------------------------------


class GraphSetup(GraphSetupVer2):  # REVISION S
    def sub_dat_multivar(self, dframe, nb_set, id_set, tag):
        i = 0  # i, k = 0, 0
        df_raw = dframe[tag].iloc[id_set[i] + 1: id_set[i + 1]]
        for i in range(1, nb_set):
            df_tmp = dframe[tag].iloc[id_set[i] + 1: id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw.reset_index(drop=True)

    def sub_dat_sen_att(self, dframe, tag_sa1, tag_sa2):
        # nb_set, id_set,
        df_no_sa1 = dframe[tag_sa1]
        df_no_sa2 = dframe[tag_sa2]
        df_no_sa2 = df_no_sa2.iloc[self._nb_iter:]
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_no_sa2 = df_no_sa2.rename(columns=columns)
        df_raw = pd.concat([df_no_sa1, df_no_sa2], axis=0)
        return df_raw.reset_index(drop=True)

    def sub_dat_sing_set(self, dframe, id_set, no_set, tag,
                         tag_sa1, tag_sa2):
        assert 1 <= no_set <= 5
        i = no_set - 1
        df_multivar = dframe[tag].iloc[id_set[i] + 1: id_set[i + 1]]
        df_multivar = df_multivar.reset_index(drop=True)
        i_row_list = list(range(id_set[i] + 1, id_set[i + 1]))
        df_no_sa1 = dframe[tag_sa1].iloc[i_row_list]
        if no_set >= 2:
            df_no_sa2 = dframe[tag_sa2].iloc[i_row_list]
            columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
            df_no_sa2 = df_no_sa2.rename(columns=columns)
            df_no_sa1 = pd.concat([df_no_sa1, df_no_sa2], axis=0)
            del columns, df_no_sa2
        df_no_sa1 = df_no_sa1.reset_index(drop=True)
        return df_multivar, df_no_sa1


# ------------------------------

# rexp1: 对比 an efficient method for Hausdorff
#   rexp1a: multivar (incl. nonbin x n_a) 对比结果
#   rexp1b: nonbin ~singularly~ 单独计算
#   rexp1c: bin 单独计算，把 multi-val 当成 bi-val 计算
# rexp2: 齐琪，考虑 embedding +y_hat 计算 df
#   rexp2a: 不是交叉验证
#   rexp2b: 交叉验证，用BaseNet,
#   rexp2c: 交叉验证，用BaseNet, 比 rexp2b 信息更多
# rexp3: 对比 statistical parity
#   rexp3c/3b: 交叉验证，不要 embedding，直接算 statistical parity; 用BaseNet
#   rexp3d: 交叉验证，用其他普通的 sklearn 的学习器，其他跟 rexp3c 一模一样
#   rexp3e|3f: 交叉验证，用一些分类器，结果都放在一起，基于 rexp3d|3c


# ------------------------------
# Rexp1


class RevPlotZ_comparison(GraphSetup):
    def sub_plt_tim(self, df_tmp, tag_col, suff, rmk='multivar',
                    omitted=True):
        tag_tim = tag_col[: 4]
        # tag_tim = tag_col[: 3]
        # tag_max = tag_col[3: 3 + 3]
        # tag_avg = tag_col[3 + 3: 3 + 3 + 2]

        scat_X = df_tmp[tag_tim[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[1]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[3]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            # annotX = r'T_{D}'
            ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
            ant_app = r'T_{ExtendDist}'
        elif rmk == 'sen-att':
            ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S,a_i)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)}'
            ant_app = r'T_{ApproxDist}'
        ant_eff = r'T_{EfficientHD}'  # r'T_{EffHD}'
        annotY = ['${}$'.format(ant_eff),  # '${}$'.format(ant_Y)
                  '${}$'.format(ant_app)]
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty4')
        if not omitted:
            multi_lin_reg_without_distr(
                df_tmp[tag_tim[0]].values.astype(DTY_FLT),
                scat_Y, annotY, annot, suff + '_alt', snspec='sty4')
        return

    def sub_plt_val(self, df_tmp, tag_col, suff, rmk='multivar',
                    omitted=True):
        tag_max = tag_col[4: 4 + 4]
        tag_avg = tag_col[4 + 4: 4 + 4 + 2]
        scat_X = df_tmp[tag_max[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_max[1]].values.astype(DTY_FLT),
                  df_tmp[tag_max[3]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'\mathbf{D}_{\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            ant_app = 'ExtendDist'
        elif rmk == 'sen-att':
            ant_X = r'\mathbf{D}_{\mathbf{a}}(S,a_i)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
            ant_app = 'ApproxDist'
        ant_eff = 'EfficientHD'
        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app)]
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty3b')
        if not omitted:
            multi_lin_reg_without_distr(
                df_tmp[tag_max[0]].values.astype(DTY_FLT),
                scat_Y, annotY, annot, suff + '_alt',
                snspec='sty3a')  # sty6
        return


class RevP_ZA_efficient(RevPlotZ_comparison):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1,
            nc_sens=0)  # each_set,each_att/gen
        csv_row_1 = unique_column(10 + 30)
        # df_raw = self.sub_dat_merged(raw_dframe, nb_set, id_set)
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])  # 10+10])

        tag_multivar = csv_row_1[10: 10 + 10]
        df_tmp = df_raw[tag_multivar]
        # self.sub_plt_tim_val(df_tmp, tag_multivar)
        tag_sa1 = csv_row_1[20: 20 + 10]
        tag_sa2 = csv_row_1[20 + 10: 20 + 10 + 10]
        df_tmp = self.sub_dat_sen_att(df_raw, tag_sa1, tag_sa2)
        # suff, rmk = 'rexp1a_whole_tim', 'sen-att'
        suff, rmk = f'rexp1a_{pre}_whole_tim', 'sen-att'
        self.sub_plt_tim(df_tmp, tag_sa1, suff, rmk)
        self.sub_plt_val(df_tmp, tag_sa1, suff.replace('tim', 'val'), rmk)

        # suff = 'rexp1a_adult_tim'
        suff = 'rexp1a_{}_adult_tim'.format(pre)
        df_tmp = self.sub_dat_sing_set(
            raw_dframe, id_set, 3, tag_multivar, tag_sa1, tag_sa2)
        # self.sub_plt_tim(df_tmp[0], tag_multivar, 'tmp')
        self.sub_plt_tim(df_tmp[1], tag_sa1, suff, 'sen-att')
        self.sub_plt_val(df_tmp[1], tag_sa1,
                         suff.replace('tim', 'val'), 'sen-att')
        return


class RevP_ZB_efficient(RevPlotZ_comparison):
    def sub_plt_tim(self, df_tmp, tag_col, suff, rmk='bin'):
        tag_tim = tag_col[: 13]
        # if rmk == 'bin':
        scat_X = df_tmp[tag_tim[6]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[3]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[11]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[9]].values.astype(DTY_FLT), ]
        if rmk == 'multival':
            scat_Y.extend([df_tmp[tag_tim[8]].values.astype(DTY_FLT),
                           df_tmp[tag_tim[12]].values.astype(DTY_FLT)])

        # if rmk == 'multivar':
        #   ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
        #   ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
        #   ant_app = r'T_{ExtendDist}'
        # elif rmk == 'sen-att':
        ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S,a_i)}'
        ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)}'
        ant_app = r'T_{ApproxDist}'
        ant_app_prev = r'T_{ApproxDist~(prev)}'
        ant_eff = r'T_{EarlyBreak}'  # r'T_{EfficientHD}'

        if rmk == 'bin':
            annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                      '${}$'.format(ant_app_prev)]
        elif rmk == 'multival':
            annotY = ['${}$'.format(ant_eff),
                      r'$T_{ApproxDist}$ bin-val',
                      r'$T_{ApproxDist~(prev)}$ bin-val',
                      r'${}$ multival'.format(ant_X),
                      r'$T_{ApproxDist}$ multival', ]
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty4')
        return

    def sub_plt_val(self, df_tmp, tag_col, suff, rmk='bin'):
        tag_max = tag_col[13: 13 + 13]
        tag_avg = tag_col[13 + 13: 13 + 13 + 5]
        scat_X = df_tmp[tag_max[6]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_max[3]].values.astype(DTY_FLT),
                  df_tmp[tag_max[11]].values.astype(DTY_FLT),
                  df_tmp[tag_max[9]].values.astype(DTY_FLT), ]
        if rmk == 'multival':  # sen-att
            scat_Y.extend([df_tmp[tag_max[8]].values.astype(DTY_FLT),
                           df_tmp[tag_max[12]].values.astype(DTY_FLT)])

        # if rmk == 'multivar':
        #   pass
        # elif rmk == 'sen-att':
        ant_X = r'\mathbf{D}_{\mathbf{a}}(S,a_i)'
        ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
        ant_app = r'ApproxDist'
        ant_app_prev = r'ApproxDist~(prev)'
        ant_eff = 'EarlyBreak'  # r'EfficientHD'

        if rmk == 'bin':
            annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                      '${}$'.format(ant_app_prev)]
        elif rmk == 'multival':
            annotY = ['${}$'.format(ant_eff),
                      r'$ApproxDist$ (bin-val)',
                      r'$ApproxDist~(prev)$ (bin-val)',
                      r'${}$ (multival)'.format(ant_X),
                      r'$ApproxDist$ (multival)']
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty3b')
        return

    def schedule_mspaint(self, raw_dframe, pre='minmax',
                         omitted=True):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4,
            nc_norm=1, nc_sens=0)  # each_set, each_att/gen
        csv_row_1 = unique_column(10 + 31 * 2)

        tag_sa1 = csv_row_1[10: 10 + 31]
        tag_sa2 = csv_row_1[10 + 31: 10 + 31 * 2]  # tag_trn/tst
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])
        df_raw = self.sub_dat_sen_att(df_raw, tag_sa1, tag_sa2)
        # suff_1 = 'rexp1b_whole_tim'
        # suff_2 = 'rexp1b_whole_val'
        suff_1 = f'rexp1b_{pre}_whole_tim'
        suff_2 = f'rexp1b_{pre}_whole_val'
        self.sub_plt_tim(df_raw, tag_sa1, suff_1 + '_bin')
        # self.sub_plt_tim(df_raw, tag_sa1, suff_1 + '_mu', 'multival')
        self.sub_plt_val(df_raw, tag_sa1, suff_2 + '_bin')
        # self.sub_plt_val(df_raw, tag_sa1, suff_2, 'multival')

        if omitted:
            return
        df_tmp = self.sub_dat_sing_set(
            raw_dframe, id_set, 3, [], tag_sa1, tag_sa2)
        _, df_tmp = df_tmp
        # tag_col = [tag_sa1[i] for i in [0, 3, 6, 9]]
        suff_1 = f'rexp1b_{pre}_adult_tim'
        suff_2 = f'rexp1b_{pre}_adult_val'
        self.sub_plt_tim(df_tmp, tag_sa1, suff_1)
        self.sub_plt_val(df_tmp, tag_sa1, suff_2)
        pdb.set_trace()
        return


class RevP_ZC_efficient(RevPlotZ_comparison):
    def sub_plt_tim(self, df_tmp, tag_col, suff):
        tag_tim = tag_col[: 9]
        scat_X = df_tmp[tag_tim[4]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[2]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[8]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[6]].values.astype(DTY_FLT)]

        ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S,a_i)}'
        ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)}'
        ant_eff = r'T_{EfficientHD}'
        ant_app = r'T_{ApproxDist}'
        ant_app_prev = r'T_{ApproxDist~(prev)}'

        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                  '${}$'.format(ant_app_prev)]
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty4')
        return

    def sub_plt_val(self, df_tmp, tag_col, suff):
        tag_max = tag_col[9: 9 + 9]
        tag_avg = tag_col[9 + 9: 9 + 9 + 3]
        scat_X = df_tmp[tag_max[4]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_max[2]].values.astype(DTY_FLT),
                  df_tmp[tag_max[8]].values.astype(DTY_FLT),
                  df_tmp[tag_max[6]].values.astype(DTY_FLT)]

        ant_X = r'\mathbf{D}_{\mathbf{a}}(S,a_i)'
        ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
        ant_eff = r'EfficientHD'
        ant_app = r'ApproxDist'
        ant_app_prev = r'ApproxDist~(prev)'

        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                  '${}$'.format(ant_app_prev)]
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty3b')
        return

    def schedule_mspaint(self, raw_dframe, pre='minmax',
                         omitted=True):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)  # each_att/gen
        csv_row_1 = unique_column(10 + 21 * 2)  # 31*2)
        # not 13+13+5 =31 anymore, it's 9+9+3 =21 this time

        tag_sa1 = csv_row_1[10: 10 + 21]
        tag_sa2 = csv_row_1[10 + 21: 10 + 21 * 2]
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])
        df_raw = self.sub_dat_sen_att(df_raw, tag_sa1, tag_sa2)
        suff_1 = f'rexp1c_{pre}_whole_tim'
        suff_2 = f'rexp1c_{pre}_whole_val'
        self.sub_plt_tim(df_raw, tag_sa1, suff_1 + '_bin')
        self.sub_plt_val(df_raw, tag_sa1, suff_2 + '_bin')

        if omitted:
            return
        df_tmp = self.sub_dat_sing_set(
            raw_dframe, id_set, 3, [], tag_sa1, tag_sa2)
        _, df_tmp = df_tmp
        suff_1 = suff_1.replace('whole', 'adult')
        suff_2 = suff_2.replace('whole', 'adult')
        self.sub_plt_tim(df_tmp, tag_sa1, suff_1)
        self.sub_plt_val(df_tmp, tag_sa1, suff_2)
        return


# ------------------------------
# Rexp2


class RevPlotY_comparison(GraphSetup):
    def sub_hfm_sing_set(self, dframe, no_set, tag_sa1, tag_sa2):
        df_raw = dframe.iloc[
            (no_set - 1) * self._nb_iter: no_set * self._nb_iter]
        # return: [sa#1] Ds, Df, Dh, Ds.avg, Df.avg, Dh.avg,
        #         [sa#2] Ds, Df, Dh, Ds.avg, Df.avg, Dh.avg,
        #         [multivar] tim, Ds,Df,Dh, Ds.avg,Df.avg,Dh.avg, df
        # ans = np.zeros((self._nb_iter, 3, 2 + 6 + 8))
        # ans = np.zeros((3, self._nb_iter, 2 + 6 + 8))
        ans = np.zeros((3, 6, self._nb_iter))  # 2+6+8
        ans_tim = np.zeros((2 * 3, self._nb_iter))
        ans_hfm = np.zeros((3, 8, self._nb_iter))

        if no_set == 1:  # 1 <= no_set <= 5
            for i in range(3):
                curr = tag_sa1[i * 17: (i + 1) * 17]  # curr_tag
                df_tmp = df_raw[curr].astype(DTY_FLT)  # [: 17]
                ans_tim[i * 2 + 0, :] = (
                    df_tmp[curr[0]] + df_tmp[curr[1]]).values
                ans_tim[i * 2 + 1, :] = (
                    df_tmp[curr[0]] + df_tmp[curr[2]]).values
                for j in range(3, 9):  # Ds,Df,Dh, Ds_avg,Df_avg,Dh_avg
                    # ans[i][:, j - 1] = df_tmp[curr[j]].values  # Ds
                    # ans[i][:, 3] = df_tmp[curr[4]].values  # Df
                    # ans[i][:, 4] = df_tmp[curr[5]].values  # Dh
                    # ans[i][:, 5] = df_tmp[curr[6]]
                    # ans[:, i, j - 1] = df_tmp[curr[j]].values
                    ans[i, j - 1 - 2, :] = df_tmp[curr[j]].values
                for j in range(self._nb_iter):  # -2-6
                    ans_hfm[i, 0, j] = fair_degree_v3(
                        ans[i, 0, j], ans[i, 1, j])[0]
                    ans_hfm[i, 1, j] = fair_degree_v3(
                        ans[i, 0, j], ans[i, 2, j])[0]
                    ans_hfm[i, 2, j] = fair_degree_v4(
                        ans[i, 0, j], ans[i, 1, j])[0]
                    ans_hfm[i, 3, j] = fair_degree_v4(
                        ans[i, 0, j], ans[i, 2, j])[0]
                    ans_hfm[i, 4, j] = fair_degree_v3(
                        ans[i, 3, j], ans[i, 4, j])[0]
                    ans_hfm[i, 5, j] = fair_degree_v3(
                        ans[i, 3, j], ans[i, 5, j])[0]
                    ans_hfm[i, 6, j] = fair_degree_v4(
                        ans[i, 3, j], ans[i, 4, j])[0]
                    ans_hfm[i, 7, j] = fair_degree_v3(
                        ans[i, 3, j], ans[i, 5, j])[0]
            return ans, ans_hfm, ans_tim

        for i in range(3):
            cur1 = tag_sa1[i * 17: (i + 1) * 17]
            cur2 = tag_sa2[i * 17: (i + 1) * 17]
            df_t1 = df_raw[cur1].astype(DTY_FLT)
            df_t2 = df_raw[cur2].astype(DTY_FLT)

            ans_tim[i * 2 + 0, :] = (
                df_t1[cur1[0]] + df_t1[cur1[1]] +
                df_t2[cur2[0]] + df_t2[cur2[1]]).values
            ans_tim[i * 2 + 1, :] = (
                df_t1[cur1[0]] + df_t1[cur1[2]] +
                df_t2[cur2[0]] + df_t2[cur2[2]]).values
            for j in range(self._nb_iter):
                for k in range(3, 6):
                    ans[i, k - 3, j] = max(df_t1[cur1[k]].iloc[j],
                                           df_t2[cur2[k]].iloc[j])
                for k in range(6, 9):
                    ans[i, k - 3, j] = (df_t1[cur1[k]].iloc[j] +
                                        df_t2[cur2[k]].iloc[j]) / 2.
                ans_hfm[i, 0, j] = fair_degree_v3(
                    ans[i, 0, j], ans[i, 1, j])[0]
                ans_hfm[i, 1, j] = fair_degree_v3(
                    ans[i, 0, j], ans[i, 2, j])[0]
                ans_hfm[i, 2, j] = fair_degree_v4(
                    ans[i, 0, j], ans[i, 1, j])[0]
                ans_hfm[i, 3, j] = fair_degree_v4(
                    ans[i, 0, j], ans[i, 2, j])[0]
                ans_hfm[i, 4, j] = fair_degree_v3(
                    ans[i, 3, j], ans[i, 4, j])[0]
                ans_hfm[i, 5, j] = fair_degree_v3(
                    ans[i, 3, j], ans[i, 5, j])[0]
                ans_hfm[i, 6, j] = fair_degree_v4(
                    ans[i, 3, j], ans[i, 4, j])[0]
                ans_hfm[i, 7, j] = fair_degree_v4(
                    ans[i, 3, j], ans[i, 5, j])[0]
        return ans, ans_hfm, ans_tim

    def sub_hfm_multivar(self, dframe, tag_sa1, tag_sa2):
        # tag_sa?: DistDirect_bin + DistDirect_nonbin (bi-val + multi-val)
        #          = ( 3+6+4*2 )*3 =17*3 =51
        # where DistDirect_bin= tim 3+ val max 3+ val avg 3+ df (2+2)+ T(df) 4
        #       DistDirect_nonbin (bi-val)= tim 3+ val 3+3+ df 4+ T(df) 4
        #       DistDirect_nonbin (multi-val)= tim 3+ val 6+ df 4+ T(df) 4

        nb_set = 5  # return: (#iter, tim)
        ans = np.zeros((self._nb_iter * nb_set, 3 * 6))
        ans_hfm = np.zeros((self._nb_iter * nb_set, 3 * 8))
        ans_tim = np.zeros((self._nb_iter * nb_set, 3 * 2))
        for no_set in range(1, 1 + nb_set):
            tmp, t_hfm, t_tim = self.sub_hfm_sing_set(
                dframe, no_set, tag_sa1, tag_sa2)  # (3,6|8,#iter), (2*3,#iter)
            start = (no_set - 1) * self._nb_iter
            for i in range(self._nb_iter):
                for j in range(6):
                    ans[start + i, 0 * 6 + j] = tmp[0, j, i]
                    ans[start + i, 1 * 6 + j] = tmp[1, j, i]
                    ans[start + i, 2 * 6 + j] = tmp[2, j, i]
                for j in range(8):
                    ans_hfm[start + i, 0 * 8 + j] = t_hfm[0, j, i]
                    ans_hfm[start + i, 1 * 8 + j] = t_hfm[1, j, i]
                    ans_hfm[start + i, 2 * 8 + j] = t_hfm[2, j, i]
                for j in range(2 * 3):
                    ans_tim[start + i, j] = t_tim[j, i]
        # ans:     Direct_bin, Direct_nonbin, Direct_nonbin where
        #             each is (Ds, Df, Dh, Ds_avg, Df_avg, Dh_avg)
        # ans_hfm:    each is (df_v3,dfh_v3,df_v4,dfh_v4, ..avg)
        # ans_tim:    each is (Ds+Df, Ds+Dh, )
        # pdb.set_trace()
        # self.sub_plt_val_tim(ans[-6:], ans_hfm[-8:], ans_tim[-2:])
        return ans, ans_hfm, ans_tim

    def sub_plt_val_tim(self, ans, ans_hfm, ans_tim, figname,
                        omitted=True):
        scat_X = np.concatenate([
            ans_tim[:, 0], ans_tim[:, 2], ans_tim[:, 4]], axis=0)
        scat_Y = np.concatenate([
            ans_tim[:, 1], ans_tim[:, 3], ans_tim[:, 5]], axis=0)
        annot = [r'$T_{HFM}$', r'$T_{HFM~via~embed}$']  # feature embedding
        annot = [r'$T_{\mathbf{df}}$', r'$T_{\mathbf{df}~via~embed}$',
                 r'$T_{\mathbf{df}~via~embed} = T_{\mathbf{df}}$']
        annot = [r'$T_{\mathbf{df}}$', r'$T_{\mathbf{df}_{emb}}$',
                 r'$T_{\mathbf{df}_{emb}} = T_{\mathbf{df}}$']
        if not omitted:
            scatter_id_chart(scat_X, scat_Y,  # 'test_tim',
                             f'{figname}_tim',
                             annots=annot[:2], identity=True)
        multi_lin_reg_without_distr(scat_X, [scat_Y], [
            ''], annot, f'{figname}_tim_alt', snspec='sty3b')
        return

    def sub_get_acc_fair(self, dframe, tag_acc, tag_far):
        df_raw = dframe[tag_acc + tag_far].astype(DTY_FLT)
        ans_acc = df_raw[tag_acc[7: 14]].values - df_raw[tag_acc[:7]].values
        ans_acc = np.abs(ans_acc)

        # ans_far = np.zeros((ans_acc.shape[0], 2 + 3 + 3))
        ans_far = np.zeros((ans_acc.shape[0], 2 + 3))
        ans_far_alt = np.zeros((ans_acc.shape[0], 3 + 3))
        ans_far[:, 0] = df_raw[tag_acc[-2]].values
        ans_far[:, 1] = df_raw[tag_acc[-1]].values
        # ans_far[:, 2:2 + 3] = df_raw[tag_far[6:9]].values
        # ans_far[:, 5:5 + 3] = df_raw[tag_far[15:18]].values
        ans_far_alt[:, 0:3] = df_raw[tag_far[6:9]].values
        ans_far_alt[:, 3:6] = df_raw[tag_far[15:18]].values
        ans_far_alt = np.nan_to_num(ans_far_alt)  # for each sen-att
        # pdb.set_trace()
        # return ans_acc, np.nan_to_num(ans_far)

        for i in range(self._nb_iter):
            ans_far[i, 2] = ans_far_alt[i, 0]
            ans_far[i, 3] = ans_far_alt[i, 1]
            ans_far[i, 4] = ans_far_alt[i, 2]
        for no_set in range(2, 1 + 5):
            start = (no_set - 1) * self._nb_iter
            for i in range(self._nb_iter):
                ans_far[start + i, 2] = (ans_far_alt[start + i, 0] +
                                         ans_far_alt[start + i, 3]) / 2.
                ans_far[start + i, 3] = (ans_far_alt[start + i, 1] +
                                         ans_far_alt[start + i, 4]) / 2.
                ans_far[start + i, 4] = (ans_far_alt[start + i, 2] +
                                         ans_far_alt[start + i, 5]) / 2.
        ans_acc = np.concatenate([
            # df_raw[tag_acc[:5]].values, ans_acc[:, :5]], axis=1)
            # df_raw[tag_acc[:1] + tag_acc[4:5] + tag_acc[1:4]].values,
            # ans_acc[:, [0, 4, 1, 2, 3]]], axis=1)
            df_raw[tag_acc[:7]].values, ans_acc], axis=1)
        return ans_acc, ans_far, ans_far_alt

    def sub_plt_val_corr(self, ans_acc, ans_far, ans_far_alt, ans_hfm,
                         figname, omitted=True):
        # ans_acc .shape:     (5*#it, 7)
        # ans_far .shape:     (5*#it, 2+3)
        # ans_far_alt .shape: (5*#it, twice 3)
        # ans_hfm .shape:     (5*#it, triple 8)

        pm = {'cmap_name': 'PuBu', 'rotate': 45}  # 15,25,35
        mat_A = ans_acc[:, :5]
        mat_B_1 = np.concatenate([
            ans_far[:, 2:], ans_far[:, 1:2], ans_hfm[:, -8:-4]], axis=1)
        key_A = ['Accuracy', r'$f_1$ score', 'Precision', 'Recall', 'Specificity']
        # mat_A = ans_acc[:, :5]  # :7]
        key_A = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                 r'$f_1$ score', r'g\_mean', r'DiscP'][:5]  # r'$g_mean$'
        key_B_1 = [r'DP', r'EO', r'PQP', r'DR',
                   r'$\mathbf{df}_{prev}$', r'$\mathbf{df}_{prev,emb}$',
                   r'$\mathbf{df}$', r'$\mathbf{df}_{emb}$']
        # analogous_confusion_extended(
        #     mat_A.T, mat_B_1.T, key_A, key_B_1, 'test_3', **pm)
        mat_B_2 = np.concatenate([ans_far[:, 2:], ans_far[
            :, 1:2], ans_hfm[:, -4:]], axis=1)
        key_B_2 = [r'DP', r'EO', r'PQP', r'DR',
                   r'$\mathbf{df}_{prev}^{avg}$', r'$\mathbf{df}_{prev,emb}^{avg}$',
                   r'$\mathbf{df}^{avg}$', r'$\mathbf{df}_{emb}^{avg}$']
        if not omitted:
            analogous_confusion_extended(
                mat_A.T, mat_B_1.T, key_A, key_B_1,
                f'{figname}_oo_max', **pm)
            analogous_confusion_extended(
                mat_A.T, mat_B_2.T, key_A, key_B_2,
                f'{figname}_oo_avg', **pm)
        mat_A = ans_acc[:, 5:]
        key_A = [r'$\Delta$Accuracy', r'$\Delta\,f_1$ score',
                 r'$\Delta$Precision', r'$\Delta$Recall',
                 r'$\Delta$Specificity']
        mat_A = ans_acc[:, 7:7 + 5]
        key_A = [r'$\Delta$Accuracy',
                 r'$\Delta$Precision', r'$\Delta$Recall',
                 r'$\Delta$Specificity', r'$\Delta\,f_1$ score',
                 r'$\Delta$g_mean', r'$\Delta$DiscP'][:5]
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}_oo_delta_max', **pm)
        analogous_confusion_extended(mat_A.T, mat_B_2.T, key_A, key_B_2,
                                     f'{figname}_oo_delta_avg', **pm)
        return

    def sub_plt_again_var_corr(self, ans_acc, ans_far, ans_far_alt, ans_hfm,
                               figname, omitted=True):
        # ans_acc .shape:     (5*#it, 7+7)
        # ans_far .shape:     (5*#it, 2+3)
        # ans_far_alt .shape: (5*#it, twice 3)  for each sen-att
        # ans_hfm .shape:     (5*#it, triple 8)
        pm = {'cmap_name': 'PuBu', 'rotate': 45}

        mat_A = ans_acc[:, 7:7 + 5]
        key_A = [r'$\Delta$Accuracy',
                 r'$\Delta$Precision', r'$\Delta$Recall',
                 r'$\Delta$Specificity', r'$\Delta\,f_1$ score',
                 r'$\Delta$g_mean', r'$\Delta$DiscP'][:5]
        key_B = [r'DP', r'EO', r'PQP', r'DR', ]
        mat_B = np.concatenate([
            ans_far[:, 2:], ans_far[:, 1:2],
            ans_hfm[:, [-8, -6, -2, -7, -5, -1]]], axis=1)
        key_B = key_B + [
            r'$\mathbf{df}_{prev}$', r'$\mathbf{df}$',
            r'$\mathbf{df}^{avg}$', r'$\mathbf{df}_{prev,emb}$',
            r'$\mathbf{df}_{emb}$', r'$\mathbf{df}_{emb}^{avg}$']
        analogous_confusion_extended(mat_A.T, mat_B.T, key_A, key_B,
                                     f'{figname}_delt_oo', **pm)
        mat_A = ans_acc[:, :5]
        key_A = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                 r'$f_1$ score', r'g\_mean', r'DiscP'][:5]  # r'$g_mean$'
        analogous_confusion_extended(mat_A.T, mat_B.T, key_A, key_B,
                                     f'{figname}_oo', **pm)
        return


class RevP_YA_embedding(RevPlotY_comparison):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 4 + 51 * 2 + 16 + 9 * 2)  # 14+136
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])

        tag_sa1 = csv_row_1[14: 14 + 51]
        tag_sa2 = csv_row_1[14 + 51: 14 + 51 * 2]
        tag_acc = csv_row_1[14 + 51 * 2: 14 + 51 * 2 + 16]
        tag_far = csv_row_1[14 + 51 * 2 + 16: 132 + 18]  # 10+140

        # self.sub_hfm_sing_set(df_raw, 1, tag_sa1, tag_sa2)
        # self.sub_hfm_sing_set(df_raw, 2, tag_sa1, tag_sa2)
        # self.sub_hfm_sing_set(df_raw, 5, tag_sa1, tag_sa2)
        ans, ar_hfm, ar_tim = self.sub_hfm_multivar(df_raw, tag_sa1, tag_sa2)
        self.sub_plt_val_tim(ans, ar_hfm, ar_tim, f'rexp2a_{pre}', False)
        # df_tmp = df_raw[tag_acc + tag_far]
        ar_acc, ar_far, ar_alt = self.sub_get_acc_fair(df_raw, tag_acc, tag_far)
        '''
        self.sub_plt_val_corr(ar_acc, ar_far, ar_alt, ar_hfm, f'rexp2a_{pre}', False)
        '''
        self.sub_plt_again_var_corr(ar_acc, ar_far, ar_alt, ar_hfm, f'rexp2a_{pre}', False)
        # pdb.set_trace()
        return


class RevP_YB_embedding(RevPlotY_comparison):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 4 + 136 * 2)  # 136= 51*2+16+9*2
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])

        tag_trn = csv_row_1[14: 14 + 136]
        tag_tst = csv_row_1[14 + 136: 14 + 136 * 2]
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp2b_{pre}')
        # omitted = True
        # if not omitted:
        self.subproc_plot(df_raw, tag_trn, 'trn', f'rexp2b_{pre}')
        return

    def subproc_plot(self, df_raw, tag, flag='tst', figname='rexp2b',
                     omitted=True):
        tag_sa1 = tag[: 51]
        tag_sa2 = tag[51: 51 * 2]
        tag_acc = tag[51 * 2: 51 * 2 + 16]
        tag_far = tag[51 * 2 + 16:]
        ans, ar_hfm, ar_tim = self.sub_hfm_multivar(df_raw, tag_sa1, tag_sa2)
        if not omitted:
            self.sub_plt_val_tim(ans, ar_hfm, ar_tim, f'{figname}_{flag}')
        ar_acc, ar_far, ar_alt = self.sub_get_acc_fair(df_raw, tag_acc, tag_far)
        '''
        self.sub_plt_val_corr(ar_acc, ar_far, ar_alt, ar_hfm,
                          f'{figname}_{flag}', False)
        '''
        self.sub_plt_again_var_corr(ar_acc, ar_far, ar_alt, ar_hfm,
                                    f'{figname}_{flag}', False)
        return


class RevP_YC_embedding(RevPlotY_comparison):
    def sub_plt_val_tim(self, ans, ans_hfm, ans_tim, figname, omitted=True):
        # ans .shape:     (5*#it, 7*3+5*3)  # Ds,Df,Dh (max 7, avg 5)
        # ans_hfm .shape: (5*#it, 7*4+5*4)  # dfv3,df(h)v3,dfv4,df(h)v4
        # ans_tim .shape: (5*#it, 7*2)      # to get df, df(h)

        scat_X = np.concatenate([ans_tim[:, 0], ans_tim[:, 6], ans_tim[:, 10]], axis=0)
        scat_Y = np.concatenate([ans_tim[:, 1], ans_tim[:, 7], ans_tim[:, 11]], axis=0)
        annot = [r'$T_{\mathbf{df}}$ (sec)', r'$T_{\mathbf{df}_{emb}}$ (sec)',
                 r'$T_{\mathbf{df}_{emb}} = T_{\mathbf{df}}$']
        multi_lin_reg_without_distr(scat_X, [scat_Y], [''], annot,
                                    f'{figname}_tim_drt', snspec='sty4')

        scat_X = np.concatenate([ans_tim[:, 2], ans_tim[:, 8], ans_tim[:, 12]], axis=0)
        scat_Y = np.concatenate([ans_tim[:, 3], ans_tim[:, 9], ans_tim[:, 13]], axis=0)
        annot = [r'$T_{\hat{\mathbf{df}}}$ (sec)', r'$T_{\hat{\mathbf{df}}_{emb}}$ (sec)',
                 r'$T_{\hat{\mathbf{df}}_{emb}} = T_{\hat{\mathbf{df}}}$']
        multi_lin_reg_without_distr(scat_X, [scat_Y], [''], annot,
                                    f'{figname}_tim_app', snspec='sty4')

        if not omitted:
            scat_X = np.concatenate([ans_tim[:, 0], ans_tim[:, 6], ans_tim[:, 10]], axis=0)
            scat_Y = [np.concatenate([ans_tim[:, 1], ans_tim[:, 7], ans_tim[:, 11]], axis=0),
                      np.concatenate([ans_tim[:, 2], ans_tim[:, 8], ans_tim[:, 12]], axis=0),
                      np.concatenate([ans_tim[:, 3], ans_tim[:, 9], ans_tim[:, 13]], axis=0)]
            annotY = [
                r'$T_{\mathbf{df}_{emb}}$', r'$T_{\hat{\mathbf{df}}}$',
                r'$T_{\hat{\mathbf{df}}_{emb}}$']
            annot = [r'$T_{\mathbf{df}}$ (sec)',
                     r'$T$ (sec)', r'$T = T_{\mathbf{df}}$']
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot,
                f'{figname}_tim', snspec='sty4')  # 'sty3b'

        res_alt = np.zeros((ans.shape[0] * 3, 7 + 5))
        for i in range(7 + 5):
            gap = 5 * self._nb_iter
            res_alt[: gap, i] = ans[:, i * 3 + 0]
            res_alt[gap: gap * 2, i] = ans[:, i * 3 + 1]
            res_alt[gap * 2: gap * 3, i] = ans[:, i * 3 + 2]
        if not omitted:
            scat_X = res_alt[:, 0]
            scat_Y = [res_alt[:, 3], res_alt[:, 5]]
            annotY = [r'$\mathbf{D}_{\cdot}$ nonbin (bin-val)',
                      r'$\mathbf{D}_{\cdot}$ nonbin (multival)']
            annot = [r'$\mathbf{D}_{\cdot}$ bin',
                     r'$\mathbf{D}_\cdot$ nonbin']
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot,
                f'{figname}_val_drt', snspec='sty3b')
            scat_X = res_alt[:, 1]
            scat_Y = [res_alt[:, 4], res_alt[:, 6]]
            annotY = [r'$\hat{\mathbf{D}}_\cdot$ nonbin (bin-val)',
                      r'$\hat{\mathbf{D}}_\cdot$ nonbin (multival)']
            annot = [r'$\hat{\mathbf{D}}_\cdot$ bin',
                     r'$\hat{\mathbf{D}}_\cdot$ nonbin']
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot,
                f'{figname}_val_app', snspec='sty3b')
        scat_X = res_alt[:, 5]
        scat_Y = [
            res_alt[:, 0], res_alt[:, 1], res_alt[:, 4], res_alt[:, 6]]
        annotY = [r'$\mathbf{D}_\cdot$ bin',
                  r'$\hat{\mathbf{D}}_\cdot$ bin',
                  r'$\hat{\mathbf{D}}_\cdot$ nonbin (bin-val)',
                  r'$\hat{\mathbf{D}}_\cdot$ nonbin (multival)']
        annot = [r'$\mathbf{D}_\cdot$ nonbin (multival)',
                 r'$\mathbf{D}_\cdot$ / $\hat{\mathbf{D}}_\cdot$ (i.e. approx $\mathbf{D}_\cdot$)']
        multi_lin_reg_without_distr(scat_X, scat_Y, annotY, annot,
                                    f'{figname}_val', snspec='sty3b')
        return

    def sub_plt_val_corr(self, ans_acc, ans_far, ans_far_alt, ans_hfm,
                         figname, omitted=True):
        # ans_acc .shape:     (5*#it, 7+7)
        # ans_far .shape:     (5*#it, 2+3)
        # ans_far_alt .shape: (5*#it, twice 3)  for each sen-att
        # ans_hfm .shape:     (5*#it, (max 7+ avg 5) *4) =(25,48)
        pm = {'cmap_name': 'PuBu', 'rotate': 55}  # 45
        res_alt_hfm_max = ans_hfm[:, :7 * 4]  # drt,app*2, drt,app, drt,app
        res_alt_hfm_avg = ans_hfm[:, 7 * 4:]  # drt, drt,app, drt,app
        pm['cmap_name'] = 'Purples'  # 'YlOrBr',BuPu  # 'PiYG',RdPu

        mat_A = ans_acc[:, 7:7 + 5]
        key_A = [r'$\Delta$Accuracy', r'$\Delta$Precision', r'$\Delta$Recall',
                 r'$\Delta$Specificity', r'$\Delta\,f_1$ score', ]
        key_B = [r'DP', r'EO', r'PQP', r'DR']
        mat_B_1 = np.concatenate([ans_far[:, 2:], ans_far[:, 1:2],
                                  res_alt_hfm_max[:, [-8, -6]],
                                  res_alt_hfm_avg[:, [-6, ]],
                                  res_alt_hfm_max[:, [-7, -5]],
                                  res_alt_hfm_avg[:, [-5, ]]], axis=1)
        mat_B_2 = np.concatenate([ans_far[:, 2:], ans_far[:, 1:2],
                                  res_alt_hfm_max[:, [-4, -2]],
                                  res_alt_hfm_avg[:, [-2, ]],
                                  res_alt_hfm_max[:, [-3, -1]],
                                  res_alt_hfm_avg[:, [-1, ]]], axis=1)
        key_B_1 = key_B + [r'$\mathbf{df}_{prev}$', r'$\mathbf{df}$',
                           r'$\mathbf{df}^{avg}$', r'$\mathbf{df}_{prev,emb}$',
                           r'$\mathbf{df}_{emb}$', r'$\mathbf{df}_{emb}^{avg}$']
        key_B_2 = key_B + [r'$\hat{\mathbf{df}}_{prev}$', r'$\hat{\mathbf{df}}$',
                           r'$\hat{\mathbf{df}}^{avg}$',
                           r'$\hat{\mathbf{df}}_{prev,emb}$',
                           r'$\hat{\mathbf{df}}_{emb}$',
                           r'$\hat{\mathbf{df}}_{emb}^{avg}$']
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}_delt_oo_max', **pm)
        analogous_confusion_extended(mat_A.T, mat_B_2.T, key_A, key_B_2,
                                     f'{figname}_delt_oohat_max', **pm)

        mat_A = ans_acc[:, :5]
        key_A = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                 r'$f_1$ score', 'g_mean', 'DiscP'][: 5]
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}_oo_max', **pm)
        analogous_confusion_extended(mat_A.T, mat_B_2.T, key_A, key_B_2,
                                     f'{figname}_oohat_max', **pm)
        return

    def sub_plt_again_var_corr(self, ans_acc, ans_far, ans_far_alt, ans_hfm,
                               ans_hfm_emb, figname, omitted=False):
        # ans_hfm .shape:     (5*#it, 19)  19=3+2+2+3*4
        # ans_hfm_emb .shape: (5*#it, 19)
        pm = {'cmap_name': 'PuBu', 'rotate': 45}

        mat_A = ans_acc[:, 7:7 + 5]
        key_A = [r'$\Delta$Accuracy', r'$\Delta$Precision', r'$\Delta$Recall',
                 r'$\Delta$Specificity', r'$\Delta\,f_1$ score', ]
        key_B = [r'DP', r'EO', r'PQP', r'DR']
        mat_B_1 = np.concatenate([ans_far[:, 2:], ans_far[:, 1:2],
                                  ans_hfm[:, -6:-3],
                                  ans_hfm_emb[:, -6:-3]], axis=1)
        mat_B_2 = np.concatenate([ans_far[:, 2:], ans_far[:, 1:2],
                                  ans_hfm[:, -3:], ans_hfm_emb[:, -3:]], axis=1)
        key_B_1 = key_B + [r'$\mathbf{df}_{prev}$', r'$\mathbf{df}$',
                           r'$\mathbf{df}^{avg}$', r'$\mathbf{df}_{prev,emb}$',
                           r'$\mathbf{df}_{emb}$',
                           r'$\mathbf{df}_{emb}^{avg}$']
        key_B_2 = key_B + [r'$\hat{\mathbf{df}}_{prev}$', r'$\hat{\mathbf{df}}$',
                           r'$\hat{\mathbf{df}}^{avg}$',
                           r'$\hat{\mathbf{df}}_{prev,emb}$',
                           r'$\hat{\mathbf{df}}_{emb}$',
                           r'$\hat{\mathbf{df}}_{emb}^{avg}$']
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}ag_delt_oo_max', **pm)
        analogous_confusion_extended(mat_A.T, mat_B_2.T, key_A, key_B_2,
                                     f'{figname}ag_delt_oohat_max', **pm)

        mat_A = ans_acc[:, :5]
        key_A = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                 r'$f_1$ score', 'g_mean', 'DiscP'][: 5]
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}ag_oo_max', **pm)
        analogous_confusion_extended(mat_A.T, mat_B_2.T, key_A, key_B_2,
                                     f'{figname}ag_oohat_max', **pm)
        return

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 4 + 328 * 2)
        # 4+?*2 =660  # ?= ??*2+16+9*2 =328  # ??= 21*5+21*2 =147

        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])
        tag_trn = csv_row_1[14: 14 + 328]
        tag_tst = csv_row_1[14 + 328: 14 + 328 * 2]
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp2c_{pre}')
        '''
        self.subproc_plot(df_raw, tag_trn, 'trn', f'rexp2c_{pre}')
        '''
        return

    def subproc_plot(self, df_raw, tag, flag='tst', figname=''):
        # hfm: DistDirect_bin 21+ ApproxDist_bin 21+ ApproxDist_alter 21+
        #      DistDirect_nonbin (bi-val) 21+ DistApprox_nonbin 21+
        #      DistDirect_nonbin (multi-val) 21+ DistApprox_nonbin (multi-val) 21
        tag_sa1 = tag[: 21 * 7]
        tag_sa2 = tag[21 * 7: 21 * 7 * 2]
        tag_acc = tag[21 * 14: 147 * 2 + 16]  # not 27*14
        tag_far = tag[147 * 2 + 16: 294 + 16 + 9 * 2]
        _, ar_hfm, _ = self.sub_hfm_multivar(
            df_raw, tag_sa1, tag_sa2)  # ans,,ar_tim
        '''
        self.sub_plt_val_tim(ans, ar_hfm, ar_tim, f'{figname}_{flag}')
        '''
        ar_acc, ar_far, ar_alt = self.sub_get_acc_fair(df_raw, tag_acc, tag_far)
        self.sub_plt_val_corr(ar_acc, ar_far, ar_alt, ar_hfm, f'{figname}_{flag}')
        return

    def sub_hfm_multivar(self, dframe, tag_sa1, tag_sa2, num=7):
        nb_set = 5
        ans = np.zeros((self._nb_iter * nb_set, (num + num - 2) * 3))
        ans_hfm = np.zeros((self._nb_iter * nb_set, (num + num - 2) * 4))
        ans_tim = np.zeros((self._nb_iter * nb_set, num * 2))
        for no_set in range(1, 1 + nb_set):
            tmp, t_hfm, t_tim = self.sub_hfm_sing_set(dframe, no_set, tag_sa1, tag_sa2, num)
            start = (no_set - 1) * self._nb_iter

            for k in range(self._nb_iter):
                for j in range(3):
                    for i in range(num + num - 2):
                        ans[start + k, i * 3 + j] = tmp[j, i, k]
                for j in range(4):
                    for i in range(num + num - 2):
                        ans_hfm[start + k, i * 4 + j] = t_hfm[j, i, k]
                for j in range(2 * num):
                    ans_tim[start + k, j] = t_tim[j, k]

        return ans, ans_hfm, ans_tim

    def sub_hfm_sing_set(self, dframe, no_set, tag_sa1, tag_sa2, num=7):
        df_raw = dframe.iloc[(no_set - 1) * self._nb_iter: no_set * self._nb_iter]

        # res_alt_max = np.zeros((num, 3, self._nb_iter))
        # res_alt_avg = np.zeros((num - 2, 3, self._nb_iter))
        # # res_hfm_alt_max = np.zeros((num, 4, self._nb_iter))
        # # res_hfm_alt_avg = np.zeros((num - 2, 4, self._nb_iter))
        res_hfm_alt = np.zeros((4, num + num - 2, self._nb_iter))
        res_alt = np.zeros((3, num + num - 2, self._nb_iter))
        res_tim_alt = np.zeros((2 * num, self._nb_iter))

        if no_set == 1:  # 1 <= no_set <= 5
            for i in range(num):
                curr = tag_sa1[i * 21: (i + 1) * 21]  # 9+12=21, not 17
                df_tmp = df_raw[curr[: 9]].astype(DTY_FLT)

                res_tim_alt[i * 2 + 0, :] = (df_tmp[curr[0]].values + df_tmp[curr[1]].values)
                res_tim_alt[i * 2 + 1, :] = (df_tmp[curr[0]].values + df_tmp[curr[2]].values)
                for j in range(3, 6):  # Ds, Df, Dh
                    res_alt[j - 3, i, :] = df_tmp[curr[j]].values
                if not (1 <= i <= 2):
                    cur_row = i if i < 1 else i - 2
                    for j in range(6, 9):  # Ds_avg, Df_avg, Dh_avg
                        res_alt[j - 6, cur_row + num, :] = df_tmp[
                            curr[j]].values
                    del cur_row

            for k in range(self._nb_iter):
                for i in range(num + num - 2):
                    res_hfm_alt[0, i, k] = fair_degree_v3(
                        res_alt[0, i, k], res_alt[1, i, k])[0]
                    res_hfm_alt[1, i, k] = fair_degree_v3(
                        res_alt[0, i, k], res_alt[2, i, k])[0]
                    res_hfm_alt[2, i, k] = fair_degree_v4(
                        res_alt[0, i, k], res_alt[1, i, k])[0]
                    res_hfm_alt[3, i, k] = fair_degree_v4(
                        res_alt[0, i, k], res_alt[2, i, k])[0]
            return res_alt, res_hfm_alt, res_tim_alt

        for i in range(num):
            cur1 = tag_sa1[i * 21: (i + 1) * 21][: 9]
            cur2 = tag_sa2[i * 21: (i + 1) * 21][: 9]
            df_t1 = df_raw[cur1].astype(DTY_FLT)
            df_t2 = df_raw[cur2].astype(DTY_FLT)

            res_tim_alt[i * 2 + 0, :] = (
                df_t1[cur1[0]].values + df_t1[cur1[1]].values +
                df_t2[cur2[0]].values + df_t2[cur2[1]].values)
            res_tim_alt[i * 2 + 1, :] = (
                df_t1[cur1[0]].values + df_t1[cur1[2]].values +
                df_t2[cur2[0]].values + df_t2[cur2[2]].values)

            for k in range(self._nb_iter):
                for j in range(3, 6):  # Ds, Df, Dh
                    res_alt[j - 3, i, k] = max(
                        df_t1[cur1[j]].iloc[k], df_t2[cur2[j]].iloc[k])
                for j in range(6, 9):  # Ds_avg, Df_avg, Dh_avg
                    if 1 <= i <= 2:
                        break  # continue
                    cur_row = i if i < 1 else i - 2
                    res_alt[j - 6, cur_row + num, k] = (df_t1[
                        cur1[j]].iloc[k] + df_t2[cur2[j]].iloc[k]) / 2.

        for k in range(self._nb_iter):
            for i in range(num + num - 2):
                res_hfm_alt[0, i, k] = fair_degree_v3(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
                res_hfm_alt[1, i, k] = fair_degree_v3(
                    res_alt[0, i, k], res_alt[2, i, k])[0]
                res_hfm_alt[2, i, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
                res_hfm_alt[3, i, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[2, i, k])[0]
        return res_alt, res_hfm_alt, res_tim_alt

    def sub_hfm_again_multivar(self, dframe, tag_sa1, tag_sa2, num=7):
        # ans = np.zeros((5 * self._nb_iter, (num + num - 2) * 3))
        # ans_hfm = np.zeros((5 * self._nb_iter, (num + num - 2) * 4))

        num_prim = 3 + 2 * 2 + 3 * 4  # 3*num-2 =19
        ans_hfm = np.zeros((5 * self._nb_iter, num_prim))
        ans_hfm_emb = np.zeros((5 * self._nb_iter, num_prim))
        for no_set in range(1, 1 + 5):
            df, df_h = self.sub_hfm_again_sing_set(dframe, no_set, tag_sa1, tag_sa2, num)
            start, end = (no_set - 1) * self._nb_iter, no_set * self._nb_iter
            for j in range(num_prim):
                ans_hfm[start: end, j] = df[j]
                ans_hfm_emb[start: end, j] = df_h[j]
        return ans_hfm, ans_hfm_emb

    def sub_hfm_again_sing_set(self, dframe, no_set, tag_sa1, tag_sa2, num=7):
        df_raw = dframe.iloc[(no_set - 1) * self._nb_iter: no_set * self._nb_iter]
        # res_alt = np.zeros((3, num + num - 2, self._nb_iter))
        # res_hfm_alt = np.zeros((4, num + num - 2, self._nb_iter))
        # res_hfm_alt = np.zeros((3, num + num - 2, self._nb_iter))
        # ans_df_sa1 = np.zeros((3, num + num - 2, self._nb_iter))
        # ans_df_sa2 = np.zeros((3, num + num - 2, self._nb_iter))

        num_prim = 3 + 2 * 2 + 3 * 4  # 3+4+12 =19
        df = np.zeros((num_prim, self._nb_iter))
        df_h = np.zeros((num_prim, self._nb_iter))

        if no_set == 1:
            for i in range(num):
                curr = tag_sa1[i * 21: (i + 1) * 21][9: 9 + 6]
                df_tmp = df_raw[curr].astype(DTY_FLT)

                if i < 1:
                    start = i * 3
                elif 1 <= i <= 2:
                    start = 3 + (i - 1) * 2
                else:
                    start = 3 + 2 * 2 + (i - 3) * 3
                if not (1 <= i <= 2):
                    for j in range(3):
                        df[start + j, :] = df_tmp[curr[j]].values
                        df_h[start + j, :] = df_tmp[curr[j + 3]].values
                else:
                    for j in range(2):
                        df[start + j, :] = df_tmp[curr[j]].values
                        df_h[start + j, :] = df_tmp[curr[j + 3]].values

                # df_prev[0, i, :] = df_tmp[curr[0]].values
                # df [0, i, :] = df_tmp[curr[1]].values
                # df_h[0, i, :] = df_tmp[curr[]]
                # df_prev[-1, i, :] = df_tmp[curr[0]].values
                # df[0, i, :] = df_tmp[]

            # res_hfm_alt = ans_df_sa1.copy()
            # res_hfm_alt = (ans_df_sa1 + ans_df_sa2) / 2.
            # return ans_df_sa1, ans_df_sa2, res_hfm_alt
            return df, df_h

        for i in range(num):
            cur1 = tag_sa1[i * 21: (i + 1) * 21][9: 9 + 6]
            cur2 = tag_sa2[i * 21: (i + 1) * 21][9: 9 + 6]
            df_t1 = df_raw[cur1].astype(DTY_FLT)
            df_t2 = df_raw[cur2].astype(DTY_FLT)

            if i < 1:
                start = i * 3
            elif 1 <= i <= 2:
                start = 3 + (i - 1) * 2
            else:
                start = 3 + 2 * 2 + (i - 3) * 3

            if not (1 <= i <= 2):
                for j in range(3):
                    df[start + j, :] = (
                        df_t1[cur1[j]] + df_t2[cur2[j]]).values / 2.
                    df_h[start + j, :] = (df_t1[
                        cur1[j + 3]] + df_t2[cur2[j + 3]]).values / 2.
            else:
                for j in range(2):
                    df[start + j, :] = (
                        df_t1[cur1[j]] + df_t2[cur2[j]]).values / 2.
                    df_h[start + j, :] = (df_t1[
                        cur1[j + 3]] + df_t2[cur2[j + 3]]).values / 2.
        return df, df_h


# ------------------------------
# Rexp3


class RevPlotX_extendSP(GraphSetup):
    def sub_hfm_multivar(self, dframe, tag_sa1, tag_sa2):
        nb_set, num = 5, 7
        ans_hfm = np.zeros((5 * self._nb_iter, num + num + num - 2))
        ans_tim = np.zeros((5 * self._nb_iter, num))
        ans = np.zeros((5 * self._nb_iter, num + num - 2, 2))

        # ans = np.zeros((5*self._nb_iter, ))
        # ans_tim = np.zeros((5*self._nb_iter,))
        for no_set in range(1, 1 + nb_set):
            tmp, t_hfm, t_tim = self.sub_hfm_sing_set(dframe, no_set, tag_sa1, tag_sa2)
            start = (no_set - 1) * self._nb_iter
            for k in range(self._nb_iter):
                for i in range(num + num + num - 2):
                    ans_hfm[start + k, i] = t_hfm[i, k]
                for i in range(num):
                    ans_tim[start + k, i] = t_tim[i, k]

                for j in range(2):
                    for i in range(num + num - 2):
                        ans[start + k, i, j] = tmp[j, i, k]
            # pdb.set_trace()
        # return ans_hfm, ans_tim
        return ans, ans_hfm, ans_tim

    def sub_hfm_sing_set(self, dframe, no_set, tag_sa1, tag_sa2):
        df_raw = dframe.iloc[(
            no_set - 1) * self._nb_iter: no_set * self._nb_iter].astype(DTY_FLT)
        num = 7  # 84=7* 12 [t_Ds,t_Df, Ds,Df, Ds_avg,Df_avg, df 3+3]
        res_alt = np.zeros((2, num + num - 2, self._nb_iter))      # Ds,Df | Ds/f_avg
        res_hfm_alt = np.zeros((num + num + num - 2, self._nb_iter))
        res_tim_alt = np.zeros((num, self._nb_iter))

        if no_set == 1:  # 1 <= no_set <= 5
            for i in range(num):
                curr = tag_sa1[i * 12: (i + 1) * 12][: 6]
                df_tmp = df_raw[curr].astype(DTY_FLT)

                res_tim_alt[i, :] = (
                    df_tmp[curr[0]].values + df_tmp[curr[1]].values)
                for j in range(2, 4):
                    res_alt[j - 2, i, :] = df_tmp[curr[j]].values
                if not (1 <= i <= 2):
                    cur_row = i if i < 1 else i - 2
                    for j in range(4, 6):
                        res_alt[j - 4, cur_row + num, :] = df_tmp[
                            curr[j]].values

            for k in range(self._nb_iter):
                for i in range(num):
                    res_hfm_alt[i, k] = fair_degree_v3(
                        res_alt[0, i, k], res_alt[1, i, k])[0]
                    res_hfm_alt[i + num, k] = fair_degree_v4(
                        res_alt[0, i, k], res_alt[1, i, k])[0]
                for i in range(num, num + num - 2):
                    res_hfm_alt[i + num, k] = fair_degree_v4(
                        res_alt[0, i, k], res_alt[1, i, k])[0]
                # df_prev, df, df_avg
            return res_alt, res_hfm_alt, res_tim_alt

        for i in range(num):
            cur1 = tag_sa1[i * 12:(i + 1) * 12][: 6]
            cur2 = tag_sa2[i * 12: (i + 1) * 12][: 6]
            df_t1 = df_raw[cur1].astype(DTY_FLT)
            df_t2 = df_raw[cur2].astype(DTY_FLT)

            res_tim_alt[i, :] = (
                df_t1[cur1[0]].values + df_t1[cur1[1]].values +
                df_t2[cur2[0]].values + df_t2[cur2[1]].values)
            for k in range(self._nb_iter):
                for j in range(2, 4):
                    res_alt[j - 2, i, k] = max(
                        df_t1[cur1[j]].iloc[k], df_t2[cur2[j]].iloc[k])
                for j in range(4, 6):
                    if 1 <= i <= 2:
                        break
                    cur_row = i if i < 1 else i - 2
                    res_alt[j - 4, cur_row + num, k] = (df_t1[
                        cur1[j]].iloc[k] + df_t2[cur2[j]].iloc[k]) / 2.

        for i in range(self._nb_iter):
            for i in range(num):
                res_hfm_alt[i, k] = fair_degree_v3(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
                res_hfm_alt[i + num, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
            for i in range(num, num + num - 2):
                res_hfm_alt[i + num, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[1, i, k])[0]

        return res_alt, res_hfm_alt, res_tim_alt

    def sub_plt_val_tim(self, ans, ans_hfm, ans_tim, figname):
        # ans_hfm .shape: (5*#it, 7+7+5)  # df_prev, df, df_avg
        # ans_tim .shape: (5*#it, 7)  # how long to get Ds & Df (incl. Df_avg)
        # ans .shape:     (5*#it, 7+5, 2) # D_cdot_max 7+ D_cdot_avg 5 | Ds/Df
        scat_X = np.concatenate([ans_tim[:, 0], ans_tim[:, 3], ans_tim[:, 5]], axis=0)
        scat_Y = np.concatenate([ans_tim[:, 1], ans_tim[:, 4], ans_tim[:, 6]], axis=0)
        annot = [r'$T_{\mathbf{df}}$ (sec)', r'$T_{\hat{\mathbf{df}}}$ (sec)',
                 r'$T_{\hat{\mathbf{df}}} = T_{\mathbf{df}}$']
        multi_lin_reg_without_distr(scat_X, [scat_Y], [''], annot,
                                    f'{figname}_tim', snspec='sty4')

        res_alt = ans.transpose(1, 0, 2).reshape(ans.shape[1], -1)  # .shape=(12,50)
        scat_X = res_alt[5]
        scat_Y = [res_alt[0], res_alt[3], res_alt[1], res_alt[4], res_alt[6]]
        annotY = [r'$\mathbf{D}_\cdot(S_1,\bar{S}_1)$ bin-val',
                  r'$\mathbf{D}_{\cdot,\mathbf{a}}(S)$ bin-val',
                  r'$\hat{\mathbf{D}}_\cdot(S_1,\bar{S}_1})$ bin-val',
                  r'$\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)$ bin-val',
                  r'$\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)$ multi-val']
        annot = [r'$\mathbf{D}_{\cdot,\mathbf{a}}(S)$ multi-val', '']
        multi_lin_reg_without_distr(scat_X, scat_Y, annotY, annot,
                                    f'{figname}_val', snspec='sty3b')
        # scat_X = ans_hfm[]
        # pdb.set_trace()
        return

    def sub_get_acc_fair(self, dframe, tag_acc, tag_far_sa1, tag_far_sa2):
        df_raw = dframe[tag_acc + tag_far_sa1 + tag_far_sa2].astype(DTY_FLT)
        ans_acc = df_raw[tag_acc[7: 14]].values - df_raw[tag_acc[:7]].values
        ans_acc = np.abs(ans_acc)

        ans_far = np.zeros((ans_acc.shape[0], 3 + 2 + 6 + 6))
        ans_far_alt = np.zeros((ans_acc.shape[0], 2, 3 + 6 + 6))
        ans_far[:, 3] = df_raw[tag_acc[-1]].values  # DR
        ans_far[:, 4] = df_raw[tag_acc[-2]].values  # loss

        a_k = self._nb_iter
        df_tmp = df_raw.iloc[: a_k]
        for a_i, tag_far in enumerate([tag_far_sa1]):
            ans_far_alt[:a_k, a_i, 0:3] = df_tmp[tag_far[6:9]].values
            ans_far_alt[:a_k, a_i, 3:9] = df_tmp[tag_far[9:15]].values
            ans_far_alt[:a_k, a_i, 9:15] = df_tmp[tag_far[15:21]].values
        df_tmp = df_raw.iloc[a_k:]
        for a_i, tag_far in enumerate([tag_far_sa1, tag_far_sa2]):
            ans_far_alt[a_k:, a_i, 0:3] = df_tmp[tag_far[6:9]].values
            ans_far_alt[a_k:, a_i, 3:9] = df_tmp[tag_far[9:15]].values
            ans_far_alt[a_k:, a_i, 9:15] = df_tmp[tag_far[15:21]].values
        del a_k, df_tmp   # np.isnan(ans_far_alt).any()
        if (ans_far_alt != ans_far_alt).any():
            ans_far_alt = np.nan_to_num(ans_far_alt)

        for k in range(self._nb_iter):
            for j in range(3):
                ans_far[k, j] = ans_far_alt[k, 0, j]
            for j in range(3, 9):
                ans_far[k, j + 2] = ans_far_alt[k, 0, j]
            for j in range(9, 15):
                ans_far[k, j + 2] = ans_far_alt[k, 0, j]
        for k in range(self._nb_iter, df_raw.shape[0]):  # #nb_iter*5):
            for j in range(3):
                ans_far[k, j] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k][1][j]) / 2.
            # extGF
            for j in range(3, 9):
                ans_far[k, j + 2] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k, 1, j]) / 2.
            for j in range(9, 15):
                ans_far[k, j + 2] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k, 1, j]) / 2.
            # alternative extGF (not original definition)

        ans_acc = np.concatenate([
            df_raw[tag_acc[:7]].values, ans_acc], axis=1)
        # pdb.set_trace()
        return ans_acc, ans_far, ans_far_alt

    def sub_plt_val_corr(self, ans_acc, ans_far, ans_far_alt, ans_hfm,
                         figname, verbose=False):
        # ans_acc .shape:     (5*#it, 7+7)
        # ans_far .shape:     (5*#it, 17= regular GF 3+ DR 2+ extGF 6+ extGFalter 6)
        # ans_far_alt .shape: (5*#it, 25, 2 sen-att, 15= 3+6+6 =15)
        # ans_hfm .shape:     (5*#it, 19= df_prev 7+ df 7+ df_avg 5)
        pm = {'cmap_name': 'OrRd', 'rotate': 45, 'figsize': 'extra'}  # PuBu
        # pdb.set_trace()

        mat_A = ans_acc[:, 7:7 + 5]
        key_A = [r'$\Delta$Accuracy', r'$\Delta\,f_1$ score',
                 r'$\Delta$Precision',
                 r'$\Delta$Recall', r'$\Delta$Specificity']
        key_B = [r'DP', r'EO', r'PQP', r'DR']

        mat_B_1 = np.concatenate([ans_far[:, :4],
                                  ans_hfm[:, [0, 12, 17, 1, 13, 18]],
                                  ans_far[:, 5:7]], axis=1)
        key_B_1 = key_B + [r'$\mathbf{df}_{prev}$', r'$\mathbf{df}$',
                           r'$\mathbf{df}^{avg}$', r'$\hat{\mathbf{df}}_{prev}$',
                           r'$\hat{\mathbf{df}}$', r'$\hat{\mathbf{df}}^{avg}$',
                           r'SP$^{max}$', r'SP$^{avg}$']
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}_delt_oo_mydfext', **pm)

        if verbose:
            mat_B_2 = np.concatenate([ans_far[:, :4],
                                      ans_hfm[:, [0, 12, 17, ]],
                                      ans_far[:, 5:11]], axis=1)
            key_B_2 = key_B + ['$\mathbf{df}_{prev}$', r'$\mathbf{df}$',
                               r'$\mathbf{df}^{avg}$',
                               r'SP$^{max}$', r'SP$^{avg}$',
                               r'extEO$^{max}$', r'extEO$^{avg}$',
                               r'extPQP$^{max}$', r'extPQP$^{avg}$']
            analogous_confusion_extended(
                mat_A.T, mat_B_2.T, key_A, key_B_2,
                f'{figname}_delt_oo_mydf', **pm)

        mat_A = ans_acc[:, :5]
        key_A = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                 r'$f_1$ score', r'g\_mean', r'DiscP'][:5]  # r'$g_mean$'
        analogous_confusion_extended(mat_A.T, mat_B_1.T, key_A, key_B_1,
                                     f'{figname}_oo_mydfext', **pm)
        if verbose:
            analogous_confusion_extended(
                mat_A.T, mat_B_2.T, key_A, key_B_2,
                f'{figname}_oo_mydf', **pm)
        col_tmp = unique_column(5 + 12)
        df_tmp = np.concatenate([mat_A, mat_B_1], axis=1)
        df_tmp = pd.DataFrame(df_tmp, columns=col_tmp)
        if verbose:
            lineplot_with_uncertainty(
                df_tmp, col_tmp[0], 'Fairness', col_tmp[4 + 5:],
                key_B_1[4:], figname=f'{figname}_delt_pc2mat_b4',
                alpha_loc='b4', alpha_rev=True, annotY=' error rate',
                cmap_name='coolwarm_r', alpha_clarity=.15)
            lineplot_with_uncertainty(
                df_tmp, col_tmp[4], 'Fairness', col_tmp[4 + 5:],
                key_B_1[4:], figname=f'{figname}_delt_pc3mat_b4',
                alpha_loc='b4', alpha_rev=True,
                annotY=r'($1-f_1\,$score)',
                cmap_name='coolwarm_r', alpha_clarity=.15)

        lineplot_with_uncertainty(
            df_tmp, col_tmp[0], 'Fairness',
            col_tmp[9:12] + col_tmp[15:17],
            key_B_1[4:7] + key_B_1[10:12],
            figname=f'{figname}_delt_pc1a',
            alpha_loc='b4', alpha_rev=True, annotY=' error rate',
            cmap_name='coolwarm_r', alpha_clarity=.15)
        lineplot_with_uncertainty(
            df_tmp, col_tmp[0], 'Fairness',
            col_tmp[12:15] + col_tmp[15:17],
            key_B_1[7:10] + key_B_1[10:12],
            figname=f'{figname}_delt_pc1b',
            alpha_loc='b4', alpha_rev=True, annotY=' error rate',
            cmap_name='coolwarm_r', alpha_clarity=.15)
        return


class RevP_XC_statsParity(RevPlotX_extendSP):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(11 + 4 + 226 * 2)  # 11+456
        # ?= 12*7 *2+ (7*2+2)+ (9+12)*2  =84*2+16+21*2 =226
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[11:])

        tag_trn = csv_row_1[15: 15 + 226]
        tag_tst = csv_row_1[15 + 226: 15 + 226 * 2]
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp3c_{pre}')
        return

    def subproc_plot(self, df_raw, tag, flag='tst', figname='rexp3c'):
        tag_sa1 = tag[: 84]
        tag_sa2 = tag[84: 84 * 2]
        tag_acc = tag[84 * 2: 84 * 2 + 16]
        tag_far_sa1 = tag[184:184 + 21]  # tag[184: 184 + 21 * 2]
        tag_far_sa2 = tag[184 + 21:]  # tag[184 + 21 * 2:]

        ans, ar_hfm, ar_tim = self.sub_hfm_multivar(df_raw, tag_sa1,
                                                    tag_sa2)
        self.sub_plt_val_tim(ans, ar_hfm, ar_tim, f'{figname}_{flag}')
        ar_acc, ar_far, ar_alt = self.sub_get_acc_fair(
            df_raw, tag_acc, tag_far_sa1, tag_far_sa2)
        self.sub_plt_val_corr(
            ar_acc, ar_far, ar_alt, ar_hfm, f'{figname}_{flag}')
        # pdb.set_trace()
        return


class RevP_XD_statsParity(RevP_XC_statsParity):
    # pass

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(11 + 4 + 226 * 2)  # 11+456
        # ?= 12*7 *2+ (7*2+2)+ (9+12)*2  =84*2+16+21*2 =226
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[11:])

        tag_trn = csv_row_1[15: 15 + 226]
        tag_tst = csv_row_1[15 + 226: 15 + 226 * 2]
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp3d_{pre}')
        return


class RevP_XE_statsParity(RevPlotX_extendSP):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, per_att, per_gen = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=4)  # ,each_set,
        csv_row_1 = unique_column(11 + 4 + 226 * 2)
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[11:])
        tag_trn = csv_row_1[15: 15 + 226]
        tag_tst = csv_row_1[15 + 226: 15 + 226 * 2]
        # pdb.set_trace()
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp3e_{pre}',
                          per_gen, per_att)  # nb_set, id_set)
        return

    def subproc_plot(self, df_raw, tag, flag='tst', figname='rexp3e',
                     # nb_set=5, id_set=list()):
                     per_gen=15, per_att=20):
        tag_sa1 = tag[: 84]
        tag_sa2 = tag[84: 84 * 2]
        tag_acc = tag[84 * 2: 84 * 2 + 16]
        tag_far_sa1 = tag[184:184 + 21]  # tag[184: 184 + 21 * 2]
        tag_far_sa2 = tag[184 + 21:]  # tag[184 + 21 * 2:]

        _, ar_hfm, ar_tim = self.sub_hfm_multivar(
            df_raw, tag_sa1, tag_sa2, per_gen, per_att)  # ans,
        ar_acc, ar_far, ar_alt = self.sub_get_acc_fair(
            df_raw, tag_acc, tag_far_sa1, tag_far_sa2, per_gen, per_att)
        self.sub_plt_val_corr(
            ar_acc, ar_far, ar_alt, ar_hfm, f'{figname}_{flag}')
        return

    def sub_hfm_multivar(self, dframe, tag_sa1, tag_sa2,
                         per_gen=15, per_att=20):
        nb_set, num = 5, 7
        nc_norm = per_gen // self._nb_iter
        nc_sens = per_att // self._nb_iter
        each_set_prim = per_att * 2 + per_gen
        overall_rows = dframe.shape[0]
        ans_hfm = np.zeros((overall_rows, num + num + num - 2))
        ans_tim = np.zeros((overall_rows, num))
        ans = np.zeros((overall_rows, num + num - 2, 2))

        for no_set in range(1, 1 + nb_set):
            if no_set == 1:
                df_raw = dframe.iloc[: per_gen + per_att]
                curr_start, curr_end = 0, per_gen + per_att
            else:
                curr_start = (no_set - 1) * each_set_prim - per_att
                curr_end = no_set * each_set_prim - per_att
                df_raw = dframe.iloc[curr_start: curr_end]
            tmp, t_hfm, t_tim = self.sub_hfm_sing_set(df_raw, no_set, tag_sa1, tag_sa2)
            present_rows = df_raw.shape[0]  # present_start =
            for k in range(present_rows):
                for i in range(num + num + num - 2):
                    ans_hfm[curr_start + k, i] = t_hfm[i, k]
                for i in range(num):
                    ans_tim[curr_start + k, i] = t_tim[i, k]
                for j in range(2):
                    for i in range(num + num - 2):
                        ans[curr_start + k, i, j] = tmp[j, i, k]
        return ans, ans_hfm, ans_tim

    def sub_hfm_sing_set(self, df_raw, no_set, tag_sa1, tag_sa2):
        num, present_rows = 7, df_raw.shape[0]  # df_raw = dframe.iloc
        res_alt = np.zeros((2, num + num - 2, present_rows))  # Ds,Df 0-dim
        res_hfm_alt = np.zeros((num + num + num - 2, present_rows))
        res_tim_alt = np.zeros((num, present_rows))

        if no_set == 1:  # 1 <= no_set <= 5
            for i in range(num):
                curr = tag_sa1[i * 12: (i + 1) * 12][: 6]
                df_tmp = df_raw[curr].astype(DTY_FLT)
                # df_tmp = df_tmp.reset_index(drop=True)

                res_tim_alt[i, :] = (
                    df_tmp[curr[0]].values + df_tmp[curr[1]].values)
                for j in range(2, 4):
                    res_alt[j - 2, i, :] = df_tmp[curr[j]].values
                if not (1 <= i <= 2):
                    cur_row = i if i < 1 else i - 2
                    for j in range(4, 6):
                        res_alt[j - 4, cur_row + num, :] = df_tmp[
                            curr[j]].values

        else:
            for i in range(num):
                cur1 = tag_sa1[i * 12: (i + 1) * 12][: 6]
                cur2 = tag_sa2[i * 12: (i + 1) * 12][: 6]
                df_t1 = df_raw[cur1].astype(DTY_FLT).reset_index(
                    drop=True)
                df_t2 = df_raw[cur2].astype(DTY_FLT).reset_index(
                    drop=True)

                res_tim_alt[i, :] = (
                    df_t1[cur1[0]].values + df_t1[cur1[1]].values +
                    df_t2[cur2[0]].values + df_t2[cur2[1]].values)
                for k in range(present_rows):
                    for j in range(2, 4):
                        res_alt[j - 2, i, k] = max(df_t1[cur1[j]].iloc[k],
                                                   df_t2[cur2[j]].iloc[k])
                    for j in range(4, 6):
                        if 1 <= i <= 2:
                            break
                        cur_row = i if i < 1 else i - 2
                        res_alt[j - 4, cur_row + num, k] = (df_t1[
                            cur1[j]].iloc[k] + df_t2[cur2[j]].iloc[k]) / 2

        for k in range(present_rows):
            for i in range(num):
                res_hfm_alt[i, k] = fair_degree_v3(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
                res_hfm_alt[i + num, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
            for i in range(num, num + num - 2):
                res_hfm_alt[i + num, k] = fair_degree_v4(
                    res_alt[0, i, k], res_alt[1, i, k])[0]
        return res_alt, res_hfm_alt, res_tim_alt

    def sub_get_acc_fair(self, dframe, tag_acc, tag_far_sa1, tag_far_sa2,
                         per_gen=15, per_att=20):
        df_raw = dframe[tag_acc + tag_far_sa1 + tag_far_sa2].astype(DTY_FLT)
        ans_acc = df_raw[tag_acc[7:14]].values - df_raw[tag_acc[:7]].values
        ans_acc = np.abs(ans_acc)
        overall_rows = dframe.shape[0]
        nc_norm = per_gen // self._nb_iter
        nc_sens = per_att // self._nb_iter
        each_set_prim = per_att * 2 + per_gen
        num = 7  # nb_set, num = 5, 7

        ans_far = np.zeros((overall_rows, 3 + 2 + 6 + 6))
        ans_far_alt = np.zeros((overall_rows, 2, 3 + 6 + 6))
        ans_far[:, 3] = df_raw[tag_acc[-1]].values  # DR
        ans_far[:, 4] = df_raw[tag_acc[-2]].values  # loss
        a_k = per_gen + per_att  # = each_set_prim - per_att
        df_tmp = df_raw.iloc[: a_k]
        for a_i, tag_far in enumerate([tag_far_sa1]):
            ans_far_alt[:a_k, a_i, 0:3] = df_tmp[tag_far[6:9]].values
            ans_far_alt[:a_k, a_i, 3:9] = df_tmp[tag_far[9:15]].values
            ans_far_alt[:a_k, a_i, 9:15] = df_tmp[tag_far[15:21]].values
        df_tmp = df_raw.iloc[a_k:]
        for a_i, tag_far in enumerate([tag_far_sa1, tag_far_sa2]):
            ans_far_alt[a_k:, a_i, 0:3] = df_tmp[tag_far[6:9]].values
            ans_far_alt[a_k:, a_i, 3:9] = df_tmp[tag_far[9:15]].values
            ans_far_alt[a_k:, a_i, 9:15] = df_tmp[tag_far[15:21]].values
        del df_tmp  # del a_k, df_tmp
        if (ans_far_alt != ans_far_alt).any():  # nan!!NOTICE
            ans_far_alt = np.nan_to_num(ans_far_alt)  # ans_far_alt[0][1] nan!!

        for k in range(a_k):
            for j in range(3):
                ans_far[k, j] = ans_far_alt[k, 0, j]
            for j in range(3, 9):
                ans_far[k, j + 2] = ans_far_alt[k, 0, j]
            for j in range(9, 15):
                ans_far[k, j + 2] = ans_far_alt[k, 0, j]
        for k in range(a_k, overall_rows):
            for j in range(3):
                ans_far[k, j] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k, 1, j]) / 2.
            for j in range(3, 9):
                ans_far[k, j + 2] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k, 1, j]) / 2.
            for j in range(9, 15):
                ans_far[k, j + 2] = (
                    ans_far_alt[k, 0, j] + ans_far_alt[k, 1, j]) / 2.
        ans_acc = np.concatenate([
            df_raw[tag_acc[:7]].values, ans_acc], axis=1)
        return ans_acc, ans_far, ans_far_alt


class RevP_XF_statsParity(RevP_XE_statsParity):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, each_set, per_att, per_gen = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=11, nc_sens=0)
        csv_row_1 = unique_column(11 + 4 + 226 * 2)
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[11:])
        tag_trn = csv_row_1[15: 15 + 226]
        tag_tst = csv_row_1[15 + 226: 15 + 226 * 2]
        self.subproc_plot(df_raw, tag_tst, 'tst', f'rexp3f_{pre}',
                          per_gen, per_att)
        return


# ------------------------------


# ------------------------------


# ------------------------------
