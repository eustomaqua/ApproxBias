# coding: utf-8


import pdb
import pandas as pd
import numpy as np

from pyfair.granite.draw_addtl import (
    multi_lin_reg_without_distr, lineplot_with_uncertainty,
    single_line_reg_with_distr)
from pyfair.granite.draw_graph import scatter_id_chart
from pyfair.granite.draw_chart import analogous_confusion_extended
from experiment.utils_empirical import GraphSetupVer2

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.hfm_df import bias_degree_bin as fair_degree_v3
from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4
from hfm.utils.recorders import BLFAIR
from hfm.hfm_df import differentiate_tim, differentiate_val


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
# rexp2: qiqi，考虑 embedding +y_hat 计算 df
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
        key_B_1 = BLFAIR + [  # [r'DP', r'EO', r'PQP', r'DR',
            r'$\mathbf{df}_{prev}$', r'$\mathbf{df}_{prev,emb}$',
            r'$\mathbf{df}$', r'$\mathbf{df}_{emb}$']
        # analogous_confusion_extended(
        #     mat_A.T, mat_B_1.T, key_A, key_B_1, 'test_3', **pm)
        mat_B_2 = np.concatenate([ans_far[:, 2:], ans_far[
            :, 1:2], ans_hfm[:, -4:]], axis=1)
        key_B_2 = BLFAIR + [  # [r'DP', r'EO', r'PQP', r'DR',
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
        key_B = BLFAIR  # [r'DP', r'EO', r'PQP', r'DR', ]
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
        key_B = BLFAIR  # [r'DP', r'EO', r'PQP', r'DR']
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
        key_B = BLFAIR  # [r'DP', r'EO', r'PQP', r'DR']
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
        key_B = BLFAIR  # [r'DP', r'EO', r'PQP', r'DR']

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
# convergence
# rexp4|5: two strategies
#   rexp4 only includes the early-stopping strategy
#   rexp5 also includes the rearrangment strategy
# rexp*b: cross-validation, several learning algorithms


class ConvPlotE_init(GraphSetup):
    # refer to RevPlotZ

    def sub_plt_tim(self, df_tmp, tag_col, suff, rmk='multivar',
                    omitted=True):
        tag_tim = tag_col[: 5]
        scat_X = df_tmp[tag_tim[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[1]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[3]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[4]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'  # .(StratES)
            ant_arr = r'T_{ExactDist (StratRA)}'  # .(StratRA)
        elif rmk == 'sen-att':
            ant_X = r'T_{\mathbf{D}_a(S,a_i)}'        # {\mathbf{a}}
            ant_Y = r'T_{\hat{\mathbf{D}}_a(S,a_i)}'  # {\mathbf{a}}
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
        ant_eff = r'T_{EarlyBreak}'
        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                  '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
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
        tag_max = tag_col[5: 5 + 5]
        tag_avg = tag_col[5 + 5: 5 + 5 + 3]
        scat_X = df_tmp[tag_max[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_max[1]].values.astype(DTY_FLT),
                  df_tmp[tag_max[3]].values.astype(DTY_FLT),
                  df_tmp[tag_max[4]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'\mathbf{D}_{\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            ant_app = 'ExtendDist'
            ant_cvg = 'ExactDist(StratES)'  # 'ExactDist (StratES)'
            ant_arr = 'ExactDist(StratRA)'  # 'ExactDist (StratRA)'
        elif rmk == 'sen-att':
            ant_X = r'\mathbf{D}_a(S,a_i)'
            ant_Y = r'\hat{\mathbf{D}}_a(S,a_i)'
            ant_app = 'ApproxDist'
            ant_cvg = 'StratES'
            ant_arr = 'StratRA'
        ant_eff = 'EarlyBreak'
        # annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
        #           '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]
        annotY = [ant_eff, ant_app, ant_cvg, ant_arr]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty3b')
        if not omitted:
            multi_lin_reg_without_distr(
                df_tmp[tag_max[0]].values.astype(DTY_FLT),
                scat_Y, annotY, annot, suff + '_alt',
                snspec='sty3a')  # 'sty6'?
        return

    def obtain_binval_senatt(self, dframe, id_set,
                             tag, tag_s1, tag_s2, dr_ptb=''):
        columns = {t2: t1 for t1, t2 in zip(tag_s1, tag_s2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag + tag_s1]
        if dr_ptb:
            df_raw[dr_ptb] = dframe.iloc[id_set[1]][dr_ptb]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag + tag_s2]
            df_tmp = df_tmp.rename(columns=columns)
            if dr_ptb:
                df_tmp[dr_ptb] = dframe.iloc[id_set[k]][dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag + tag_s1]
            if dr_ptb:
                df_tmp[dr_ptb] = dframe.iloc[id_set[k]][dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set,
                               tag, tag_s1, tag_s2, dr_ptb='',
                               first_incl=False):
        columns = {t2: t1 for t1, t2 in zip(tag_s1, tag_s2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag + tag_s1]
        if dr_ptb:
            df_raw[dr_ptb] = dframe.iloc[id_set[2]][dr_ptb]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[
                k + 1]][tag + tag_s2]
            df_tmp = df_tmp.rename(columns=columns)
            if dr_ptb:
                df_tmp[dr_ptb] = dframe.iloc[id_set[k]][dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        if not first_incl:
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_s1]
        if dr_ptb:
            df_tmp[dr_ptb] = dframe.iloc[id_set[0]][dr_ptb]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw


class ConvFig_4E_exact(ConvPlotE_init):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 39)
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])

        tag_multivar = csv_row_1[10: 10 + 13]
        df_tmp = df_raw[tag_multivar]
        tag_sa1 = csv_row_1[23: 23 + 13]
        tag_sa2 = csv_row_1[23 + 13: 23 + 13 * 2]
        df_tmp = self.sub_dat_sen_att(df_raw, tag_sa1, tag_sa2)
        suff, rmk = f'rexp8e_{pre}_whole_tim', 'sen-att'
        self.sub_plt_tim(df_tmp, tag_sa1, suff, rmk)
        self.sub_plt_val(df_tmp, tag_sa1, suff.replace('tim', 'val'), rmk)

        # suff = 'rexp8e_{}_adult_tim'.format(pre)
        # df_tmp = self.sub_dat_sing_set(
        #     raw_dframe, id_set, 3, tag_multivar, tag_sa1, tag_sa2)
        # self.sub_plt_tim(df_tmp[1], tag_sa1, suff, 'sen-att')
        # self.sub_plt_val(df_tmp[1], tag_sa1, suff.replace(
        #     'tim', 'val'), 'sen-att')

        suff = suff.replace('whole', 'multivar')
        self.sub_plt_tim(df_raw, tag_multivar, suff)
        self.sub_plt_val(df_raw, tag_multivar, suff.replace('tim', 'val'))
        '''
        df_tmp = self.obtain_binval_senatt(
            raw_dframe, id_set, tag_multivar, tag_sa1, tag_sa2)
        self.sub_plt_tim(df_tmp, tag_sa1, suff, 'sen-att')
        self.sub_plt_val(df_tmp, tag_sa1, suff.replace(
            'tim', 'val'), 'sen-att')
        df_tmp = self.obtain_multival_senatt(
            raw_dframe, id_set, tag_multivar, tag_sa1, tag_sa2,
            first_incl=True)
        self.sub_plt_tim(df_tmp, tag_sa1, suff)
        self.sub_plt_val(df_tmp, tag_sa1, suff.replace('tim', 'val'))
        # df_tmp = self.obtain_multival_senatt(raw_dframe, )
        '''
        # pdb.set_trace()
        return


# class ConvFig_4C_exact(ConvPlotE_init):
#     def schedule_mspaint(self, raw_dframe, pre='minmax'):
#         nb_set, id_set, _, _, _ = self.recap_sub_data(
#             raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
#         csv_row_1 = unique_column(10 + 28 * 2)  # 34*2)
#         df_raw = self.sub_dat_multivar(raw_dframe, nb_set, id_set, csv_row_1[10:])
#         pdb.set_trace()
#         return


# class ConvFig_4D_exact(ConvPlotE_init):
#     def schedule_mspaint(self, raw_dframe, pre='minmax'):
#         pass


class ConvFig_4B_exact(ConvPlotE_init):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        pass


# class ConvP_5C_exact(ConvPlotE_init):
#     def schedule_mspaint(self, raw_dframe, pre='minmax'):
#         pass
class ConvPlotF_init(ConvPlotE_init):
    def sub_plt_tim(self, df_tmp, tag_col, suff, rmk='multivar',
                    omitted=True):
        tag_tim = tag_col[: 6]
        scat_X = df_tmp[tag_tim[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[1]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[3]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[4]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[5]].values.astype(DTY_FLT),
                  ]  # df_tmp[tag_tim[1]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StratRA)}'
            # ant_Z1 = r'\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} }-1'
            ant_Z2 = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
        elif rmk == 'sen-att':
            ant_X = r'T_{\mathbf{D}_a(S,a_i)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_a(S,a_i)}'
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
            # ant_Z1 = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
            # ant_Z2 = r'\lg(\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) })'
            ant_Z2 = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
        ant_eff = r'T_{EarlyBreak}'
        annotY = [
            '${}$'.format(ant_eff), '${}$'.format(ant_app),
            '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        annot = ['${}$ (sec)'.format(ant_X),
                 '${}$ (sec)'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty4')
        if not omitted:
            multi_lin_reg_without_distr(
                df_tmp[tag_tim[0]].values.astype(DTY_FLT),
                scat_Y, annotY, annot, suff + '_alt', snspec='sty4')

        # scat_Z = [k / scat_X - 1 for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z1)
        kws = {'snspec': 'sty6'}  # 'linreg': True,
        # # single_line_reg_with_distr(
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, suff + '_s6a', **kws)
        scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z2)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, suff + '_s6b', **kws)
        return

    def sub_plt_val(self, df_tmp, tag_col, suff, rmk='multivar',
                    omitted=True):
        tag_max = tag_col[6: 6 + 6]
        scat_X = df_tmp[tag_max[2]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_max[1]].values.astype(DTY_FLT),
                  df_tmp[tag_max[3]].values.astype(DTY_FLT),
                  df_tmp[tag_max[4]].values.astype(DTY_FLT),
                  df_tmp[tag_max[5]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'\mathbf{D}_{\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            ant_app = 'ExtendDist'
            ant_cvg = 'ExactDist(StratES)'  # 'ExactDist (StratES)'
            ant_arr = 'ExactDist(StratRA)'  # 'ExactDist (StratRA)'
            ant_Z1 = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'
            # ant_Z2 = r'\lg(\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) })'
        elif rmk == 'sen-att':
            ant_X = r'\mathbf{D}_a(S,a_i)'
            ant_Y = r'\hat{\mathbf{D}}_a(S,a_i)'
            ant_app = 'ApproxDist'
            ant_cvg = 'StratES'
            ant_arr = 'StratRA'
            ant_Z1 = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
            # ant_Z2 = r'\lg(\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) })'
        ant_eff = 'EarlyBreak'
        annotY = [
            # '${}$'.format(ant_eff), '${}$'.format(ant_app),
            # '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
            ant_eff, ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, snspec='sty3b')
        if not omitted:
            multi_lin_reg_without_distr(
                df_tmp[tag_max[0]].values.astype(DTY_FLT),
                scat_Y, annotY, annot, suff + '_alt',
                snspec='sty3a')  # 'sty6'?

        kws = {'snspec': 'sty6'}
        scat_Z = [k / scat_X - 1 for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z1)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, suff + '_s6a', **kws)
        # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z2)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, suff + '_s6b', **kws)
        return

    def sub_plt_avg(self, df_tmp, tag_col, suff, rmk='multivar'):
        tag_avg = tag_col[6 + 6: 6 + 6 + 4]
        scat_X = df_tmp[tag_avg[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_avg[1]].values.astype(DTY_FLT),
                  df_tmp[tag_avg[2]].values.astype(DTY_FLT),
                  df_tmp[tag_avg[3]].values.astype(DTY_FLT)]

        if rmk == 'multivar':
            ant_X = r'\mathbf{D}_{\mathbf{a}}^\text{avg}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S)'
            ant_app = 'ExtendDist'
            ant_cvg = 'ExactDist(StratES)'  # 'ExactDist (StratES)'
            ant_arr = 'ExactDist(StratRA)'  # 'ExactDist (StratRA)'
            ant_Z1 = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }-1'
            ant_Z2 = r'\lg(\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) })'
        elif rmk == 'sen-att':
            ant_X = r'\mathbf{D}_a^\text{avg}(S,a_i)'
            ant_Y = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
            ant_app = 'ApproxDist'
            ant_cvg = 'StratES'
            ant_arr = 'StratRA'
            ant_Z1 = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
            ant_Z2 = r'\lg(\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) })'
        annotY = [  # '${}$'.format(ant_app), '${}$'.format(
            # ant_cvg), '${}$'.format(ant_arr)]
            ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, suff, 
            snspec='sty3c')  # snspec='sty3b')
        kws = {'snspec': 'sty6c'}  # 'sty6'}
        scat_Z = [k / scat_X - 1 for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z1)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, suff + '_s6a', **kws)
        # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z2)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, suff + '_s6b', **kws)
        return

    def obtain_multival_senatt(self, dframe, id_set, tag,
                               tag_s1, tag_s2, first_incl=False):
        columns = {t2: t1 for t1, t2 in zip(tag_s1, tag_s2)}
        k = 2 if len(id_set) == 6 else 0
        df_raw = dframe.iloc[id_set[
            k] + 1: id_set[k + 1]][tag + tag_s1]
        for k in ([3, 4] if len(id_set) == 6 else [2, 3]):
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag + tag_s2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)
        if not (first_incl and len(id_set) == 6):
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_s1]
        return pd.concat([
            df_raw, df_tmp], axis=0).reset_index(drop=True)

    def obtain_binval_senatt(self, dframe, id_set, tag,
                             tag_s1, tag_s2):
        columns = {t2: t1 for t1, t2 in zip(tag_s1, tag_s2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag + tag_s1]
        for k in ([1, 2] if len(id_set) == 6 else [1]):
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag + tag_s2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        for k in ([3, 4] if len(id_set) == 6 else [2, 3]):
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag + tag_s1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw


class ConvFig_5C_exact(ConvPlotF_init):  # E_init
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 34 * 2)  # 28*2)
        df_raw = self.sub_dat_multivar(raw_dframe, nb_set, id_set, csv_row_1[10:])
        tag_sa1 = csv_row_1[10: 10 + 34]
        tag_sa2 = csv_row_1[10 + 34:]
        pdb.set_trace()
        return


class ConvFig_5D_exact(ConvPlotF_init):  # E_init
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 41 * 2)
        df_raw = self.sub_dat_multivar(raw_dframe, nb_set, id_set, csv_row_1[10:])
        tag_sa1 = csv_row_1[10: 10 + 41]
        tag_sa2 = csv_row_1[10 + 41:]
        pdb.set_trace()
        return


class ConvFig_5E_exact(ConvPlotF_init):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=1, nc_sens=0)
        csv_row_1 = unique_column(10 + 48)  # 34*2)  # 28*2)
        df_raw = self.sub_dat_multivar(
            raw_dframe, nb_set, id_set, csv_row_1[10:])

        tag_multivar = csv_row_1[10: 10 + 16]
        tag_sa1 = csv_row_1[26: 26 + 16]
        tag_sa2 = csv_row_1[26 + 16: 26 + 16 * 2]
        df_tmp = df_raw[tag_multivar]
        df_tmp = self.sub_dat_sen_att(df_raw, tag_sa1, tag_sa2)
        suff, rmk = f'rexp9e_{pre}_sing_tim', 'sen-att'
        self.sub_plt_tim(df_tmp, tag_sa1, suff, rmk)
        self.sub_plt_val(
            df_tmp, tag_sa1, suff.replace('tim', 'val'), rmk)
        self.sub_plt_avg(
            df_tmp, tag_sa1, suff.replace('tim', 'avg'), rmk)
        suff = suff.replace('sing', 'multivar')  # 'sapl')
        self.sub_plt_tim(df_raw, tag_multivar, suff)
        self.sub_plt_val(df_raw, tag_multivar,
                         suff.replace('tim', 'val'))
        self.sub_plt_avg(df_raw, tag_multivar,
                         suff.replace('tim', 'avg'))

        # df_tmp = self.obtain_binval_senatt(
        #     raw_dframe, id_set, tag_multivar, tag_sa1, tag_sa2)
        # suff = suff.replace('sapl', 'bin')
        # self.sub_plt_tim(df_tmp, tag_sa1, suff, rmk)
        # self.sub_plt_val(df_tmp, tag_sa1, suff.replace(
        #     'tim', 'val'), rmk)
        # suff = suff.replace('bin', 'mut')
        # df_tmp = self.obtain_multival_senatt(
        #     raw_dframe, id_set, tag_multivar, tag_sa1, tag_sa2,
        #     first_incl=True)
        # self.sub_plt_tim(df_tmp, tag_sa1, suff)
        # self.sub_plt_val(df_tmp, tag_sa1, suff.replace('tim', 'val'))
        # suff = suff.replace('mut', 'muf')
        # df_tmp = self.obtain_multival_senatt(
        #     raw_dframe, id_set, tag_multivar, tag_sa1, tag_sa2)
        # self.sub_plt_tim(df_tmp, tag_sa1, suff)
        # self.sub_plt_val(df_tmp, tag_sa1, suff.replace('tim', 'val'))
        # pdb.set_trace()
        return


class ConvFig_5B_exact(ConvPlotF_init):  # E_init
    def prepare_graph(self):
        csv_row_1 = unique_column(10 + 2 + 374 * 2)
        tmp_norm, tmp_gf = 7, 11   # tmp_egf=0,tmp_if
        tmp_hfm = 12 + 5 + 27 + 5  # hfm
        tmp_hfm_ext = (9 + 18) * 4 + 5

        pms = csv_row_1[: 11 + 1]  # params last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + 374]
        tag_tst = csv_row_1[12 + 374:]
        return pms, tag_trn, tag_tst

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=4)
        pdb.set_trace()
        return


class ConvFig_5H_exact(ConvPlotF_init):
    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 1 + 274 * 2)
        pms = csv_row_1[: 11 + 1]
        tag_trn = csv_row_1[12: 12 + 274]
        tag_tst = csv_row_1[12 + 274:]
        return pms, tag_trn, tag_tst

    def incise_graph(self, tag):           # tmp_norm, tmp_gf = 7, 11
        tag_norm = [tag[: 7], tag[7 + 7: 7 * 3]]  # normal,delta(perf)
        tag_gf = [tag[21: 21 + 5] + tag[31: 31 + 16],
                  tag[26: 21 + 10] + tag[47: 47 + 16]]
        # tag_gf each sa:   DP,EOpp,PP,DR,hat_L(loss),
        #                   extG*(max|avg *3), altExtG*(max|avg *3),
        #               T(extG1),T(extG2),T(extG3), T(extGrp* total)
        tmp = [tag[63: 63 + 12 + 5 + 9 + 9 * 2 + 5],
               tag[63 + 49: 63 + 49 * 2]]
        tag_hfm = {
            'dist': [
                j[:4] + j[12:12 + 2] + j[17:17 + 4] +
                j[26:26 + 4] + j[35:35 + 4] for j in tmp],
            # 'dist' sa:    DistDirect_bin{Ds,Ds_avg,Df,Df_avg},
            #               ApproxDist_bin{Ds,Df},
            #               StratVacant{Ds,Ds_avg,Df,Df_avg},
            #               StratES{Ds,Ds_avg,Df,Df_avg},
            #               StratRA{Ds,Ds_avg,Df,Df_avg},
            'df': [j[
                4:7] + [j[12 + 2], ] + j[17 + 4:17 + 7] + j[
                26 + 4:26 + 7] + j[35 + 4:35 + 7] for j in tmp],
            # 'df' sa  :    DistDirect_bin{df_prev,df,df_avg},
            #               ApproxDist_bin{df_prev},
            #               StratVacant{df_prev,df,df_avg},
            #               StratES{df_prev,df,df_avg},
            #               StratRA{df_prev,df,df_avg},
            'tim': [j[
                7:12] + j[12 + 3:12 + 5] + j[17 + 7:17 + 9] + j[
                26 + 7:26 + 9] + j[35 + 7:35 + 9 + 5] for j in tmp],
            # 'tim' sa :    DistDirect_bin{t_Ds,t_Df,
            #                           T(df_prev),T(df),T(df_avg)},
            #               ApproxDist_bin{t_Ds,t_Df},
            #               StratVacant{t_Ds,t_Df},
            #               StratES{t_Ds,t_Df}, StratRA{t_Ds,t_Df},
            #               T(DistDirect_bin), T(ApproxDist_bin),
            #               T(StratVacant /DistApprox_nonbin),
            #               T(StratES), T(StratRA),
            #
            # 'dist': 4+2+4*3=18, 'df':3+1+3*3=13, 'tim': 5+2*4+5=18
        }  # 'dist_avg': [j[] for j in tmp]}  # .siz=(2,18|13|13+5)

        tmp = tag[(21 + 10 + 16 * 2) + 49 * 2: 63 + 98 + (
            9 + 18 + 9 + 18 + 9 + 18 + 9 + 18 + 5)]
        tag_df_conv = {
            'multivar': tmp[:9] + tmp[
                27:27 + 9] + tmp[54:54 + 9] + tmp[81:81 + 9],
            # 'multivar':   DistDirect_multivar{Ds,Ds_avg,Df,Df_avg,
            #                          df_prev,df,df_avg,t_Ds,t_Df},
            #               EffExact-Vacant{.... ... ..},
            #               EffExact-StratES{. x9},
            #               EffExact-StratRA{. x9},
            'sa1': tmp[9:18] + tmp[
                9 * 4:9 * 5] + tmp[9 * 7:9 * 8] + tmp[9 * 10:9 * 11],
            'sa2': tmp[18:27] + tmp[
                9 * 5:9 * 6] + tmp[9 * 8:9 * 9] + tmp[9 * 11:9 * 12],
            # 'sa' :    DistDirect_nonbin{.... ... ..},
            #           StratVacant /DistApprox_nonbin{. x9},
            #           StratES{. x9}, StratRA{. x9},
            'tim': tmp[9 * 12: 9 * 12 + 5],
            # 'tim':  T(DistDirect_multivar), T(DistExtend_multivar),
            #         T(EffExact-StratES), T(EffExact-StratRA),
            #         T(computing normal-perf +grp fair dr),
            #
            # 'multivar /sa': 9*4=36, 'tim': 4+1=5
        }  # tag_df_multivar = []
        del tmp  # pdb.set_trace()
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        pms, _, tag_tst = self.prepare_graph()
        tag_norm, tag_dr, tag_hfm, tag_conv = self.incise_graph(tag_tst)
        fgn = f'{self._figname}{pre}'  # _multivar
        # self.subfig_conv_multivar(raw_dframe, tag_conv, fgn,
        #                           tag_hfm=tag_hfm)
        # self.subfig_conv_singvar(raw_dframe, tag_hfm, tag_conv, fgn)

        self.subfig_fair_sp(raw_dframe, tag_norm, tag_dr, tag_hfm,
                            tag_conv, fgn + '_esp')
        return

    def subfig_fair_sp(self, dframe, tag_norm,  # tag_fair,
                       tag_dr, tag_hfm, tag_conv, fgn):
        tag_fair = [tag_dr[i][:7] + tag_dr[i][11:13] + tag_hfm[
            'df'][i] for i in range(2)]  # hfm:3+1+3+3*2
        tag_fair = [i[:4] + i[5:7] + i[9:][
            :1] + i[9 + 3:][:1] + i[9 + 4:][1:3] + i[
            13 + 3:][1:3] + i[16 + 3:][1:3] for i in tag_fair]
        tmp_dr = [i[:3] + i[4:6] + i[3:4] + tag_dr[ti][
            4:5] for ti, i in enumerate(tag_fair)]  # tmp_hfm:1+1+2+2*2=8
        # tmp_dr  : DP,EOpp,PP, SP,SP_avg, DR, hat_L(loss),  # 3+2+2=7
        tmp_hfm = [[i[6:][j] for j in [
            0, 1, 2, 4, 6, 3, 5, 7]] for i in tag_fair]
        # tmp_hfm : Direct_bin df_prev, ApproxDist_bin df_prev,
        #           StratVacant,StratES,StratRA, ..df_avg
        tmp_conv = [[i[5:7], i[9:][5:7], i[18:][5:7], i[27:][
            5:7]] for i in [tag_conv['sa1'], tag_conv['sa2']]]
        tmp_conv = [np.array(tmp_conv[i]).T.reshape(
            -1).tolist() for i in range(2)]
        # tmp_conv: Direct_nonbin,StratVacant,StratES,StratRA, ..df_avg
        tmp_multivar = np.array([tag_conv['multivar'][5:7], tag_conv[
            'multivar'][9:][5:7], tag_conv['multivar'][18:][
            5:7], tag_conv['multivar'][27:][5:7]]).T.reshape(
            -1).tolist()   # Direct_mv,Extend,ExactES,ExactRA, ..df_avg

        tag_sa1 = tmp_dr[0] + tmp_hfm[0] + tmp_conv[0]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1] + tmp_conv[1]
        tb = tag_norm[0] + tag_norm[1]  # tmp_multivar+ #tsa=
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        # df_alt = self.sub_dat_multivar(
        #     dframe, nb_set, id_set,
        #     tb + tmp_multivar)  # tag_sa1+tag_sa2+tb)
        df_alt = self.obtain_multival_senatt(
            # df_alt = self.obtain_binval_senatt(
            dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        # df = self.sub_dat_grpfair(
        #     dframe, nb_set, id_set, tag_sa1, tag_sa2)
        '''
        df = df_alt  # df=self.sub_dat_grpfair(df_alt, tag_sa1, tag_sa2)
        # TODO!!
        # key_A = [r'accuracy',  # r'precision', r'recall',
        #          r'$\mathrm{f}_1$ score', r'specificity',
        #          r'g_mean', r'dp']
        # key_B = [r'$\Delta$(accuracy)',  # r'$\Delta$(precision)',
        #          # r'$\Delta$(recall)', r'$\Delta$(specificity)',
        #          # r'$\Delta(\mathrm{f}_1\text{ score})$',
        #          r'$\Delta$($\mathrm{f}_1$ score)',
        #          r'$\Delta$(specificity)',
        #          r'$\Delta$(g_mean)', r'$\Delta$(dp)']
        # tag_norm = [i[:1] + i[4:5] + i[3:4] + i[-2:] for i in tag_norm]

        key_C = BLFAIR[:3] + [  # r'$\text{ESP}^\text{max}$',
            # r'$\mathrm{ESP}$', r'$\mathrm{ESP}^\text{avg}$'] + BLFAIR[
            r'$\mathrm{SP}$', r'$\mathrm{SP}^\text{avg}$'] + BLFAIR[
            -1:] + [r'$\mathbf{df}_\text{prev}$',
                    r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        #     r'$\hat{\mathbf{df}}_\text{prev}$',
        #     r'$\hat{\mathbf{df}}$', r'$\hat{\mathbf{df}}^\text{avg}$']
        key_D = tmp_hfm[0][:1] + tmp_conv[0][:1] + tmp_conv[0][4:5]  # tmp_multivar[:1] + tmp_multivar[4:5]
        mat_C = np.concatenate([
            df[tag_sa1[:6]].values.astype(DTY_FLT).T,
            # df_alt[tmp_multivar[:1] + tmp_multivar[4:5]].values.astype(
            #     DTY_FLT).T,
            #
            # df_alt[tmp_hfm[0][0]].values.astype(DTY_FLT).reshape(1, -1),
            # df_alt[tmp_conv[0][:1] + tmp_conv[0][4:5]].values.astype(
            #     DTY_FLT).T, df_alt[tag_hfm[0][1]].values.astype(
            #     DTY_FLT).reshape(1, -1), ], axis=0)[:-1]
            df_alt[key_D].values.astype(DTY_FLT).T], axis=0)
        #
        # key_C = BLFAIR[:3] + [r'$\text{ESP}^\text{max}$', ] + BLFAIR[
        #     -1:] + [r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        # mat_C = np.concatenate([
        #     df[tag_sa1[:4] + tag_sa1[5:6]].values.astype(DTY_FLT).T,
        #     df_alt[tmp_multivar[:1] + tmp_multivar[4:5]].values.astype(
        #         DTY_FLT).T], axis=0)

        pms = {'cmap_name': 'BuGn', 'rotate': 25}  # ,'OrRd','figsize':'L-WS'}
        analogous_confusion_extended(df_alt[
            tag_norm[0][:-1]].values.astype(DTY_FLT).T, mat_C,
            key_A, key_C, f'{fgn}_oo', **pms)
        analogous_confusion_extended(df_alt[
            tag_norm[1][:-1]].values.astype(DTY_FLT).T, mat_C,
            key_B, key_C, f'{fgn}_delt', **pms)
        mat_D = np.concatenate([mat_C.T, df_alt[
            # tag_norm[0][:2]
            tag_norm[0][:1] + tag_norm[0][-1:]
        ].values.astype(DTY_FLT)], axis=1)
        # key_D = tag_sa1[:6] + tmp_hfm[0][:1] + tmp_multivar[
        #     :1] + tmp_multivar[4:5] + tag_norm[
        #     0][:1] + tag_norm[0][-1:]  # :2] acc,f1_score
        key_D = tag_sa1[:6] + key_D + tag_norm[0][:1] + tag_norm[0][-1:]
        mat_D = pd.DataFrame(mat_D, columns=key_D)
        # lineplot_with_uncertainty(
        #     mat_D, key_D[-2], 'Fairness', key_D[:4] + key_D[6:8],
        #     key_C[:4] + key_C[6:8], figname=fgn + '_pc1b',
        #     alpha_loc='b4', alpha_rev=True, annotY=' error rate',
        #     cmap_name='coolwarm_r', alpha_clarity=.15)
        kws = {'alpha_loc': 'b4', 'alpha_rev': True,
               'alpha_clarity': .15, 'cmap_name': 'coolwarm_r',
               'annotY': ' error rate'}
        # lineplot_with_uncertainty(
        #     mat_D, key_D[-2], 'Fairness', key_D[:3] + key_D[6:8],
        #     key_C[:3] + key_C[6:8], figname=fgn + '_pc1b', **kws)
        # lineplot_with_uncertainty(
        #     mat_D, key_D[-2], 'Fairness', key_D[3:8], key_C[3:8],
        #     figname=fgn + '_pc2b', **kws)
        # kD_sep1 = [0, 1, 2, 6, 7];  kD_sep2 = [3, 4, 5, 7, 8]
        
        lineplot_with_uncertainty(
            mat_D, key_D[-2], 'Fairness', key_D[:3] + key_D[6:8],
            key_C[:3] + key_C[6:8], figname=fgn + '_pc1b', **kws)
        lineplot_with_uncertainty(
            mat_D, key_D[-2], 'Fairness', key_D[3:6] + key_D[7:9],
            key_C[3:6] + key_C[7:9], figname=fgn + '_pc2b', **kws)
        kws['annotY'] = '(1$-$performance)'  # del
        lineplot_with_uncertainty(
            mat_D, key_D[-1], 'Fairness', key_D[:3] + key_D[6:8],
            key_C[:3] + key_C[6:8], figname=fgn + '_pc3b', **kws)
        lineplot_with_uncertainty(
            mat_D, key_D[-1], 'Fairness', key_D[3:6] + key_D[7:9],
            key_C[3:6] + key_C[7:9], figname=fgn + '_pc4b', **kws)
        pdb.set_trace()
        '''

        df = df_alt
        key_A = [r'accuracy', r'precision', r'recall', r'specificity',
                 r'g_mean', r'dp']  # r'$\mathrm{f}_1$ score',
        key_B = [r'$\Delta$(accuracy)', r'$\Delta$(precision)',
                 r'$\Delta$(recall)', r'$\Delta$(specificity)',
                 # r'$\Delta$($\mathrm{f}_1$ score)',
                 r'$\Delta$(g_mean)', r'$\Delta$(dp)']
        tag_norm = [i[:4] + i[5:] + i[4:5] for i in tag_norm]
        key_C = BLFAIR[:3] + [r'$\mathrm{SP}$',
                              r'$\mathrm{SP}^\text{avg}$'] + BLFAIR[
            -1:] + [r'$\mathbf{df}_\text{prev}$',
                    r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        key_D = tmp_hfm[0][:1] + tmp_conv[0][:1] + tmp_conv[0][4:5]
        mat_C = df[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        pms = {'cmap_name': 'Greens', 'rotate': 34}  # 'BuGn'
        # analogous_confusion_extended(df[tag_norm[0][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_A, key_C, f'{fgn}_mv_oo', **pms)
        analogous_confusion_extended(df[tag_norm[1][:-1]].values.astype(
            DTY_FLT).T, mat_C, key_B, key_C, f'{fgn}_mv_delt', **pms)
        key_D = tag_sa1[:6] + key_D + tag_norm[0][:1] + tag_norm[0][-1:]
        df = self.obtain_binval_senatt(
            dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        pms['cmap_name'] = 'Blues'  # 'Spectral'  # 'GnBu_r'
        mat_C = df[key_D[:-2]].values.astype(DTY_FLT).T
        # analogous_confusion_extended(df[tag_norm[0][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_A, key_C, f'{fgn}_bi_oo', **pms)
        analogous_confusion_extended(df[tag_norm[1][:-1]].values.astype(
            DTY_FLT).T, mat_C, key_B, key_C, f'{fgn}_bi_delt', **pms)

        # tmp = self.sub_dat_sen_att(  # .siz=(7+8+8)+8+7*2=23+22=45
        #     dframe, nb_set, id_set, tag_sa1 + tb, tag_sa2 + tb)
        tmp = df_alt
        key_C = BLFAIR[:3] + [  # r'$\text{SP}^\text{max}$',
            r'$\mathrm{SP}$', r'$\mathrm{SP}^\text{avg}$'] + BLFAIR[
            -1:] + [r'$\mathbf{df}_\text{prev}$',
                    r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        key_D = tag_sa1[:6] + tag_sa1[7:][:1] + tag_sa1[
            7 + 8:][:1] + tag_sa1[15:][4:5]
        mat_C = tmp[key_D].values.astype(DTY_FLT).T
        # analogous_confusion_extended(
        #     tmp[tag_norm[0]].values.astype(DTY_FLT).T, mat_C,
        #     key_A, key_C, f'{fgn}_sing_oo', **pms)
        analogous_confusion_extended(
            tmp[tag_norm[1]].values.astype(DTY_FLT).T, mat_C,
            key_B, key_C, f'{fgn}_sing_dt', **pms)
        # mat_D = np.concatenate([mat_C.T, tmp[
        #     # tag_norm[0][:2]].values.astype(DTY_FLT)], axis=1)
        #     tag_norm[0][:1] + tag_norm[0][-1:]
        # ].values.astype(DTY_FLT), ], axis=1)
        # mat_D = pd.DataFrame(
        #     mat_D, columns=key_D + tag_norm[0][:1] + tag_norm[0][-1:])
        # # mat_D = pd.DataFrame(mat_D, columns=key_D + tag_norm[0][:2])
        # kws = {'alpha_loc': 'b4', 'alpha_rev': True,
        #        'alpha_clarity': .15, 'cmap_name': 'coolwarm_r',
        #        'annotY': ' error rate'}
        # lineplot_with_uncertainty(
        #     mat_D, tag_norm[0][0], 'Fairness',
        #     key_D[:3] + key_D[6:9], key_C[:3] + key_C[6:9],
        #     figname=fgn + '_sing_pc1', **kws)
        # lineplot_with_uncertainty(
        #     mat_D, tag_norm[0][0], 'Fairness', key_D[3:9],
        #     key_C[3:9], figname=fgn + '_sing_pc2', **kws)
        # kws['annotY'] = '(1$-$performance)'  # del
        # lineplot_with_uncertainty(
        #     mat_D, tag_norm[0][-1], 'Fairness',
        #     key_D[:3] + key_D[6:9], key_C[:3] + key_C[6:9],
        #     figname=fgn + '_sing_pc3', **kws)
        # lineplot_with_uncertainty(
        #     mat_D, tag_norm[0][-1], 'Fairness', key_D[3:9],
        #     key_C[3:9], figname=fgn + '_sing_pc4', **kws)
        # del kws, pms
        pdb.set_trace()  # pdb.set_option()
        return

    # def sub_dat_grpfair(self, dframe, nb_set, id_set, tag_sa1, tag_sa2):
    #     df_raw = dframe[tag_sa1].iloc[id_set[0] + 1: id_set[0 + 1]]
    #     columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
    #     operator_max = tag_sa1[3:4] + tag_sa1[
    #         7:][:5] + tag_sa1[7 + 8:][:4]
    #     operator_avg = tag_sa1[:3] + tag_sa1[4:7] + tag_sa1[
    #         7:][5:8] + tag_sa1[7 + 8:][4:]
    #     for i in range(1, nb_set):
    #         df_tmp = dframe[tag_sa1].iloc[id_set[i] + 1: id_set[i + 1]]
    #         df_alt = dframe[tag_sa2].iloc[id_set[i] + 1: id_set[i + 1]]
    #         df_alt = df_alt.rename(columns=columns)
    #
    #         tmp = df_alt.copy()
    #         for j in operator_avg:
    #             tmp[j] = (df_tmp[j] + df_alt[j]) / 2.
    #         for j in operator_max:
    #             tmp[j] = np.max([
    #                 df_tmp[j].values.astype(DTY_FLT),
    #                 df_alt[j].values.astype(DTY_FLT)], axis=0)
    #         df_raw = pd.concat([df_raw, tmp], axis=0)
    #         del tmp, df_tmp, df_alt
    #     return df_raw.reset_index(drop=True)

    # def sub_dat_grpfair(self, df_alt, tag_sa1, tag_sa2):
    #     columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
    #     op_max = tag_sa1[3:4] + tag_sa1[7:][:5] + tag_sa1[7 + 8:][:4]
    #     op_avg = tag_sa1[:3] + tag_sa1[4:7] + tag_sa1[
    #         7:][5:8] + tag_sa1[7 + 8:][4:]
    #     tmp = df_alt.copy()
    #     for j in op_avg:
    #         tmp[j] = (df_)
    #     pdb.set_trace()
    #     return

    def subfig_conv_singvar(self, dframe, tag_hfm, tag_conv, fgn):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        tag_sa1 = tag_conv['sa1'][:9]  # [:4] *=tag_sa1[:4]+tag_sa1[-2:]
        tag_sa2 = tag_conv['sa2'][:9]  # [:4] *=tag_sa2[:4]+tag_sa2[-2:]
        tag_sa1 = tag_hfm['dist'][0] + tag_sa1[:4] + tag_hfm[
            'tim'][0][:-5] + tag_sa1[-2:]
        tag_sa2 = tag_hfm['dist'][1] + tag_sa2[:4] + tag_hfm[
            'tim'][1][:-5] + tag_sa2[-2:]
        # tag_sa*:  DistDirect_bin x4, ApproxDist_bin x2,
        #           StratVacant x4, StratES x4, StratRA x4,
        #           DistDirect_nonbin x4,   # .siz=18+4=22
        # tag_tim:  DistDirect_bin x5, ApproxDist_bin x2,
        #           StratVacant x2, StratES x2, StratRA x2,
        #           DistDirect_nonbin x2,   # .siz=13+2=15
        # tmp = self.sub_dat_sen_att(
        #     dframe, nb_set, id_set, tag_sa1, tag_sa2)
        tmp = self.obtain_binval_senatt(
            dframe, id_set, [], tag_sa1, tag_sa2)

        tag_tim = tag_sa1[18 + 4:]
        tag_tim = tag_tim[:2] + tag_tim[5:]
        tag_tim = np.array(tag_tim).reshape(-1, 2).T.tolist()
        fgn += '_singvar'  # '_whole'
        # self.sub_plt_tim_prev(tmp, tag_tim, fgn + '_tim', 't_Ds')
        tag_val = tag_sa1[: 18 + 4]
        tag_avg = np.array([tag_val[i] for i in [
            1, 3, 7, 9, 11, 13, 15, 17,
            19, 21]]).reshape(-1, 2).T.tolist()
        tag_val = np.array([tag_val[i] for i in [
            0, 2, 4, 5, 6, 8, 10, 12, 14, 16,
            18, 20]]).reshape(-1, 2).T.tolist()
        # self.sub_plt_val_prev(tmp, tag_val, fgn + '_val', 'Ds')
        # self.sub_plt_avg_prev(tmp, tag_avg, fgn + '_avg', 'Ds_avg')
        self.thread_previous(tmp, tag_tim, fgn, tag_val, tag_avg)
        return

    def thread_previous(self, tmp, tag_tim, fgn, tag_val, tag_avg,
                        verbose=False):
        if verbose:
            self.sub_plt_tim_prev(tmp, tag_tim[0], fgn + '_tim', 't_Ds')
            self.sub_plt_tim_prev(tmp, tag_tim[1], fgn + '_timf', 't_Df')
        df_new = tmp[tag_tim[1]].rename(columns={
            t2: t1 for t1, t2 in zip(*tag_tim)})
        df_new = pd.concat([tmp[tag_tim[0]], df_new], axis=0)
        self.sub_plt_tim_prev(df_new, tag_tim[0], fgn + '_timdt')
        if verbose:
            self.sub_plt_val_prev(tmp, tag_val[0], fgn + '_val', 'Ds')
            self.sub_plt_val_prev(tmp, tag_val[1], fgn + '_valf', 'Df')
        df_new = tmp[tag_val[1]].rename(columns={
            t2: t1 for t1, t2 in zip(*tag_val)})
        df_new = pd.concat([tmp[tag_val[0]], df_new], axis=0)
        self.sub_plt_val_prev(df_new, tag_val[0], fgn + '_valdt')
        if not tag_avg:
            return
        if verbose:
            self.sub_plt_avg_prev(tmp, tag_avg[0], fgn + '_avg', 'Ds_avg')
            self.sub_plt_avg_prev(tmp, tag_avg[1], fgn + '_avgf', 'Df_avg')
        df_new = tmp[tag_avg[1]].rename(columns={
            t2: t1 for t1, t2 in zip(*tag_avg)})
        df_new = pd.concat([tmp[tag_avg[0]], df_new], axis=0)
        self.sub_plt_avg_prev(df_new, tag_avg[0], fgn + '_avgdt')
        return

    def sub_plt_tim_prev(self, df_tmp, tag, fgn, sgn='t_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)

        # if sgn.startswith('t_Ds') or (sgn == 't_D'):
        #     scat_X = df_tmp[tag[0][5]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][3]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][4]].values.astype(DTY_FLT)]
        #     ant_X = r'T_{\mathbf{D}_a(S,a_i)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_a(S,a_i)}'
        #     ant_Xp = r'T_{\mathbf{D}(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
        # elif sgn.startswith('t_Df'):
        #     scat_X = df_tmp[tag[1][5]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[1][0]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][3]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][4]].values.astype(DTY_FLT)]
        #     ant_X = r'T_{\mathbf{D}_{f,a}(S,a_i)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_{f,a}(S,a_i)}'
        #     ant_Xp = r'T_{\mathbf{D}_f(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}_f(S_1,\bar{S}_1)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,a}(S,a_i)} }{ T_{\mathbf{D}_{f,a}(S,a_i)} })'
        # else:
        #     scat_X = np.concatenate([df_tmp[tag[0][
        #         5]].values, df_tmp[tag[1][5]].values]).astype(DTY_FLT)
        #     scat_Y = [np.concatenate([
        #         df_tmp[tag[0][0]].values, df_tmp[
        #             tag[1][0]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][1]].values, df_tmp[
        #             tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][2]].values, df_tmp[
        #             tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][3]].values, df_tmp[
        #             tag[1][3]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][4]].values, df_tmp[
        #             tag[1][4]].values]).astype(DTY_FLT)]
        #     ant_X = r'T_{\mathbf{D}_{\cdot,a}(S,a_i)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)}'
        #     ant_Xp = r'T_{\mathbf{D}_{\cdot}(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)} }{ T_{\mathbf{D}_{\cdot,a}(S,a_i)} })'
        # annotY = ['${}$'.format(ant_Xp), '${}$'.format(ant_Yq),
        #           r'$T_{ApproxDist}$', r'$T_{StratES}$',
        #           r'$T_{StratRA}$']  # r'$T_{DirectDist_nonbin}$']
        #           # '${}$ {}'.format(ant_Y, 'ApproxDist'),
        #           # '${}$ {}'.format(ant_Y, 'StratES'),
        #           # '${}$ {}'.format(ant_Y, 'StratRA')]
        annotY = [r'$T_{ApproxDist}$', r'$T_{StratES}$', r'$T_{StratRA}$',
                  '${}$'.format(ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Y), '${}$ (bin-val)$= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty4c')  # 'sty4')

        # kws = {'snspec': 'sty6c'}  # 'sty6'}
        # # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn = 'D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)  # [5]
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  # df_tmp[tag[0]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)

        # if sgn.startswith('Ds') or (sgn == 'D'):
        #     scat_X = df_tmp[tag[0][5]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][3]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][4]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_a(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_a(S,a_i)'
        #     ant_Xp = r'\mathbf{D}(S_1,\bar{S}_1)'
        #     ant_Yq = r'\hat{\mathbf{D}}(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
        # elif sgn.startswith('Df'):
        #     scat_X = df_tmp[tag[1][5]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[1][0]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][3]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][4]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{f,a}(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_{f,a}(S,a_i)'
        #     ant_Xp = r'\mathbf{D}_f(S_1,\bar{S}_1)'
        #     ant_Yq = r'\hat{\mathbf{D}}_f(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}(S,a_i) }{ \mathbf{D}_{f,a}(S,a_i) }-1'
        # else:
        #     scat_X = np.concatenate([df_tmp[tag[0][
        #         5]].values, df_tmp[tag[1][5]].values]).astype(DTY_FLT)
        #     scat_Y = [np.concatenate([
        #         df_tmp[tag[0][0]].values, df_tmp[
        #             tag[1][0]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][1]].values, df_tmp[
        #             tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][2]].values, df_tmp[
        #             tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][3]].values, df_tmp[
        #             tag[1][3]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][4]].values, df_tmp[
        #             tag[1][4]].values]).astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{\cdot,a}(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}(S,a_i)'
        #     ant_Xp = r'\mathbf{D}_{\cdot}(S_1,\bar{S}_1)'
        #     ant_Yq = r'\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}(S,a_i) }{ \mathbf{D}_{\cdot,a}(S,a_i) }-1'
        # annotY = ['${}$'.format(ant_Xp), '${}$'.format(ant_Yq),
        #           # r'ApproxDist', r'StratES',
        #           # r'StratRA']  # r'$DirectDist_nonbin$']
        #           # '${}$ {}'.format(ant_Y, 'ApproxDist'),
        #           # '${}$ {}'.format(ant_Y, 'StratES'),
        #           # '${}$ {}'.format(ant_Y, 'StratRA')]
        #           'ApproxDist', 'StratES', 'StratRA']
        annotY = ['ApproxDist', 'StratES', 'StratRA', '${}$'.format(
            ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$'.format(ant_Xp), '${}$ (bin-val)'.format(
            ant_Y), '${}$ (bin-val) $= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty3c')  # 'sty3b')

        # kws = {'snspec': 'sty6c'}  # 'sty6'}
        # # scat_Z = [k / scat_X - 1. for k in scat_Y]
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def sub_plt_avg_prev(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[4]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_avg(fgn, sgn)

        # if sgn.startswith('Ds') or (sgn == 'D_avg'):
        #     scat_X = df_tmp[tag[0][4]].values.astype(DTY_FLT)
        #     scat_Y = [  # df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #         df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #         df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #         df_tmp[tag[0][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_a^\text{avg}(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
        #     # ant_Xp = r'\mathbf{D}^{avg}(S_1,\bar{S}_1)'
        #     # ant_Yq = r'\hat{\mathbf{D}}^{avg}(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
        # elif sgn.startswith('Df'):  # 'Df_avg'
        #     scat_X = df_tmp[tag[1][4]].values.astype(DTY_FLT)
        #     scat_Y = [  # df_tmp[tag[1][0]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{f,a}^\text{avg}(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i)'
        #     # ant_Xp = r'\mathbf{D}_f^{avg}(S_1,\bar{S}_1)'
        #     # ant_Yq = r'\hat{\mathbf{D}}_f^{avg}(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{f,a}^\text{avg}(S,a_i) }-1'
        # else:
        #     scat_X = np.concatenate([df_tmp[tag[0][
        #         4]].values, df_tmp[tag[1][4]].values]).astype(DTY_FLT)
        #     scat_Y = [  # np.concatenate([
        #         # df_tmp[tag[0][0]].values, df_tmp[
        #         #     tag[1][0]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][1]].values, df_tmp[
        #             tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][2]].values, df_tmp[
        #             tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([df_tmp[tag[0][3]].values, df_tmp[
        #             tag[1][3]].values]).astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{\cdot,a}^\text{avg}(S,a_i)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i)'
        #     # ant_Xp = r'\mathbf{D}_{\cdot}^{avg}(S_1,\bar{S}_1)'
        #     # ant_Yq = r'\hat{\mathbf{D}}_{\cdot}^{avg}(S_1,\bar{S}_1)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{\cdot,a}^\text{avg}(S,a_i) }-1'
        annotY = [  # '${}$'.format(ant_Xp), # '${}$'.format(ant_Yq),
            # r'ApproxDist', r'StratES',
            # r'StratRA']  # r'$DirectDist_nonbin$']
            # '${}$ {}'.format(ant_Y, 'ApproxDist'),
            # '${}$ {}'.format(ant_Y, 'StratES'),
            # '${}$ {}'.format(ant_Y, 'StratRA')]
            r'ApproxDist', r'StratES', r'StratRA']
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$ (bin-val)'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty3c')  # 'sty3d')# snspec='sty3b')

        # kws = {'snspec': 'sty6c'}  # 'sty6d'}#'sty6'}
        # # scat_Z = [k / scat_X - 1. for k in scat_Y]
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def hfm_dict_prev(self, sgn='t_D.'):
        if sgn.startswith('t_Ds') or (sgn == 't_D'):
            ant_Xp = r'T_{\mathbf{D}(S_1,\bar{S}_1)}'
            ant_Yq = r'T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)}'
        elif sgn.startswith('t_Df'):
            ant_Xp = r'T_{\mathbf{D}_f(S_1,\bar{S}_1)}'
            ant_Yq = r'T_{\hat{\mathbf{D}}_f(S_1,\bar{S}_1)}'
        elif sgn.startswith('t_'):
            ant_Xp = r'T_{\mathbf{D}_{\cdot}(S_1,\bar{S}_1)}'
            ant_Yq = r'T_{\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)}'
        elif sgn.startswith('Ds') or (sgn == 'D'):
            ant_Xp = r'\mathbf{D}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}(S_1,\bar{S}_1)'
        elif sgn.startswith('Df'):
            ant_Xp = r'\mathbf{D}_f(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_f(S_1,\bar{S}_1)'
        else:
            ant_Xp = r'\mathbf{D}_{\cdot}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)'
        return ant_Xp, ant_Yq

    def subfig_conv_multivar(self, dframe, tag_conv, fgn,
                             tag_hfm):  # ,obtn_fig_conv_multivar
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)

        tag_multivar = tag_conv['multivar'] + tag_conv['tim'][: -1]  # +tag_conv['sa1']+tag_conv['sa2']
        # df_alt = self.sub_dat_multivar(
        #     dframe, nb_set, id_set, tag_multivar)
        tag1 = tag_hfm['dist'][0][:1] + tag_hfm['dist'][0][
            2:3] + tag_hfm['dist'][0][4:6] + tag_hfm['tim'][
            0][:2] + tag_hfm['tim'][0][5:7]
        tag2 = tag_hfm['dist'][1][:1] + tag_hfm['dist'][1][
            2:3] + tag_hfm['dist'][1][4:6] + tag_hfm['tim'][
            1][:2] + tag_hfm['tim'][1][5:7]
        df_alt = self.obtain_multival_senatt(
            dframe, id_set, tag_multivar, tag_conv[
                'sa1'] + tag1, tag_conv['sa2'] + tag2)
        tag_tim = [tag_multivar[7: 9], tag_multivar[9 + 7: 18],
                   tag_multivar[18 + 7: 27], tag_multivar[27 + 7: 36]]
        tag_tim = np.array(tag_tim).T.tolist()
        fgn += '_multivar'  # _mvc
        # self.sub_plt_tim(df_alt, tag_tim, fgn + '_tim', 't_Ds')
        # self.sub_plt_tim(df_alt, tag_tim, fgn + '_timDf', 't_Df')
        # self.sub_plt_tim(df_alt, tag_tim, fgn + '_timdot')
        tag_val = np.array([tag_multivar[i] for i in [
            0, 2, 9, 11, 18, 20, 27, 29, ]]).reshape(4, 2).T.tolist()
        tag_avg = np.array([tag_multivar[i] for i in [
            1, 3, 10, 12, 19, 21, 28, 30, ]]).reshape(4, 2).T.tolist()
        # self.sub_plt_val(df_alt, tag_val, fgn + '_val', 'Ds')
        # self.sub_plt_avg(df_alt, tag_avg, fgn + '_avg', 'Ds_avg')
        self.thread_multivar(df_alt, tag_tim, fgn, tag_val, tag_avg)

        tag_sa1 = tag_conv['sa1']
        # tmp = self.sub_dat_sen_att(
        #     dframe,  # or df_alt,
        #     nb_set, id_set, tag_sa1, tag_conv['sa2'])
        tag_tim = np.array([tag_sa1[7: 9], tag_sa1[9 + 7: 18], tag_sa1[
            18 + 7: 27], tag_sa1[27 + 7: 36],
            tag1[4:][:2], tag1[4:][2:]]).T.tolist()
        fgn = fgn.replace('multivar', 'whole')
        # self.sub_plt_tim(tmp, tag_tim, fgn + '_tim', 't_Ds')
        # self.sub_plt_tim(tmp, tag_tim, fgn + '_timDf', 't_Df')
        # self.sub_plt_tim(tmp, tag_tim, fgn + '_timdot', 't_D')  # _D,Dd
        tag_val = np.array([tag_sa1[i] for i in [
            0, 2, 9, 11, 18, 20, 27, 29, ]] + tag1[
            :4]).reshape(4 + 2, 2).T.tolist()
        tag_avg = np.array([tag_sa1[i] for i in [
            1, 3, 10, 12, 19, 21, 28, 30, ]]).reshape(4, 2).T.tolist()
        # self.sub_plt_val(tmp, tag_val, fgn + '_val', 'Ds')
        # self.sub_plt_avg(tmp, tag_avg, fgn + '_avg', 'Ds_avg')
        self.thread_singvar(df_alt, tag_tim, fgn, tag_val, tag_avg)
        del tag_tim, tag_val, tag_avg, fgn, tag_sa1  # , tmp
        del df_alt, tag_multivar, nb_set, id_set
        return

    def sub_dat_sen_att(self, dframe, nb_set, id_set, tag_sa1, tag_sa2):
        df_raw = dframe[tag_sa1].iloc[id_set[0] + 1: id_set[0 + 1]]
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        for i in range(1, nb_set):
            df_tmp = dframe[tag_sa1].iloc[id_set[i] + 1: id_set[i + 1]]
            df_alt = dframe[tag_sa2].iloc[id_set[i] + 1: id_set[i + 1]]
            df_alt = df_alt.rename(columns = columns)
            df_raw = pd.concat([df_raw, df_tmp, df_alt], axis = 0)
            del df_tmp, df_alt
        return df_raw.reset_index(drop = True)

    def thread_singvar(self, tmp, tag_tim, fgn, tag_val, tag_avg,
                       verbose = False):
        if verbose:
            self.sub_plt_tim(tmp, tag_tim[0], fgn + '_tim', 't_Ds')
            self.sub_plt_tim(tmp, tag_tim[1], fgn + '_timf', 't_Df')
        df_new = tmp[tag_tim[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(*tag_tim)})
        df_new = pd.concat([tmp[tag_tim[0]], df_new], axis = 0)
        self.sub_plt_tim(df_new, tag_tim[0], fgn + '_timdt')
        if verbose:
            self.sub_plt_val(tmp, tag_val[0], fgn + '_val', 'Ds')
            self.sub_plt_val(tmp, tag_val[1], fgn + '_valf', 'Df')
        df_new = tmp[tag_val[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(*tag_val)})
        df_new = pd.concat([tmp[tag_val[0]], df_new], axis = 0)
        self.sub_plt_val(df_new, tag_val[0], fgn + '_valdt')
        if verbose:
            self.sub_plt_avg(tmp, tag_avg[0], fgn + '_avg', 'Ds_avg')
            self.sub_plt_avg(tmp, tag_avg[1], fgn + '_avgf', 'Df_avg')
        df_new = tmp[tag_avg[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(*tag_avg)})
        df_new = pd.concat([tmp[tag_avg[0]], df_new], axis = 0)
        self.sub_plt_avg(df_new, tag_avg[0], fgn + '_avgdt')
        return

    def thread_multivar(self, df_alt, tag_tim, fgn,  # subcore_multivar:
                        tag_val, tag_avg, verbose = False):
        if verbose:
            self.sub_plt_tim(df_alt, tag_tim[0], fgn + '_tim', 't_Ds')
            self.sub_plt_tim(df_alt, tag_tim[1], fgn + '_timDf', 't_Df')
        df_tmp = df_alt[tag_tim[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(tag_tim[0], tag_tim[1])})
        df_tmp = pd.concat([df_alt[tag_tim[0]], df_tmp], axis = 0)
        self.sub_plt_tim(df_tmp, tag_tim[0], fgn + '_timdot')
        if verbose:
            self.sub_plt_val(df_alt, tag_val[0], fgn + '_val', 'Ds')
            self.sub_plt_val(df_alt, tag_val[1], fgn + '_valDf', 'Df')
        df_tmp = df_alt[tag_val[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(tag_val[0], tag_val[1])})
        df_tmp = pd.concat([df_alt[tag_val[0]], df_tmp], axis = 0)
        self.sub_plt_val(df_tmp, tag_val[0], fgn + '_valdot')
        if verbose:
            self.sub_plt_avg(df_alt, tag_avg[0], fgn + '_avg', 'Ds_avg')
            self.sub_plt_avg(df_alt, tag_avg[1], fgn + '_avgDf', 'Df_avg')
        df_tmp = df_alt[tag_avg[1]].rename(columns = {
            t2: t1 for t1, t2 in zip(tag_avg[0], tag_avg[1])})
        df_tmp = pd.concat([df_alt[tag_avg[0]], df_tmp], axis = 0)
        self.sub_plt_avg(df_tmp, tag_avg[0], fgn + '_avgdot')
        return

    def hfm_dict_tim(self, fgn, sgn):
        if sgn == 't_Ds' or (sgn == 't_D'):
            ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
            if 'multivar' not in fgn:
                ant_X = r'T_{\mathbf{D}_a(S,a_i)}'
                ant_Y = r'T_{\hat{\mathbf{D}}_a(S,a_i)}'
                ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
        elif sgn == 't_Df':
            ant_X = r'T_{\mathbf{D}_{f,\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S)} })'
            if 'multivar' not in fgn:
                ant_X = r'T_{\mathbf{D}_{f,a}(S,a_i)}'
                ant_Y = r'T_{\hat{\mathbf{D}}_{f,a}(S,a_i)}'
                ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,a}(S,a_i)} }{ T_{\mathbf{D}_{f,a}(S,a_i)} })'
        else:  # if sgn.startswith('t_'):
            ant_X = r'T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)}'
            ant_Y = r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)} })'
            if 'multivar' not in fgn:
                ant_X = r'T_{\mathbf{D}_{\cdot,a}(S,a_i)}'
                ant_Y = r'T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)}'
                ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)} }{ T_{\mathbf{D}_{\cdot,a}(S,a_i)} })'
        return ant_X, ant_Y, ant_Z

    def hfm_dict_val(self, fgn, sgn):
        if sgn.startswith('Ds') or (sgn == 'D'):
            ant_X = r'\mathbf{D}_{\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_a(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_a(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
        elif sgn.startswith('Df'):
            ant_X = r'\mathbf{D}_{f,\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{f,\mathbf{a}}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S) }{ \mathbf{D}_{f,\mathbf{a}}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_{f,a}(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_{f,a}(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}(S,a_i) }{ \mathbf{D}_{f,a}(S,a_i) }-1'
        else:
            ant_X = r'\mathbf{D}_{\cdot,\mathbf{a}}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_{\cdot,a}(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}(S,a_i) }{ \mathbf{D}_{\cdot,a}(S,a_i) }-1'
        return ant_X, ant_Y, ant_Z

    def hfm_dict_avg(self, fgn, sgn):
        if sgn.startswith('Ds') or (sgn == 'D_avg'):
            ant_X = r'\mathbf{D}_{\mathbf{a}}^\text{avg}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_a^\text{avg}(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
        elif sgn.startswith('Df'):
            ant_X = r'\mathbf{D}_{f,\mathbf{a}}^\text{avg}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{f,\mathbf{a}}^\text{avg}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_{f,a}^\text{avg}(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{f,a}^\text{avg}(S,a_i) }-1'
        else:
            ant_X = r'\mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S)'
            ant_Y = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^\text{avg}(S)'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S) }-1'
            if 'multivar' not in fgn:
                ant_X = r'\mathbf{D}_{\cdot,a}^\text{avg}(S,a_i)'
                ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i)'
                ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{\cdot,a}^\text{avg}(S,a_i) }-1'
        return ant_X, ant_Y, ant_Z

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn = 't_D.'):
        # tag_tim = tag_col[-4:]  # rmk='multivar',
        # scat_X = df_tmp[tag_tim[0]].values.astype(DTY_FLT)
        # scat_Y = [df_tmp[tag_tim[1]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[2]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[3]].values.astype(DTY_FLT)]

        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)

        # if sgn == 't_Ds' or (sgn == 't_D'):
        #     scat_X = df_tmp[tag[0][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][3]].values.astype(DTY_FLT)]
        #     ant_X = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
        #     if 'multivar' not in fgn:
        #         ant_X = r'T_{\mathbf{D}_a(S,a_i)}'
        #         ant_Y = r'T_{\hat{\mathbf{D}}_a(S,a_i)}'
        #         ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
        # elif sgn == 't_Df':
        #     scat_X = df_tmp[tag[1][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][3]].values.astype(DTY_FLT)]
        #     ant_X = r'T_{\mathbf{D}_{f,\mathbf{a}}(S)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S)} })'
        #     if 'multivar' not in fgn:
        #         ant_X = r'T_{\mathbf{D}_{f,a}(S,a_i)}'
        #         ant_Y = r'T_{\hat{\mathbf{D}}_{f,a}(S,a_i)}'
        #         ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,a}(S,a_i)} }{ T_{\mathbf{D}_{f,a}(S,a_i)} })'
        # else:
        #     scat_X = np.concatenate([
        #         df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][0]].values.astype(DTY_FLT)], axis=0)
        #     scat_Y = [np.concatenate([
        #         df_tmp[tag[0][1]].values,
        #         df_tmp[tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][2]].values,
        #             df_tmp[tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][3]].values,
        #             df_tmp[tag[1][3]].values]).astype(DTY_FLT), ]
        #     ant_X = r'T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)}'
        #     ant_Y = r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)}'
        #     ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)} })'
        #     if 'multivar' not in fgn:
        #         ant_X = r'T_{\mathbf{D}_{\cdot,a}(S,a_i)}'
        #         ant_Y = r'T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)}'
        #         ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,a}(S,a_i)} }{ T_{\mathbf{D}_{\cdot,a}(S,a_i)} })'

        if 'multivar' in fgn:  # if rmk == 'multivar':
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StratRA)}'
        else:  # rmk=='sen-att'
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
        annotY = ['${}$'.format(ant_app), '${}$'.format(
            ant_cvg), '${}$'.format(ant_arr)]  # '${}$ (sec)'
        if 'multivar' not in fgn:
            # scat_Y.extend([df_tmp[tag[-2]].values.astype(DTY_FLT),
            #                df_tmp[tag[-1]].values.astype(DTY_FLT)])
            scat_Y.extend([df_tmp[tag[4]].values.astype(DTY_FLT),
                           df_tmp[tag[5]].values.astype(DTY_FLT)])
            ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        # annotY = ['${}$ {}'.format(ant_Y, ant_app[3:-1]),
        #           '${}$ {}'.format(ant_Y, ant_cvg[3:-1]),
        #           '${}$ {}'.format(ant_Y, ant_arr[3:-1])]
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty4c')  # 'sty4d')'sty4')

        # # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s',
        #     snspec='sty6c')  # 'sty6d')#'sty6')
        # del scat_Z, ant_Z, annot
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn = 'D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)

        # if sgn.startswith('Ds') or (sgn == 'D'):  # sgn=='Ds'
        #     scat_X = df_tmp[tag[0][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{\mathbf{a}}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_a(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_a(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
        # elif sgn.startswith('Df'):             # sgn == 'Df':
        #     scat_X = df_tmp[tag[1][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{f,\mathbf{a}}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{f,\mathbf{a}}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S) }{ \mathbf{D}_{f,\mathbf{a}}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_{f,a}(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_{f,a}(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}(S,a_i) }{ \mathbf{D}_{f,a}(S,a_i) }-1'
        # else:
        #     scat_X = np.concatenate([
        #         df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][0]].values.astype(DTY_FLT)], axis=0)
        #     scat_Y = [np.concatenate([
        #         df_tmp[tag[0][1]].values,
        #         df_tmp[tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][2]].values,
        #             df_tmp[tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][3]].values,
        #             df_tmp[tag[1][3]].values]).astype(DTY_FLT), ]
        #     ant_X = r'\mathbf{D}_{\cdot,\mathbf{a}}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_{\cdot,a}(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}(S,a_i) }{ \mathbf{D}_{\cdot,a}(S,a_i) }-1'
        if 'multivar' in fgn:  # if rmk == 'multivar':
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist (StratES)'
            ant_arr = r'ExactDist (StratRA)'
        else:  # rmk=='sen-att'
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        annotY = [
            # '${}$'.format(ant_app), '${}$'.format(
            #     ant_cvg), '${}$'.format(ant_arr)]
            ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty3c')  # 'sty3d')'sty3b')

        # # scat_Z = [k / scat_X - 1. for k in scat_Y]
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s',
        #     snspec='sty6c')  # 'sty6d')#'sty6')
        # del scat_Z, ant_Z, annot
        return

    def sub_plt_avg(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_avg(fgn, sgn)

        # if sgn.startswith('Ds') or (sgn == 'D_avg'):  # =='Ds_avg'
        #     scat_X = df_tmp[tag[0][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[0][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[0][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{\mathbf{a}}^\text{avg}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_a^\text{avg}(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
        # elif sgn.startswith('Df'):              # sgn == 'Df_avg':
        #     scat_X = df_tmp[tag[1][0]].values.astype(DTY_FLT)
        #     scat_Y = [df_tmp[tag[1][1]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][2]].values.astype(DTY_FLT),
        #               df_tmp[tag[1][3]].values.astype(DTY_FLT)]
        #     ant_X = r'\mathbf{D}_{f,\mathbf{a}}^\text{avg}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{f,\mathbf{a}}^\text{avg}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_{f,a}^\text{avg}(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_{f,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{f,a}^\text{avg}(S,a_i) }-1'
        # else:
        #     scat_X = np.concatenate([
        #         df_tmp[tag[0][0]].values.astype(DTY_FLT),
        #         df_tmp[tag[1][0]].values.astype(DTY_FLT)], axis=0)
        #     scat_Y = [np.concatenate([
        #         df_tmp[tag[0][1]].values,
        #         df_tmp[tag[1][1]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][2]].values,
        #             df_tmp[tag[1][2]].values]).astype(DTY_FLT),
        #         np.concatenate([
        #             df_tmp[tag[0][3]].values,
        #             df_tmp[tag[1][3]].values]).astype(DTY_FLT), ]
        #     ant_X = r'\mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S)'
        #     ant_Y = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^\text{avg}(S)'
        #     ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S) }-1'
        #     if 'multivar' not in fgn:
        #         ant_X = r'\mathbf{D}_{\cdot,a}^\text{avg}(S,a_i)'
        #         ant_Y = r'\hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i)'
        #         ant_Z = r'\frac{ \hat{\mathbf{D}}_{\cdot,a}^\text{avg}(S,a_i) }{ \mathbf{D}_{\cdot,a}^\text{avg}(S,a_i) }-1'
        if 'multivar' in fgn:  # if rmk == 'multivar':
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'  # r'ExactDist (StratES)'
            ant_arr = r'ExactDist(StratRA)'  # r'ExactDist (StratRA)'
        else:  # rmk=='sen-att'
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        annotY = [
            # '${}$'.format(ant_app), '${}$'.format(
            #     ant_cvg), '${}$'.format(ant_arr)]
            ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty3c')  # 'sty3d')#'sty3b')

        # # scat_Z = [k / scat_X - 1. for k in scat_Y]
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s',
        #     snspec='sty6c')  # 'sty6d')#'sty6c')
        # del scat_Z, ant_Z, annot
        return


class ConvFig_5F_exact(ConvFig_5H_exact):  # ConvPlotF_init):
    # def schedule_mspaint(self, raw_dframe, pre='minmax'):
    #     csv_row_1 = unique_column(11 + 1 + 438 * 2)
    #     pdb.set_trace()
    #     return

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 1 + 438 * 2)
        pms = csv_row_1[: 11 + 1]
        tag_trn = csv_row_1[12: 12 + 438]
        tag_tst = csv_row_1[12 + 438:]
        return pms, tag_trn, tag_tst

    def incise_graph(self, tag):
        tag_norm = [tag[: 7], tag[7 + 7: 7 * 3]]
        tag_gf = [
            tag[21 + 6: 21 + 6 + 5] + tag[(21 + 11 * 4): 65 + 16],
            tag[21 + 11 + 6: 21 + 11 * 2] + tag[
                (21 + 44) + 16: 65 + 16 * 2]]
        tmp = [tag[21 + 11 * 4 + 16 * 4: (
            21 + 44 + 64) + 12 + 5 + 9 + 9 * 2 + 5],
            tag[129 + 49: 129 + 49 * 2]]
        tag_hfm = {
            'dist': [j[: 4] + j[12: 12 + 2] + j[17: 17 + 4] +
                     j[26: 26 + 4] + j[35: 35 + 4] for j in tmp],
            'df': [j[
                4: 7] + [j[12 + 2], ] + j[17 + 4: 17 + 7] + j[
                26 + 4: 26 + 7] + j[35 + 4: 35 + 7] for j in tmp],
            'tim': [j[
                7: 12] + j[12 + 3: 12 + 5] + j[17 + 7: 17 + 9] + j[
                26 + 7: 26 + 9] + j[35 + 7: 35 + 9 + 5] for j in tmp],
        }

        tmp = tag[(21 + 11 * 4 + 16 * 4) + 49 * 4: 129 + 196 + (
            (9 + 18) * 4 + 5)]  # 12+5+9+9*2+5=49
        tag_df_conv = {
            'multivar': tmp[: 9] + tmp[
                27: 27 + 9] + tmp[54: 54 + 9] + tmp[81: 81 + 9],
            'sa1': tmp[9: 18] + tmp[
                9 * 4: 9 * 5] + tmp[9 * 7: 9 * 8] + tmp[9 * 10: 9 * 11],
            'sa2': tmp[18: 27] + tmp[
                9 * 5: 9 * 6] + tmp[9 * 8: 9 * 9] + tmp[9 * 11: 9 * 12],
            'tim': tmp[9 * 12: 9 * 12 + 5], }
        del tmp  # pdb.set_trace()
        return tag_norm, tag_gf, tag_hfm, tag_df_conv


class ConvFig_5Iprev_exact(ConvFig_5H_exact):
    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 1 + 280 * 2)
        pms = csv_row_1[: 11 + 1]
        tag_trn = csv_row_1[12: 12 + 280]
        tag_tst = csv_row_1[12 + 280:]
        return pms, tag_trn, tag_tst

    def incise_graph(self, tag):
        tag_norm = [tag[: 7], tag[7 + 7: 7 * 3]]
        tag_gf = [tag[21: 21 + 5] + tag[31: 31 + 16],
                  tag[26: 21 + 10] + tag[47: 47 + 16]]
        tmp = [tag[63: 63 + (6 + 5 + 8 + 8 * 2 + 6) + 6],
               tag[63 + 47: 63 + 47 * 2]]
        tag_hfm = {'dist': [
            j[: 2] + j[6: 6 + 2] + j[11: 11 + 4] + j[19: 19 + 4] +
            j[27: 27 + 4] + j[35: 35 + 2] for j in tmp],
            'df': [j[2: 4] + [j[6 + 2], ] + j[11 + 4: 11 + 6] +
                   j[19 + 4: 19 + 6] + j[27 + 4: 27 + 6] +
                   j[35 + 2: 35 + 4] for j in tmp],
            'tim': [j[4: 6] + j[6 + 3: 6 + 5] + j[11 + 6: 11 + 8] +
                    j[19 + 6: 19 + 8] + j[27 + 6: 27 + 8] +
                    j[35 + 4: 35 + 6 + 6] for j in tmp]}
        # 'dist' sa:  DistDirect_bin{Ds,Df}, ApproxDist_bin{Ds,Df},
        #             StratVacant{Ds,Ds_avg,Df,Df_avg}, StratES{x4},
        #             StratRA{x4}, EarlyBreak/EffHD_bin{Ds,Df}
        # 'df' sa:    DistDirect_bin{df_prev,df},
        #             ApproxDist_bin{df_prev}, StratVacant{df,df_avg},
        #             StratES{x2}, StratRA{x2}, EffHD_bin{df_prev,df}
        # 'tim' sa:   DistDirect_bin{t_Ds,t_Df}, ApproxDist_bin{x2},
        #             StratVacant{x2}, StratES{x2}, StratRA{x2},
        #             EffHD_bin{x2}         # siz=18,11,12+6=18

        tmp = tag[(21 + 10 + 16 * 2) + 47 * 2: 63 + 94 + (
            9 + 18 + 8 + 16 + (8 + 16) * 2 + 6 + 12 + 6)]
        tag_df_conv = {'multivar': tmp[: 9] + tmp[27: 27 + 8] + tmp[
            51: 51 + 8] + tmp[75: 75 + 8] + tmp[99: 99 + 6],
            'sa1': tmp[9: 18] + tmp[35: 35 + 8] + tmp[
            59: 59 + 8] + tmp[83: 83 + 8] + tmp[105: 105 + 6],
            'sa2': tmp[18: 27] + tmp[43: 43 + 8] + tmp[
            67: 67 + 8] + tmp[91: 91 + 8] + tmp[111: 111 + 6],
            'tim': tmp[117: 117 + 6], }
        # 'multivar':   DistDirect_multivar{Ds,Ds_avg,Df,Df_avg,df_prev,
        #                 df,df_avg,t_Ds,t_Df}, DistExtend_multivar_mp{
        #                 Ds,Ds_avg,Df,Df_avg,df,df_avg,t_Ds,t_Df}/Vacant,
        #               Exact-StratES{x8}, Exact-StratRA{x8},
        #               EffHD_multivar{Ds,Df,df_prev,df,t_Ds,t_Df}
        #                       # 9+8+8*3+6=17+24+6=41+6=47
        #                       # 9+8+8*2+6=17+16+6=39 corrected
        # 'sa':         DistDirect_nonbin{x9}, DistApprox_nonbin{x8},
        #               StratES{x8}, StratRA{x8}, EffHD_nonbin{x6}
        #                       # 9+8+16+6=17+22=39
        # 'tim':        T(DistDirect_multivar), T(EffExact-Vacant),
        #               T(EffExact-StratES), T(EffExact-StratRA),
        #               T(EarlyBreak _multivar), T(comp. acc+extGrp)
        del tmp
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

    def subfig_conv_singvar(self, dframe, tag_hfm, tag_conv, fgn):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_sa1 = tag_conv['sa1'][: 9]
        tag_sa2 = tag_conv['sa2'][: 9]
        tag_sa1 = tag_hfm['dist'][0] + tag_sa1[: 4] + tag_hfm[
            'tim'][0][: -6] + tag_sa1[-2:]
        tag_sa2 = tag_hfm['dist'][1] + tag_sa2[: 4] + tag_hfm[
            'tim'][1][: -6] + tag_sa2[-2:]
        # tag_sa*: DistDirect_bin x2, ApproxDist_bin x2, StratVacant
        #          x4, StratES x4, StratRA x4, EarlyBreak x2,
        #          DistDirect_nonbin x4            # .siz=6+12+4=22
        # tag_tim: DistDirect_bin x2, ApproxDist_bin x2, StratVacant
        #          x2, StratES x2, StratRA x2, EffHD_bin x2,
        #          DistDirect_nonbin x2            # .siz=12+2  =14

        tag_sa1 = tag_sa1[: 4] + tag_sa1[16: 18] + tag_sa1[
            4: 16] + tag_sa1[18: 18 + 4 + 4] + tag_sa1[
            32: 34] + tag_sa1[26: 26 + 6] + tag_sa1[34:]
        tag_sa2 = tag_sa2[: 4] + tag_sa2[16: 18] + tag_sa2[
            4: 4 + 12] + tag_sa2[18: 18 + 4 + 4] + tag_sa2[
            32: 34] + tag_sa2[26: 26 + 6] + tag_sa2[34:]
        # tmp = self.sub_dat_sen_att(
        #     dframe, nb_set, id_set, tag_sa1, tag_sa2)
        tmp = self.obtain_binval_senatt(
            dframe, id_set, [], tag_sa1, tag_sa2)
        pdb.set_trace()
        tag_tim = tag_sa1[6 + 12 + 4:]
        tag_tim = np.array(tag_tim).reshape(-1, 2).T.tolist()
        fgn += '_singvar'
        tag_val = tag_sa1[: 6 + 12 + 4]
        tag_avg = np.array([tag_val[i] for i in [
            7, 9, 11, 13, 15, 17, 19, 21]]).reshape(-1, 2).T.tolist()
        tag_val = np.array([tag_val[i] for i in [
            0, 1, 2, 3, 4, 5,
            6, 8, 10, 12, 14, 16, 18, 20]]).reshape(-1, 2).T.tolist()
        pdb.set_trace()
        self.thread_previous(tmp, tag_tim, fgn, tag_val, tag_avg)
        return

    def sub_plt_tim_prev(self, df_tmp, tag_tim, fgn, sgn = 't_D.'):
        # scat_X = df_tmp[tag_tim[6]].values.astype(DTY_FLT)
        # scat_Y = [df_tmp[tag_tim[2]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[3]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[4]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[5]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[0]].values.astype(DTY_FLT),
        #           df_tmp[tag_tim[1]].values.astype(DTY_FLT)]
        scat_X = df_tmp[tag_tim[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[2]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[3]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[4]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[5]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[1]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[6]].values.astype(DTY_FLT)]

        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)
        # if sgn.startswith('t_Ds') or (sgn == 't_D'):
        #     ant_Xp = r'T_{\mathbf{D}(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)}'
        # elif sgn.startswith('t_Df'):
        #     ant_Xp = r'T_{\mathbf{D}_f(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}_f(S_1,\bar{S}_1)}'
        # else:
        #     ant_Xp = r'T_{\mathbf{D}_{\cdot}(S_1,\bar{S}_1)}'
        #     ant_Yq = r'T_{\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)}'
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
        # annotY = ['${}$'.format(ant_Xp), '${}$'.format(ant_Yq),
        #           # 'EarlyBreak', 'ApproxDist', 'StratES', 'StratRA']
        #           r'$T_{EarlyBreak}$', r'$T_{ApproxDist}$',
        #           r'$T_{StratES}$', r'$T_{StratRA}$']
        annotY = [r'$T_{EarlyBreak}$', r'$T_{ApproxDist}$',
                  r'$T_{StratES}$', r'$T_{StratRA}$', '${}$'.format(
                      ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Y), '${}$ (bin-val)$= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty4')
        # kws = {'snspec': 'sty6'}
        # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn = 'D.'):
        # scat_X = df_tmp[tag[5]].values.astype(DTY_FLT)
        # scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
        #           df_tmp[tag[3]].values.astype(DTY_FLT),
        #           df_tmp[tag[4]].values.astype(DTY_FLT),
        #           df_tmp[tag[5]].values.astype(DTY_FLT),
        #           df_tmp[tag[0]].values.astype(DTY_FLT),
        #           df_tmp[tag[1]].values.astype(DTY_FLT)]
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[6]].values.astype(DTY_FLT)]

        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)
        if sgn.startswith('Ds') or (sgn == 'D'):
            ant_Xp = r'\mathbf{D}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}(S_1,\bar{S}_1)'
        elif sgn.startswith('Df'):
            ant_Xp = r'\mathbf{D}_f(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_f(S_1,\bar{S}_1)'
        else:
            ant_Xp = r'\mathbf{D}_{\cdot}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)'
        annotY = [  # '${}$'.format(ant_Xp),'${}$'.format(ant_Yq),
            'EarlyBreak', 'ApproxDist', 'StratES', 'StratRA',
            '${}$'.format(ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$'.format(ant_Xp), '${}$ (bin-val)'.format(
            ant_Y), '${}$ (bin-val) $= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty3b')
        # kws = {'snspec': 'sty6'}
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def sub_plt_avg_prev(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[3]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[0]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_avg(fgn, sgn)
        annotY = [r'ApproxDist', r'StratES', r'StratRA']
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$ (bin-val)'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec = 'sty3c')      # 'sty3b','sty3d')
        # kws = {'snspec': 'sty6c'}  # 'sty6'}#'sty6d'}
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        # del scat_Z, kws, ant_Z
        return

    def subfig_conv_multivar(self, dframe, tag_conv, fgn,
                             tag_hfm=None):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_multivar = tag_conv['multivar'] + tag_conv['tim'][: -1]
        # df_alt = self.sub_dat_multivar(
        #     dframe, nb_set, id_set, tag_multivar)
        tag1 = tag_hfm['dist'][0][:4] + tag_hfm['dist'][0][
            -2:] + tag_hfm['tim'][0][:4] + tag_hfm['tim'][0][-8:-6]
        tag2 = tag_hfm['dist'][1][:4] + tag_hfm['dist'][1][
            -2:] + tag_hfm['tim'][1][:4] + tag_hfm['tim'][1][-8:-6]
        df_alt = self.obtain_multival_senatt(
            dframe, id_set, tag_multivar,
            tag_conv['sa1'] + tag1, tag_conv['sa2'] + tag2)
        pdb.set_trace()
        tag_tim = [tag_multivar[7: 9], tag_multivar[33 + 4: 33 + 6],
                   tag_multivar[9 + 6: 9 + 8], tag_multivar[17 + 6: 17 + 8],
                   tag_multivar[25 + 6: 25 + 8], ]  # EarlyBreak,Vacant|..
        # tag_tim = [tag_multivar[7:9], tag_multivar[9 + 6: 9 + 8],
        #            tag_multivar[17 + 6:17 + 8],
        #            tag_multivar[25 + 6:25 + 8],
        #            tag_multivar[33 + 4:33 + 6]]  # StratES/RA,EarlyBreak
        tag_tim = np.array(tag_tim).T.tolist()

        tag_val = np.array([tag_multivar[i] for i in [
            0, 2, 33, 34,
            9, 11, 17, 19, 25, 27]]).reshape(5, 2).T.tolist()
        tag_avg = np.array([tag_multivar[i] for i in [
            1, 3, 10, 12, 18, 20, 26, 28]]).reshape(4, 2).T.tolist()
        # self.subcore_multivar(df_alt, tag_tim)
        # self.thread_multivar(df_alt, tag_tim, fgn)
        fgn += '_multivar'
        self.thread_multivar(df_alt, tag_tim, fgn, tag_val, tag_avg)

        tag_sa1 = tag_conv['sa1']
        # tmp = self.sub_dat_sen_att(dframe, nb_set, id_set, tag_sa1,
        #                            tag_conv['sa2'])
        tag_tim = np.array([tag_sa1[7: 9], tag_sa1[9 + 6: 9 + 8],
                            tag_sa1[17 + 6: 17 + 8],
                            tag_sa1[25 + 6: 25 + 8],
                            tag_sa1[33 + 4: 33 + 6],
                            tag1[-6:][:2], tag1[-6:][2:4],
                            tag1[-6:][4:]]).T.tolist()
        tag_val = np.array([tag_sa1[i] for i in [
            0, 2, 33, 34,
            9, 11, 17, 19, 25, 27]] + tag1[
            :6]).reshape(5 + 3, 2).T.tolist()
        tag_avg = np.array([tag_sa1[i] for i in [
            1, 3, 10, 12, 18, 20, 26, 28]]).reshape(4, 2).T.tolist()
        fgn = fgn.replace('multivar', 'whole')
        self.thread_singvar(df_alt, tag_tim, fgn, tag_val, tag_avg)
        pdb.set_trace()
        return

    # def hfm_dict_multivar(self, sgn):
    #     return ant_X, ant_Y, ant_Z
    # def hfm_dict_sen_att(self):
    #     return ant_X, ant_Y, ant_Z

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn = 't_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        # ant_X, ant_Y, ant_Z = self.hfm_dict_multivar(sgn)
        # if 'multivar' not in fgn:
        #     ant_X, ant_Y, ant_Z = self.hfm_dict_sen_att(sgn)
        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)

        if 'multivar' in fgn:
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StratRA)}'
        else:
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
        ant_eff = r'T_{EarlyBreak}'
        # annotY = ['${}$ {}'.format(ant_Y, ant_eff[3:-1]),
        #           '${}$ {}'.format(ant_Y, ant_app[3:-1]),
        #           '${}$ {}'.format(ant_Y, ant_cvg[3:-1]),
        #           '${}$ {}'.format(ant_Y, ant_arr[3:-1])]
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        annotY = [  # '${}$ (multival)'.format(ant_eff),
            '${}$ (multi)'.format(ant_eff), '${}$'.format(ant_app),
            '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]

        if 'multivar' not in fgn:
            # scat_Y.extend([df_tmp[tag[-3]].values.astype(DTY_FLT),
            #                df_tmp[tag[-2]].values.astype(DTY_FLT),
            #                df_tmp[tag[-1]].values.astype(DTY_FLT)])
            scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                           df_tmp[tag[6]].values.astype(DTY_FLT)])
            scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
            ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
            annotY[0] = '${}$'.format(ant_eff)  # r'T_{EarlyBreak}')

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec ='sty4')  # 'sty4d')
        # # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s',
        #     snspec ='sty6')  # 'sty6d')
        # del scat_Z, ant_Z, annot
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn='D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        # ant_X, ant_Y, ant_Z = self.hfm_dict_multivar(sgn)
        # if 'multivar' not in fgn:
        #     ant_X, ant_Y, ant_Z = self.hfm_dict_sen_att(sgn)
        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)

        if 'multivar' in fgn:
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'
            ant_arr = r'ExactDist(StratRA)'
        else:
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        ant_eff = 'EarlyBreak'
        annotY = [r'EarlyBreak$^{(multi)}$',  # ant_eff,multival
                  ant_app, ant_cvg, ant_arr]  # '${}$'
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]

        if 'multivar' not in fgn:
            scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                           df_tmp[tag[6]].values.astype(DTY_FLT)])
            # pdb.set_trace()
            scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
            ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
            annotY[0] = ant_eff  # '${}$'.format(ant_eff[:10])

        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec = 'sty3b')  # 'sty3d')
        # # scat_Z = [k / scat_X - 1. for k in scat_Y]
        # scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z, annotY, annot, fgn + '_s',
        #     snspec = 'sty6')  # 'sty6d')
        # del scat_Z, ant_Z, annot
        return

    # def sub_plt_avg(self, df_tmp, tag, fgn, sgn='D._avg'):
    #     scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
    #     scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
    #               df_tmp[tag[2]].values.astype(DTY_FLT),
    #               df_tmp[tag[3]].values.astype(DTY_FLT)]
    #     ant_X, ant_Y, ant_Z = self.hfm_dict_avg(fgn, sgn)
    #
    #     if 'multivar' in fgn:
    #         ant_app = r'ExtendDist'
    #         ant_cvg = r'ExactDist(StratES)'
    #         ant_arr = r'ExactDist(StratRA)'
    #     else:
    #         ant_app = r'ApproxDist'
    #         ant_cvg = r'StratES'
    #         ant_arr = r'StratRA'
    #     annotY = [ant_app, ant_cvg, ant_arr]  # '${}$'.format()
    #     annot = ['${}$'.format(ant_X), '${}$'.format(
    #         ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
    #     multi_lin_reg_without_distr(
    #         scat_X, scat_Y, annotY, annot, fgn,
    #         snspec='sty3b')  # 'sty3d')
    #     # scat_Z = [k / scat_X - 1. for k in scat_Y]
    #     scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
    #     annot[1] = '${}$'.format(ant_Z)
    #     multi_lin_reg_without_distr(
    #         scat_X, scat_Z, annotY, annot, fgn + '_s',
    #         snspec='sty6c')  # 'sty6d')
    #     del scat_Z, ant_Z, annot
    #     return


class ConvFig_5I_exact(ConvFig_5Iprev_exact):
    def incise_graph(self, tag):
        tag_norm = [tag[: 7], tag[7 + 7: 7 * 3]]
        tag_gf = [tag[21: 21 + 5] + tag[31: 31 + 16],
                  tag[26: 21 + 10] + tag[47: 47 + 16]]
        tmp = [tag[63: 63 + (6 + 5 + 6 + 8 + 8 * 2) + 6],
               tag[63 + 47: 63 + 47 * 2]]
        tag_hfm = {'dist': [
            j[: 2] + j[6: 6 + 2] + j[11: 11 + 2] + j[17: 17 + 4] +
            j[25: 25 + 4] + j[33: 33 + 4] for j in tmp],
            'df': [j[2: 4] + [j[6 + 2], ] + j[11 + 2: 11 + 4] +
                   j[17 + 4: 17 + 6] + j[25 + 4: 25 + 6] +
                   j[33 + 4: 33 + 6] for j in tmp],
            'tim': [j[4: 6] + j[6 + 3: 6 + 5] + j[11 + 4: 11 + 6] +
                    j[17 + 6: 17 + 8] + j[25 + 6: 25 + 8] +
                    j[33 + 6: 33 + 8 + 6] for j in tmp]}
        # 'dist' sa:  DistDirect_bin{Ds,Df}, ApproxDist_bin{Ds,Df},
        #             EarlyBreak/EffHD_bin{Ds,Df}, StratVacant{Ds,
        #             Ds_avg,Df,Df_avg}, StratES{x4}, StratRA{x4}
        # 'df' sa:    DistDirect_bin{df_prev,df}, ApproxDist_bin{
        #             df_prev}, EffHD_bin{df_prev, df}, StratVacant{
        #             df,df_avg}, StratES{x2}, StratRA{x2}
        # 'tim' sa:   DistDirect_bin{t_Ds,t_Df}, ApproxDist_bin{x2},
        #             EffHD_bin{x2}, StratVacant{x2}, StratES{x2},
        #             StratRA{x2}               # siz=18,11,12+6=18

        tmp = tag[(21 + 10 + 16 * 2) + 47 * 2: 63 + 94 + (
            9 + 18 + 6 + 12 + 8 + 16 + (8 + 16) * 2 + 6)]
        tag_df_conv = {'multivar': tmp[: 9] + tmp[27: 27 + 6] + tmp[
            45: 45 + 8] + tmp[69: 69 + 8] + tmp[93: 93 + 8],
            'sa1': tmp[9: 18] + tmp[33: 33 + 6] + tmp[
            53: 53 + 8] + tmp[77: 77 + 8] + tmp[101: 101 + 8],
            'sa2': tmp[18: 27] + tmp[39: 39 + 6] + tmp[
            61: 61 + 8] + tmp[85: 85 + 8] + tmp[109: 109 + 8],
            'tim': tmp[117: 117 + 6], }
        # 'multivar':   DistDirect_multivar{Ds,Ds_avg,Df,Df_avg,df_prev,
        #                 df,df_avg,t_Ds,t_Df}, EffHD_multivar{Ds,Df,
        #                   df_prev,df,t_Ds,t_Df}, DistExtend_multivar_mp{
        #                 Ds,Ds_avg,Df,Df_avg,df,df_avg,t_Ds,t_Df}/Vacant,
        #               Exact-StratES{x8}, Exact-StratRA{x8}
        #                       # 9+6+8+8*2=15+24=39 corrected
        # 'sa':         DistDirect_nonbin{x9}, EffHD_nonbin{x6},
        #               DistApprox_nonbin{x8}, StratES{x8}, StratRA{x8}
        #                       # 9+6+8+16=15+24=39
        # 'tim':        T(DistDirect_multivar), T(EarlyBreak _multivar),
        #               T(EffExact-Vacant), T(EffExact-StratES),
        #               T(EffExact-StratRA), T(comp. acc+extGrp)
        del tmp  # pdb.set_trace()
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

    def subfig_conv_singvar(self, dframe, tag_hfm, tag_conv, fgn):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_sa1 = tag_conv['sa1'][: 9]
        tag_sa2 = tag_conv['sa2'][: 9]
        tag_sa1 = tag_hfm['dist'][0] + tag_sa1[: 4] + tag_hfm[
            'tim'][0][: -6] + tag_sa1[-2:]
        tag_sa2 = tag_hfm['dist'][1] + tag_sa2[: 4] + tag_hfm[
            'tim'][1][: -6] + tag_sa2[-2:]
        # tag_sa*: DistDirect_bin x2, ApproxDist_bin x2, EarlyBreak
        #          x2, StratVacant x4, StratES x4, StratRA x4,
        #          DistDirect_nonbin x4            # .siz=6+12+4=22
        # tag_tim: DistDirect_bin x2, ApproxDist_bin x2, EffHD_bin
        #          x2, StratVacant x2, StratES x2, StratRA x2,
        #          DistDirect_nonbin x2            # .siz=12+2  =14

        # tmp = self.sub_dat_sen_att(
        #     dframe, nb_set, id_set, tag_sa1, tag_sa2)
        # tmp = self.obtain_multival_senatt(
        #     dframe, id_set, [], tag_sa1, tag_sa2)
        tmp = self.obtain_binval_senatt(
            dframe, id_set, [], tag_sa1, tag_sa2)
        tag_tim = tag_sa1[6 + 12 + 4:]
        tag_tim = np.array(tag_tim).reshape(-1, 2).T.tolist()
        fgn += '_singvar'
        tag_val = tag_sa1[: 6 + 12 + 4]
        tag_avg = np.array([tag_val[i] for i in [
            7, 9, 11, 13, 15, 17, 19, 21]]).reshape(-1, 2).T.tolist()
        tag_val = np.array([tag_val[i] for i in [
            0, 1, 2, 3, 4, 5,
            6, 8, 10, 12, 14, 16, 18, 20]]).reshape(-1, 2).T.tolist()
        self.thread_previous(tmp, tag_tim, fgn, tag_val, tag_avg)
        return

    def subfig_conv_multivar(self, dframe, tag_conv, fgn,
                             tag_hfm=None):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_multivar = tag_conv['multivar'] + tag_conv['tim'][: -1]
        # df_alt = self.sub_dat_multivar(
        #     dframe, nb_set, id_set, tag_multivar)
        # df_alt = self.obtain_multival_senatt(
        #     dframe, id_set, tag_multivar, tag_conv[
        #         'sa1'], tag_conv['sa2'])
        tag1 = tag_hfm['dist'][0][:6] + tag_hfm['tim'][0][:6]
        tag2 = tag_hfm['dist'][1][:6] + tag_hfm['tim'][1][:6]
        df_alt = self.obtain_multival_senatt(
            dframe, id_set, tag_multivar,
            tag_conv['sa1'] + tag1, tag_conv['sa2'] + tag2)
        tag_tim = [tag_multivar[7: 9], tag_multivar[9 + 4: 9 + 6],
                   tag_multivar[15 + 6: 15 + 8], tag_multivar[
                   23 + 6: 23 + 8], tag_multivar[31 + 6: 31 + 8]]
        tag_tim = np.array(tag_tim).T.tolist()

        tag_val = np.array([tag_multivar[i] for i in [
            0, 2, 9, 10,
            15, 17, 23, 25, 31, 33]]).reshape(5, 2).T.tolist()
        tag_avg = np.array([tag_multivar[i] for i in [
            1, 3, 16, 18, 24, 26, 32, 34]]).reshape(4, 2).T.tolist()
        fgn += '_multivar'
        self.thread_multivar(df_alt, tag_tim, fgn, tag_val, tag_avg)

        tag_sa1 = tag_conv['sa1']
        # tmp = self.sub_dat_sen_att(dframe, nb_set, id_set, tag_sa1,
        #                            tag_conv['sa2'])
        tmp = df_alt
        tag_tim = np.array([tag_sa1[7: 9], tag_sa1[9 + 4: 9 + 6],
                            tag_sa1[15 + 6: 15 + 8],
                            tag_sa1[23 + 6: 23 + 8],
                            tag_sa1[31 + 6: 31 + 8],
                            tag1[-6:][:2], tag1[-6:][2:4],
                            tag1[-6:][4:]]).T.tolist()
        tag_val = np.array([tag_sa1[i] for i in [
            0, 2, 9, 10,
            15, 17, 23, 25, 31, 33]] + tag1[
            :6]).reshape(8, 2).T.tolist()
        # tag_val = np.array([tag_sa1[i] for i in [
        #     0, 2, 9, 10,
        #     15, 17, 23, 25, 31, 33]]).reshape(5, 2).T.tolist()
        tag_avg = np.array([tag_sa1[i] for i in [
            1, 3, 16, 18, 24, 26, 32, 34]]).reshape(4, 2).T.tolist()
        fgn = fgn.replace('multivar', 'whole')
        # pdb.set_trace()
        self.thread_singvar(tmp, tag_tim, fgn, tag_val, tag_avg)
        return


class ConvFig_5Isimpl(ConvFig_5I_exact):
    def prepare_graph(self, omitted=True):
        tmp = 161 if not omitted else 123
        num = 21 + 10 + 16 * 2 + 25 * 2 + tmp
        csv_row_1 = unique_column(11 + 1 + num * 2)
        pms = csv_row_1[: 11 + 1]
        tag_trn = csv_row_1[12: 12 + num]
        tag_tst = csv_row_1[12 + num:]
        return pms, tag_trn, tag_tst

    def incise_graph(self, tag):
        tag_norm = [tag[: 7], tag[7 + 7: 7 * 3]]
        tag_gf = [tag[21: 21 + 5] + tag[31: 31 + 16],
                  tag[26: 21 + 10] + tag[47: 47 + 16]]
        tmp = [tag[63: 63 + (4 + 6 * 2 + 5 + 4)],
               tag[63 + 25: 63 + 25 * 2]]
        tag_hfm = {'dist': [j[:2] + j[
            4:6] + j[10:12] + j[16:18] for j in tmp],
            'df': [j[6:8] + j[12:14] + j[18:19] for j in tmp],
            'tim': [j[2:4] + j[
                8:10] + j[14:16] + j[19:21] for j in tmp]}
        # 'dist' sa:  Naive{Ds,Df}, EarlyBreak{x2}, DistDirect_bin{
        #             x2}, ApproxDist_bin{x2}
        # 'df' sa:    EarlyBreak_bin{df_prev,df}, DistDirect_bin{x2},
        #             ApproxDist_bin{df_prev}
        # 'tim' sa:   Naive{t_Ds,t_Df}, EarlyBreak{x2}, DistDirect
        #             _bin{x2}, ApproxDist_bin{x2}
        tmp = [tag[(63 + 25 * 2): 63 + 50 + (
            6 + 9 + 8 + 8 * 2)], tag[113 + 39: 113 + 39 * 2]]
        tag_df_conv = {'multivar': tag[113 + 39 * 2: -6],
                       # 'sa1': tmp[0], 'sa2': tmp[1],
                       'tim_mv': tag[-6:]}
        # 'multivar': EffHD_multivar{Ds,Df,df_prev,df,t_Ds,t_Df},
        #             DistDirect_multivar{Ds,Ds_avg,Df,Df_avg,df_prev,
        #             df,df_avg,t_Ds,t_Df}, Eff-Vacant{Ds,Ds_avg,Df,
        #             Df_avg,df,df_avg,t_Ds,t_Df}, EffExact-StratES{x8},
        #             EffExact-StratRA{x8}
        # 'sa':       EarlyBreak_nonbin{Ds,Df,df_prev,df,t_Ds,t_Df},
        #             DistDirect_nonbin{Ds,Ds_avg,Df,Df_avg,df_prev,df,
        #             df_avg,t_Ds,t_Df}, StratVacant{Ds,Ds_avg,Df,Df_avg,
        #             df,df_avg,t_Ds,t_Df}, StratES{x8}, StratRA{x8}
        # 'tim':      T(EarlyBreak |EffHD_multivar), T(DistDirect_multivar),
        #             T(Eff.Vacant), T(EffExact-StratES), T(EffExact-StratRA),
        #             T(computing performance |+gf)

        tag_df_conv['dist'] = [j[:2] + j[6:10] + j[
            15:19] + j[23:27] + j[31:35] for j in tmp]
        tag_df_conv['df'] = [j[2:4] + j[10:13] + j[
            19:21] + j[27:29] + j[35:37] for j in tmp]
        tag_df_conv['tim'] = [j[4:6] + j[13:15] + j[
            21:23] + j[29:31] + j[37:39] for j in tmp]
        tmp = tag_df_conv['dist']
        # tmp:  EarlyBreak_nonbin{Ds,Df}, DistDirect_non{Ds,Ds_avg,Df,Df_avg},
        #       StratVacant{x4}, StratES{x4}, StratRA{x4}  # 2+4+4*3=18
        tag_df_conv['dist_avg'] = [[j[k] for k in [
            3, 5, 7, 9, 11, 13, 15, 17]] for j in tmp]
        tag_df_conv['dist'] = [[j[k] for k in [
            0, 1, 2, 4, 6, 8, 10, 12, 14, 16, ]] for j in tmp]
        del tmp  # pdb.set_trace()
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

    def subfig_conv_singvar(self, dframe, tag_hfm, tag_conv, fgn,
                            omitted=False):  # True):
        if omitted:
            return
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_sa1 = tag_hfm['dist'][0][4:6] + tag_hfm['dist'][0][
            2:4] + tag_hfm['dist'][0][6:8] + tag_conv[
            'dist'][0][2:4] + tag_conv['dist_avg'][0][:2]
        tag_sa2 = tag_hfm['dist'][1][4:6] + tag_hfm['dist'][1][
            2:4] + tag_hfm['dist'][1][6:8] + tag_conv[
            'dist'][1][2:4] + tag_conv['dist_avg'][1][:2]
        tag_tim = [i[4:6] + i[2:4] + i[6:] + j[
            2:4] for i, j in zip(tag_hfm['tim'], tag_conv['tim'])]
        # tag_sa*:  DistDirect_bin{Ds,Df}, EarlyBreak{x2}, ApproxDist
        #           _bin{x2}, DistDirect_nonbin{Ds,Df, Ds_avg,Df_avg}
        tmp = self.obtain_binval_senatt(
            # tmp = self.obtain_multival_senatt(
            dframe, id_set, [
            ], tag_sa1 + tag_tim[0], tag_sa2 + tag_tim[1])
        t_tim = np.array(tag_tim[0]).reshape(-1, 2).T.tolist()
        fgn += '_singvar'
        self.thread_previous(tmp, t_tim, fgn, tag_val=np.array(
            tag_sa1).reshape(-1, 2).T.tolist(), tag_avg=[])
        return

    def sub_plt_tim_prev(self, df_tmp, tag, fgn, sgn='t_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
        annotY = [r'$T_{EarlyBreak}$',  # '${}$'.format(ant_Yq),
                  r'$T_{ApproxDist \text{(prev)}}$',
                  # '${}$ (bin-val)'.format(ant_X)]
                  '${}$'.format(ant_X)]
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Yq), '${} = {}$'.format(ant_Yq, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty4')
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn='D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
        pdb.set_trace()
        annotY = ['EarlyBreak', '${}$'.format(ant_Yq),
                  '${}$'.format(ant_X),  # (multival)
                  '${}$'.format(self.hfm_dict_avg(fgn, sgn)[0])]
        annot = ['${}$'.format(ant_Xp), '${}$'.format(
            ant_Yq), '${} = {}$'.format(ant_Yq, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3b')
        return

    def subfig_conv_multivar(self, dframe, tag_conv, fgn, tag_hfm):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_multivar = tag_conv['multivar'] + tag_conv['tim_mv'][:-1]
        tag1 = tag_hfm['dist'][0][4:8] + tag_hfm['dist'][0][
            2:4] + tag_hfm['tim'][0][4:8] + tag_hfm['tim'][0][2:4]
        tag2 = tag_hfm['dist'][1][4:8] + tag_hfm['dist'][1][
            2:4] + tag_hfm['tim'][1][4:8] + tag_hfm['tim'][1][2:4]
        tag_sa1 = tag_conv['dist'][0] + tag_conv['dist_avg'][
            0] + tag_conv['tim'][0]
        tag_sa2 = tag_conv['dist'][1] + tag_conv['dist_avg'][
            1] + tag_conv['tim'][1]
        df_alt = self.obtain_multival_senatt(
            dframe, id_set, tag_multivar, tag_sa1 + tag1, tag_sa2 + tag2)

        tag_tim = [tag_multivar[6 + 7:6 + 9], tag_multivar[4:6],
                   tag_multivar[15 + 6:15 + 8], tag_multivar[
                   23 + 6:23 + 8], tag_multivar[31 + 6:31 + 8]]
        tag_tim = np.array(tag_tim).T.tolist()
        tag_val = np.array([tag_multivar[i] for i in [
            6, 8, 0, 1,
            15, 17, 23, 25, 31, 33]]).reshape(5, 2).T.tolist()
        tag_avg = np.array([tag_multivar[i] for i in [
            7, 9, 16, 18, 24, 26, 32, 34]]).reshape(4, 2).T.tolist()
        fgn += '_multivar'
        self.thread_multivar(df_alt, tag_tim, fgn, tag_val, tag_avg)

        tag_sa1 = tag_conv['tim'][0]  # + tag1[-6:]
        tag_tim = np.array([tag_sa1[2:4], tag_sa1[:2], tag_sa1[
            4:6], tag_sa1[6:8], tag_sa1[8:10], tag1[
            -6:-4], tag1[-4:-2], tag1[-2:]]).T.tolist()
        tag_sa1 = tag_conv['dist'][0]
        tag_val = np.array(tag_sa1[2:4] + tag_sa1[:2] + tag_sa1[
            4:] + tag1[:6]).reshape(-1, 2).T.tolist()
        tag_avg = np.array(tag_conv[
            'dist_avg'][0]).reshape(-1, 2).T.tolist()
        fgn = fgn.replace('multivar', 'whole')
        self.thread_singvar(df_alt, tag_tim, fgn, tag_val, tag_avg)
        return

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn='t_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_tim(fgn, sgn)

        if 'multivar' in fgn:
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StratRA)}'
        else:
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
        ant_eff = r'T_{EarlyBreak}'
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        annotY = [  # '${}$ (multival)'.format(ant_eff),
            '${}$ (multi)'.format(ant_eff), '${}$'.format(ant_app),
            '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        if 'multivar' in fgn:
            multi_lin_reg_without_distr(
                scat_X, scat_Y[1:], annotY[1:], annot, fgn,
                snspec ='sty4c')
            return
        scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                       df_tmp[tag[6]].values.astype(DTY_FLT)])
        scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
        annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        annotY[0] = '${}$'.format(ant_eff)
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec ='sty4')
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn='D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = self.hfm_dict_val(fgn, sgn)
        if 'multivar' in fgn:
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'
            ant_arr = r'ExactDist(StratRA)'
        else:
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        ant_eff = 'EarlyBreak'
        annotY = [r'EarlyBreak$^{(multi)}$',  # ant_eff,multival
                  ant_app, ant_cvg, ant_arr]  # '${}$'
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        if 'multivar' in fgn:
            multi_lin_reg_without_distr(
                scat_X, scat_Y[1:], annotY[1:], annot, fgn,
                snspec='sty3c')
            return
        scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                       df_tmp[tag[6]].values.astype(DTY_FLT)])
        scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
        ant_Xp, ant_Yq = self.hfm_dict_prev(sgn)
        annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        annotY[0] = ant_eff 
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn,
            snspec='sty3b', figsize='L-WS')  # 'sty3b','L-NT')
        return


# ------------------------------


# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp8e -pre min_max
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9e -pre min_max
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9f -pre min_max --nb-cls 7
# ------------------------------
