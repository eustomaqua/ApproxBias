# coding: utf-8
# cor_mext_plt.py

import pdb
import csv
import numpy as np
import pandas as pd

# from experiment.utils_empirical import GraphSetupVer2
from experiment.df_nonbin.rev_mext_plt import GraphSetup
from pyfair.granite.draw_addtl import (
    hyper_params_lin_reg, multi_lin_reg_without_distr)
from pyfair.granite.draw_chart import analogous_confusion_extended
from pyfair.marble.draw_hypos import _encode_sign, _avg_and_stdev

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.utils.recorders import BLFAIR
from hfm.hfm_df import differentiate_tim, differentiate_val


# ==============================
# Division


# class GraphSetup(GraphSetupVer2):
#     def sub_dat_multivar(self, dframe, nb_set, id_set, tag):
#         pass


# ------------------------------
# cf.
#   HyperEA_renew_m1fix
#   HyperEB_renew_m2fix


class Plot5_hyperpm(GraphSetup):
    def __init__(self, nk, mp_cores=3, omitted=True,
                 m1=25, m2=11, n_e=2, figname='exp5_'):
        self._nb_iter = 1 if nk <= 0 else nk
        self._mpc, self._omit = mp_cores, omitted
        gen = rep = None  # False
        super().__init__(gen, rep, m1, m2, n_e, figname)
        # Hyper-parameters
        self._m2_set = list(range(2, 23, 1))  # ConvHP_EA_anal
        self._m1_set = list(range(3, 50, 2))  # ConvHP_EB_anal
        if omitted:
            self._m2_set = list(range(2, 14, 1))  # len= 21|12
            self._m1_set = list(range(3, 34, 2))  # len= 24|16
        # end of Hyper-parameters

    def prepare_graph(self, n_l):
        n_ell = (3 + n_l * 3) * 3
        csv_row_1 = unique_column(10 + n_ell * 2)
        params = csv_row_1[: 10]
        tag_sa1 = csv_row_1[10: 10 + n_ell]
        tag_sa2 = csv_row_1[10 + n_ell:]
        return params, tag_sa1, tag_sa2

    def incise_graph(self, n_l, tag):
        n_ell = 3 + n_l * 3
        direct = {'bin': tag[: 3],
                  'nonbin': tag[n_ell: n_ell + 3],
                  'cvg': tag[n_ell * 2: n_ell * 2 + 3]}
        approx = {'bin': tag[3: 3 + n_l],
                  'nonbin': tag[n_ell + 3: n_ell + 3 + n_l],
                  'arr': tag[n_ell * 2 + 3: n_ell * 2 + 3 + n_l]}
        approx_avg = {
            'bin': tag[3 + n_l: 3 + n_l * 2],
            'nonbin': tag[n_ell + 3 + n_l: n_ell + 3 + n_l * 2],
            'arr': tag[n_ell * 2 + 3 + n_l: n_ell * 2 + 3 + n_l * 2]}
        approx_tim = {
            'bin': tag[3 + n_l * 2: 3 + n_l * 3],
            'nonbin': tag[n_ell + 3 + n_l * 2: n_ell + n_ell],
            'arr': tag[n_ell * 2 + 3 + n_l * 2: n_ell * 2 + n_ell]}
        return direct, approx, approx_avg, approx_tim

    # def sub_dat_multivar(self, dframe, nb_set, id_set, tag):
    #     pass
    # def sub_dat_sen_att(self):
    #     pass
    # def sub_dat_sing_set(self):
    #     pass
    #
    # def obtn_sa_binval/multival: tag,
    #     # if dr_ptb:
    #     #     df_tmp[dr_ptb] = dframe.iloc[id_set[k]][dr_ptb]

    def obtn_sa_bin(self, dframe, id_set, pos, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_sa1]
        start = id_set[1] + 1 + pos
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
            df_tmp = df_tmp[tag_sa2].rename(columns=columns)
            start = id_set[k] + 1 + pos
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        for k in [3, 4]:
            df_tmp = dframe.iloc[
                id_set[k] + 1: id_set[k + 1]][tag_sa1]
            start = id_set[k] + 1 + pos
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw  # .reset_index(drop=True)

    def obtn_sa_nonbin(self, dframe, id_set, pos, tag_sa1, tag_sa2,
                       first_incl=True):  # , dr_ptb=''):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_sa1]
        start = id_set[2] + 1 + pos
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
            df_tmp = df_tmp[tag_sa2].rename(columns=columns)
            start = id_set[k] + 1 + pos
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        if not first_incl:
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_sa1]
        start = id_set[0] + 1 + pos
        df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw  # .reset_index(drop=True)

    def obtn_sa_multivar(self, dframe, id_set, pos, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        # df_raw= dframe.iloc[id_set[0]+1: id_set[1]][tag+ tag_sa1]
        df_raw = dframe.iloc[id_set[0] + 1: id_set[1]][tag_sa1]
        start = id_set[0] + 1 + pos
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for i in range(1, len(id_set) - 1):
            df_tmp = dframe.iloc[id_set[i] + 1: id_set[i + 1]]
            df_no_sa1 = df_tmp[tag_sa1]  # tag + tag_sa1]
            df_no_sa2 = df_tmp[tag_sa2]  # tag + tag_sa2]
            df_no_sa2 = df_no_sa2.rename(columns=columns)
            start = id_set[i] + 1 + pos
            df_no_sa1 = df_no_sa1.loc[start: start + self._nb_iter - 1]
            df_no_sa2 = df_no_sa2.loc[start: start + self._nb_iter - 1]
            # df_raw = pd.concat([df_raw, df_no_sa1, df_no_sa2], axis=0)
            tmp = [df_raw, df_no_sa1]
            if len(tag_sa2) > 0:
                tmp.append(df_no_sa2)
            df_raw = pd.concat(tmp, axis=0)
        return df_raw  # .reset_index(drop=True)

    def quick_rmk(self, rmk='tim', interm=False):
        if interm:
            if rmk.endswith('tim'):
                annotX = r'T_{ \mathbf{D}_{a}(S,a_i) }'
                annotY = r'T_{ \hat{\mathbf{D}}_a(S,a_i) }'
                anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
            elif rmk.endswith('avg'):
                annotX = r'\mathbf{D}_a^\text{avg}(S,a_i)'
                annotY = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
                anotAP = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
            else:
                annotX = r'\mathbf{D}_a(S,a_i)'
                annotY = r'\hat{\mathbf{D}}_a(S,a_i)'
                anotAP = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
            return annotX, annotY, anotAP

        if rmk.endswith('tim'):
            annotX = r'T_{ \mathbf{D}_{\mathbf{a}}(S) }'
            annotY = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }'
            anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
        elif rmk.endswith('avg'):
            annotX = r'\mathbf{D}_{\mathbf{a}}^\text{avg}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }-1'
        else:
            annotX = r'\mathbf{D}_{\mathbf{a}}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'
        return annotX, annotY, anotAP

    def present_ext_multivar(self, df, tag_drt,
                             tag_app, tag_app_avg, tag_app_tim,
                             tag_arr, tag_arr_avg, tag_arr_tim,
                             ms_set, picked_m, suff, interm=False):
        #                      # rmk='tim', alt=True):
        antX, antY, antAP = self.quick_rmk('tim', interm)
        annots = ['${}$ (sec)'.format(antX), '${}$ (sec)'.format(
            antY), '${} = {}$'.format(antY, antX)]
        kw = {'snspec': 'sty5b', 'corr': False}
        tX = df[tag_drt[2]].values.astype(DTY_FLT)
        hyper_params_lin_reg(
            tX, df[tag_app_tim].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_app_tim', **kw)
        hyper_params_lin_reg(
            tX, df[tag_arr_tim].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_arr_tim', **kw)

        kw['corr'] = True
        kw['curr_legend_nb_split'] = 6
        tX = df[tag_drt[0]].values.astype(DTY_FLT)
        antX, antY, _ = self.quick_rmk('', interm)
        annots = ['${}$'.format(antX), '${}$'.format(
            antY), '${} = {}$'.format(antY, antX)]
        hyper_params_lin_reg(
            tX, df[tag_app].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_app_max', **kw)
        hyper_params_lin_reg(
            tX, df[tag_arr].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_arr_max', **kw)
        tX = df[tag_drt[1]].values.astype(DTY_FLT)
        antX, antY, _ = self.quick_rmk('avg', interm)
        annots = ['${}$'.format(antX), '${}$'.format(
            antY), '${} = {}$'.format(antY, antX)]
        hyper_params_lin_reg(
            tX, df[tag_app_avg].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_app_avg', **kw)
        hyper_params_lin_reg(
            tX, df[tag_arr_avg].values.astype(DTY_FLT).T,
            ms_set, picked_m, annots, suff + '_arr_avg', **kw)
        return

    # def present_ext_midterm(self, df, tag_drt,
    #                         tag_app, tag_app_avg, tag_app_tim,
    #                         tag_arr, tag_arr_avg, tag_arr_tim,
    #                         ms_set, picked_m, suff):
    #     pdb.set_trace()
    #     return


class HPEA_m1fix(Plot5_hyperpm):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m2_set)
        pms, tag_sa1, tag_sa2 = self.prepare_graph(n_l)
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=0)
        ms_set = [r'$m_2$={}'.format(i) for i in self._m2_set]
        picked_m = [2, 4, 6, 8, 10]
        # this_suf = '{}{}_mp_m2fix'.format(self._figname, pre)

        n_k = self._nb_iter
        n_ell = 3 + n_l * 3
        # df_bin = self.obtn_sa_bin(
        #     raw_dframe, id_set, 0, tag_sa1, tag_sa2)
        df_nonbin = self.obtn_sa_nonbin(
            raw_dframe, id_set, n_k, tag_sa1[n_ell:], tag_sa2[n_ell:])
        df_multivar = self.obtn_sa_multivar(
            raw_dframe, id_set, n_k * 2, tag_sa1[n_ell:], [])

        t_drt, t_app, t_avg, t_tim = self.incise_graph(n_l, tag_sa1)
        # this_suf = f'{self._figname}{pre}_multivar'  # _mp_m2fix
        this_suf = f'{self._figname}{pre}_mv'  # '_mvar'
        self.present_ext_multivar(
            df_multivar, t_drt['nonbin'],
            t_app['nonbin'], t_avg['nonbin'], t_tim['nonbin'],
            t_app['arr'], t_avg['arr'], t_tim['arr'],
            ms_set, picked_m, this_suf)
        self.present_ext_multivar(
            df_nonbin, t_drt['nonbin'],
            t_app['nonbin'], t_avg['nonbin'], t_tim['nonbin'],
            t_app['arr'], t_avg['arr'], t_tim['arr'],
            ms_set, picked_m, this_suf.replace(
                # '_multivar', '_whole'), interm=True)
                '_mv', '_wh'), interm=True)
        # pdb.set_trace()
        return


class HPEB_m2fix(Plot5_hyperpm):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m1_set)
        pms, tag_sa1, tag_sa2 = self.prepare_graph(n_l)
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=0)
        ms_set = [r'$m_1$={}'.format(i) for i in self._m1_set]
        picked_m = [0, 3, 6, 9, 12]
        # this_suf = '{}{}_mp_m1fix'.format(self._figname, pre)

        n_k = self._nb_iter
        n_ell = 3 + n_l * 3
        # df_bin = self.obtn_sa_bin(
        #     raw_dframe, id_set, 0, tag_sa1, tag_sa2)
        df_nonbin = self.obtn_sa_nonbin(
            raw_dframe, id_set, n_k, tag_sa1[n_ell:], tag_sa2[n_ell:])
        df_multivar = self.obtn_sa_multivar(
            raw_dframe, id_set, n_k * 2, tag_sa1[n_ell:], [])

        t_drt, t_app, t_avg, t_tim = self.incise_graph(n_l, tag_sa1)
        this_suf = f'{self._figname}{pre}_mv'  # '_multivar',_mp_m2fix
        self.present_ext_multivar(
            df_multivar, t_drt['nonbin'],
            t_app['nonbin'], t_avg['nonbin'], t_tim['nonbin'],
            t_app['arr'], t_avg['arr'], t_tim['arr'],
            ms_set, picked_m, this_suf)
        self.present_ext_multivar(
            df_nonbin, t_drt['nonbin'],
            t_app['nonbin'], t_avg['nonbin'], t_tim['nonbin'],
            t_app['arr'], t_avg['arr'], t_tim['arr'],
            ms_set, picked_m, this_suf.replace(
                # '_multivar', '_whole'), interm=True)
                '_mv', '_wh'), interm=True)
        # pdb.set_trace()
        return


# ------------------------------
# cf.
#   ConvFig_5H_exact (ConvPlotF_init)
#   ConvFig_5I_exact (ConvFig_5Iprev_exact)
#   ConvFig_5Isimpl


class ConvPlotF_init(GraphSetup):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        _, _, tag_tst = self.prepare_graph()
        # pms, _, tag_tst = self.prepare_graph()
        tag_norm, tag_dr, tag_hfm, tag_conv = self.incise_graph(tag_tst)
        fgn = f'{self._figname}{pre}'  # _multivar
        self.subfig_conv_multivar(raw_dframe, tag_conv, fgn,
                                  tag_hfm=tag_hfm)
        self.subfig_conv_singvar(raw_dframe, tag_hfm, tag_conv, fgn)
        self.subfig_fair_sp(raw_dframe, tag_norm, tag_dr, tag_hfm,
                            tag_conv, fgn)  # +'_delt') #'_esp')
        return

    def obtn_sa_nonbin(self, dframe, id_set, tag, tag_sa1, tag_sa2,
                       first_incl=True):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        nb_set = len(id_set) - 1
        k = 2 if nb_set == 5 else 0
        df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1]
        df_raw = [df_raw, ]
        for k in ([3, 4] if nb_set == 5 else [2, 3]):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw.append(df_tmp)
        if first_incl and nb_set == 5:
            df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_sa1]
            df_raw = [df_tmp, ] + df_raw
        return df_raw

    def obtn_sa_bin(self, dframe, id_set, tag, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        nb_set = len(id_set) - 1  # k = 1
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag + tag_sa1]
        df_raw = [df_raw, ]
        for k in ([1, 2] if nb_set == 5 else [1]):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw.append(df_tmp)
        for k in ([3, 4] if nb_set == 5 else [2, 3]):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1]
            df_raw.append(df_tmp)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set, tag, tag_sa1, tag_sa2,
                               first_incl=True):  # False):
        # columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        # nb_set = len(id_set) - 1
        # k = 2 if nb_set == 5 else 0
        # df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1]
        # for k in ([3, 4] if nb_set == 5 else [2, 3]):
        #     df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa2]
        #     df_tmp = df_tmp.rename(columns=columns)
        #     df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # if first_incl and nb_set == 5:
        #     df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_sa1]
        #     df_raw = pd.concat([df_tmp, df_raw], axis=0)
        # # pdb.set_trace()
        # return df_raw.reset_index(drop=True)

        df_raw = self.obtn_sa_nonbin(dframe, id_set, tag, tag_sa1, tag_sa2, first_incl)
        return pd.concat(df_raw, axis=0).reset_index(drop=True)

    def obtain_binval_senatt(self, dframe, id_set, tag, tag_sa1, tag_sa2):
        # columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        # nb_set = len(id_set) - 1  # k = 1
        # df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag + tag_sa1]
        # for k in ([1, 2] if nb_set == 5 else [1]):
        #     df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa2]
        #     df_tmp = df_tmp.rename(columns=columns)
        #     df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # for k in ([3, 4] if nb_set == 5 else [2, 3]):
        #     df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1]
        #     df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # # pdb.set_trace()
        # return df_raw.reset_index(drop=True)

        df_raw = self.obtn_sa_bin(dframe, id_set, tag, tag_sa1, tag_sa2)
        return pd.concat(df_raw, axis=0).reset_index(drop=True)

    # def obtn_sa_bin(self):
    #     return
    # def obtn_sa_nonbin(self):
    #     return
    # def obtn_sa_multivar(self):
    #     return

    def obtn_fulfil_ricci(self, dframe, id_set, tag, tag_sa1, tag_sa2):
        # # k = 0  # columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        # # for t1, t2 in zip(tag_sa1, tag_sa2):
        # #     df_raw[t2] = df_raw[t1]
        # df_raw = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_sa1 + tag_sa2]
        # df_raw = df_raw.assign(**{t2: df_raw[t1] for t1, t2 in zip(tag_sa1, tag_sa2)})
        # # pdb.set_trace()
        # for k in range(1, 5):
        #     df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1 + tag_sa2]
        #     df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # return df_raw.reset_index(drop=True)

        df_raw = dframe.iloc[id_set[0] + 1: id_set[1]][tag + tag_sa1 + tag_sa2]
        df_raw = df_raw.assign(**{t2: df_raw[t1] for t1, t2 in zip(tag_sa1, tag_sa2)})
        df_raw = [df_raw, ]
        for k in range(1, 5):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag + tag_sa1 + tag_sa2]
            df_raw.append(df_tmp)  # df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw  # return df_raw.reset_index(drop=True)
        # after return:  df = pd.concat(df, axis=0).reset_index(drop=True)

    def thread_multivar(self, df_alt, tag_tim, fgn,
                        tag_val, tag_avg, verbose=False):
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

    def thread_singvar(self, tmp, tag_tim, fgn, tag_val, tag_avg,
                       verbose=False):
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


def _hfm_dict_tim(fgn, sgn):
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


def _hfm_dict_val(fgn, sgn):
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


def _hfm_dict_avg(fgn, sgn):
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


def _hfm_dict_prev(sgn='t_D.'):
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
        del tmp
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

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

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn = 't_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
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
            scat_Y.extend([df_tmp[tag[4]].values.astype(DTY_FLT),
                           df_tmp[tag[5]].values.astype(DTY_FLT)])
            ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(  # 'sty4d')'sty4')
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty4c')

        # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(  # 'sty6d')#'sty6')
            scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6c')
        del scat_Z, ant_Z, annot
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn = 'D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        if 'multivar' in fgn:  # if rmk == 'multivar':
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist (StratES)'
            ant_arr = r'ExactDist (StratRA)'
        else:  # rmk=='sen-att'
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        annotY = [ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(  # 'sty3d')'sty3b')
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3c')

        # scat_Z = [k / scat_X - 1. for k in scat_Y]
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(  # 'sty6d')#'sty6')
            scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6c')
        del scat_Z, ant_Z, annot
        return

    def sub_plt_avg(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_avg(fgn, sgn)
        if 'multivar' in fgn:  # if rmk == 'multivar':
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'  # r'ExactDist(w/ StratES)'
            ant_arr = r'ExactDist(StratRA)'  # r'ExactDist(w/ StratRA)'
        else:  # rmk=='sen-att'
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        annotY = [ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(  # 'sty3d')#'sty3b')
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3c')

        # scat_Z = [k / scat_X - 1. for k in scat_Y]
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(  # 'sty6d')#'sty6c')
            scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6c')
        del scat_Z, ant_Z, annot
        return

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
        tmp = self.obtain_binval_senatt(
            dframe, id_set, [], tag_sa1, tag_sa2)
        tag_tim = tag_sa1[18 + 4:]
        tag_tim = tag_tim[:2] + tag_tim[5:]
        tag_tim = np.array(tag_tim).reshape(-1, 2).T.tolist()
        fgn += '_singvar'  # '_whole'
        tag_val = tag_sa1[: 18 + 4]
        tag_avg = np.array([tag_val[i] for i in [
            1, 3, 7, 9, 11, 13, 15, 17,
            19, 21]]).reshape(-1, 2).T.tolist()
        tag_val = np.array([tag_val[i] for i in [
            0, 2, 4, 5, 6, 8, 10, 12, 14, 16,
            18, 20]]).reshape(-1, 2).T.tolist()
        self.thread_previous(tmp, tag_tim, fgn, tag_val, tag_avg)
        return

    def sub_plt_tim_prev(self, df_tmp, tag, fgn, sgn='t_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY = [r'$T_{ApproxDist}$', r'$T_{StratES}$', r'$T_{StratRA}$',
                  '${}$'.format(ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Y), '${}$ (bin-val)$= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty4c')  # 'sty4')

        kws = {'snspec': 'sty6c'}  # 'sty6'}
        # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn = 'D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)  # [5]
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY = ['ApproxDist', 'StratES', 'StratRA', '${}$'.format(
            ant_Yq), '${}$ (multival)'.format(ant_X)]
        annot = ['${}$'.format(ant_Xp), '${}$ (bin-val)'.format(
            ant_Y), '${}$ (bin-val) $= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3c')  # 'sty3b')

        kws = {'snspec': 'sty6c'}  # 'sty6'}
        # scat_Z = [k / scat_X - 1. for k in scat_Y]
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def sub_plt_avg_prev(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[4]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_avg(fgn, sgn)
        annotY = [r'ApproxDist', r'StratES', r'StratRA']
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$ (bin-val)'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(  # 'sty3d')'sty3b')
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3c')

        kws = {'snspec': 'sty6c'}  # 'sty6d'}#'sty6'}
        # scat_Z = [k / scat_X - 1. for k in scat_Y]
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def thread_fair(self, dframe, tb, tag_sa1, tag_sa2, fgn):
        key_A = [r'accuracy', r'precision', 'recall', r'specificity',
                 r'g_mean', r'dp']  # r'$\mathrm{f}_1$ score'
        key_B = [r'$\Delta$(accuracy)', r'$\Delta$(precision)',
                 r'$\Delta$(recall)', r'$\Delta$(specificity)',
                 # r'$\Delta$($\mathrm{f}_1$ score)',
                 r'$\Delta$(g_mean)', r'$\Delta$(dp)']
        key_C = BLFAIR[:3] + [r'$\mathrm{SP}$',
                              r'$\mathrm{SP}^\text{avg}$'] + BLFAIR[
            -1:] + [r'$\mathbf{df}_\text{prev}$',
                    r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        key_D = tag_sa1[:6 + 1] + tag_sa1[-2:]
        # pdb.set_trace()
        pms = {'rotate': 34, 'cmap_name': 'Blues'}
        df_bin = self.obtain_binval_senatt(
            dframe, id_set, tb, tag_sa1, tag_sa2)
        mat_D = df_bin[key_D].values.astype(DTY_FLT).T
        mat_B = df_bin[tb[:6]].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            mat_B, mat_D, key_B, key_C, f'{fgn}_dt_bin', **pms)
        pms['cmap_name'] = 'Greens'
        df_nonbin = self.obtain_multival_senatt(
            dframe, id_set, tb, tag_sa1, tag_sa2)
        mat_E = df_nonbin[key_D].values.astype(DTY_FLT).T
        mat_A = df_nonbin[tb[:6]].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            mat_A, mat_E, key_B, key_C, f'{fgn}_dt_nonbin', **pms)

        pms['cmap_name'] = 'OrRd'
        df = self.obtn_fulfil_ricci(dframe, id_set, tb, tag_sa1, tag_sa2)
        df = pd.concat(df, axis=0).reset_index(drop=True)
        key_E = tag_sa2[:6 + 1] + tag_sa2[-2:]
        mat_D = df[key_D].values.astype(DTY_FLT).T
        mat_E = df[key_E].values.astype(DTY_FLT).T
        key_C[3] = r'$\mathrm{ESP}$'
        key_C[4] = r'$\mathrm{ESP}^\text{avg}$'
        key_C = key_C[:6] + key_C[-2:]
        mat_C = np.zeros_like(mat_D, dtype=DTY_FLT)[:-1]
        for k in [0, 1, 2, 4, 5]:
            mat_C[k] = (mat_D[k] + mat_E[k]) / 2.
        for k in range(mat_C.shape[1]):
            mat_C[3][k] = max(mat_D[3][k], mat_E[3][k])
        mat_C[6] = df[tb[12:][0]].values.astype(DTY_FLT)
        mat_C[7] = df[tb[12:][4]].values.astype(DTY_FLT)
        analogous_confusion_extended(df[tb[:6]].values.astype(
            DTY_FLT).T, mat_C, key_B, key_C, f'{fgn}_delt_mv', **pms)
        return

    def subfig_fair_sp(self, dframe, tag_norm, tag_dr, tag_hfm, tag_conv, fgn):
        tag_fair = [tag_dr[i][:7] + tag_dr[i][11:13] + tag_hfm[
            'df'][i] for i in range(2)]  # hfm:3+1+3+3*2
        tag_fair = [i[:4] + i[5:7] + i[9:][
            :1] + i[9 + 3:][:1] + i[9 + 4:][1:3] + i[
            13 + 3:][1:3] + i[16 + 3:][1:3] for i in tag_fair]
        tmp_dr = [i[:3] + i[4:6] + i[3:4] + tag_dr[ti][
            4:5] for ti, i in enumerate(tag_fair)]  # tmp_hfm:1+1+2+2*2=8
        tmp_hfm = [[i[6:][j] for j in [
            0, 1, 2, 4, 6, 3, 5, 7]] for i in tag_fair]
        tmp_conv = [[i[5:7], i[9:][5:7], i[18:][5:7], i[27:][
            5:7]] for i in [tag_conv['sa1'], tag_conv['sa2']]]
        tmp_conv = [np.array(tmp_conv[i]).T.reshape(
            -1).tolist() for i in range(2)]
        tmp_multivar = np.array([tag_conv['multivar'][5:7], tag_conv[
            'multivar'][9:][5:7], tag_conv['multivar'][18:][
            5:7], tag_conv['multivar'][27:][5:7]]).T.reshape(
            -1).tolist()   # Direct_mv,Extend,ExactES,ExactRA, ..df_avg
        # tmp_dr  : DP,EOpp,PP, SP,SP_avg, DR, hat_L(loss),  # 3+2+2=7
        # tmp_hfm : Direct_bin df_prev, ApproxDist_bin df_prev,
        #           StratVacant,StratES,StratRA, ..df_avg
        # tmp_conv: Direct_nonbin,StratVacant,StratES,StratRA, ..df_avg

        tag_sa1 = tmp_hfm[0][:2] + tmp_conv[0][:1] + tmp_conv[0][4:5]
        tag_sa2 = tmp_hfm[1][:2] + tmp_conv[1][:1] + tmp_conv[1][4:5]
        tag_sa1 = tmp_dr[0][:-1] + tag_sa1
        tag_sa2 = tmp_dr[1][:-1] + tag_sa2
        tmp_norm = [i[:4] + i[5:] + i[4:5] for i in tag_norm]
        tb = tmp_norm[1][:-1] + tmp_norm[0][:-1]  # delta,oo
        self.thread_fair(dframe, tb + tmp_multivar, tag_sa1, tag_sa2, fgn)
        '''
        pdb.set_trace()

        tag_sa1 = tmp_dr[0] + tmp_hfm[0] + tmp_conv[0]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1] + tmp_conv[1]
        tb = tag_norm[0] + tag_norm[1]
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)

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
        # key_E = tmp_hfm[1][:1] + tmp_conv[1][:1] + tmp_conv[1][4:5]
        # key_F = tmp_multivar[:1] + tmp_multivar[4:5]
        # key_C.extend([r'$\mathbf{df}_{ext}$', r'$\mathbf{df}^\text{avg}_{ext}$'])

        df_bin = self.obtain_binval_senatt(
            dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        df_nonbin = self.obtain_multival_senatt(
            dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        mat_D = df_bin[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        mat_E = df_nonbin[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        mat_B = df_bin[tag_norm[1][:-1]].values.astype(DTY_FLT).T
        mat_A = df_nonbin[tag_norm[1][:-1]].values.astype(DTY_FLT).T
        pms = {'rotate': 34, 'cmap_name': 'Blues'}
        analogous_confusion_extended(mat_B, mat_D, key_B, key_C, f'{fgn}_dt_bin', **pms)
        pms['cmap_name'] = 'Greens'
        analogous_confusion_extended(mat_A, mat_E, key_B, key_C, f'{fgn}_dt_nonbin', **pms)
        pms['cmap_name'] = 'OrRd'
        # mat_C = np.zeros_like(mat_D, dtype=DTY_FLT)[:-1]
        # analogous_confusion_extended(mat_B, )
        pdb.set_trace()
        '''
        return

    def subfig_fair_sp_prev(self, dframe, tag_norm, tag_dr, tag_hfm, tag_conv, fgn):
        tag_fair = [tag_dr[i][:7] + tag_dr[i][11:13] + tag_hfm[
            'df'][i] for i in range(2)]  # hfm:3+1+3+3*2
        tag_fair = [i[:4] + i[5:7] + i[9:][
            :1] + i[9 + 3:][:1] + i[9 + 4:][1:3] + i[
            13 + 3:][1:3] + i[16 + 3:][1:3] for i in tag_fair]
        tmp_dr = [i[:3] + i[4:6] + i[3:4] + tag_dr[ti][
            4:5] for ti, i in enumerate(tag_fair)]  # tmp_hfm:1+1+2+2*2=8
        tmp_hfm = [[i[6:][j] for j in [
            0, 1, 2, 4, 6, 3, 5, 7]] for i in tag_fair]
        tmp_conv = [[i[5:7], i[9:][5:7], i[18:][5:7], i[27:][
            5:7]] for i in [tag_conv['sa1'], tag_conv['sa2']]]
        tmp_conv = [np.array(tmp_conv[i]).T.reshape(
            -1).tolist() for i in range(2)]
        # tmp_dr  : DP,EOpp,PP, SP,SP_avg, DR, hat_L(loss),  # 3+2+2=7
        # tmp_hfm : Direct_bin df_prev, ApproxDist_bin df_prev,
        #           StratVacant,StratES,StratRA, ..df_avg
        # tmp_conv: Direct_nonbin,StratVacant,StratES,StratRA, ..df_avg
        tmp_multivar = np.array([tag_conv['multivar'][5:7], tag_conv[
            'multivar'][9:][5:7], tag_conv['multivar'][18:][
            5:7], tag_conv['multivar'][27:][5:7]]).T.reshape(
            -1).tolist()   # Direct_mv,Extend,ExactES,ExactRA, ..df_avg

        tag_sa1 = tmp_dr[0] + tmp_hfm[0] + tmp_conv[0]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1] + tmp_conv[1]
        tb = tag_norm[0] + tag_norm[1]
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        # df_alt = self.obtain_multival_senatt(
        #     dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        # df = df_alt
        df = self.obtn_fulfil_ricci(
            dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        df = pd.concat(df, axis=0).reset_index(drop=True)
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
        key_E = tmp_hfm[1][:1] + tmp_conv[1][:1] + tmp_conv[1][4:5]
        mat_D = df[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        mat_E = df[tag_sa2[:6] + key_E].values.astype(DTY_FLT).T
        pms = {'cmap_name': 'Blues', 'rotate': 14}
        # analogous_confusion_extended(df[tag_norm[0][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_A, key_C, f'{fgn}_bi_oo', **pms)
        # analogous_confusion_extended(df[tag_norm[1][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_B, key_C, f'{fgn}_mv_delt', **pms)
        mat_B = df[tag_norm[1][:-1]].values.astype(DTY_FLT).T
        analogous_confusion_extended(mat_B, mat_D, key_B, key_C, f'{fgn}_dt_sing_sa1', **pms)
        pms['cmap_name'] = 'Greens'
        analogous_confusion_extended(mat_B, mat_E, key_B, key_C, f'{fgn}_dt_sing_sa2', **pms)

        key_D = tag_sa1[:6] + key_D + tag_norm[0][:1] + tag_norm[0][-1:]
        key_E = tag_sa2[:6] + key_D + tag_norm[1][:1] + tag_norm[1][-1:]
        # df = self.obtain_binval_senatt(
        #     dframe, id_set, tb + tmp_multivar, tag_sa1, tag_sa2)
        # pms['cmap_name'] = 'Blues'  # 'Spectral'  # 'GnBu_r'
        # mat_C = df[key_D[:-2]].values.astype(DTY_FLT).T
        # analogous_confusion_extended(df[tag_norm[0][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_A, key_C, f'{fgn}_bi_oo', **pms)
        # analogous_confusion_extended(df[tag_norm[1][:-1]].values.astype(
        #     DTY_FLT).T, mat_C, key_B, key_C, f'{fgn}_bi_delt', **pms)

        # tmp = df  # tmp = df_alt
        key_C = BLFAIR[:3] + [  # r'$\text{SP}^\text{max}$',
            r'$\mathrm{ESP}$', r'$\mathrm{ESP}^\text{avg}$'] + BLFAIR[
            -1:] + [  # r'$\mathbf{df}_\text{prev}$',
            r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        # key_D = tag_sa1[:6] + tag_sa1[7:][:1] + tag_sa1[
        #     7 + 8:][:1] + tag_sa1[15:][4:5]
        # mat_C = tmp[key_D].values.astype(DTY_FLT).T
        # analogous_confusion_extended(
        #     tmp[tag_norm[1]].values.astype(DTY_FLT).T, mat_C,
        #     key_B, key_C, f'{fgn}_sing_dt', **pms)
        # pdb.set_trace()
        mat_C = np.zeros_like(mat_D, dtype=DTY_FLT)[:-1]
        for k in [0, 1, 2, 4, 5]:
            mat_C[k] = (mat_D[k] + mat_E[k]) / 2.
        for k in range(mat_C.shape[1]):
            mat_C[3][k] = max(mat_D[3][k], mat_E[3][k])
        mat_C[6] = df[tmp_multivar[0]].values.astype(DTY_FLT)
        mat_C[7] = df[tmp_multivar[4]].values.astype(DTY_FLT)
        pms['cmap_name'] = 'OrRd'
        analogous_confusion_extended(mat_B, mat_C, key_B, key_C, f'{fgn}_delt_mv', **pms)
        return


class ConvFig_5I_exact(ConvFig_5H_exact):
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
        del tmp
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

    def subfig_conv_multivar(self, dframe, tag_conv, fgn,
                             tag_hfm):  # tag_hfm=None):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row = 4, nc_norm = 3, nc_sens = 4)
        tag_multivar = tag_conv['multivar'] + tag_conv['tim'][: -1]
        # df_alt = self.sub_dat_multivar(
        #     dframe, nb_set, id_set, tag_multivar)
        # tag1 = tag_hfm['dist'][0][:4] + tag_hfm['dist'][0][
        #     -2:] + tag_hfm['tim'][0][:4] + tag_hfm['tim'][0][-8:-6]
        # tag2 = tag_hfm['dist'][1][:4] + tag_hfm['dist'][1][
        #     -2:] + tag_hfm['tim'][1][:4] + tag_hfm['tim'][1][-8:-6]
        # tag_tim = [tag_multivar[7: 9], tag_multivar[33 + 4: 33 + 6],
        #            tag_multivar[9 + 6: 9 + 8], tag_multivar[17 + 6: 17 + 8],
        #            tag_multivar[25 + 6: 25 + 8], ]  # EarlyBreak,Vacant|..
        tag1 = tag_hfm['dist'][0][:6] + tag_hfm['tim'][0][:6]
        tag2 = tag_hfm['dist'][1][:6] + tag_hfm['tim'][1][:6]
        df_alt = self.obtain_multival_senatt(
            dframe, id_set, tag_multivar,
            tag_conv['sa1'] + tag1, tag_conv['sa2'] + tag2)
        tag_tim = [tag_multivar[7: 9], tag_multivar[9 + 4: 9 + 6],
                   tag_multivar[15 + 6: 15 + 8], tag_multivar[
                   23 + 6: 23 + 8], tag_multivar[31 + 6: 31 + 8]]
        tag_tim = np.array(tag_tim).T.tolist()

        # tag_val = np.array([tag_multivar[i] for i in [
        #     0, 2, 33, 34,
        #     9, 11, 17, 19, 25, 27]]).reshape(5, 2).T.tolist()
        # tag_avg = np.array([tag_multivar[i] for i in [
        #     1, 3, 10, 12, 18, 20, 26, 28]]).reshape(4, 2).T.tolist()
        tag_val = np.array([tag_multivar[i] for i in [
            0, 2, 9, 10,
            15, 17, 23, 25, 31, 33]]).reshape(5, 2).T.tolist()
        tag_avg = np.array([tag_multivar[i] for i in [
            1, 3, 16, 18, 24, 26, 32, 34]]).reshape(4, 2).T.tolist()
        fgn += '_multivar'
        self.thread_multivar(df_alt, tag_tim, fgn, tag_val, tag_avg)

        tag_sa1 = tag_conv['sa1']
        # tag_tim = np.array([tag_sa1[7: 9], tag_sa1[9 + 6: 9 + 8],
        #                     tag_sa1[17 + 6: 17 + 8],
        #                     tag_sa1[25 + 6: 25 + 8],
        #                     tag_sa1[33 + 4: 33 + 6],
        #                     tag1[-6:][:2], tag1[-6:][2:4],
        #                     tag1[-6:][4:]]).T.tolist()
        # tag_val = np.array([tag_sa1[i] for i in [
        #     0, 2, 33, 34,
        #     9, 11, 17, 19, 25, 27]] + tag1[
        #     :6]).reshape(5 + 3, 2).T.tolist()
        # tag_avg = np.array([tag_sa1[i] for i in [
        #     1, 3, 10, 12, 18, 20, 26, 28]]).reshape(4, 2).T.tolist()
        # tmp = df_alt
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
        tag_avg = np.array([tag_sa1[i] for i in [
            1, 3, 16, 18, 24, 26, 32, 34]]).reshape(4, 2).T.tolist()
        fgn = fgn.replace('multivar', 'whole')
        pdb.set_trace()
        self.thread_singvar(df_alt, tag_tim, fgn, tag_val, tag_avg)
        return

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn = 't_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
        if 'multivar' in fgn:
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StratRA)}'
        else:
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StratRA}'
        ant_eff = r'T_{EarlyBreak^{multi}}'  # r'T_{EarlyBreak}'
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        annotY = [  # '${}$ (multi)'.format(ant_eff),
            '${}$'.format(ant_eff), '${}$'.format(ant_app),
            '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        if 'multivar' not in fgn:
            scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                           df_tmp[tag[6]].values.astype(DTY_FLT)])
            scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
            ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
            annotY[0] = '${}$'.format(ant_eff)
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec ='sty4')  # 'sty4d')

        # scat_Z = [np.log10(k / scat_X) for k in scat_Y]
        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(  # 'sty6d')
            scat_X, scat_Z, annotY, annot, fgn + '_s', snspec ='sty6')
        del scat_Z, ant_Z, annot
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn='D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        if 'multivar' in fgn:
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'
            ant_arr = r'ExactDist(StratRA)'
        else:
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        ant_eff = 'EarlyBreak'
        # annotY = [r'EarlyBreak$^{(multi)}$', ant_app, ant_cvg, ant_arr]
        annotY = [r'$\mathrm{EarlyBreak}^{multi}$', ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        if 'multivar' not in fgn:
            scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                           df_tmp[tag[6]].values.astype(DTY_FLT)])
            # scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
            ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
            annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
            annotY[0] = ant_eff
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty3b')  # 'sty3d')

        # scat_Z = [k / scat_X - 1. for k in scat_Y]
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s',
            snspec = 'sty6')  # 'sty6d')
        del scat_Z, ant_Z, annot
        return

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
        pdb.set_trace()
        self.thread_previous(tmp, tag_tim, fgn, tag_val, tag_avg)
        return

    def sub_plt_tim_prev(self, df_tmp, tag_tim, fgn, sgn = 't_D.'):
        scat_X = df_tmp[tag_tim[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag_tim[2]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[3]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[4]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[5]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[1]].values.astype(DTY_FLT),
                  df_tmp[tag_tim[6]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY = [r'$T_{EarlyBreak}$', r'$T_{ApproxDist}$',
                  r'$T_{StratES}$', r'$T_{StratRA}$', '${}$'.format(
                      ant_Yq), '${}$ (multival)'.format(ant_X)]
        annotY[4] += r'$\,_{ApproxDist(prev)}$'
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Y), '${}$ (bin-val)$= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty4')

        kws = {'snspec': 'sty6'}
        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn = 'D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT),
                  df_tmp[tag[5]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[6]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        if sgn.startswith('Ds') or (sgn == 'D'):
            ant_Xp = r'\mathbf{D}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}(S_1,\bar{S}_1)'
        elif sgn.startswith('Df'):
            ant_Xp = r'\mathbf{D}_f(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_f(S_1,\bar{S}_1)'
        else:
            ant_Xp = r'\mathbf{D}_{\cdot}(S_1,\bar{S}_1)'
            ant_Yq = r'\hat{\mathbf{D}}_{\cdot}(S_1,\bar{S}_1)'
        annotY = ['EarlyBreak', 'ApproxDist', 'StratES', 'StratRA',
                  '${}$'.format(ant_Yq), '${}$ (multival)'.format(ant_X)]
        annotY[4] += r' ApproxDist(prev)'
        annot = ['${}$'.format(ant_Xp), '${}$ (bin-val)'.format(
            ant_Y), '${}$ (bin-val) $= {}$'.format(ant_Y, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty3b')

        kws = {'snspec': 'sty6'}
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def sub_plt_avg_prev(self, df_tmp, tag, fgn, sgn = 'D._avg'):
        scat_X = df_tmp[tag[3]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[0]].values.astype(DTY_FLT),
                  df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_avg(fgn, sgn)
        annotY = [r'ApproxDist', r'StratES', r'StratRA']
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$ (bin-val)'.format(ant_Y, ant_X)]
        multi_lin_reg_without_distr(  # 'sty3b','sty3d')
            scat_X, scat_Y, annotY, annot, fgn, snspec = 'sty3c')

        kws = {'snspec': 'sty6c'}  # 'sty6'}#'sty6d'}
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, kws, ant_Z
        return

    def subfig_fair_sp(self, dframe, tag_norm, tag_dr, tag_hfm, tag_conv, fgn):
        tmp_dr = [i[:3] + i[5:7] + i[3:4] for i in tag_dr]  # GFx3,SPx2,DR
        tmp_hfm = [[i[j] for j in [
            0, 3, 2, 1, 5, 7, 9, 6, 8, 10]] for i in tag_hfm['df']]
        tmp_conv = [i[4:7] + i[9:][3:4] + i[
            9 + 6:][4:6] + i[9 + 6 + 8:][4:6] + i[9 + 6 + 8 + 8:][
            4:6] for i in [tag_conv['sa1'], tag_conv['sa2']]]
        tmp_multivar = tag_conv['multivar'][4:7] + tag_conv['multivar'][
            9:][3:4] + tag_conv['multivar'][9 + 6:][4:6] + tag_conv['multivar'][
            9 + 6 + 8:][4:6] + tag_conv['multivar'][9 + 6 + 8 + 8:][4:6]
        # tmp_hfm:      DistDirect_bin df_prev, EarlyBreak df_prev,
        #               ApproxDist_bin df_prev, DistDirect_bin df,
        #               Vacant|StratES|StratRA df, Vacant|StratES|StratRA df_avg
        # tmp_conv:     DistDirect_nonbin {df_prev, df,df_avg}, EffHD_nonbin {df},
        #               Vacant {df,df_avg}, StratES x2, StratRA x2
        # tmp_multivar: DistDirect_multivar {df_prev, df,df_avg}, EffHD_multivar{
        #               df}, Vacant {df,df_avg}, StratES x2, StratRA x2
        tmp_norm = [i[:4] + i[5:] + i[4:5] for i in tag_norm]
        tag_sa1 = tmp_dr[0] + tmp_hfm[0][:4] + tmp_conv[0][:3]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1][:4] + tmp_conv[1][:3]
        tb = tmp_norm[1][:-1] + tmp_norm[0][:-1]  # delta,oo
        self.thread_fair(dframe, tb + tmp_multivar, tag_sa1, tag_sa2, fgn)

        tb = tmp_norm[1] + tmp_norm[0]  # add f1_score
        tag_sa1 = tmp_dr[0] + tmp_hfm[0][:1] + tmp_conv[0][1:3]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1][:1] + tmp_conv[0][1:3]
        self.thread_tabulate(dframe, tb + tmp_multivar[1:3], tag_sa1, tag_sa2,
                             fgn.replace('exp', 'tab'))  # f'{fgn}tab_')
        return

    def thread_fair(self, dframe, tb, tag_sa1, tag_sa2, fgn, verbose=False):
        key_A = [r'accuracy', r'precision', 'recall', r'specificity',
                 r'g_mean', r'dp']  # r'$\mathrm{f}_1$ score'
        key_B = [r'$\Delta$(accuracy)', r'$\Delta$(precision)',
                 r'$\Delta$(recall)', r'$\Delta$(specificity)',
                 # r'$\Delta$($\mathrm{f}_1$ score)',
                 r'$\Delta$(g_mean)', r'$\Delta$(dp)']
        key_C = BLFAIR[:3] + [r'$\mathrm{SP}$',
                              r'$\mathrm{SP}^\text{avg}$'] + BLFAIR[
            -1:] + [r'$\mathbf{df}_\text{prev}$'] + ([
                r'', r'', ] if verbose else []) + [
            r'$\mathbf{df}$', r'$\mathbf{df}^\text{avg}$']
        key_C[3] = r'SP'
        key_C[4] = r'SP$^\text{avg}$'
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        df_bin = self.obtain_binval_senatt(dframe, id_set, tb, tag_sa1, tag_sa2)
        df_nonbin = self.obtain_multival_senatt(dframe, id_set, tb, tag_sa1, tag_sa2)
        key_D = tag_sa1[6:][:1] + tag_sa1[6:][3:4] + tag_sa1[6 + 4:][:3]
        if not verbose:
            key_D = tag_sa1[6:][:1] + tag_sa1[6 + 4:][1:3]

        mat_D = df_bin[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        mat_E = df_nonbin[tag_sa1[:6] + key_D].values.astype(DTY_FLT).T
        mat_B = df_bin[tb[:6]].values.astype(DTY_FLT).T
        mat_A = df_nonbin[tb[:6]].values.astype(DTY_FLT).T
        pms = {'rotate': 34, 'cmap_name': 'Blues', 'figsize': 'L-ET'}  # 'extra'}
        if not verbose:
            del pms['figsize']
        analogous_confusion_extended(mat_B, mat_D, key_B, key_C, f'{fgn}_dt_bin', **pms)
        # analogous_confusion_extended(df_bin[tb[6:12]].values.astype(DTY_FLT).T, mat_D, key_A, key_C, f'{fgn}_oo_bin', **pms)
        pms['cmap_name'] = 'Greens'
        analogous_confusion_extended(mat_A, mat_E, key_B, key_C, f'{fgn}_dt_nonbin', **pms)
        # analogous_confusion_extended(df_nonbin[tb[6:12]].values.astype(DTY_FLT).T, mat_E, key_A, key_C, f'{fgn}_oo_nonbin', **pms)
        # pdb.set_trace()  # tb[6:12] --> tb[7:13] no longer needed

        pms['cmap_name'] = 'OrRd'
        df = self.obtn_fulfil_ricci(dframe, id_set, tb, tag_sa1, tag_sa2)
        df = pd.concat(df, axis=0).reset_index(drop=True)
        key_E = tag_sa2[6:][:1] + tag_sa2[6:][3:4] + tag_sa2[6 + 4:][:3]
        if not verbose:
            key_E = tag_sa2[6:][:1] + tag_sa2[6 + 4:][1:3]
        key_D = tag_sa1[:6] + key_D
        key_E = tag_sa2[:6] + key_E
        mat_D = df[key_D].values.astype(DTY_FLT).T
        mat_E = df[key_E].values.astype(DTY_FLT).T
        for k in [0, 1, 2]:
            key_C[k] += r'$^\prime$'
        key_C[3] = r'ESP'  # r'$\mathrm{ESP}$'
        key_C[4] = r'ESP$^\text{avg}$'  # r'$\mathrm{ESP}^\text{avg}$'
        key_C = key_C[:6] + key_C[-2:]
        mat_C = np.zeros_like(mat_D, dtype=DTY_FLT)  # [:-1]
        for k in [0, 1, 2, 4, 5]:
            mat_C[k] = (mat_D[k] + mat_E[k]) / 2.
        for k in range(mat_C.shape[1]):
            mat_C[3][k] = max(mat_D[3][k], mat_E[3][k])
        mat_C[6] = df[tb[12:][1]].values.astype(DTY_FLT)
        mat_C[7] = df[tb[12:][2]].values.astype(DTY_FLT)
        mat_C[8] = df[tb[12:][0]].values.astype(DTY_FLT)
        # pdb.set_trace()  # (mat_D[5] == mat_E[5]).all()
        # key_C.append(r'$\mathbf{df}_\text{prev}$\n(multival)')
        # key_C.append(r'$\mathbf{df}_\text{prev}$ (multival)')
        key_C.append(r'$\mathbf{df}_\text{prev}^{(multival)}$')
        analogous_confusion_extended(df[tb[:6]].values.astype(
            DTY_FLT).T, mat_C[:-1], key_B, key_C[:-1], f'{fgn}_delt_mv', **pms)
        return

    def thread_tabulate(self, dframe, tb, tag_sa1, tag_sa2, fgn,
                        ddof=1, rez=4, clf=3):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            dframe, nb_row=4, nc_norm=3, nc_sens=4)
        # df_bin = self.obtain_binval_senatt(dframe, id_set, tb, tag_sa1, tag_sa2)
        # df_nonbin = self.obtain_multival_senatt(dframe, id_set, tb, tag_sa1, tag_sa2)
        df = self.obtn_fulfil_ricci(dframe, id_set, tb, tag_sa1, tag_sa2)
        df_bin = self.obtn_sa_bin(dframe, id_set, tb, tag_sa1, tag_sa2)
        df_nonbin = self.obtn_sa_nonbin(dframe, id_set, tb, tag_sa1, tag_sa2)

        kw = {'nc_norm': 3, 'nc_sens': 4}  # ,'id_set':id_set,'nb_set':nb_set}
        # t = self.obtn_clf_result(df, tb, tag_sa1, tag_sa2, 3, **kw) #lightGBM
        # dt = self.obtn_clf_result(df, tb, tag_sa1, tag_sa2, 3 + 3, **kw)
        # dt = self.obtn_clf_result(df, tb, tag_sa1, tag_sa2, 7 + 3, **kw)
        #   3: lightGBM;    3|7+3: FairGBM|FPR,FNR; 4|7+4: AdaFair
        dt = self.obtn_clf_result(df, clf, **kw)  # tb, tag_sa1, tag_sa2,
        ta = [tb[7:14], tb[:7], tb[14:]]  # acc,f1,delta(acc,f1),hfm_ext
        ta = ta[0][:1] + ta[0][-1:] + ta[1][:1] + ta[1][-1:] + ta[2]
        ta = ta[:4] + tag_sa1[5:6] + tag_sa2[5:6] + ta[-2:]  # +DR x2, hfm x2
        ans_tex, cmp_tex = [], []  # 对照组
        for j in range(nb_set):
            tmp_tex, tmp_my = [''], ['']
            for k, v in enumerate(ta):
                ii = dt[j][v].values.astype(DTY_FLT)
                if k < 4:
                    ii *= 100.
                tmp_tex.append(_encode_sign(
                    ii.mean().tolist(), ii.std(ddof=ddof).tolist(), rez))
                mu, _, sigma = _avg_and_stdev(ii.tolist(), self._nb_iter)
                tmp_my.append(_encode_sign(mu, sigma, rez))
            ans_tex.append(tmp_tex)
            cmp_tex.append(tmp_my)

        # pdb.set_trace()
        log_document = f'{fgn}_fifth.csv'  # suff + 'tab_fifth.csv'
        csv_t = open(log_document, 'w')
        csv_w = csv.writer(csv_t)
        csv_w.writerow(['df'])
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerows(cmp_tex)
        csv_w.writerows([[''], ] * 4)  # [['', ] * 4])

        dt = self.obtn_clf_result(df_nonbin, clf, **kw)  # tb,tag_sa1,[],
        ta = ta[:4] + tag_sa1[:4] + tag_sa1[-4:]
        ans_tex, cmp_tex = [], []
        for jj in dt:  # for jj in range(len(dt)):
            tmp_tex, tmp_my = [''], ['']
            for k, v in enumerate(ta):
                ii = jj[v].values.astype(DTY_FLT)  # dt[jj][v]
                if k < 4:
                    ii *= 100.
                tmp_tex.append(_encode_sign(
                    ii.mean().tolist(), ii.std(ddof=ddof).tolist(), rez))
                mu, _, sigma = _avg_and_stdev(ii.tolist(), self._nb_iter)
                tmp_my.append(_encode_sign(mu, sigma, rez))
            ans_tex.append(tmp_tex)
            cmp_tex.append(tmp_my)
        csv_w.writerow(['df_nonbin'])
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerows(cmp_tex)
        csv_w.writerows([[''], ] * 4)
        # pdb.set_trace()
        dt = self.obtn_clf_result(df_bin, clf, **kw)
        ans_tex, cmp_tex = [], []
        for jj in dt:  # range(len(dt)):  # nb_set):
            tmp_tex, tmp_my = [''], ['']
            for k, v in enumerate(ta):
                ii = jj[v].values.astype(DTY_FLT)  # dt[jj][v]
                if k < 4:
                    ii *= 100.
                tmp_tex.append(_encode_sign(
                    ii.mean().tolist(), ii.std(ddof=ddof).tolist(), rez))
                mu, _, sigma = _avg_and_stdev(ii.tolist(), self._nb_iter)
                tmp_my.append(_encode_sign(mu, sigma, rez))
            ans_tex.append(tmp_tex)
            cmp_tex.append(tmp_my)
        csv_w.writerow(['df_bin'])
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerows(cmp_tex)
        csv_w.writerows([[''], ] * 4)

        csv_t.close()
        del csv_t, csv_w, log_document
        return

    # def obtn_clf_result(self, df,  # tb, tag_sa1, tag_sa2, clf,
    #                     clf, nb_set, id_set, nc_norm=3, nc_sens=4):
    #     df_alt = []  # nk = self._nb_iter
    #     if clf <= nc_norm + nc_sens:
    #         k = id_set[0] + 1
    #         start_i = k + (clf - 1) * self._nb_iter
    #         end_i = k + clf * self._nb_iter - 1  # start_i+nk-1
    #         df_alt = df[0].loc[start_i: end_i]   # df[0][:nk]
    #         df_alt = [df_alt, ]
    #     for k in range(1, nb_set):
    #         start_i = id_set[k] + 1 + (clf - 1) * self._nb_iter
    #         end_i = start_i + self._nb_iter - 1
    #         pdb.set_trace()
    #         df_alt.append(df[k].loc[start_i: end_i])
    #     return df_alt  # pdb.set_trace()

    def obtn_clf_result(self, df, clf, nc_norm=3, nc_sens=4):
        df_alt = []
        if clf <= nc_norm + nc_sens:
            start_i = (clf - 1) * self._nb_iter
            end_i = clf * self._nb_iter
            df_alt = df[0][start_i: end_i]  # df[0].iloc[start_i:end_i]
            df_alt = [df_alt, ]
        for k in df[1:]:
            start_i = (clf - 1) * self._nb_iter
            df_alt.append(k[start_i: start_i + self._nb_iter])
        return df_alt


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
        del tmp
        return tag_norm, tag_gf, tag_hfm, tag_df_conv

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

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn='t_D.', eb=True):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
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
        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                  '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        if 'multivar' in fgn:
            # multi_lin_reg_without_distr(
            #     scat_X, scat_Y[1:], annotY[1:], annot, fgn,
            #     snspec ='sty4c')
            # if eb:
            annotY[0] = r'$T_{EarlyBreak^{multi}}$'
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot, fgn, snspec='sty4')

            scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
            annot[1] = '${}$'.format(ant_Z)
            multi_lin_reg_without_distr(
                scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6')
            del scat_Z, ant_Z, annot
            return
        scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                       df_tmp[tag[6]].values.astype(DTY_FLT)])
        scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        annotY[0] = '${}$'.format(ant_eff)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Y, annotY, annot, fgn, snspec ='sty4')
        multi_lin_reg_without_distr(
            scat_X, scat_Y[:4], annotY[:4], annot, fgn, snspec='sty4')
        multi_lin_reg_without_distr(
            scat_X, scat_Y[1:2] + scat_Y[4:], annotY[1:2] + annotY[4:],
            annot, fgn + '_bin', snspec='sty4e')  # 'sty4d'

        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(scat_X, scat_Z[:4], annotY[:4],
                                    annot, fgn + '_s', snspec='sty6')
        multi_lin_reg_without_distr(
            scat_X, scat_Z[1:2] + scat_Z[4:], annotY[1:2] + annotY[4:],
            annot, fgn + '_s_bin', snspec='sty6e')
        del scat_Z, ant_Z, annot
        return

    def sub_plt_val(self, df_tmp, tag, fgn, sgn='D.', eb=True):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        if 'multivar' in fgn:
            ant_app = r'ExtendDist'
            ant_cvg = r'ExactDist(StratES)'
            ant_arr = r'ExactDist(StratRA)'
        else:
            ant_app = r'ApproxDist'
            ant_cvg = r'StratES'
            ant_arr = r'StratRA'
        ant_eff = 'EarlyBreak'
        # annotY = [r'EarlyBreak$^{(multi)}$', ant_app, ant_cvg, ant_arr]
        annotY = [r'$\mathrm{EarlyBreak}^{multi}$', ant_app, ant_cvg, ant_arr]
        annot = ['${}$'.format(ant_X), '${}$'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        if 'multivar' in fgn:
            # multi_lin_reg_without_distr(
            #     scat_X, scat_Y[1:], annotY[1:], annot, fgn,
            #     snspec='sty3c')
            # if eb:
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot, fgn, snspec='sty3b')

            scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
            annot[1] = '${}$'.format(ant_Z)
            multi_lin_reg_without_distr(
                scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6')
            return
        scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
                       df_tmp[tag[6]].values.astype(DTY_FLT)])
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        annotY[0] = ant_eff
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Y, annotY, annot, fgn,
        #     snspec='sty3b', figsize='L-WS')  # 'sty3b','L-NT')
        multi_lin_reg_without_distr(
            scat_X, scat_Y[:4], annotY[:4], annot, fgn,
            snspec='sty3b')  # , figsize='L-WS')
        multi_lin_reg_without_distr(
            scat_X, scat_Y[1:2] + scat_Y[4:], annotY[1:2] + annotY[4:],
            annot, fgn + '_bin', snspec='sty3e')  # , figsize='L-WS')  # 3d

        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z[:4], annotY[:4], annot, fgn + '_s', snspec='sty6')
        multi_lin_reg_without_distr(
            scat_X, scat_Z[1:2] + scat_Z[4:], annotY[1:2] + annotY[4:],
            annot, fgn + '_s_bin', snspec='sty6e')
        return

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
        tmp = self.obtain_binval_senatt(
            dframe, id_set, [], tag_sa1 + tag_tim[0], tag_sa2 + tag_tim[1])
        t_tim = np.array(tag_tim[0]).reshape(-1, 2).T.tolist()
        fgn += '_singvar'
        # pdb.set_trace()
        self.thread_previous(tmp, t_tim, fgn, tag_val=np.array(
            tag_sa1).reshape(-1, 2).T.tolist(), tag_avg=[])
        return

    def sub_plt_tim_prev(self, df_tmp, tag, fgn, sgn='t_D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        annotY = [r'$T_{EarlyBreak}$',
                  r'$T_{ApproxDist \text{(prev)}}$',
                  '${}$'.format(ant_X)]
        annot = ['${}$ (sec)'.format(ant_Xp), '${}$ (sec)'.format(
            ant_Yq), '${} = {}$'.format(ant_Yq, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty4')

        kws = {'snspec': 'sty6'}
        scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s', **kws)
        del scat_Z, ant_Z, kws
        return

    def sub_plt_val_prev(self, df_tmp, tag, fgn, sgn='D.'):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_val(fgn, sgn)
        ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        # pdb.set_trace()
        annotY = ['EarlyBreak', '${}$'.format(ant_Yq),
                  '${}$'.format(ant_X),
                  '${}$'.format(_hfm_dict_avg(fgn, sgn)[0])]
        # annotY[1] += r' ApproxDist(prev)'
        annot = ['${}$'.format(ant_Xp), '${}$'.format(
            ant_Yq), '${} = {}$'.format(ant_Yq, ant_Xp)]
        multi_lin_reg_without_distr(
            scat_X, scat_Y, annotY, annot, fgn, snspec='sty3b')

        kws = {'snspec': 'sty6'}
        scat_Z = [differentiate_val(scat_X, k) for k in scat_Y]
        annot[1] = '${}$'.format(ant_Z)
        multi_lin_reg_without_distr(
            scat_X, scat_Z, annotY, annot, fgn + '_s_bin', **kws)
        del scat_Z, kws, ant_Z
        return

    def subfig_fair_sp(self, dframe, tag_norm, tag_dr, tag_hfm, tag_conv, fgn):
        tmp_dr = [i[:3] + i[5:7] + i[3:4] for i in tag_dr]  # GFx3,SPx2,DR
        tmp_hfm = [[i[j] for j in [2, 0, 4, 3]] for i in tag_hfm['df']]
        tmp_conv = [i[2:5] + i[1:2] + i[5:] for i in tag_conv['df']]
        tmp_multivar = tag_conv['multivar'][6:][4:7] + tag_conv['multivar'][
            3:4] + tag_conv['multivar'][6 + 9:][4:6] + tag_conv['multivar'][
            6 + 9 + 8:][4:6] + tag_conv['multivar'][6 + 9 + 8 + 8:][4:6]
        # tmp_hfm:      {DistDirect_bin|EarlyBreak|ApproxDist_bin} df_prev,
        #               {DistDirect_bin} df
        # tmp_conv:     DistDirect_nonbin {df_prev,df,df_avg}, EffHD_nonbin{
        #               df}, Vacant {df,df_avg}, StratES x2, StratRA x2
        # tmp_multivar: DistDirect_multivar {df_prev, df,df_avg},
        #               EffHD_multivar{df}, Vacant {df,df_avg}, StratES x2,
        #               StratRA x2
        tmp_norm = [i[:4] + i[5:] + i[4:5] for i in tag_norm]
        tag_sa1 = tmp_dr[0] + tmp_hfm[0][:4] + tmp_conv[0][:3]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1][:4] + tmp_conv[1][:3]
        tb = tmp_norm[1][:-1] + tmp_norm[0][:-1]  # delta,oo
        self.thread_fair(dframe, tb + tmp_multivar, tag_sa1, tag_sa2, fgn)

        tb = tmp_norm[1] + tmp_norm[0]  # add f1_score
        tag_sa1 = tmp_dr[0] + tmp_hfm[0][:1] + tmp_conv[0][1:3]
        tag_sa2 = tmp_dr[1] + tmp_hfm[1][:1] + tmp_conv[0][1:3]
        self.thread_tabulate(dframe, tb + tmp_multivar[1:3], tag_sa1, tag_sa2,
                             fgn.replace('exp', 'tab'))  # f'{fgn}tab_')
        return


# ------------------------------


# ------------------------------


# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9e -pre min_max  # 8e|9e,
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9g -pre min_max  # 9f|c|d|b
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9h|i # -pre min_max


# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9i
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9h
# python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9g
