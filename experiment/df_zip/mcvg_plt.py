# coding: utf-8


import pdb
import csv
import numpy as np
import pandas as pd

from hfm.hfm_df import differentiate_tim, differentiate_val
from hfm.utils.verifiers import unique_column, DTY_FLT
from experiment.utils_empirical import GraphSetupVer2
from experiment.df_nonbin.rev_mext_plt_cor import (
    _hfm_dict_tim, _hfm_dict_val, _hfm_dict_avg, _hfm_dict_prev)

from pyfair.granite.draw_addtl import (
    hyper_params_lin_reg, multi_lin_reg_without_distr)
from experiment.df_zip.cont_draw import hyper_pms_lin_reg_gather


class GraphSetup(GraphSetupVer2):
    # def sub_dat_multival(self):
    #     pass
    # def sub_dat_binval(self):
    #     pass

    def obtn_sa_bin(self, dframe, id_set, pos, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_sa1]
        start = id_set[1] + 1 + pos * self._nb_iter
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
            df_tmp = df_tmp[tag_sa2].rename(columns=columns)
            start = id_set[k] + 1 + pos * self._nb_iter
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_sa1]
            start = id_set[k] + 1 + pos * self._nb_iter
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw  # .reset_index(drop=True)

    def obtn_sa_nonbin(self, dframe, id_set, pos, tag_sa1, tag_sa2,
                       first_incl=True):  # ,dr_ptb=''):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_sa1]
        start = id_set[2] + 1 + pos * self._nb_iter
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
            df_tmp = df_tmp[tag_sa2].rename(columns=columns)
            start = id_set[k] + 1 + pos * self._nb_iter
            df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        if not first_incl:
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_sa1]
        start = id_set[0] + 1 + pos * self._nb_iter
        df_tmp = df_tmp.loc[start: start + self._nb_iter - 1]
        df_raw = pd.concat([df_tmp, df_raw], axis=0)
        return df_raw  # .reset_index(drop=True)

    def obtn_sa_multivar(self, dframe, id_set, pos, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[0] + 1: id_set[1]][tag_sa1]
        start = id_set[0] + 1 + pos * self._nb_iter
        df_raw = df_raw.loc[start: start + self._nb_iter - 1]
        for i in range(1, len(id_set) - 1):
            df_tmp = dframe.iloc[id_set[i] + 1: id_set[i + 1]]
            df_no_sa1 = df_tmp[tag_sa1]
            df_no_sa2 = df_tmp[tag_sa2]
            df_no_sa2 = df_no_sa2.rename(columns=columns)
            start = id_set[i] + 1 + pos * self._nb_iter
            df_no_sa1 = df_no_sa1.loc[start: start + self._nb_iter - 1]
            df_no_sa2 = df_no_sa2.loc[start: start + self._nb_iter - 1]
            tmp = [df_raw, df_no_sa1]
            if len(tag_sa2) > 0:
                tmp.append(df_no_sa2)
            df_raw = pd.concat(tmp, axis=0)
        return df_raw  # .reset_index(drop=True)


# ------------------------------
# cf. Plot5_hyperpm
#           [rev_mext_plt_cor.py]
#


class DistPerf_draw(GraphSetup):
    def __init__(self, nk, m1=20, m2=8, n_e=2, n_p=3, figname='exp1_'):
        self._nb_cv = 1 if nk <= 0 else nk
        self._n_p = n_p
        gen = rep = None  # False
        super().__init__(gen, rep, m1, m2, n_e, figname)
        # Hyper-parameters
        # self._m2_set = list(range(2, 14, 1))  # len=12
        # self._m1_set = list(range(3, 34, 2))  # len=16
        # end of Hyper-parameters

    def prepare_graph(self, n_l):
        n_ell = (1 + n_l) * 2 * 3
        csv_row_1 = unique_column(10 + n_ell * 3)
        # params = csv_row_1[: 10]
        d_max = csv_row_1[10: 10 + n_ell]
        d_avg = csv_row_1[10 + n_ell: 10 + n_ell * 2]
        tag_tim = csv_row_1[10 + n_ell * 2:]
        n_ell = (1 + n_l) * 2
        tag_d_max = [d_max[:n_ell], d_max[n_ell: -n_ell], d_max[-n_ell:]]
        tag_d_avg = [d_avg[:n_ell], d_avg[n_ell: -n_ell], d_avg[-n_ell:]]
        tag_tim = [tag_tim[:n_ell], tag_tim[n_ell: -n_ell], tag_tim[-n_ell:]]
        # if n_l < 14:
        #     idx = [2, 4, 6, 8, 10]
        #     flag = [f'$m_2$={self._m2_set[i]}' for i in picked_m]
        # else:
        #     idx = [0, 3, 6, 9, 12]
        #     flag = [f'$m_1$={self._m1_set[i]}' for i in picked_m]
        picked_m = [2, 4, 6, 8, 10] if n_l < 14 else [0, 3, 6, 9, 12]
        return tag_d_max, tag_d_avg, tag_tim, picked_m  # idx, flag

    def incise_graph(self, tag, n_l, picked_m):  # n_l=12|16
        # idx = [2, 4, 6, 8, 10] if n_l < 14 else [0, 3, 6, 9, 12]
        tag_app = tag[1: 1 + n_l]
        tag_app = [tag_app[i] for i in picked_m]
        tag_arr = tag[-n_l:]
        tag_arr = [tag_arr[i] for i in picked_m]
        return tag[:1], tag_app, tag[1 + n_l: 2 + n_l], tag_arr

    def sub_plt_tim(self):
        return

    def sub_plt_val(self):
        return

    def quick_rmk(self, rmk='tim', interm=False):
        if interm:
            if rmk.endswith('tim'):
                annotX = r'T_{ \mathbf{D}_{a}(S,a_i) }'
                annotY = r'T_{ \hat{\mathbf{D}}_a(S,a_i) }'
                anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_a(S,a_i)} }{ T_{\mathbf{D}_a(S,a_i)} })'
                anotID = r'T_{ \hat{\mathbf{D}}_a }=T_{ \mathbf{D}_a }'
            elif rmk.endswith('avg'):
                annotX = r'\mathbf{D}_a^\text{avg}(S,a_i)'
                annotY = r'\hat{\mathbf{D}}_a^\text{avg}(S,a_i)'
                anotAP = r'\frac{ \hat{\mathbf{D}}_a^\text{avg}(S,a_i) }{ \mathbf{D}_a^\text{avg}(S,a_i) }-1'
                anotID = r'\hat{\mathbf{D}}_a^\text{avg} = \mathbf{D}_a^\text{avg}'
            else:
                annotX = r'\mathbf{D}_a(S,a_i)'
                annotY = r'\hat{\mathbf{D}}_a(S,a_i)'
                anotAP = r'\frac{ \hat{\mathbf{D}}_a(S,a_i) }{ \mathbf{D}_a(S,a_i) }-1'
                anotID = r'\hat{\mathbf{D}}_a = \mathbf{D}_a'
            return annotX, annotY, anotAP, anotID

        if rmk.endswith('tim'):
            annotX = r'T_{ \mathbf{D}_{\mathbf{a}}(S) }'
            annotY = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }'
            anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
            anotID = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}} }=T_{ \mathbf{D}_{\mathbf{a}} }'
        elif rmk.endswith('avg'):
            annotX = r'\mathbf{D}_{\mathbf{a}}^\text{avg}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }-1'
            anotID = r'\hat{\mathbf{D}}_{\mathbf{a}}^\text{avg} = \mathbf{D}_{\mathbf{a}}^\text{avg}'
        else:
            annotX = r'\mathbf{D}_{\mathbf{a}}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'
            anotID = r'\hat{\mathbf{D}}_{\mathbf{a}} = \mathbf{D}_{\mathbf{a}}'
        return annotX, annotY, anotAP, anotID


class cvgPlt1_anal_gather(DistPerf_draw):
    _m2_set = list(range(2, 14, 1))  # len=12
    _m1_set = list(range(3, 34, 2))  # len=16

    def schedule_mspaint(self, raw_df_mA, raw_df_mB, pre='minmax'):
        ma_n_l, mb_n_l = len(self._m2_set), len(self._m1_set)
        ma_max, ma_avg, ma_tim, ma_pick = self.prepare_graph(ma_n_l)
        ma_picked_set = [r'$m_2$={}'.format(self._m2_set[i]) for i in ma_pick]
        mb_max, mb_avg, mb_tim, mb_pick = self.prepare_graph(mb_n_l)
        mb_picked_set = [r'$m_1$={}'.format(self._m1_set[i]) for i in mb_pick]
        id_set = self.recap_sub_data(raw_df_mA, nb_row=4, nc_norm=6, nc_sens=0)[1]
        id_set = self.recap_sub_data(raw_df_mB, nb_row=4, nc_norm=6, nc_sens=0)[1]
        n_l = {'ma': ma_n_l, 'mb': mb_n_l}
        picked_m = {'ma': ma_pick, 'mb': mb_pick}
        tag_d_max = {'ma': ma_max, 'mb': mb_max}
        tag_d_avg = {'ma': ma_avg, 'mb': mb_avg}
        tag_tim = {'ma': ma_tim, 'mb': mb_tim}
        picked_set = {'ma': ma_picked_set, 'mb': mb_picked_set}
        del ma_n_l, ma_max, ma_avg, ma_tim, ma_pick, ma_picked_set
        del mb_n_l, mb_max, mb_avg, mb_tim, mb_pick, mb_picked_set
        raw_df = {'ma': raw_df_mA, 'mb': raw_df_mB}

        pos = 2
        fgn = f'{self._figname}_{pre}_pos{pos}_multivar'
        self.present_multivar_gather(
            n_l, tag_tim, picked_m, picked_set, fgn, raw_df, id_set, pos, 'tim')
        self.present_multivar_gather(
            n_l, tag_d_max, picked_m, picked_set, fgn, raw_df, id_set, pos, 'dmax')
        self.present_multivar_gather(
            n_l, tag_d_avg, picked_m, picked_set, fgn, raw_df, id_set, pos, 'davg')

        fgn = fgn.replace('multivar', 'sasing')  # 'sa_non')
        self.present_sa_sing_gather(
            n_l, tag_tim, picked_m, picked_set, fgn, raw_df, id_set, pos, 'tim')
        self.present_sa_sing_gather(
            n_l, tag_d_max, picked_m, picked_set, fgn, raw_df, id_set, pos, 'dmax')
        self.present_sa_sing_gather(
            n_l, tag_d_avg, picked_m, picked_set, fgn, raw_df, id_set, pos, 'davg')
        return

    # def present_multivar_gather(self, ma_tags, ma_picked, mb_tags, mb_picked,
    #                             fgn, interm=False):  # intermediate=
    #     return
    def present_multivar_gather(self, n_l, tags, picked_m, picked_set, fgn,
                                dframe, id_set, pos, rmk):
        tag_ma = self.incise_graph(tags['ma'][0], n_l['ma'], picked_m['ma'])
        tag_mb = self.incise_graph(tags['mb'][0], n_l['mb'], picked_m['mb'])
        tag_ma = tag_ma[0] + tag_ma[1] + tag_ma[2] + tag_ma[3]
        tag_mb = tag_mb[0] + tag_mb[1] + tag_mb[2] + tag_mb[3]
        df_ma = self.obtn_sa_multivar(dframe['ma'], id_set, pos, tag_ma, [])
        df_mb = self.obtn_sa_multivar(dframe['mb'], id_set, pos, tag_mb, [])

        tX_a = df_ma[tag_ma[0]].values.astype(DTY_FLT)
        tX_b = df_mb[tag_mb[0]].values.astype(DTY_FLT)
        tt = 5 + 1
        antX, antY, _, antZ = self.quick_rmk(rmk, interm=False)  # ,antAP
        annots = ['${}$ (sec)'.format(antX), '${}$ (sec)'.format(
            antY), '${}$'.format(antZ)]  # '${} = {}$'.format(antY, antX)]
        strt = 8 if rmk == 'tim' else (4 if rmk.endswith('avg') else 0)
        hyper_pms_lin_reg_gather([tX_a, tX_a, tX_b, tX_b], [
            df_ma[tag_ma[1:tt]].values.astype(DTY_FLT).T,
            df_ma[tag_ma[-5:]].values.astype(DTY_FLT).T,
            df_mb[tag_mb[1:tt]].values.astype(DTY_FLT).T,
            df_mb[tag_mb[-5:]].values.astype(DTY_FLT).T, ], [
            picked_set['ma'], picked_set['ma'], picked_set['mb'], picked_set['mb']],
            annots, f'{fgn}_{rmk}', strt=strt, snspec='sty5b',
        )  # snspec='sty6', differentiate=True)

        # strt = 4 if rmk == 'tim' else (2 if rmk.endswith('avg') else 0)
        # hyper_pms_lin_reg_gather([tX_a, tX_b], [
        #     df_ma[tag_ma[tt]].values.astype(DTY_FLT).T,
        #     df_mb[tag_mb[tt]].values.astype(DTY_FLT).T, ],
        #     ['', ''], annots, f'{fgn}_my_{rmk}', strt=strt,
        #     # snspec='sty3{}'.format('b' if rmk != 'tim' else 'a'))
        #     snspec=('sty8b' if rmk != 'tim' else 'sty8a'))
        return

    def present_sa_sing_gather(self, n_l, tags, picked_m, picked_set, fgn,
                               dframe, id_set, pos, rmk, sa_bin=False):
        tag_sa = {}
        tag_sa1 = self.incise_graph(tags['ma'][1], n_l['ma'], picked_m['ma'])
        tag_sa2 = self.incise_graph(tags['ma'][2], n_l['ma'], picked_m['ma'])
        tag_sa1 = tag_sa1[0] + tag_sa1[1] + tag_sa1[2] + tag_sa1[3]
        tag_sa2 = tag_sa2[0] + tag_sa2[1] + tag_sa2[2] + tag_sa2[3]
        df_ma = self.obtn_sa_multivar(dframe['ma'], id_set, pos, tag_sa1, tag_sa2)
        tag_sa['ma_sa1'] = tag_sa1
        tag_sa['ma_sa2'] = tag_sa2
        tag_sa1 = self.incise_graph(tags['mb'][1], n_l['mb'], picked_m['mb'])
        tag_sa2 = self.incise_graph(tags['mb'][2], n_l['mb'], picked_m['mb'])
        tag_sa1 = tag_sa1[0] + tag_sa1[1] + tag_sa1[2] + tag_sa1[3]
        tag_sa2 = tag_sa2[0] + tag_sa2[1] + tag_sa2[2] + tag_sa2[3]
        df_mb = self.obtn_sa_multivar(dframe['mb'], id_set, pos, tag_sa1, tag_sa2)
        tag_sa['mb_sa1'] = tag_sa1
        tag_sa['mb_sa2'] = tag_sa2

        tX_a = df_ma[tag_sa['ma_sa1'][0]].values.astype(DTY_FLT)
        tX_b = df_mb[tag_sa['mb_sa1'][0]].values.astype(DTY_FLT)
        tt = 5 + 1
        antX, antY, antY_diff, antZ = self.quick_rmk(rmk, interm=True)
        annots = ['${}$ (sec)'.format(antX), '${}$ (sec)'.format(
            antY), '${}$'.format(antZ)]  # '${} = {}$'.format(antY, antX)]
        strt = 8 if rmk == 'tim' else (4 if rmk.endswith('avg') else 0)
        curr_Ys = [df_ma[tag_sa['ma_sa1'][1:tt]].values.astype(DTY_FLT).T,
                   df_ma[tag_sa['ma_sa1'][-5:]].values.astype(DTY_FLT).T,
                   df_mb[tag_sa['mb_sa1'][1:tt]].values.astype(DTY_FLT).T,
                   df_mb[tag_sa['mb_sa1'][-5:]].values.astype(DTY_FLT).T, ]
        curr_Zs = [picked_set['ma'], picked_set['ma'], picked_set['mb'], picked_set['mb']]
        hyper_pms_lin_reg_gather([tX_a, tX_a, tX_b, tX_b], curr_Ys, curr_Zs,
                                 annots, f'{fgn}_{rmk}', strt=strt, snspec='sty5b')
        # annots[1] = '${}$'.format(antY_diff)
        # hyper_pms_lin_reg_gather([tX_a, tX_a, tX_b, tX_b], curr_Ys, curr_Zs,
        #                          annots, f'{fgn}_{rmk}6', strt=strt,  # snspec='sty5b',
        #                          snspec='sty6', differentiate=True)
        # annots[1] = '${}$ (sec)'.format(antY)

        if not sa_bin:
            return
        fgn = fgn.replace('sasing', 'sa_non')
        df_ma = self.obtn_sa_nonbin(
            dframe['ma'], id_set, pos, tag_sa['ma_sa1'], tag_sa['ma_sa2'])
        df_mb = self.obtn_sa_nonbin(
            dframe['mb'], id_set, pos, tag_sa['mb_sa1'], tag_sa['mb_sa2'])
        tX_a = df_ma[tag_sa['ma_sa1'][0]].values.astype(DTY_FLT)
        tX_b = df_mb[tag_sa['mb_sa1'][0]].values.astype(DTY_FLT)
        hyper_pms_lin_reg_gather([tX_a, tX_a, tX_b, tX_b], [
            df_ma[tag_sa['ma_sa1'][1:tt]].values.astype(DTY_FLT).T,
            df_ma[tag_sa['ma_sa1'][-5:]].values.astype(DTY_FLT).T,
            df_mb[tag_sa['mb_sa1'][1:tt]].values.astype(DTY_FLT).T,
            df_mb[tag_sa['mb_sa1'][-5:]].values.astype(DTY_FLT).T, ], [
            picked_set['ma'], picked_set['ma'], picked_set['mb'], picked_set['mb']],
            annots, f'{fgn}_{rmk}', strt=strt, snspec='sty5b')

        fgn = fgn.replace('sa_non', 'sa_bin')  # 'sasing'
        df_ma = self.obtn_sa_bin(
            dframe['ma'], id_set, pos, tag_sa['ma_sa1'], tag_sa['ma_sa2'])
        df_mb = self.obtn_sa_bin(
            dframe['mb'], id_set, pos, tag_sa['mb_sa1'], tag_sa['mb_sa2'])
        tX_a = df_ma[tag_sa['ma_sa1'][0]].values.astype(DTY_FLT)
        tX_b = df_mb[tag_sa['mb_sa1'][0]].values.astype(DTY_FLT)
        hyper_pms_lin_reg_gather([tX_a, tX_a, tX_b, tX_b], [
            df_ma[tag_sa['ma_sa1'][1:tt]].values.astype(DTY_FLT).T,
            df_ma[tag_sa['ma_sa1'][-5:]].values.astype(DTY_FLT).T,
            df_mb[tag_sa['mb_sa1'][1:tt]].values.astype(DTY_FLT).T,
            df_mb[tag_sa['mb_sa1'][-5:]].values.astype(DTY_FLT).T, ], [
            picked_set['ma'], picked_set['ma'], picked_set['mb'], picked_set['mb']],
            annots, f'{fgn}_{rmk}', strt=strt, snspec='sty5b')
        return


class cvgPlt1A_anal(DistPerf_draw):
    _m2_set = list(range(2, 14, 1))  # len=12

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m2_set)
        tag_d_max, tag_d_avg, tag_tim, picked_m = self.prepare_graph(n_l)
        ms_set = [r'$m_2$={}'.format(self._m2_set[i]) for i in picked_m]
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=6, nc_sens=0)

        # tag = self.incise_graph(tag_tim[0], n_l, picked_m)
        # tag = tag[0] + tag[1] + tag[2] + tag[3]
        # df_multivar = self.obtn_sa_multivar(raw_dframe, id_set, 2, tag, [])
        # fgn = f'{self._figname}_{pre}'
        # kw = dict(snspec='sty5b', corr=False)
        # tX = df_multivar[tag[0]].values.astype(DTY_FLT)
        # antX, antY, antAP = self.quick_rmk('tim', False)
        # annots = ['${}$ (sec)'.format(antX), '${}$ (sec)'.format(
        #     antY), '${} = {}$'.format(antY, antX)]
        # pdb.set_trace()
        # hyper_params_lin_reg(tX, df_multivar[tag[1:6]].values.astype(DTY_FLT).T,
        #                      ms_set, picked_m, annots, fgn, **kw)
        return

    # def prepare_graph(self):
    #     csv_row_1 = unique_column(10 + 78 * 3)
    #     return


class cvgPlt1B_anal(DistPerf_draw):
    _m1_set = list(range(3, 34, 2))  # len=16

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m1_set)
        tag_d_max, tag_d_avg, tag_tim, picked_m = self.prepare_graph(n_l)
        ms_set = [r'$m_1$={}'.format(self._m1_set[i]) for i in picked_m]
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=6, nc_sens=0)
        pdb.set_trace()
        return

    # def prepare_graph(self):
    #     csv_row_1 = unique_column(10 + 102 * 3)
    #     return


class cvgPlt1C_take(DistPerf_draw):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, _, _ = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=6, nc_sens=0)
        tag_wh, tag_sa1, tag_sa2 = self.prepare_graph()
        df_raw = self.obtn_sa_multivar(raw_dframe, id_set, 2, tag_sa1, tag_sa2)
        df_tmp = self.obtn_sa_multivar(raw_dframe, id_set, 2, tag_wh, [])
        fgn = self._figname + f'_{pre}'
        self.sub_plt_tim(df_tmp, tag_wh[:5], self._figname + '_multivar', 't_D')
        self.sub_plt_tim(df_raw, tag_sa1[:5], self._figname + '_sa1', 't_D')
        pdb.set_trace()
        return

    def prepare_graph(self):
        csv_row_1 = unique_column(10 + 42)
        # tag_pm = csv_row_1[:10]
        tag_tim = csv_row_1[10: 10 + 5 * 3]
        tag_d_max = csv_row_1[25: 25 + 5 * 3]
        tag_d_avg = csv_row_1[40: 40 + 4 * 3]
        tag_wh = tag_tim[:5] + tag_d_max[:5] + tag_d_avg[:4]
        tag_sa1 = tag_tim[5:10] + tag_d_max[5:10] + tag_d_avg[4:8]
        tag_sa2 = tag_tim[10:] + tag_d_max[10:] + tag_d_avg[8:]
        return tag_wh, tag_sa1, tag_sa2

    def sub_plt_tim(self, df_tmp, tag, fgn, sgn='t_D.'):  # ,eb=True):
        scat_X = df_tmp[tag[0]].values.astype(DTY_FLT)
        scat_Y = [df_tmp[tag[1]].values.astype(DTY_FLT),
                  df_tmp[tag[2]].values.astype(DTY_FLT),
                  df_tmp[tag[3]].values.astype(DTY_FLT),
                  df_tmp[tag[4]].values.astype(DTY_FLT)]
        ant_X, ant_Y, ant_Z = _hfm_dict_tim(fgn, sgn)
        if 'multivar' in fgn:
            ant_app = r'T_{ExtendDist}'
            ant_cvg = r'T_{ExactDist (StratES)}'
            ant_arr = r'T_{ExactDist (StartRA)}'
            ant_eff = r'T_{EarlyBreak^{multi}}'
        else:
            ant_app = r'T_{ApproxDist}'
            ant_cvg = r'T_{StratES}'
            ant_arr = r'T_{StartRA}'
            ant_eff = r'T_{EarlyBreak}'
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        annotY = ['${}$'.format(ant_eff), '${}$'.format(ant_app),
                  '${}$'.format(ant_cvg), '${}$'.format(ant_arr)]
        if 'multivar' in fgn:
            multi_lin_reg_without_distr(
                scat_X, scat_Y, annotY, annot, fgn, snspec='sty4')
            # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
            # annot[1] = '${}$'.format(ant_Z)
            # multi_lin_reg_without_distr(
            #     scat_X, scat_Z, annotY, annot, fgn + '_s', snspec='sty6')
            # del scat_Z, ant_Z, annot
            return

        # scat_Y.extend([df_tmp[tag[5]].values.astype(DTY_FLT),
        #                df_tmp[tag[6]].values.astype(DTY_FLT)])
        # scat_Y[0] = df_tmp[tag[7]].values.astype(DTY_FLT)
        # ant_Xp, ant_Yq = _hfm_dict_prev(sgn)
        # annotY.extend(['${}$'.format(ant_Xp), '${}$'.format(ant_Yq)])
        # annotY[0] = '${}$'.format(ant_eff)
        multi_lin_reg_without_distr(
            scat_X, scat_Y[:4], annotY[:4], annot, fgn, snspec='sty4')
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Y[1:2] + scat_Y[4:], annotY[1:2] + annotY[4:],
        #     annot, fgn + '_bin', snspec='sty4e')  # 'sty4d'

        # scat_Z = [differentiate_tim(scat_X, k) for k in scat_Y]
        # annot[1] = '${}$'.format(ant_Z)
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z[:4], annotY[:4], annot, fgn + '_s', snspec='sty6')
        # multi_lin_reg_without_distr(
        #     scat_X, scat_Z[1:2] + scat_Z[4:], annotY[1:2] + annotY[4:],
        #     annot, fgn + '_s_bin', snspec='sty6e')
        # del scat_Z, ant_Z, annot
        return


# ------------------------------
#
