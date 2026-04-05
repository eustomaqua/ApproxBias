# coding: utf-8


from experiment.utils_empirical import GraphSetupVer1
from experiment.df_bin.manf_plt import (
    _sub_depict_scat, _sub_depict_tim, _sub_depict_sep,
    _sub_depict_sep_alt, check_zero)

# from pyfair.utils_empirical import GraphSetup
from pyfair.utils_empirical import DAT_EXPT_NMS, DAT_EXPT_ORG
from pyfair.facil.utils_const import unique_column, DTY_FLT

from pyfair.facil.draw_prelim import DTY_PLT
from pyfair.granite.draw_fancy import (
    radar_chart, grped_radar_cht, tabular_chart)
from pyfair.granite.draw_chart import analogous_confusion_extended
from pyfair.granite.draw_addtl import (
    single_line_reg_with_distr, multi_lin_reg_without_distr)

import numpy as np
import pandas as pd
import os

import pdb
GRP_FAIR_COMMON = ['DP', 'EO', 'PQP']


class GraphSetup(GraphSetupVer1):
    _cmap_name = 'muted'
    _nb_cv = 5
    _datasets = {'abbr': DAT_EXPT_NMS, 'full': DAT_EXPT_ORG}

    @property
    def nb_cv(self):
        return self._nb_cv

    @nb_cv.setter
    def nb_cv(self, value):
        self._nb_cv = value

    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, figname=''):
        raise NotImplementedError

    def recap_sub_data(self, dframe, nb_row=4,
                       sa_ir=3, sa_r=4, n_a=2):
        # sa_irrelevant, sa_relevant: classifiers/learners
        each = (sa_ir + sa_r * n_a) * self._nb_cv + 1
        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set + sa_r * self._nb_cv) // each
        each_sing = (sa_ir + sa_r) * self._nb_cv + 1
        id_set = [0] + [(
            i * each + each_sing) for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]
        # id_set: [3, 39,  95, 151, 207, 263] clf. 3+4
        #         [3, 59, 115, 171, 227, 283] clf.11+0
        return nb_set, id_set  # id_set[-1] for bounding


# -----------------------------
# Exp1: bin-val vs. multi-val
#


class PlotA_initial(GraphSetup):
    _dr_ptb = 'K'
    _perf_metric = [
        'Accuracy', 'Precision', 'Recall', 'Specificity',
        r'$\mathrm{f}_1$ score', 'G mean', 'bal. acc', 'discr power']
    _dal_metric = [
        r'$\Delta$(Accuracy)', r'$\Delta$(Precision)',
        r'$\Delta$(Recall)', r'$\Delta$(Specificity)',
        r'$\Delta$($\mathrm{f}_1$ score)', r'$\Delta$(G mean)',
        r'$\Delta$(bal. acc)', r'$\Delta$(discr power)']

    def obtain_tag_col(self, tag='tst'):
        csv_row_1 = unique_column(12 + 158 * 2)
        tag_trn = csv_row_1[12: 12 + 158]
        tag_tst = csv_row_1[-158:]
        tag_col = tag_trn if tag == 'trn' else tag_tst

        # sub-tags
        st_acc = tag_col[: 24]
        st_grp = [tag_col[24: 24 + 29], tag_col[53: 24 + 58]]
        st_dr = tag_col[24 + 58: 24 + 58 + 18]  # 24+76=100
        st_hfm_drt = tag_col[100: 100 + 29]
        st_hfm_app = tag_col[100 + 29: 100 + 58]

        # tag_sa1 = st_grp[0][6:22] + st_grp[0][-1:] + st_hfm_drt[
        #     :4] + st_hfm_drt[15:22] + st_hfm_app[:4] + st_hfm_app[
        #     15:22] + st_grp[0][-7: -1]
        # tag_sa2 = st_grp[1][6:22] + st_grp[1][-1:] + st_hfm_drt[
        #     4:8] + st_hfm_drt[22:29] + st_hfm_app[4:8] + st_hfm_app[
        #     22:29] + st_grp[1][-7: -1]

        tag_ind = [st_dr[4], st_dr[6], st_dr[9], st_dr[12],
                   st_dr[14]] + st_dr[-3:] + st_dr[:4]
        # tag_df = st_hfm_drt[8:15] + st_hfm_app[8:15]
        tag_common = st_acc[:8] + st_acc[-8:] + tag_ind  # +tag_df
        tag_sa1 = st_grp[0][6:10] + st_hfm_drt[:4] + st_hfm_app[
            :4]  # + st_grp[0][-7:-1]
        tag_sa2 = st_grp[1][6:10] + st_hfm_drt[4:8] + st_hfm_app[
            4:8]  # + st_grp[1][-7:-1]
        # pdb.set_trace()

        # tag_common: perf 8+ delta(perf) 8+ GEI(alph=0|.2|.5|.8|1,Theil,
        #             T(Theil),T(GEIx11)) 6+ dr(loss,ut, hat_bias,ut) 4+
        #             hfm direct multiver 7+ hfm approx multiver 7
        # tag_sa1/2 : DP,EO,PQP,tim, hfm drt (bin 4), hfm app (bin 4)
        #             +tim (extGrp1|2|3, altGrp1|2|3, **tim)
        # siz: 16+8+4=28, 4+4*2(+6)=12(+6)=18|12
        # return tag_common, tag_sa1, tag_sa2
        return tag_common + csv_row_1[10:12], tag_sa1, tag_sa2

    def obtain_binval_senatt(self, dframe, id_set,  # nb_set,
                             tag='tst'):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[1]][self._dr_ptb]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set,  # nb_set,
                               tag='tst', first_incl=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[2]][self._dr_ptb]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[
                k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # df_raw = df_raw.reset_index(drop=True)
        # np.isnan(df_raw.values.astype('float')).any()
        if not first_incl:  # if not first: first_incl.
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_tmp[self._dr_ptb] = dframe.iloc[id_set[0]][self._dr_ptb]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_whole_bin_sa(self, dframe, id_set, tag='tst',
                            first_incl=True):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}

        tsa1 = tag_acc + tag_sa1
        tsa2 = tag_acc + tag_sa2
        k = 0
        df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tsa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
        for k in [1, 2, 3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tsa1]
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tsa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        if not first_incl:
            k = 0
            df_tmp = id_set[k + 1] - (id_set[k] + 1)
            df_raw = df_raw.iloc[df_tmp:]
        return df_raw  # pdb.set_trace()


class PlotA_drawing(PlotA_initial):
    # def __init__(self):
    #     pass

    def obtain_sing_dat_cls(self, pick_set, pick_clf, tag_acc,
                            tag_sa1, tag_sa2, dframe, id_set):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        picked_a = id_set[pick_set] + 1 + pick_clf * self._nb_cv
        picked_b = id_set[pick_set] + 1 + (pick_clf + 1) * self._nb_cv

        df_tmp = dframe.iloc[picked_a: picked_b]
        df_tmp = df_tmp[tag_acc + tag_sa1]
        if pick_set == 0:
            return df_tmp, None

        df_alt = dframe.iloc[picked_a: picked_b]
        df_alt = df_alt[tag_acc + tag_sa2]
        df_alt = df_alt.rename(columns=columns)
        return df_tmp, df_alt

    def depict_separately(self, pick_set, pick_clf, df, id_set,
                          mk='tst', fgn='', verbose=True):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(mk)
        df_at_1, df_at_2 = self.obtain_sing_dat_cls(
            pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2, df, id_set)
        if pick_set > 0 and pick_clf > 2:
            df_as_1, df_as_2 = self.obtain_sing_dat_cls(
                pick_set, pick_clf + 4,
                tag_acc, tag_sa1, tag_sa2, df, id_set)

        sub_my = tag_acc[16:][5 + 3:]
        sub_my = [sub_my[2], sub_my[0]]  # L_fair, L_loss
        sub_grp = tag_sa1[:3] + sub_my[:-1]
        sub_idv = tag_acc[16:][: 5 + 1]
        sub_idv = [sub_idv[2], sub_idv[5]]
        del sub_my
        sub_my = [tag_sa1[4 + 2], tag_sa1[4 + 4 + 2]]  # hfm

        labels = GRP_FAIR_COMMON + ['DR', r'GEI ($\gamma$=0.5)'] + [
            # 0, 0.2, 0.5, 0.8, 1]] + [
            # 0.5]] + [  # 0.2, 0.5, 0.8]] + [
            # 'Theil', 'DR', '0/1 loss'][:-1] + [
            'Theil', r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        if verbose:
            # labels = labels[:4] + [r'GEI ($\alpha$={:.1f})'.format(
            #     i) for i in [.2, .5, .8]] + labels[-3:]
            labels = labels[:4] + [
                r'{:7s}GEI ($\gamma$=0.2)'.format(''),
                r'GEI ($\gamma$=0.5)',
                r'GEI ($\gamma$=0.8){:7s}'.format('')] + labels[-3:]
            sub_idv = tag_acc[16:][: 5 + 1]
            sub_idv = [sub_idv[i] for i in [1, 2, 3, 5]]
        currX = sub_grp + sub_idv + sub_my
        # pdb.set_trace()

        df_tmp = df_at_1[currX].reset_index(drop=True)
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        radar_chart(df_tmp.iloc[:1], currX, labels, ['SA #1'],
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_sa1',
                    clockwise=True)
        if pick_set == 0:
            return
        os.remove(f'{fgn}_s{pick_set}c{pick_clf}_sa1' + DTY_PLT)

        df_tmp = df_tmp.iloc[:2]
        df_alt = df_at_2[currX]
        for i in currX:
            df_tmp.iloc[1][i] = float(df_alt[i].mean())
        radar_chart(df_tmp, currX, labels, ['SA #1', 'SA #2'],
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_sa2',
                    clockwise=True)
        if pick_clf <= 2:
            return
        # dtt = df_tmp

        df_tmp = df_as_1[currX].reset_index(drop=True)
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        df_tmp = df_tmp.iloc[:2]
        df_alt = df_as_2[currX]
        for i in currX:
            df_tmp.iloc[1][i] = float(df_alt[i].mean())
        radar_chart(df_tmp, currX, labels, ['SA #1', 'SA #2'],
                    # figname=f'{fgn}_s{pick_set}c{pick_clf+4}_sa2',
                    figname=f'{fgn}_s{pick_set}c{pick_clf}p_sa2',
                    clockwise=True)
        # pdb.set_trace()
        return

    def depict_sep_tabular(self, pick_set, pick_clf, df, id_set,
                           mk='tst', fgn=''):  # ,verbose=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(mk)
        sub_my = tag_acc[16:][5 + 3:]
        sub_my = [sub_my[2], sub_my[0]]  # L_fair, L_loss
        sub_grp = tag_sa1[:3] + sub_my[:-1]
        sub_idv = tag_acc[16:][: 5 + 1]
        sub_idv = [sub_idv[2], sub_idv[5]]
        del sub_my
        sub_my = [tag_sa1[4 + 2], tag_sa1[4 + 4 + 2]]  # hfm
        labels = GRP_FAIR_COMMON + ['DR', r'GEI ($\gamma$=0.5)', 'Theil',
                                    r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # if verbose:
        #     labels = labels[:4] + [
        #         r'{:7s}GEI ($\gamma$=0.2)'.format(''),
        #         r'GEI ($\gamma$=0.5)',
        #         r'GEI ($\gamma$=0.8){:7s}'.format('')] + labels[-3:]
        #     sub_idv = tag_acc[16:][: 5 + 1]
        #     sub_idv = [sub_idv[i] for i in [1, 2, 3, 5]]
        currX = sub_grp + sub_idv + sub_my

        def _internal(ps, pc):
            df_at_1, df_at_2 = self.obtain_sing_dat_cls(
                ps, pc, tag_acc, tag_sa1, tag_sa2, df, id_set)
            df_tmp = df_at_1[currX].reset_index(drop=True)
            for i in currX:
                df_tmp.loc[:, i] = float(df_tmp[i].mean())
            df_tmp = df_tmp.iloc[:2]
            df_alt = df_at_2[currX]
            for i in currX:
                df_tmp.iloc[1][i] = float(df_alt[i].mean())
            return df_tmp

        anotY = ['SA #1', 'SA #2']
        # labels[4] = 'GEI'
        nm_set = ['Ricci', 'Credit', 'Income', 'PPR', 'PPVR']
        nm_clf = ['bagging', 'AdaBoost', 'LightGBM',
                  'FairGBM', 'FairGBM', 'FairGBM', 'AdaFair (SA #1)',
                  'FairGBM', 'FairGBM', 'FairGBM', 'AdaFair (SA #2)']
        for ps in pick_set:
            for pc in pick_clf:
                # df_at_1, df_at_2 = self.obtain_sing_dat_cls(
                #     ps, pc, tag_acc, tag_sa1, tag_sa2, df, id_set)
                # if ps > 0 and pc > 2:
                #     df_as_1, df_as_2 = self.obtain_sing_dat_cls(
                #         ps, pc + 4, tag_acc, tag_sa1, tag_sa2, df, id_set)
                df_tmp = _internal(ps, pc)
                # radar_chart(df_tmp, currX, labels, anotY,
                #             figname=f'{fgn}_s{ps}c{pc}_radar', clockwise=True)
                tabular_chart(df_tmp, currX, labels[:4] + ['GEI'] + labels[5:],
                              anotY, figname=f'{fgn}_s{ps}c{pc}', data=nm_set[ps],
                              algo=nm_clf[pc], cumulate=True)
            # pdb.set_trace()
        return

    def depict_grouped_invt(self, pick_set, pick_clf_pl, df, id_set,
                            mk='tst', fgn='', verbose=True):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col()
        sub_my = tag_acc[16:][5 + 3:][2]  # dr
        sub_grp = tag_sa1[:3] + [sub_my]
        sub_idv = tag_acc[16:][:5 + 1]
        sub_idv = [sub_idv[2], sub_idv[5]]
        sub_my = [tag_sa1[4 + 2], tag_sa1[4 + 4 + 2]]  # hfm
        labels = GRP_FAIR_COMMON + [
            'DR', r'GEI ($\gamma$=0.5)', 'Theil',
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        if verbose:
            labels = labels[:4] + [
                r'{:7s}GEI ($\gamma$=0.2)'.format(''),
                r'GEI ($\gamma$=0.5)',
                r'GEI ($\gamma$=0.8){:7s}'.format('')] + labels[-3:]
            sub_idv = tag_acc[16:][:5 + 1]
            sub_idv = [sub_idv[i] for i in [1, 2, 3, 5]]
        currX = sub_grp + sub_idv + sub_my

        df_tmp_pl = []
        for pick_clf in pick_clf_pl:
            df_at_1, df_at_2 = self.obtain_sing_dat_cls(
                pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2, df, id_set)
            df_tmp = df_at_1[currX].reset_index(drop=True)
            for i in currX:
                df_tmp.loc[:, i] = float(df_tmp[i].mean())
            if pick_set == 0:
                df_tmp_pl.append(df_tmp.iloc[:1])
                continue
            df_tmp = df_tmp.iloc[:2]
            df_alt = df_at_2[currX]
            for i in currX:
                df_tmp.iloc[1][i] = float(df_alt[i].mean())
            df_tmp_pl.append(df_tmp)

        anotY = ['$Att_{sen}$ #1', '$Att_{sen}$ #2']
        fgn = f'{fgn}_s{pick_set}c_sa'
        grped_radar_cht(df_tmp_pl, currX, labels, anotY,
                        clockwise=True, figname=fgn)
        return

    def depict_grouped(self, pick_set_pl, pick_clf, df, id_set,
                       mk='tst', fgn='', verbose=True):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(mk)
        sub_my = tag_acc[16:][5 + 3:][2]  # dr
        sub_grp = tag_sa1[:3] + [sub_my]
        sub_idv = tag_acc[16:][:5 + 1]
        sub_idv = [sub_idv[2], sub_idv[5]]
        sub_my = [tag_sa1[4 + 2], tag_sa1[4 + 4 + 2]]  # hfm
        labels = GRP_FAIR_COMMON + [
            'DR', r'GEI ($\gamma$=0.5)', 'Theil',
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        if verbose:
            labels = labels[:4] + [
                r'{:7s}GEI ($\gamma$=0.2)'.format(''),
                r'GEI ($\gamma$=0.5)',
                r'GEI ($\gamma$=0.8){:7s}'.format('')] + labels[-3:]
            sub_idv = tag_acc[16:][:5 + 1]
            sub_idv = [sub_idv[i] for i in [1, 2, 3, 5]]
        currX = sub_grp + sub_idv + sub_my

        df_tmp_pl = []
        for pick_set in pick_set_pl:
            df_at_1, df_at_2 = self.obtain_sing_dat_cls(
                pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2,
                df, id_set)

            df_tmp = df_at_1[currX].reset_index(drop=True)
            for i in currX:
                df_tmp.loc[:, i] = float(df_tmp[i].mean())
            if pick_set == 0:
                df_tmp_pl.append(df_tmp.iloc[:1])
                continue
            df_tmp = df_tmp.iloc[:2]
            df_alt = df_at_2[currX]
            for i in currX:
                df_tmp.iloc[1][i] = float(df_alt[i].mean())
            df_tmp_pl.append(df_tmp)

        anotY = ['$Att_{sen}$ #1', '$Att_{sen}$ #2']
        fgn = f'{fgn}_sc{pick_clf}_sa'  # len(pick_set_pl)
        grped_radar_cht(df_tmp_pl, currX, labels, anotY,
                        clockwise=True, figname=fgn)
        return


class Ver2_PlotA_fair_ens(PlotA_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=3, sa_r=4)
        mk, first_incl = 'tst', True
        df_bin = self.obtain_binval_senatt(raw_dframe, id_set, mk)
        df_all = self.obtain_whole_bin_sa(
            raw_dframe, id_set, mk, first_incl)

        # pdb.set_trace()
        fgn = f'{figname}_radar'
        # for pkc in [0, 1, 2, 5, 6]:  # [3,4,]
        #     for pks in [1, 2, 3, 4]:  # [0,]
        #         self.depict_separately(
        #             pks, pkc, raw_dframe, id_set, mk, fgn)
        # # self.depict_separately(2, 2, raw_dframe, id_set, mk, fgn)

        # for pkc in [0, 1, 2, 5, 6, 5 + 4, 6 + 4]:
        #     self.depict_grouped([
        #         1, 2, 3, 4], pkc, raw_dframe, id_set, mk, fgn)
        # # self.depict_grouped([
        # #     0, 1, 2, 3, 4], 2, raw_dframe, id_set, mk, fgn)

        for pks in [1, 2, 3, 4]:
            self.depict_grouped_invt(pks, [
                0, 1, 2, 6, 6 + 4],
                raw_dframe, id_set, mk, fgn, verbose=False)

        fgn = f'{figname}_tab'
        self.depict_sep_tabular([1, 2, 3, 4], [
            # 0, 1, 2, 5, 6, 5 + 4, 6 + 4],
            0, 1, 2, 6, 6 + 4], raw_dframe, id_set, mk, fgn)
        return


class Ver2_PlotA_norm_cls(PlotA_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=11, sa_r=0)
        mk = 'tst'
        first_incl = True
        df_bin = self.obtain_binval_senatt(raw_dframe, id_set, mk)
        df_all = self.obtain_whole_bin_sa(
            raw_dframe, id_set, mk, first_incl)

        pdb.set_trace()
        return


# -----------------------------
#


def _ver4sub_depict_scat(df_raw, tYs, suff, diff=False):
    scat_X = np.concatenate([
        df_raw[tYs[0]].values.astype(DTY_FLT),
        df_raw[tYs[2]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[1]].values.astype(DTY_FLT),
        df_raw[tYs[3]].values.astype(DTY_FLT)], axis=0)
    annotX = r'\mathbf{D}_\cdot'
    annotY = r'\hat{\mathbf{D}}_\cdot'
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + '_sty3',
                               linreg=True, snspec='sty4')  # 'sty3b')
    if not diff:
        return
    scat_Z = np.zeros_like(scat_Y) - 1.
    for i, (xi, yi) in enumerate(zip(scat_X, scat_Y)):
        scat_Z[i] = np.abs(yi - xi) / check_zero(xi)
    annotZ = r'\frac{abs(\hat{\mathbf{D}}_\cdot-\mathbf{D}_\cdot)}{\mathbf{D}_\cdot}'
    annots = ['${}$'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_sty6',
                               linreg=True, snspec='sty6')
    return


def _ver4sub_depict_tim(df_raw, tYs, suff, diff=False, leq=False):
    scat_X = np.concatenate([
        df_raw[tYs[6]].values.astype(DTY_FLT),
        df_raw[tYs[8]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[7]].values.astype(DTY_FLT),
        df_raw[tYs[9]].values.astype(DTY_FLT)], axis=0)
    annotX = r'T_{\mathbf{D}_{\cdot}}'
    annotY = r'T_{\hat{\mathbf{D}}_{\cdot}}'
    annots = ['${}$ (sec)'.format(annotX),
              '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws = {'linreg': True, 'snspec': 'sty4'}
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + '_sty4', **kws)
    if not diff:
        return
    kws['snspec'] = 'sty6'
    scat_Z = scat_Y / scat_X - 1.

    annotZ = r'\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} }-1'
    annots = ['${}$  (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    if leq:
        annots[2] += ' ($\!\leq${:.2f}%)'.format(
            (scat_Z <= 0.).mean().tolist() * 100.)
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_sty6a', **kws)

    scat_Z = np.log10(scat_Z + 1)
    annotZ = r'\lg(\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} })'
    annots = ['${}$ (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    if leq:
        annots[2] += ' ($\!\leq${:.2f}%)'.format(
            (scat_Z <= 0.).mean().tolist() * 100.)
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_sty6b', **kws)
    return


def _ver4sub_depict_earlybreak(dfs, tYs, suff, mark='tD.'):
    scat_X = dfs[tYs[0]].values.astype(DTY_FLT)
    scat_Y = [dfs[tYs[1]].values.astype(DTY_FLT),
              dfs[tYs[2]].values.astype(DTY_FLT)]

    if mark.startswith('tD'):
        if mark == 'tDs':
            ant_X, ant_Y = r'T_{\mathbf{D}}', r'T_{\hat{\mathbf{D}}}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} })'
        elif mark == 'tDf':
            ant_X, ant_Y = r'T_{\mathbf{D}_f}', r'T_{\hat{\mathbf{D}}_f}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} })'
        else:
            ant_X = r'T_{\mathbf{D}_\cdot}'
            ant_Y = r'T_{\hat{\mathbf{D}}_\cdot}'
            ant_Z = r'\lg(\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} })'
        annotY = ['$T_{EarlyBreak}$', '$T_{ApproxBias}$']
        annot = ['${}$ (sec)'.format(ant_X), '${}$ (sec)'.format(
            ant_Y), '${} = {}$'.format(ant_Y, ant_X)]
        annotY[0] += ' $\leq {}$  {:.2f}%'.format(ant_X, (
            scat_Y[0] <= scat_X).mean().tolist() * 100.)
        annotY[1] += ' $\leq {}${:.2f}%'.format(ant_X, (
            scat_Y[1] <= scat_X).mean().tolist() * 100.)
    else:
        if mark == 'Ds':
            ant_X, ant_Y = r'\mathbf{D}', r'\hat{\mathbf{D}}'
            ant_Z = r'\frac{ \hat{\mathbf{D}} }{ \mathbf{D} }-1'
        elif mark == 'Df':
            ant_X, ant_Y = r'\mathbf{D}_f', r'\hat{\mathbf{D}}_f'
            ant_Z = r'\frac{ \hat{\mathbf{D}}_f }{ \mathbf{D}_f }-1'
        else:
            ant_X = r'\mathbf{D}_\cdot'
            ant_Y = r'\hat{\mathbf{D}}_\cdot'
            ant_Z = r'\frac{ \hat{\mathbf{D}_\cdot} }{ T_{\mathbf{D}_\cdot} }-1'
        annotY = ['EarlyBreak', 'ApproxBias']
        annot = ['${}$'.format(ant_X), '${}$'.format(ant_Y),
                 '${} = {}$'.format(ant_Y, ant_X)]

    multi_lin_reg_without_distr(
        scat_X, scat_Y, annotY, annot, suff,
        snspec='sty4' if mark.startswith('tD') else 'sty3b')  # 'sty3bf')

    if mark.startswith('tD'):
        scat_Z = [np.log10(i / scat_X) for i in scat_Y]
    else:
        scat_Z = [i / scat_X - 1. for i in scat_Y]
    annot[1] = '${}$'.format(ant_Z)  # ant_Z
    multi_lin_reg_without_distr(
        scat_X, scat_Z, annotY, annot, suff + 'p',
        snspec='sty6')
    # pdb.set_trace()
    return


class PlotH_drawing(GraphSetup):  # PlotA_initial):
    _dr_ptb = 'K'
    _perf_metric = [
        'Accuracy', 'Precision', 'Recall', 'Specificity',
        r'$\mathrm{f}_1$ score',  # 'G mean', 'bal. acc', 'discr power']
        'g mean', 'dp']
    _dal_metric = [
        r'$\Delta$(Accuracy)', r'$\Delta$(Precision)',
        r'$\Delta$(Recall)', r'$\Delta$(Specificity)',
        r'$\Delta$($\mathrm{f}_1$ score)',  # r'$\Delta$(G mean)',
        # r'$\Delta$(bal. acc)', r'$\Delta$(discr power)']
        r'$\Delta$(g mean)', r'$\Delta$(dp)']

    def obtain_binval_senatt(self, dframe, id_set, tag_acc, tag_sa1, tag_sa2):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_acc + tag_sa1]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        pdb.set_trace()
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set, tag_acc, tag_sa1, tag_sa2,
                               first_incl=True):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_acc + tag_sa1]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        if not first_incl:
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_raw = pd.concat([df_tmp, df_raw], axis=0)
        pdb.set_trace()
        return df_raw

    def obtain_whole_bin_sa(self, dframe, id_set, tag_acc, tag_sa1, tag_sa2,
                            first_incl=True):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        k = 0
        df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
        for k in range(1, len(id_set) - 1):  # [1, 2, 3]:  # ,4]
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        if not first_incl:
            k = 0
            df_tmp = id_set[k + 1] - (id_set[k] + 1)
            df_raw = df_raw.iloc[df_tmp:]
        # pdb.set_trace()
        return df_raw

    # def obtain_whole_bin_more(self, dframe, id_set, tag_acc, tag_sa1, tag_sa2,
    #                           tag_jta, tag_jto, first_incl=True):
    #     columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
    #     col_jta = {t2: t1 for t1, t2 in zip(tag_sa1, tag_jta)}
    #     col_jto = {t2: t1 for t1, t2 in zip(tag_sa1, tag_jto)}
    #     k = 0
    #     df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
    #     for k in range(1, len(id_set) - 1):
    #         df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
    #         df_raw = pd.concat([df_raw, df_tmp[tag_acc + tag_sa1],
    #                             df_tmp[tag_acc + tag_sa2].rename(columns=columns),
    #                             df_tmp[tag_acc + tag_jta].rename(columns=col_jta),
    #                             df_tmp[tag_acc + tag_jto].rename(columns=col_jto)
    #                             ], axis=0)
    #     if not first_incl:
    #         k = 0
    #         df_tmp = id_set[k + 1] - (id_set[k] + 1)
    #         df_raw = df_raw.iloc[df_tmp:]
    #     # pdb.set_trace()
    #     return df_raw

    def obtain_whole_bin_more(self, dframe, id_set, tag_acc, tag_sa1, tag_sa2,
                              tag_jta, tag_jto, first_incl=True):
        curr_tag = tag_acc + tag_sa1 + tag_sa2 + tag_jta + tag_jto
        k = 0
        df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
        df_raw = df_raw[curr_tag]
        df_raw[tag_sa2] = df_raw[tag_sa1]
        df_raw[tag_jta] = df_raw[tag_sa1]
        df_raw[tag_jto] = df_raw[tag_sa1]
        df_raw = [df_raw]
        for k in range(1, len(id_set) - 1):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]]
            # df_raw = pd.concat([df_raw, df_tmp[curr_tag]], axis=0)
            df_raw.append(df_tmp[curr_tag])
        if not first_incl:
            # k = 0
            # df_tmp = id_set[k + 1] - (id_set[k] + 1)
            # df_raw = df_raw.iloc[df_tmp:]
            df_raw = df_raw[1:]
        # pdb.set_trace()
        return df_raw

    def obtain_tag_col(self, tag='tst'):
        csv_row_1 = unique_column(11 + 112 * 2)
        tag_trn = csv_row_1[11: 11 + 112]
        tag_tst = csv_row_1[-112:]
        tag_col = tag_trn if tag == 'trn' else tag_tst

        # sub-tags
        st_acc = tag_col[: 7 * 2]
        st_grp = tag_col[14: 14 + 3 * 4]
        st_dr = tag_col[14 + 3 * 4: 14 + 12 + 2]
        st_hfm = tag_col[14 + 12 + 2:]
        st_grp = np.array(st_grp).reshape(4, 3).tolist()

        st_hfm = np.array(st_hfm).reshape(4, 3, -1)  # .tolist()
        st_hm_drt = st_hfm[:, 0, :].tolist()
        st_hm_eff = st_hfm[:, 1, :].tolist()
        st_hm_app = st_hfm[:, 2, :].tolist()
        del st_hfm
        # return st_acc, st_grp, st_dr, st_hm_drt, st_hm_eff, st_hm_app

        st_acc += st_dr  # tag_common
        tag_sa1 = st_grp[0] + st_hm_drt[0] + st_hm_eff[0] + st_hm_app[0]
        tag_sa2 = st_grp[1] + st_hm_drt[1] + st_hm_eff[1] + st_hm_app[1]
        tag_jta = st_grp[2] + st_hm_drt[2] + st_hm_eff[2] + st_hm_app[2]
        tag_jto = st_grp[3] + st_hm_drt[3] + st_hm_eff[3] + st_hm_app[3]
        del st_dr, st_grp, st_hm_drt, st_hm_eff, st_hm_app
        return st_acc, tag_sa1, tag_sa2, tag_jta, tag_jto

    # def schedule_mspaint_prime(self, raw_dframe, nb_set, id_set,
    #                            figname, mk='tst', first_incl=True):
    #     tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = self.obtain_tag_col(tag=mk)
    #
    #     # df_bin = self.obtain_binval_senatt(
    #     #     raw_dframe, id_set, tag_com, tag_sa1, tag_sa2)
    #     # df_multi = self.obtain_multival_senatt(
    #     #     raw_dframe, id_set, tag_com, tag_sa1, tag_sa2, first_incl)
    #     # df_all = self.obtain_whole_bin_sa(
    #     #     raw_dframe, id_set, tag_com, tag_sa1, tag_sa2, first_incl)
    #     df_alt = self.obtain_whole_bin_more(
    #         raw_dframe, id_set, tag_com, tag_sa1, tag_sa2,
    #         tag_jta, tag_jto, first_incl)
    #     fgn = f'{figname}_n_{mk}_'

    def schedule_mspaint_prime(self, df_alt, fgn, tags):
        tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = tags

        for k, df_raw in enumerate(df_alt):
            if k != 2:  # k not in [2, 3, 4]:
                continue
            # self.depict_hfm_approximation(df_raw, tag_sa1, fgn + f'set{k}_')
            self.depict_hfm_subset(df_raw, tag_sa1, fgn[:-6] + f'set{k}_')

        df_alt = pd.concat(df_alt, axis=0).reset_index(drop=True)
        # '''
        # self.depict_hfm_approximation(df_alt, tag_sa1, fgn)
        # self.depict_hfm_fairness(df_alt, tag_com, tag_sa1, tag_sa2, fgn,
        #                          tag_jta, tag_jto)
        #
        # columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        # col_jta = {t2: t1 for t1, t2 in zip(tag_sa1, tag_jta)}
        # col_jto = {t2: t1 for t1, t2 in zip(tag_sa1, tag_jto)}
        # df_tmp = pd.concat([df_alt[tag_sa1],
        #                     df_alt[tag_sa2].rename(columns=columns),
        #                     df_alt[tag_jta].rename(columns=col_jta),
        #                     df_alt[tag_jto].rename(columns=col_jto)], axis=0)
        # del columns, col_jta, col_jto
        # pdb.set_trace()
        # '''
        return

    def depict_hfm_subset(self, df_all, tag_sa1, fgn, verbose=False):
        hfm_drt = tag_sa1[3: 3 + 7]
        hfm_eff = tag_sa1[3 + 7: 3 + 7 * 2]
        hfm_app = tag_sa1[3 + 7 * 2:]
        tYs = [hfm_drt[0], hfm_app[0], hfm_drt[1], hfm_app[1],
               hfm_drt[6], hfm_app[6],
               hfm_drt[4], hfm_app[4], hfm_drt[5], hfm_app[5]]
        if verbose:
            _ver4sub_depict_scat(df_all, tYs, fgn + 'scat', diff=True)
            _ver4sub_depict_tim(df_all, tYs, fgn + 'tim', diff=True,
                                leq=True)  # log_taken=True,
            os.remove(f'{fgn}tim_sty6a.pdf')

        tYs = [hfm_drt[0], hfm_eff[0], hfm_app[0],
               hfm_drt[1], hfm_eff[1], hfm_app[1],
               hfm_drt[4], hfm_eff[4], hfm_app[4],
               hfm_drt[5], hfm_eff[5], hfm_app[5]]
        # df_raw = df_all[tYs]
        tYs = np.array(tYs).reshape(4, 3).tolist()
        df_Ds_tDs = df_all[tYs[0] + tYs[2]]
        df_Df_tDf = df_all[tYs[1] + tYs[3]].rename(columns={
            t2: t1 for t1, t2 in zip(tYs[0] + tYs[2], tYs[1] + tYs[3])})
        if verbose:
            _ver4sub_depict_earlybreak(df_Ds_tDs, tYs[0], fgn + 'ebs', 'Ds')
            _ver4sub_depict_earlybreak(df_Df_tDf, tYs[0], fgn + 'ebf', 'Df')
            _ver4sub_depict_earlybreak(df_Ds_tDs, tYs[2], fgn + 'ebts', 'tDs')
            _ver4sub_depict_earlybreak(df_Df_tDf, tYs[2], fgn + 'ebtf', 'tDf')

        df_raw = pd.concat([df_Ds_tDs, df_Df_tDf], axis=0)
        del df_Ds_tDs, df_Df_tDf
        # tYs = tYs[0] + tYs[2]
        # _ver4sub_depict_earlybreak(df_raw, tYs, fgn + 'eb06')
        _ver4sub_depict_earlybreak(df_raw, tYs[2], fgn + 'eb06_tim', 'tD.')
        _ver4sub_depict_earlybreak(df_raw, tYs[0], fgn + 'eb06_scat', 'D.')
        return

    def depict_hfm_approximation(self, df_all, tag_sa1, fgn):
        # tag_grp = tag_sa1[:3]
        hfm_drt = tag_sa1[3: 3 + 7]
        hfm_eff = tag_sa1[3 + 7: 3 + 7 * 2]
        hfm_app = tag_sa1[3 + 7 * 2:]

        tYs = [hfm_drt[0], hfm_app[0], hfm_drt[1], hfm_app[1],
               hfm_drt[6], hfm_app[6],
               hfm_drt[4], hfm_app[4], hfm_drt[5], hfm_app[5]]
        _sub_depict_scat(df_all, tYs, fgn + 'scat', diff=True)
        _sub_depict_tim(df_all, tYs, fgn + 'tim', diff=True,
                        log_taken=True, leq=True)
        os.remove(f'{fgn}tim_sty6a.pdf')
        # if 'set' in fgn:
        #     return
        _sub_depict_sep(df_all, tYs, fgn + 'sep', '_Ds')
        _sub_depict_sep(df_all, tYs, fgn + 'sep', '_Df')
        _sub_depict_sep_alt(df_all, tYs, fgn + 'altsep', '_TDs', double_sep=False)
        _sub_depict_sep_alt(df_all, tYs, fgn + 'altsep', '_TDf', double_sep=False)
        # pdb.set_trace()
        return

    def depict_hfm_fairness(self, df_all, tag_acc, tag_sa1, tag_sa2,
                            fgn, tag_jta, tag_jto):
        df_all['Grp1'] = (df_all[tag_sa1[0]] + df_all[tag_sa2[0]]) / 2.
        df_all['Grp2'] = (df_all[tag_sa1[1]] + df_all[tag_sa2[1]]) / 2.
        df_all['Grp3'] = (df_all[tag_sa1[2]] + df_all[tag_sa2[2]]) / 2.
        df_all['hfm_drt'] = (df_all[tag_sa1[
            3 + 2]] + df_all[tag_sa2[3 + 2]]) / 2.
        df_all['hfm_app'] = (df_all[tag_sa1[
            3 + 7 + 7 + 2]] + df_all[tag_sa2[3 + 7 + 7 + 2]]) / 2.
        # for ij in range(df_all.shape[0]):
        #     df_all['hfm_drt'].loc[ij] = max(
        #         df_all[tag_sa1[5]].loc[ij], df_all[tag_sa2[5]].loc[ij])
        #     df_all['hfm_app'].loc[ij] = max(
        #         df_all[tag_sa1[19]].loc[ij], df_all[tag_sa2[19]].loc[ij])
        # pdb.set_trace()

        Mat_A = df_all[tag_acc[:7]].values.astype(DTY_FLT).T
        Mat_C = df_all[tag_acc[7:14]].values.astype(DTY_FLT).T
        key_B = GRP_FAIR_COMMON + ['DR', r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        curr_tag = ['Grp1', 'Grp2', 'Grp3'] + tag_acc[-2:-1] + ['hfm_drt', 'hfm_app']
        Mat_B = df_all[curr_tag].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            Mat_A, Mat_B, self._perf_metric, key_B,
            figname=fgn + 'each_confusion', cmap_name='Oranges', rotate=0)
        analogous_confusion_extended(
            Mat_C, Mat_B, self._dal_metric, key_B,
            figname=fgn + 'delt_confusion', cmap_name='Oranges', rotate=0)
        pdb.set_trace()
        return


class Ver4_PlotH_fair_ens(PlotH_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk, first_incl = 'tst', True
        # self.schedule_mspaint_prime(raw_dframe, nb_set, id_set, figname)

        tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = self.obtain_tag_col(tag=mk)
        df_alt = self.obtain_whole_bin_more(
            raw_dframe, id_set, tag_com, tag_sa1, tag_sa2,
            tag_jta, tag_jto, first_incl)
        fgn = f'{figname}_n_{mk}_'
        self.schedule_mspaint_prime(df_alt, fgn, [
            tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto])
        return


class Ver4_PlotH_norm_cls(PlotH_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(raw_dframe, sa_ir=11, sa_r=0)
        mk, first_incl = 'tst', True
        # self.schedule_mspaint_prime(raw_dframe, nb_set, id_set, figname)

        tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = self.obtain_tag_col(tag=mk)
        df_alt = self.obtain_whole_bin_more(
            raw_dframe, id_set, tag_com, tag_sa1, tag_sa2,
            tag_jta, tag_jto, first_incl)
        fgn = f'{figname}_n_{mk}_'
        self.schedule_mspaint_prime(df_alt, fgn, [
            tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto])
        return


class Ver4_PlotH_gather(PlotH_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        # xlsx_name, sheet_name = raw_dframe
        rdf_fair, rdf_norm = raw_dframe
        mk, first_incl = 'tst', True
        tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = self.obtain_tag_col(tag=mk)

        nb_set, id_set = self.recap_sub_data(rdf_fair, sa_ir=3, sa_r=4)
        rdf_fair = self.obtain_whole_bin_more(
            rdf_fair, id_set, tag_com, tag_sa1, tag_sa2,
            tag_jta, tag_jto, first_incl)
        nb_set, id_set = self.recap_sub_data(rdf_norm, sa_ir=11, sa_r=0)
        rdf_norm = self.obtain_whole_bin_more(
            rdf_norm, id_set, tag_com, tag_sa1, tag_sa2,
            tag_jta, tag_jto, first_incl)

        # pdb.set_trace()
        df_alt = [pd.concat([i, j], axis=0).reset_index(
            drop=True) for i, j in zip(rdf_fair, rdf_norm)]
        fgn = f'{figname}_n_{mk}_'
        self.schedule_mspaint_prime(df_alt, fgn, [
            tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto])
        return


class Ver4_PlotH_gather_prep(PlotH_drawing):
    def schedule_mspaint(self, raw_dframe, figname=''):
        xlsx_name, sheet_name = raw_dframe
        mk, first_incl = 'tst', True
        tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto = self.obtain_tag_col(tag=mk)
        curr_set, df_tmp = 2, []

        for prep in ['minmax', 'normalize', 'standard']:  # 'none',
            # sheet_name = f'{figname}_{prep}'
            rdf_fair = self.load_raw_dataset(xlsx_name, f'exp8a_{prep}')
            rdf_norm = self.load_raw_dataset(xlsx_name, f'exp8b_{prep}')
            nb_set, id_set = self.recap_sub_data(rdf_fair, sa_ir=3, sa_r=4)
            rdf_fair = self.obtain_whole_bin_more(
                rdf_fair, id_set, tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto,
                first_incl)[curr_set]
            nb_set, id_set = self.recap_sub_data(rdf_norm, sa_ir=11, sa_r=0)
            rdf_norm = self.obtain_whole_bin_more(
                rdf_norm, id_set, tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto,
                first_incl)[curr_set]
            tmp = pd.concat([rdf_fair, rdf_norm], axis=0).reset_index(drop=True)
            df_tmp.append(tmp)

        df_tmp = pd.concat(df_tmp, axis=0).reset_index(drop=True)
        # pdb.set_trace()
        fgn = f'{figname}_n_{mk}_'
        # self.schedule_mspaint_prime(df_tmp, fgn, [
        #     tag_com, tag_sa1, tag_sa2, tag_jta, tag_jto])
        self.depict_hfm_subset(df_tmp, tag_sa1, fgn + f'set{curr_set}_')
        return


# -----------------------------
#
