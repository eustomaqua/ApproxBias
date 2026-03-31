# coding: utf-8


from experiment.utils_empirical import GraphSetupVer1
from pyfair.utils_empirical import DAT_EXPT_NMS, DAT_EXPT_ORG
# from pyfair.utils_empirical import GraphSetup
from pyfair.facil.utils_const import unique_column  # ,DTY_FLT

from pyfair.granite.draw_fancy import (
    radar_chart, grped_radar_cht, tabular_chart)
from pyfair.facil.draw_prelim import DTY_PLT

# import numpy as np
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
