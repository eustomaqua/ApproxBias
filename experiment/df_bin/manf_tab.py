# coding: utf-8
#
# TARGET:
#   8a) Measuring fairness via data manifolds
#       that is, manf_simulation.py
#
#       need to update, because something changed in excel
#


import csv
import numpy as np

from hfm.utils.verifiers import DTY_FLT
from experiment.df_bin.manf_plt import Plot2C_comparison

from pyfair.marble.draw_hypos import (
    Friedman_init, _encode_sign,
    comp_t_sing, comp_t_prep, cmp_paired_wtl, cmp_paired_avg)
from pyfair.granite.draw_graph import (
    Friedman_chart, stat_chart_stack)


# ===============================
# Experiments


# -------------------------------
# Plot2C_comparison


class Table2C_comparison(Plot2C_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, figname='exp2c_'):
        super().__init__(nb_iter, nb_cls, m1, m2, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']
        self._abbr_ensf = ['bagging', 'AdaBoost', 'LightGBM',
                           'FairGBM (fpr)', 'FairGBM (fnr)',
                           'FairGBM (fpr,fnr)', 'AdaFair']

    def schedule_spreadsheet(self, raw_dframe, logger=None):
        nb_set, id_set, each_gen, \
            each_att = self.recap_sub_data(raw_dframe, nb_row=4)
        _, tag_trn, tag_tst = self.prepare_graph()  # tag_pm,

        suff = self._figname.replace('exp2c_', 'exp1c_')
        log_document = suff + '_table_third.csv'
        # log_document = self._figname + '_table_third.csv'
        csv_t = open(log_document, 'w')
        csv_w = csv.writer(csv_t)
        '''
        self.tabulating_third(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set,
            each_gen, each_att, ind=[0, 3], ddof=0,
            alpha=.1, dist_df='both', csv_w=csv_w)
        '''

        self.tabulating_third(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set,
            each_gen, each_att, ind=[0, 0], alpha=.05,
            ddof=0, dist_df='both', csv_w=csv_w)
        self.tabulating_forth(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set,
            each_gen, each_att, ind=[0, 3, 0, 1, 2], alpha=.05,
            ddof=0, dist_df='both', csv_w=csv_w)  # 3:DR

        csv_t.close()
        del csv_t, csv_w

        # accuracy↑, ↓f1_score
        suff = suff.replace('1c', '1d')
        log_document = suff + '_table_third.csv'
        csv_t = open(log_document, 'w')
        csv_w = csv.writer(csv_t)
        self.tabulating_third(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set,
            each_gen, each_att, ind=[3, 3], alpha=.05,
            ddof=0, dist_df='both', csv_w=csv_w)
        self.tabulating_forth(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set,
            each_gen, each_att, ind=[3, 3, 0, 1, 2], alpha=.05,
            ddof=0, dist_df='both', csv_w=csv_w)  # 3:DR
        # ind=[3,3] ind=[3,3,0,1,2]
        csv_t.close()
        del csv_t, csv_w

        '''
        log_document = self._figname + '_table_second.csv'
        csv_t = open(log_document, 'w')
        csv_w = csv.writer(csv_t)
        csv_t.close()
        del csv_t, csv_w
        '''
        return

    def picking_tab_tags(self, tag, ind=[0, 1, 2, 3, 7],
                         dist_df='both'):
        acc_orgin = tag[: 13 - 1]  # origin
        acc_delta = tag[13 * 2: 13 * 3 - 1]
        tag_f_vot = tag[39: 39 + 7 * 4]
        tag_f_man = tag[67: 67 + 16 * 4]  # 67 + 8 * 4]

        tag_f_vot = [tag_f_vot[: 7], tag_f_vot[7: 14],
                     tag_f_vot[14: 21], tag_f_vot[21:]]
        tag_f_man = [tag_f_man[: 16], tag_f_man[16: 32],
                     tag_f_man[32: 48], tag_f_man[48:]]
        tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9, ]  # [2, 5, ]
        if dist_df == 'direct':
            tYs_k2 = [2, ]
        elif dist_df == 'approx':
            tYs_k2 = [9, ]  # [5, ]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]

        tmp_f_vm = [t1 + t2 for t1, t2 in zip(tmp_f_vot, tmp_f_man)]
        del tmp_f_vot, tmp_f_man

        tmp_a1 = [acc_orgin[k] for k in ind]
        tmp_a2 = [acc_delta[k] for k in ind]
        # tmp_a = tmp_a_det + tmp_a_org
        # del tmp_a_org, tmp_a_det
        return tmp_a1, tmp_a2, tmp_f_vm

    def tabulating_first(self, dframe, tag, nb_set, id_set,
                         ind=[0, 1, 2, 3, 7], csv_w=None):
        tmp_a_org, tmp_a_det, tmp_f_vm = self.picking_tab_tags(tag, ind)
        tag_a_f = tmp_a_det + tmp_a_org + tmp_f_vm[0] + tmp_f_vm[1]

        for i in range(nb_set):
            # j = 0
            df_tmp = dframe[tag_a_f].iloc[
                id_set[i] + 1: id_set[i + 1]]

            if i == 0:
                for col_Y in tag_a_f[-6:]:
                    df_tmp[col_Y].fillna(0, inplace=True)
                # df_tmp[].fillna(0, inplace=True)
                # continue

            ans_tex = self.tabulating_first_sub1(df_tmp, ind_acc=ind)
            ans_tex[0][0] = dframe['A'].iloc[id_set[i]]
            ans_tex[1][0] = dframe['A'].iloc[id_set[i]]
            csv_w.writerows(ans_tex)
            csv_w.writerow([''])
        return

    def tabulating_first_sub1(self, df_set, ddof=0,
                              ind_acc=[0, 1, 2, 3, 7]):
        nb_row, nb_col = df_set.shape
        nb_clf = nb_row // self._nb_iter
        df_tmp = df_set.values.astype(DTY_FLT)

        U_raw = np.zeros((nb_clf, nb_col, self._nb_iter))
        for i in range(nb_clf):
            loc_a = i * self._nb_iter
            loc_b = (i + 1) * self._nb_iter
            for j in range(nb_col):
                U_raw[i][j] = df_tmp[loc_a: loc_b][:, j]
        # # U_raw.shape= (nb_clf, nb_col, nb_iter =5)
        # U_avg = U_raw.mean(axis=2)
        # U_std = U_raw.std(axis=2, ddof=ddof)

        UA_raw, UF_raw = U_raw[:, :-12] * 100, U_raw[:, -12:]
        UA_avg = UA_raw.mean(axis=2)
        UF_avg = UF_raw.mean(axis=2)
        UA_std = UA_raw.std(axis=2, ddof=ddof)
        UF_std = UF_raw.std(axis=2, ddof=ddof)

        '''
        _, idx_bar_UA = Friedman_init(UA_avg, mode='descend')
        _, idx_bar_UF = Friedman_init(UF_avg, mode='ascend')
        idx_bar_U = np.concatenate([idx_bar_UA, idx_bar_UF], axis=1)
        idx_bar_UA = idx_bar_UA.mean(axis=0).tolist()
        idx_bar_UF = idx_bar_UF.mean(axis=0).tolist()
        '''

        # U_avg = np.concatenate([UA_avg, UF_std], axis=1)
        # U_std = np.concatenate([UA_std, UF_std], axis=1)
        ans_tex = [['', ''] + df_set.columns.tolist()]
        af_t1 = ['δ({})'.format(self._pick_metric[i]) for i in ind_acc]
        af_t1 += [self._pick_metric[i] for i in ind_acc]
        af_t1 += self._picked_keys * 2
        ans_tex.append(['', ''] + af_t1)
        del af_t1

        for i in range(nb_clf):
            af_t2 = ['', '']
            for j in range(nb_col - 12):
                af_t2.append(_encode_sign(
                    UA_avg[i, j], UA_std[i, j], rez=2))
            for j in range(12):  # nb_col - 12, nb_col):
                af_t2.append(_encode_sign(
                    UF_avg[i, j], UF_std[i, j], rez=4))
            ans_tex.append(af_t2)

        return ans_tex

    def tabulating_third(self,
                         dframe, tag_trn, tag_tst, nb_set, id_set,
                         each_gen, each_att, ind=[0, 3], ddof=0,
                         alpha=.7, dist_df='both', csv_w=None):
        tmp_a_org, _, tmp_f_vm = self.picking_tab_tags(tag_trn, ind,
                                                       dist_df)
        tag_trn_a_f = tmp_a_org + tmp_f_vm[0], tmp_a_org + tmp_f_vm[1]
        tmp_a_org, _, tmp_f_vm = self.picking_tab_tags(tag_tst, ind,
                                                       dist_df)
        tag_tst_a_f = tmp_a_org + tmp_f_vm[0], tmp_a_org + tmp_f_vm[1]

        nb_col = 6 if dist_df == 'both' else 5
        nb_att = (nb_set - 1) * 2 + 1
        U_f1_raw = np.zeros((nb_att, 1 + nb_col, self._nb_iter))
        U_cp_raw = np.zeros((nb_att, 1 + nb_col, self._nb_iter))

        i, k = 0, 0
        df_trn = dframe[tag_trn_a_f[0]].iloc[id_set[i] + 1:
                                             id_set[i + 1]]
        df_tst = dframe[tag_tst_a_f[0]].iloc[id_set[i] + 1:
                                             id_set[i + 1]]
        (U_trn_raw, U_tst_raw,
         U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
            df_trn, df_tst, each_gen, each_att, alpha=alpha,
            dist_df=dist_df)
        (pick_by_avg, pick_by_bar, _,  # choose_avg,
         choose_clf) = self.tabulating_third_sub2(
            U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp, ddof=0)
        U_cp_raw[k] = pick_by_avg
        U_f1_raw[k] = pick_by_bar
        k += 1

        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(curr_set, curr_set + each_gen + each_att))
            df_trn = dframe[tag_trn_a_f[0]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[0]].iloc[curr_loc]
            (U_trn_raw, U_tst_raw,
             U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
                df_trn, df_tst, each_gen, each_att, alpha=alpha, dist_df=dist_df)
            (pick_by_avg, pick_by_bar, choose_avg,
             choose_clf) = self.tabulating_third_sub2(
                U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp, ddof=0)
            U_cp_raw[k] = pick_by_avg
            U_f1_raw[k] = pick_by_bar
            k += 1

            curr_loc = list(range(
                curr_set, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_trn = dframe[tag_trn_a_f[1]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[1]].iloc[curr_loc]
            (U_trn_raw, U_tst_raw,
             U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
                df_trn, df_tst, each_gen, each_att, alpha=alpha, dist_df=dist_df)
            (pick_by_avg, pick_by_bar, choose_avg,
             choose_clf) = self.tabulating_third_sub2(
                U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp, ddof=0)
            U_cp_raw[k] = pick_by_avg
            U_f1_raw[k] = pick_by_bar
            k += 1

        # U_f1_raw .shape (1+2*4 set/att, picked f1_score/accuracy using*, #iter)
        #                 based on (1-performance)*alpha +(1-alpha)*fairness
        suff = self._figname.replace('exp2c_', 'exp1c_')
        if ind[0] == 3:
            suff = suff.replace('1c', '1d')  # f1_score
        suff = suff.replace(
            'iter5_cls7', 'iter5cls7').replace('min_max', 'minmax')
        mode = 'descend'
        (ans_tex, ans_std, ans_wtl,
         ans_cmp) = self.tabulating_third_sub3(
            U_f1_raw, ind, dist_df, ddof, rez=4, mode=mode,
            figname=suff + '_tab3_f1')
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerow([''])
        csv_w.writerows(ans_std)
        csv_w.writerow([''])
        csv_w.writerows(ans_wtl)
        csv_w.writerow([''])
        csv_w.writerows(ans_cmp)
        csv_w.writerows([[''], [''], [''], ['']])
        (ans_tex, ans_std, ans_wtl,
         ans_cmp) = self.tabulating_third_sub3(
            U_cp_raw, ind, dist_df, ddof, rez=2, mode=mode,
            figname=suff + '_tab3_cp')
        csv_w.writerows(ans_tex)
        csv_w.writerows(ans_std)
        csv_w.writerows(ans_wtl)
        csv_w.writerows(ans_cmp)
        return

    def tabulating_third_sub1(self, df_trn, df_tst,
                              each_gen, each_att,
                              alpha=.5, dist_df='both'):
        nb_clf = (each_gen + each_att) // self._nb_iter
        df_trn = df_trn.values.astype(DTY_FLT)
        df_tst = df_tst.values.astype(DTY_FLT)
        nb_col = df_trn.shape[1]  # = df_tst.shape[1]

        U_trn_raw = np.zeros((nb_clf, nb_col, self._nb_iter))
        U_tst_raw = np.zeros((nb_clf, nb_col, self._nb_iter))
        for i in range(nb_clf):
            loc_a = i * self._nb_iter
            loc_b = (i + 1) * self._nb_iter
            for j in range(nb_col):
                U_trn_raw[i, j] = df_trn[loc_a: loc_b][:, j]
                U_tst_raw[i, j] = df_tst[loc_a: loc_b][:, j]
        # U_trn/tst_raw .shape= (14+4, 8|7, 5)

        k = nb_col - (6 if dist_df == 'both' else 5)
        U_trn_tmp = np.zeros((nb_clf, nb_col - k, self._nb_iter))
        U_tst_tmp = np.zeros((nb_clf, nb_col - k, self._nb_iter))
        for j in range(k, nb_col):
            U_trn_tmp[:, j - k, :] = (1 - U_trn_raw[
                :, 0, :]) * alpha + (1 - alpha) * U_trn_raw[:, j, :]
            U_tst_tmp[:, j - k, :] = (1 - U_tst_raw[
                :, 0, :]) * alpha + (1 - alpha) * U_tst_raw[:, j, :]
        # U_trn/tst_tmp .shape= (14+4, 6, 5)
        # the smaller the better: alpha* error rate + (1-alpha)* fairness

        return U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp

    def tabulating_third_sub2(self, U_trn_raw, U_tst_raw,
                              U_trn_tmp, U_tst_tmp, ddof=0):
        U_avg = U_trn_tmp.mean(axis=2)  # shape= (14+4, 6= #col) #, 1<-5=#iter)
        _, idx_bar = Friedman_init(U_avg.T, mode='ascend')
        choose_avg = np.argsort(idx_bar, axis=1)[:, 0].tolist()
        nb_clf, nb_col = U_avg.shape

        choose_clf = []
        for j in range(nb_col):
            curr_trn = U_trn_tmp[:, j, :]   # shape= (14+4, 5= nb_iter)
            _, idx_bar = Friedman_init(curr_trn.T, mode='ascend')
            idx_bar = idx_bar.mean(axis=0)  # shape= (14+4,)
            curr_idx = np.argsort(idx_bar)[0].tolist()
            choose_clf.append(curr_idx)

        U_avg = U_trn_raw[:, 0, :]  # curr_trn. shape= (14+4, 5= #iter)
        _, idx_bar = Friedman_init(U_avg.T, mode='descend')  # Accuracy
        idx_bar = idx_bar.mean(axis=0).argsort().tolist()
        choose_avg = [idx_bar[0]] + choose_avg
        choose_clf = [idx_bar[0]] + choose_clf

        # U_trn/tst_raw   .shape= (#clf, 2+#fair, #iter) =(14+4, 8, 5)
        # U_trn/tst_tmp   .shape= (#clf,   #fair, #iter) =(14+4, 6, 5)
        # choose_avg/clf  .shape= (1+#fair,)       =(7,)
        # pick_by_avg/clf .shape= (1+#fair, #iter) =(7, 5)
        pick_by_avg = np.zeros((1 + nb_col, self._nb_iter))
        pick_by_bar = np.zeros((1 + nb_col, self._nb_iter))
        for j in range(1 + nb_col):
            pick_by_avg[j] = U_tst_raw[choose_avg[j], 1, :]
            pick_by_bar[j] = U_tst_raw[choose_avg[j], 1, :]

        return pick_by_avg, pick_by_bar, choose_avg, choose_clf

    def tabulating_third_sub3(self, U_f1_raw, ind, dist_df,
                              ddof=0, rez=2, mode='descend',
                              figname='_fried5'):
        # U_cp_raw,U_f1_raw.shape= (#set*2-1, 1+#col, #iter) =(9,7,5)
        U_f1_raw *= 100.  # f1_score, or accuracy
        nb_att, nb_col, _ = U_f1_raw.shape

        U_avg = U_f1_raw.mean(axis=2)  # shape= (#set*2-1, 1+#col) =(9,7)
        U_std = U_f1_raw.std(axis=2, ddof=ddof)
        _, idx_bar = Friedman_init(U_avg, mode=mode)

        ans_tex = []
        af_t1 = [self._pick_metric[i] for i in ind]
        af_t2 = [0, 1, 2, 3, 4, 5]
        if dist_df == 'direct':
            af_t2 = [0, 1, 2, 3, 4, ]
        elif dist_df == 'approx':
            af_t2 = [0, 1, 2, 3, 5, ]
        af_t3 = [self._picked_keys[i] for i in af_t2]
        ans_tex.append(af_t1[:: -1] + af_t3)

        name_baseline = [af_t1[0]] + af_t3
        del af_t1, af_t3  # , af_t2
        _, idx_bar = Friedman_init(U_f1_raw.mean(axis=2), mode=mode)
        if not figname.endswith('cp'):
            Friedman_chart(
                idx_bar, name_baseline, figname + '_fried5',
                alpha=.05, logger=None, anotCD=True, offset=-1)
        kwargs = {'cmap_name': 'GnBu', 'rotation': 60}
        kwargs['cmap_name'] = 'PuBu'  # 'OrRd,RdPu'
        kwargs['annots'] = r'aggr.rank.{}'.format(
            self._pick_metric[ind[-1]].lower())
        if not figname.endswith('cp'):
            stat_chart_stack(
                idx_bar, name_baseline, figname + '_stack',
                **kwargs)

        for i in range(nb_att):
            af_t4 = ['']
            for j in range(nb_col):
                af_t4.append(_encode_sign(
                    U_avg[i, j], U_std[i, j], rez))
            ans_tex.append(af_t4)
        ans_tex.append([''])
        ans_tex.extend(idx_bar.tolist())
        ans_tex.append([''])
        ans_tex.append([''] + idx_bar.mean(axis=0).tolist())

        ans_wtl, ans_cmp = [['tmp_wtl']], [['tmp_cmp']]
        ans_std = [['tmp_ _avg_and_stdev']]
        if dist_df == 'both':
            ans_wtl.append([''])
            ans_cmp.append([''])
            ans_std.append([''])
            for i in range(nb_att):
                proposed = U_f1_raw[i, -2, :]  # shape= (#iter,) =(5,)
                sign_A, G_A = comp_t_sing(proposed, self._nb_iter, rez)
                tmp_wtl, tmp_cmp = [''], ['']
                tmp_std = ['']
                for j in [0] + [k + 1 for k in [0, 1, 2, 3, 5]]:
                    compared = U_f1_raw[i, j, :]  # shape= (#iter,) =(5,)
                    sign_B, G_B = comp_t_sing(compared,
                                              self._nb_iter, rez)
                    mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                    tmp_wtl.append(cmp_paired_wtl(
                        G_A, G_B, mk_mu, mk_s2, mode=mode))
                    tmp_cmp.append(cmp_paired_avg(G_A, G_B, mode=mode))
                    tmp_std.append(sign_B)
                ans_wtl.append(tmp_wtl)
                ans_cmp.append(tmp_cmp)
                tmp_std.extend(['', sign_A])
                ans_std.append(tmp_std)
            ans_wtl.append([''])
            ans_cmp.append([''])
            ans_std.append([''])
        for i in range(nb_att):
            # proposed = U_f1_raw[i, -1, :]  # shape= (#iter,) =(5,)
            proposed = U_f1_raw[i, (af_t2[-1] + 1), :]
            sign_A, G_A = comp_t_sing(proposed, self._nb_iter, rez)
            tmp_wtl, tmp_cmp = [''], ['']
            tmp_std = ['']
            for j in [0] + [k + 1 for k in af_t2[: -1]]:
                compared = U_f1_raw[i, j, :]  # shape= (#iter,) =(5,)
                sign_B, G_B = comp_t_sing(compared, self._nb_iter, rez)
                mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                tmp_wtl.append(cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2,
                                              mode=mode))
                tmp_cmp.append(cmp_paired_avg(G_A, G_B, mode=mode))
                tmp_std.append(sign_B)
            ans_wtl.append(tmp_wtl)
            ans_cmp.append(tmp_cmp)
            tmp_std.append(sign_A)
            ans_std.append(tmp_std)
        return ans_tex, ans_std, ans_wtl, ans_cmp

    def tabulating_forth(self,
                         dframe, tag_trn, tag_tst, nb_set, id_set,
                         each_gen, each_att, ind=[0, 3], ddof=0,
                         alpha=.1, dist_df='both', csv_w=None):
        # fourth (fairness), is relevant to third (accuracy / performance)
        tmp_a_org, _, tmp_f_vm = self.picking_tab_tags(
            tag_trn, ind[:1], dist_df)
        tag_trn_a_f = tmp_a_org + tmp_f_vm[0], tmp_a_org + tmp_f_vm[1]
        tmp_a_org, _, tmp_f_vm = self.picking_tab_tags(
            tag_tst, ind[:1], dist_df)
        tag_tst_a_f = tmp_a_org + tmp_f_vm[0], tmp_a_org + tmp_f_vm[1]
        nb_col = 6 if dist_df == 'both' else 5
        nb_att = (nb_set - 1) * 2 + 1
        nb_row = len(ind) - 1
        ls_row = ind[1:]  # list of numbers or indices / indexes
        U_f1_raw = np.zeros((
            nb_att, nb_row, 1 + nb_col, self._nb_iter))
        U_cp_raw = np.zeros((
            nb_att, nb_row, 1 + nb_col, self._nb_iter))

        i, k = 0, 0
        df_trn = dframe[tag_trn_a_f[0]].iloc[id_set[i] + 1:
                                             id_set[i + 1]]
        df_tst = dframe[tag_tst_a_f[0]].iloc[id_set[i] + 1:
                                             id_set[i + 1]]
        (U_trn_raw, U_tst_raw,
         U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
            df_trn, df_tst, each_gen, each_att, alpha=alpha,
            dist_df=dist_df)
        (pick_by_avg, pick_by_bar, choose_avg,
         choose_clf) = self.tabulating_forth_sub2(
            U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp,
             ind=ls_row, ddof=0)  # ind[1:]
        U_cp_raw[k] = pick_by_avg
        U_f1_raw[k] = pick_by_bar
        k += 1

        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(
                curr_set, curr_set + each_gen + each_att))
            df_trn = dframe[tag_trn_a_f[0]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[0]].iloc[curr_loc]
            (U_trn_raw, U_tst_raw,
             U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
                df_trn, df_tst, each_gen, each_att, alpha=alpha, dist_df=dist_df)
            (pick_by_avg, pick_by_bar, choose_avg,
             choose_clf) = self.tabulating_forth_sub2(
                U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp,
                 ind=ls_row, ddof=0)
            U_cp_raw[k] = pick_by_avg
            U_f1_raw[k] = pick_by_bar
            k += 1

            curr_loc = list(range(
                curr_set, curr_set + each_gen)) + list(range(
                    curr_set + each_gen + each_att,
                    curr_set + each_gen + each_att * 2))
            df_trn = dframe[tag_trn_a_f[1]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[1]].iloc[curr_loc]
            (U_trn_raw, U_tst_raw,
             U_trn_tmp, U_tst_tmp) = self.tabulating_third_sub1(
                df_trn, df_tst, each_gen, each_att, alpha=alpha,
                 dist_df=dist_df)
            (pick_by_avg, pick_by_bar, choose_avg,
             choose_clf) = self.tabulating_forth_sub2(
                U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp,
                 ind=ls_row, ddof=0)
            U_cp_raw[k] = pick_by_avg
            U_f1_raw[k] = pick_by_bar
            k += 1

        suff = self._figname.replace('exp2c_', 'exp1c_')
        if ind[0] == 3:
            suff = suff.replace('1c', '1d')  # f1_score
        suff = suff.replace(
            'iter5_cls7', 'iter5cls7').replace('min_max', 'minmax')
        mode = 'ascend'
        # offset = -2 if dist_df == 'both' else -1
        csv_w.writerows([[''], [''], [''], ['']])
        for k in ls_row:
            (ans_tex, ans_std, ans_wtl,
             ans_cmp) = self.tabulating_forth_sub3(
                U_f1_raw, k, ls_row, dist_df, ddof,
                mode=mode, offset=-1,
                figname=suff + '_tab4_fairk{}'.format(k))
            # offset=-4,arxiv
            csv_w.writerows([[''], [''], ])
            csv_w.writerow(ans_tex[1][: 2])
            csv_w.writerows(ans_tex + [[''], ['']])
            csv_w.writerows(ans_std + [['']])
            csv_w.writerows(ans_wtl + [['']])
            csv_w.writerows(ans_cmp)
        return

    def tabulating_forth_sub2(self, U_trn_raw, U_tst_raw,
                              U_trn_tmp, U_tst_tmp,
                              ind=[0, 1, 2, 3], ddof=0):
        # U_trn/tst_raw .shape= (#clf, 1+#fair, #iter) =(14+4, 7, 5)
        # U_trn/tst_tmp .shape= (#clf,   #fair, #iter) =(14+4, 6, 5)

        U_avg = U_trn_tmp.mean(axis=2)  # shape= (14,6)
        _, idx_bar = Friedman_init(U_avg.T, mode='ascend')
        choose_avg = np.argsort(idx_bar, axis=1)[:, 0].tolist()
        nb_clf, nb_col = U_avg.shape
        choose_clf = []
        for j in range(nb_col):
            curr_trn = U_trn_tmp[:, j, :]   # shape= (14+4, 5= #iter)
            _, idx_bar = Friedman_init(curr_trn.T, mode='ascend')
            idx_bar = idx_bar.mean(axis=0)  # shape= (14+4,)
            curr_idx = np.argsort(idx_bar)[0].tolist()  # int
            choose_clf.append(curr_idx)
        # Accuracy/Performance
        U_avg = U_trn_raw[:, 0, :]  # curr_trn.shape= (14+4, 5= #iter)
        _, idx_bar = Friedman_init(U_avg.T, mode='descend')  # sz(5,18)
        idx_bar = idx_bar.mean(axis=0).argsort().tolist()  # size (18,)
        choose_avg = [idx_bar[0]] + choose_avg
        choose_clf = [idx_bar[0]] + choose_clf

        # U_trn/tst_raw   .shape= (#clf, 1+#fair, #iter) =(14+4, 7, 5)
        # U_trn/tst_tmp   .shape= (#clf,   #fair, #iter) =(14+4, 6, 5)
        # choose_avg/clf  .shape= (1+#fair,)   = (7,)  # first one is accuracy
        # pick_by_avg/clf .shape= (#row, 1+#fair, #iter) =( 3/4, 7, 5)
        #     nb_col =6/5  # number of fairness measures, used as rules
        nb_row = len(ind)  # number of fairness measures (fairness performance)
        pick_by_avg = np.zeros((nb_row, 1 + nb_col, self._nb_iter))
        pick_by_bar = np.zeros((nb_row, 1 + nb_col, self._nb_iter))
        for i in range(nb_row):
            k_ij = 1 + ind[i]
            for j in range(nb_col + 1):
                pick_by_avg[i][j] = U_tst_raw[choose_avg[j], k_ij, :]
                pick_by_bar[i][j] = U_tst_raw[choose_clf[j], k_ij, :]

        return pick_by_avg, pick_by_bar, choose_avg, choose_clf

    def tabulating_forth_sub3(self, U_f1_raw, k_i, ind, dist_df,
                              ddof=0, rez=4, mode='ascend',
                              offset=-1, figname='_fried5',
                              verbose=False):
        # U_cp_raw, U_f1_raw .shape= (#set*2-1,1-#row,1+#col,#iter) =(9,3|4,7|6,5)
        nb_att, nb_row, nb_col, _ = U_f1_raw.shape        # shape= (9?, 3|4, ..)
        k_j = ind.index(k_i)  # i_k)
        nb_att, nb_col, _ = U_f1_raw[:, k_j, :, :].shape  # shape= (9?, 7|6, 5)
        curr_U_f1_raw = U_f1_raw[:, k_j, :, :]  # shape= (9?, 7|6, 5)

        U_avg = curr_U_f1_raw.mean(axis=2)  # shape= (#set*2-1, 1+col) =(9,7|6)
        U_std = curr_U_f1_raw.std(axis=2, ddof=ddof)
        _, idx_bar = Friedman_init(U_avg, mode=mode)
        # pdb.set_trace()
        ans_tex = []
        af_t1 = ['Comparison via/on fairness: ' + self._picked_keys[
            k_i], '']
        af_t2 = [0, 1, 2, 3, 4, 5]
        # af_t5 = -1
        if dist_df == 'direct':
            af_t2 = [0, 1, 2, 3, 4, ]
            # af_t5 = 5
        elif dist_df == 'approx':
            af_t2 = [0, 1, 2, 3, 5, ]
            # af_t5 = 4
        af_t3 = [self._picked_keys[i] for i in af_t2]
        ans_tex.extend([[''], af_t1 + af_t3])

        name_baseline = ['Accuracy? performance'] + af_t3
        name_baseline[0] = 'Accuracy'  # self._pick_metric[ind[0]]
        if 'exp1d_' in figname:
            name_baseline[0] = '$f_1$ score'
        del af_t1, af_t3
        _, idx_bar = Friedman_init(curr_U_f1_raw.mean(axis=2),
                                   mode=mode)
        if verbose:
            Friedman_chart(
                idx_bar, name_baseline, figname + '_fried5',
                alpha=0.05, logger=None, anotCD=True,
                offset=offset)  # arxiv
        kwargs = {'cmap_name': 'PuBu', 'rotation': 60}
        kwargs['cmap_name'] = 'RdPu'
        kwargs['annots'] = r'aggr.rank.{}'.format(
            self._picked_keys[k_i])
        stat_chart_stack(idx_bar, name_baseline,
                         figname + '_stack', **kwargs)

        for i in range(nb_att):
            af_t4 = ['']
            for j in range(nb_col):
                af_t4.append(
                    _encode_sign(U_avg[i, j], U_std[i, j], rez))
            ans_tex.append(af_t4)
        ans_tex.append([''])
        ans_tex.extend(idx_bar.tolist())
        ans_tex.append([''])
        ans_tex.append([''] + idx_bar.mean(axis=0).tolist())

        #      U_f1_raw .shape= (9?, 3|4, 7|6, 5)
        # curr_U_f1_raw .shape= (9?,      7|6, 5)
        ans_std = [['tmp_ _avg_and_stdev']]
        ans_wtl, ans_cmp = [['tmp_wtl']], [['tmp_cmp']]
        if dist_df == 'both':
            ans_std.append([''])
            ans_wtl.append([''])
            ans_cmp.append([''])
            for i in range(nb_att):
                # idx/loc= 5 vs. [0, 1, 2, 3, 4, 6]  # 'both'
                proposed = curr_U_f1_raw[i, -2, :]  # shape= (#iter,) =(5,)
                sign_A, G_A = comp_t_sing(proposed, self._nb_iter, rez)
                tmp_wtl, tmp_cmp = [''], ['']
                tmp_std = ['']
                for j in [0] + [k + 1 for k in [0, 1, 2, 3, 5]]:
                    compared = curr_U_f1_raw[i, j, :]  # shape= (#iter,) =(5,)
                    sign_B, G_B = comp_t_sing(compared, self._nb_iter, rez)
                    mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                    tmp_wtl.append(cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2, mode=mode))
                    tmp_cmp.append(cmp_paired_avg(G_A, G_B, mode=mode))
                    tmp_std.append(sign_B)
                ans_wtl.append(tmp_wtl)
                ans_cmp.append(tmp_cmp)
                tmp_std.extend(['', sign_A])
                ans_std.append(tmp_std)
            ans_std.append([''])
            ans_wtl.append([''])
            ans_cmp.append([''])
        # pdb.set_trace()
        for i in range(nb_att):
            # idx/loc= 6 vs. [0, 1, 2, 3, 4, 5]  # 'both'
            #          5 vs. [0, 1, 2, 3, 4, ]  'direct'
            #          6 vs. [0, 1, 2, 3, 4, ]  'approx'
            # proposed = curr_U_f1_raw[i, -1, :]  # shape= (#iter,) =(5,)
            proposed = curr_U_f1_raw[i, (af_t2[-1] + 1), :]
            sign_A, G_A = comp_t_sing(proposed, self._nb_iter, rez)
            tmp_wtl, tmp_cmp = [''], ['']
            tmp_std = ['']
            for j in [0] + [k + 1 for k in af_t2[: -1]]:
                compared = curr_U_f1_raw[i, j, :]  # shape= (#iter,) =(5,)
                sign_B, G_B = comp_t_sing(compared, self._nb_iter, rez)
                mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                tmp_wtl.append(cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2, mode=mode))
                tmp_cmp.append(cmp_paired_avg(G_A, G_B, mode=mode))
                tmp_std.append(sign_B)
            ans_wtl.append(tmp_wtl)
            ans_cmp.append(tmp_cmp)
            tmp_std.append(sign_A)
            ans_std.append(tmp_std)
        # return ans_tex, ans_wtl, ans_cmp
        return ans_tex, ans_std, ans_wtl, ans_cmp


# -------------------------------
#


# ===============================
#


# -------------------------------
#


# -------------------------------
#
