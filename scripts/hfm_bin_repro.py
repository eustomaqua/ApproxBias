# coding:utf-8

import json
import csv

# from hfm.utils.verifiers import unique_column
from experiment.df_bin.rev_manf_sim import PartH1_earlybreak


def printer_expt8a(res_data, csv_w, sens_att, abbr_clf, n_cv=5):
    for t_b, tmp_b in enumerate(abbr_clf[:3]):
        csv_w.writerow([''] * 7 + ['', tmp_b, 0] + res_data[0][t_b])
        for k in range(1, n_cv):
            csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
    for t_a, tmp_a in enumerate(sens_att):
        for t_b, tmp_b in enumerate(abbr_clf[3:]):
            curr = t_b + 3 + t_a * 4
            csv_w.writerow([''] * 7 + [tmp_a, tmp_b, 0] + res_data[0][curr])
            for k in range(1, n_cv):
                csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][curr])
    return


def printer_expt8b(res_data, csv_w, abbr_clf, n_cv=5):  # sens_att,
    for t_b, tmp_b in enumerate(abbr_clf):
        csv_w.writerow([''] * 7 + ['---', tmp_b, 0] + res_data[0][t_b])
        for k in range(1, n_cv):
            csv_w.writerow([''] * 7 + ['', '', k] + res_data[k][t_b])
    return


log_document = 'mCV_expt8a_iter5_none_cls9_adult_pms_rat99_rep'
log_document = 'mCV_expt8b_iter5_none_cls9_adult_pms_rat99_rep'
nb_cls, nk = 9, 5
iterator = PartH1_earlybreak(nb_cls, [], [])


json_r = open(log_document + '.json', 'r')
json_saver = json.load(json_r)
res_aux = json_saver['res_aux']
res_data = json_saver['res_data']
json_r.close()
del json_r, json_saver
csv_t = open(log_document + '.csv', 'w')
csv_w = csv.writer(csv_t)

csv_row_2ab = ['dat_name', 'binary', '#sen-att', 'nk', 'gen', 'rep',
               'm1', 'm2'] + ['fair_ens', '#iter']
csv_row_1, csv_r2c, csv_r3c, csv_r4c = iterator.prepare_trial()
csv_row_2 = csv_row_2ab + csv_r2c
csv_row_2[-1] = '[END]'
csv_w.writerows([
    csv_row_1, csv_row_2, [''] * 10 + csv_r3c, [''] * 10 + csv_r4c])
del csv_r4c, csv_r3c, csv_r2c, csv_row_2ab, csv_row_2, csv_row_1

csv_w.writerow(res_aux[0])
sens_att = res_aux[1]
abbr_clf = res_aux[-1]
if 'expt8a' in log_document:
    printer_expt8a(res_data, csv_w, sens_att, abbr_clf, nk)
elif 'expt8b' in log_document:
    printer_expt8b(res_data, csv_w, abbr_clf, nk)
del sens_att, abbr_clf
csv_t.close()
del csv_t, csv_w
