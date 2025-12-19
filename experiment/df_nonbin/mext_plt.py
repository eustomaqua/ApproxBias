# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#   Measuring fairness with multiple attributes/values
#       fairmanf extension (ext.)
#


import csv
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import pdb

from pyfair.granite.draw_addtl import (
    scatter_with_marginal_distrib, lineplot_with_uncertainty,
    line_reg_with_marginal_distr, single_line_reg_with_distr,
    multi_lin_reg_with_distr, multi_lin_reg_without_distr,
    scatter_parl_chart_renew, hyper_params_lin_reg,
    _uncertainty_plotting)

from pyfair.marble.draw_hypos import _avg_and_stdev, _encode_sign
from pyfair.granite.draw_chart import (
    analogous_confusion_extended)  # analogous_confusion,
# from pyfair.granite.draw_graph import scatter_parl_chart

from experiment.utils_empirical import GraphSetupVer2 as GraphSetup
from hfm.utils.verifiers import unique_column, check_zero, DTY_FLT
from hfm.utils.recorders import BLFAIR


# ===============================
# Benchmarks


# -------------------------------
#
# tYs:  # Ds,hat_Ds, Df,hat_Df, t(),t(hat), t(Ds),t(hat_Ds),t(Df),t(hat_Df)


def _sub_depict_scat(df_raw, tYs, suff, diff=False):
    scat_X = np.concatenate([
        df_raw[tYs[0]].values.astype(DTY_FLT),
        df_raw[tYs[2]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[1]].values.astype(DTY_FLT),
        df_raw[tYs[3]].values.astype(DTY_FLT)], axis=0)
    # annots = ['Distance via direct computation', 'Distance via approximation']
    annotX, annotY = r'\mathbf{D}_\cdot', r'\hat{\mathbf{D}}_\cdot'
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(
        scat_X, scat_Y, annots, suff + '_sty3',
        linreg=True, snspec='sty3b')
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
        scat_X, scat_Z, annots, suff + '_sty6',
        linreg=True, snspec='sty6')
    return


def _sub_depict_tim(df_raw, tYs, suff, diff=False, log_taken=False):
    scat_X = df_raw[tYs[4]].values.astype(DTY_FLT)  # direct,ut
    scat_Y = df_raw[tYs[5]].values.astype(DTY_FLT)  # approx,ut
    annotX = r'T_{\mathbf{D}}+T_{\mathbf{D}_f}'
    annotY = r'T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f}'
    annots = ['${}$ (sec)'.format(annotX), '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws = {'linreg': True, 'snspec': 'sty4'}
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + '_sty4', **kws)
    if not diff:
        return

    scat_Z = scat_Y / scat_X - 1.
    # annotZ = r'\frac{T_{\hat{\mathbf{D}}_\cdot}}{T_{\mathbf{D}_\cdot}}-1'
    annotZ = r'\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} }-1'
    annots = [
        '${}$  (sec)'.format(annotX), '${}$'.format(annotZ),
        '${}={}$'.format(annotY, annotX)]
    kws['snspec'] = 'sty6'
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_sty6a', **kws)
    if not log_taken:
        return
    scat_Z = np.log10(scat_Z + 1.)
    annotZ = r'\lg(\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} })'
    annots = ['${}$ (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_sty6b', **kws)
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
        annotZ = r'\frac{ abs(\hat{\mathbf{D}}_f-\mathbf{D}_f) }{ \mathbf{D}_f }'

    scat_X = df_raw[col_X].values.astype(DTY_FLT)
    scat_Y = df_raw[col_Y].values.astype(DTY_FLT)
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + fig,
                               linreg=True, snspec='sty3b')
    if not diff:
        return
    scat_Z = np.abs(scat_Y - scat_X) / scat_X
    annots = ['${}$'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_diff' + fig,
                               linreg=True, snspec='sty6')
    return


def _sub_depict_sep_alt(df_raw, tYs, suff, fig='_Ds', diff=True,
                        double_sep=True):
    scat_X = (df_raw[tYs[6]].values.astype(DTY_FLT) +
              df_raw[tYs[8]].values.astype(DTY_FLT))
    scat_Y = (df_raw[tYs[7]].values.astype(DTY_FLT) +
              df_raw[tYs[9]].values.astype(DTY_FLT))
    annotX = r'T_{\mathbf{D}}+T_{\mathbf{D}_f}'
    annotY = r'T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f}'
    annots = ['${}$ (sec)'.format(annotX), '${}$  (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    kws = {'linreg': True, 'snspec': 'sty4'}
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + '_as4', **kws)
    scat_Z = np.log10(scat_Y / scat_X)
    annotZ = r'\lg(\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} })'
    annots[1] = '${}$'.format(annotZ)  # annots[1])
    kws['snspec'] = 'sty6'
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_as6', **kws)

    scat_X = np.concatenate([
        df_raw[tYs[6]].values.astype(DTY_FLT),
        df_raw[tYs[8]].values.astype(DTY_FLT)], axis=0)
    scat_Y = np.concatenate([
        df_raw[tYs[7]].values.astype(DTY_FLT),
        df_raw[tYs[9]].values.astype(DTY_FLT)], axis=0)
    annotX = r'T_{\mathbf{D}_\cdot}'
    annotY = r'T_{\hat{\mathbf{D}}_\cdot}'
    scat_Z = np.log10(scat_Y / scat_X)
    annotZ = r'\lg(\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} })'
    # annots = ['${}$ (sec)'.format(annotX), '${}$  (sec)'.format(annotY),
    #           '${}={}$'.format(annotY, annotX)]
    # annots[1] = '${}$'.format(annotZ)  # annots[1])
    annots = ['${}$ (sec)'.format(annotX), '${}$'.format(annotZ),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Z, annots, suff + '_cs6', **kws)
    annots[1] = '${}$  (sec)'.format(annotY)
    kws['snspec'] = 'sty4'
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + '_cs4', **kws)
    if not diff:
        return

    if fig.endswith('TDs'):
        col_X, col_Y = tYs[6], tYs[7]
        annotX = r'T_{\mathbf{D}}'
        annotY = r'T_{\hat{\mathbf{D}}}'
        annotZ = [r'\frac{ T_{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} }-1',
                  r'\lg(\frac{ T_{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} })']
    elif fig.endswith('TDf'):
        col_X, col_Y = tYs[8], tYs[9]
        annotX = r'T_{\mathbf{D}_f}'
        annotY = r'T_{\hat{\mathbf{D}}_f}'
        annotZ = [r'\frac{ T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} }-1',
                  r'\lg(\frac{ T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} })']
    scat_X = df_raw[col_X].values.astype(DTY_FLT)
    scat_Y = df_raw[col_Y].values.astype(DTY_FLT)
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              # '${}$ (sec)'.format(annotX), '${}$ (sec)'.format(annotY),
              '${}={}$'.format(annotY, annotX)]
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + fig, **kws)
    kws['snspec'] = 'sty6'
    scat_Z = scat_Y / scat_X - 1.
    annots[0] = '${}$  (sec)'.format(annotX)
    annots[1] = '${}$'.format(annotZ[0])
    if double_sep:
        single_line_reg_with_distr(
            scat_X, scat_Z, annots, suff + fig + '_s6a', **kws)
    scat_Z = np.log10(scat_Z + 1.)
    annots[1] = '${}$'.format(annotZ[1])
    single_line_reg_with_distr(
        scat_X, scat_Z, annots, suff + fig + '_s6b', **kws)
    return


# -------------------------------
#
# tYs:  # 19 manf (=6+4+6+3) + 14 manf_ext (=6+6+2)
#
# tYs:  # Ds,Df,t(Ds),t(Df),Ds_avg,Df_avg                `for 6 direct_bin`
#         hat_Ds,hat_Df,t(hat_Ds),t(hat_Df)                    `for 4 manf`
#         hat_Ds,hat_Df,t(hat_Ds),t(hat_Df),hat_Ds_avg,hat_Df_avg  `6 mext`
#         t(direct_bin), t(manf approx), t(manf_ext approx bin)
#         Ds,Df,t(Ds),t(Df),Ds_avg,Df_avg           `for 6 direct_multivar`
#         hat_Ds,hat_Df,t(hat_Ds),t(hat_Df),hat_Ds_avg,hat_Df_avg  `6 mext`
#         t(direct_multivar), t(manf_ext approx multivar)
#


def _ext_sub_show_sep_alt(df_raw, tYs, suff, fig='_TD', diff=True,
                          double_sep=True):
    if fig.endswith('TDs'):
        scat_X_bin = df_raw[tYs[2]].values.astype(DTY_FLT)
        scat_Y_bin = df_raw[tYs[8]].values.astype(DTY_FLT)
        scat_Z_bin = df_raw[tYs[12]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[21]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[27]].values.astype(DTY_FLT)
        # annotX = r'T_{\mathbf{D}}'
        # annotY = r'T_{\hat{\mathbf{D}}}'
        # annotZ = [r'\frac{ T_{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} }-1',
        #           r'\lg(\frac{ T_{\hat{\mathbf{D}}} }{ T_{\mathbf{D}} })']
        annotX = r'T_{\mathbf{D}_\mathbf{a}(S,a_i)}'
        annotY = r'T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)} }{ T_{\mathbf{D}_\mathbf{a}(S,a_i)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)} }{ T_{\mathbf{D}_\mathbf{a}(S,a_i)} })',
            # r'T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)} = T_{\mathbf{D}_\mathbf{a}(S,a_i)}']
            r'T_{\hat{\mathbf{D}}_\mathbf{a}} = T_{\mathbf{D}_\mathbf{a}}']
        annotYp_abbr = r'T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)}'  # r'T_{\hat{\mathbf{D}}}'
    elif fig.endswith('TDf'):
        scat_X_bin = df_raw[tYs[3]].values.astype(DTY_FLT)
        scat_Y_bin = df_raw[tYs[9]].values.astype(DTY_FLT)
        scat_Z_bin = df_raw[tYs[13]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[22]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[28]].values.astype(DTY_FLT)
        # annotX = r'T_{\mathbf{D}_f}'
        # annotY = r'T_{\hat{\mathbf{D}}_f}'
        # annotZ = [r'\frac{ T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} }-1',
        #           r'\lg(\frac{ T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}_f} })']
        annotX = r'T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)}'
        annotY = r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)} })',
            # r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)} = T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)}']
            r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}} = T_{\mathbf{D}_{f,\mathbf{a}}}']
        annotYp_abbr = r'T_{\hat{\mathbf{D}}_f(S_1,\bar{S}_1)}'  # r'T_{\hat{\mathbf{D}}_f}'
    else:
        scat_X_bin = np.concatenate([
            df_raw[tYs[2]].values.astype(DTY_FLT),
            df_raw[tYs[3]].values.astype(DTY_FLT), ], axis=0)
        scat_Y_bin = np.concatenate([
            df_raw[tYs[8]].values.astype(DTY_FLT),
            df_raw[tYs[9]].values.astype(DTY_FLT), ], axis=0)
        scat_Z_bin = np.concatenate([
            df_raw[tYs[12]].values.astype(DTY_FLT),
            df_raw[tYs[13]].values.astype(DTY_FLT), ], axis=0)
        scat_X_mu = np.concatenate([
            df_raw[tYs[21]].values.astype(DTY_FLT),
            df_raw[tYs[22]].values.astype(DTY_FLT), ], axis=0)
        scat_Y_mu = np.concatenate([
            df_raw[tYs[27]].values.astype(DTY_FLT),
            df_raw[tYs[28]].values.astype(DTY_FLT), ], axis=0)
        # annotX = r'T_{\mathbf{D}_\cdot}'
        # annotY = r'T_{\hat{\mathbf{D}}_\cdot}'
        # annotZ = [r'\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} }-1',
        #           r'\lg(\frac{ T_{\hat{\mathbf{D}}_\cdot} }{ T_{\mathbf{D}_\cdot} })']
        annotX = r'T_{\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)}'
        annotY = r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)} })',
            # r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i)} = T_{\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)}']
            r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}} = T_{\mathbf{D}_{\cdot,\mathbf{a}}}']
        annotYp_abbr = r'T_{\hat{\mathbf{D}}_\cdot(S_1,\bar{S}_1)}'  # r'T_{\hat{\mathbf{D}}_\cdot}'

    Xs = [scat_X_bin, scat_X_mu]
    Ys = [[np.log10(scat_Y_bin / scat_X_bin),
           np.log10(scat_Z_bin / scat_X_bin)],
          np.log10(scat_Y_mu / scat_X_mu)]
    Zs = [['${}$  prev'.format(annotYp_abbr),  # '${}$     prev',
           # '${}$  prev'.format(annotY),
           # '${}$  prev work'.format(annotY),
           # '${}$  previous'.format(annotY),
           '${}$  bin-val'.format(annotY)],
          '${}$  multival'.format(annotY)]
    annots = ['${}$'.format(annotX), '${}$'.format(annotZ[1]),
              '${}={}$'.format(annotY, annotX)]
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig, snspec='sty6')
    # multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig + '_s6b', snspec='sty6')
    if not diff:
        return
    Ys = [[scat_Y_bin, scat_Z_bin], scat_Y_mu]
    annots[1] = '${}$'.format(annotY)
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig + '_s4', snspec='sty4')
    if not double_sep:
        return
    Ys = [[scat_Y_bin / scat_X_bin - 1,
           scat_Z_bin / scat_X_bin - 1],
          scat_Y_mu / scat_X_mu - 1]
    annots[1] = '${}$'.format(annotZ[0])
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig + '_s6a', snspec='sty6')
    return


def _ext_sub_show_tim(df_raw, tYs, suff, diff=False):
    scat_X_bin = df_raw[tYs[16]].values.astype(DTY_FLT)  # direct.bin ,ut
    scat_Y_bin = df_raw[tYs[17]].values.astype(DTY_FLT)  # manf approx.bin
    scat_Z_bin = df_raw[tYs[18]].values.astype(DTY_FLT)  # mext approx.bin
    scat_X_mu = df_raw[tYs[31]].values.astype(DTY_FLT)  # direct.multivar
    scat_Y_mu = df_raw[tYs[32]].values.astype(DTY_FLT)  # approx.multivar
    '''
    annotX = r'T_{\mathbf{D}} + T_{\mathbf{D}_f}'
    annotY = r'T_{\hat{\mathbf{D}}} + T_{\hat{\mathbf{D}}_f}'
    annotZ = [
        r'\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} }-1',
        r'\lg(\frac{ T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f} }{ T_{\mathbf{D}}+T_{\mathbf{D}_f} })']
    '''
    annotX = r'T_{\mathbf{D}_\mathbf{a}(S,a_i)} + T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)}'
    annotY = r'T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)} + T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)}'
    annotZ = [
        r'\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_\mathbf{a}(S,a_i)}+T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)} }-1',
        r'\lg(\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S,a_i)}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_\mathbf{a}(S,a_i)}+T_{\mathbf{D}_{f,\mathbf{a}}(S,a_i)} })',
        r'T_{\hat{\mathbf{D}}_\mathbf{a}}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}} = T_{\mathbf{D}_\mathbf{a}}+T_{\mathbf{D}_{f,\mathbf{a}}}']

    Xs = [scat_X_bin, scat_X_mu]
    Ys = [[scat_Y_bin, scat_Z_bin], scat_Y_mu]
    Zs = [[r'$T_{\hat{\mathbf{D}}}+T_{\hat{\mathbf{D}}_f}$      prev',
           r'$T_{\hat{\mathbf{D}}_\mathbf{a}}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}}$  bin-val'],
          r'$T_{\hat{\mathbf{D}}_\mathbf{a}}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}}$  multival']
    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}$'.format(annotZ[2])]  # '${}={}$'.format(annotY, annotX)]
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff, snspec='sty4')
    if not diff:
        return
    Ys = [[np.log10(scat_Y_bin / scat_X_bin),
           np.log10(scat_Z_bin / scat_X_bin)],
          np.log10(scat_Y_mu / scat_X_mu)]
    annots[1] = '${}$'.format(annotZ[1])
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + '_s6b', snspec='sty6')
    Ys = [[scat_Y_bin / scat_X_bin - 1,
           scat_Z_bin / scat_X_bin - 1],
          scat_Y_mu / scat_X_mu - 1]
    annots[1] = '${}$'.format(annotZ[0])
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + '_s6a', snspec='sty6')
    return


def _ext_sub_show_scat(df_raw, tYs, suff, fig='_max', diff=False,
                       abbr=True):  # abbr=False):
    if fig.endswith('max'):
        scat_X_bin = np.concatenate([
            df_raw[tYs[0]].values.astype(DTY_FLT),
            df_raw[tYs[1]].values.astype(DTY_FLT)], axis=0)
        scat_Y_bin = np.concatenate([
            df_raw[tYs[6]].values.astype(DTY_FLT),
            df_raw[tYs[7]].values.astype(DTY_FLT)], axis=0)
        scat_Z_bin = np.concatenate([
            df_raw[tYs[10]].values.astype(DTY_FLT),
            df_raw[tYs[11]].values.astype(DTY_FLT)], axis=0)
        scat_X_mu = np.concatenate([
            df_raw[tYs[19]].values.astype(DTY_FLT),
            df_raw[tYs[20]].values.astype(DTY_FLT)], axis=0)
        scat_Y_mu = np.concatenate([
            df_raw[tYs[25]].values.astype(DTY_FLT),
            df_raw[tYs[26]].values.astype(DTY_FLT)], axis=0)
        annotX = r'\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S,a_i) }{ \mathbf{D}_{\cdot,\mathbf{a}}(S,a_i) }-1'
        annotYp_abbr = r'\hat{\mathbf{D}}_\cdot(S_1,\bar{S}_1)'
        Zs_abbr_y1 = r'\hat{\mathbf{D}}_\cdot'  # (S_1,\bar{S}_1)
        Zs_abbr_y2 = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}'  # (S,a_i)
        Zs_abbr_x2 = r'\mathbf{D}_{\cdot,\mathbf{a}}'  # (S,a_i)
    elif fig.endswith('avg'):

        scat_X_bin = np.concatenate([
            df_raw[tYs[4]].values.astype(DTY_FLT),
            df_raw[tYs[5]].values.astype(DTY_FLT)], axis=0)
        scat_Y_bin = None
        scat_Z_bin = np.concatenate([
            df_raw[tYs[14]].values.astype(DTY_FLT),
            df_raw[tYs[15]].values.astype(DTY_FLT)], axis=0)
        scat_X_mu = np.concatenate([
            df_raw[tYs[23]].values.astype(DTY_FLT),
            df_raw[tYs[24]].values.astype(DTY_FLT)], axis=0)
        scat_Y_mu = np.concatenate([
            df_raw[tYs[29]].values.astype(DTY_FLT),
            df_raw[tYs[30]].values.astype(DTY_FLT)], axis=0)
        # annotX = r'\mathbf{D}_{\cdot,\mathbf{a}}^{\text{avg}}(S,a_i)'
        # annotY = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^\text{avg}(S,a_i)'
        annotX = r'\mathbf{D}_{\cdot,\mathbf{a}}^{avg}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg}(S,a_i)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg}(S,a_i) }{ \mathbf{D}_{\cdot,\mathbf{a}}^{avg}(S,a_i) }-1'
        annotYp_abbr = r'\hat{\mathbf{D}}_\cdot(S_1,\bar{S}_1)'
        Zs_abbr_y1 = r'\hat{\mathbf{D}}_\cdot'  # (S_1,\bar{S}_1)
        Zs_abbr_y2 = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg}'  # (S,a_i)
        Zs_abbr_x2 = r'\mathbf{D}_{\cdot,\mathbf{a}}^{avg}'  # (S,a_i)  # Zs_abbr_y3

    Xs = [scat_X_bin, scat_X_mu]
    Ys = [[scat_Y_bin, scat_Z_bin], scat_Y_mu]
    Zs = [[' (prev)', ' (bin-val)'], ' (multival)']  # [[' prev', ' bin-val'], ' multival']
    annots = ['${}$'.format(annotX), '${}$'.format(annotY), '${}={}$'.format(annotY, annotX)]
    if abbr:
        annots[2] = '${}={}$'.format(Zs_abbr_y2, Zs_abbr_x2)
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig, snspec='sty3a')  # 'sty4')
    if not diff:
        return
    Ys = [[scat_Y_bin / scat_X_bin - 1 if fig.endswith('max') else None,
           scat_Z_bin / scat_X_bin - 1], scat_Y_mu / scat_X_mu - 1]
    Zs = [['${}$  prev'.format(annotYp_abbr),  # '${}$  prev'.format(annotY),
           '${}$  bin-val'.format(annotY)], '${}$  multival'.format(annotY)]
    annots[1] = '${}$'.format(annotZ)
    if abbr:
        Zs = [['${}$    prev'.format(Zs_abbr_y1), '${}$  bin-val'.format(Zs_abbr_y2)
               ], '${}$  multival'.format(Zs_abbr_y2)]
    del Zs_abbr_y1, Zs_abbr_y2, Zs_abbr_x2
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + '_diff' + fig, snspec='sty6')


def _ext_sub_show_sep(df_raw, tYs, suff, fig='_D_max', diff=False):
    if fig.endswith('Ds_max'):
        scat_X_bin = df_raw[tYs[0]].values.astype(DTY_FLT)
        scat_Y_bin = df_raw[tYs[6]].values.astype(DTY_FLT)
        scat_Z_bin = df_raw[tYs[10]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[19]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[25]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_\mathbf{a}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_\mathbf{a}(S,a_i)'
        # annotZ = [r'\frac{ \hat{\mathbf{D}}_\mathbf{a}(S,a_i) }{ \mathbf{D}_\mathbf{a}(S,a_i) }-1',
        #           r'\lg(\frac{ \hat{\mathbf{D}}_\mathbf{a}(S,a_i) }{ \mathbf{D}_\mathbf{a}(S,a_i) })']
        annotZ = r'\frac{ \hat{\mathbf{D}}_\mathbf{a}(S,a_i) }{ \mathbf{D}_\mathbf{a}(S,a_i) }-1'
    elif fig.endswith('Df_max'):
        scat_X_bin = df_raw[tYs[1]].values.astype(DTY_FLT)
        scat_Y_bin = df_raw[tYs[7]].values.astype(DTY_FLT)
        scat_Z_bin = df_raw[tYs[11]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[20]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[26]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_{f,\mathbf{a}}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i)'
        # annotZ = [r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i) }{ \mathbf{D}_{f,\mathbf{a}}(S,a_i) }-1',
        #           r'\lg(\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i) }{ \mathbf{D}_{f,\mathbf{a}}(S,a_i) })']
        annotZ = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S,a_i) }{ \mathbf{D}_{f,\mathbf{a}}(S,a_i) }-1'
    elif fig.endswith('Ds_avg'):
        scat_X_bin = df_raw[tYs[4]].values.astype(DTY_FLT)  # not 3
        scat_Z_bin = df_raw[tYs[14]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[23]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[29]].values.astype(DTY_FLT)
        # annotX = r'\mathbf{D}_\mathbf{a}^\text{avg}(S,a_i)'
        # annotY = r'\hat{\mathbf{D}}_\mathbf{a}^\text{avg}(S,a_i)'
        # annotZ = r'\frac{ \hat{\mathbf{D}}_\mathbf{a}^\text{avg}(S,a_i) }{ \mathbf{D}_\mathbf{a}^\text{avg}(S,a_i) }-1'
        scat_Y_bin = None
        annotX = r'\mathbf{D}_\mathbf{a}^{avg}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_\mathbf{a}^{avg}(S,a_i)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_\mathbf{a}^{avg}(S,a_i) }{ \mathbf{D}_\mathbf{a}^{avg}(S,a_i) }-1'
    elif fig.endswith('Df_avg'):
        scat_X_bin = df_raw[tYs[5]].values.astype(DTY_FLT)  # not 4
        scat_Z_bin = df_raw[tYs[15]].values.astype(DTY_FLT)
        scat_X_mu = df_raw[tYs[24]].values.astype(DTY_FLT)
        scat_Y_mu = df_raw[tYs[30]].values.astype(DTY_FLT)
        # annotX = r'\mathbf{D}_{f,\mathbf{a}}^\text{avg}(S,a_i)'
        # annotY = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S,a_i)'
        # annotZ = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}^\text{avg}(S,a_i) }{ \mathbf{D}_{f,\mathbf{a}}^\text{avg}(S,a_i) }-1'
        scat_Y_bin = None
        annotX = r'\mathbf{D}_{f,\mathbf{a}}^{avg}(S,a_i)'
        annotY = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^{avg}(S,a_i)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}^{avg}(S,a_i) }{ \mathbf{D}_{f,\mathbf{a}}^{avg}(S,a_i) }-1'

    Xs = [scat_X_bin, scat_X_mu]
    Ys = [[scat_Y_bin, scat_Z_bin], scat_Y_mu]
    Zs = [[' (prev)', ' (bin-val)'], ' (multival)']
    annots = ['${}$'.format(annotX), '${}$'.format(annotY), '${}={}$'.format(annotY, annotX)]
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + fig, snspec='sty3a')
    if not diff:
        return
    Ys = [[None if scat_Y_bin is None else scat_Y_bin / scat_X_bin - 1,
           scat_Z_bin / scat_X_bin - 1], scat_Y_mu / scat_X_mu - 1]
    Zs = [['${}$  prev'.format(annotY), '${}$  bin-val'.format(annotY)
           ], '${}$  multival'.format(annotY)]
    annots[1] = '${}$'.format(annotZ)
    multi_lin_reg_with_distr(Xs, Ys, Zs, annots, suff + '_diff' + fig, snspec='sty6')


# -------------------------------
#
# tYs_k4_multivar:  # 14 manf_ext Extend (=6 direct +6 approx +2)
#
# tYs:  # Ds,Df,t(Ds),t(Df),Ds_avg,Df_avg,            `for 6 direct multivar`
#         hat_Ds,hat_Df,t(hat_Ds),t(hat_Df),hat_Ds_avg,hat_Df_avg  `6 approx`
#         t(direct_multivar), t(manf_ext approx multivar)
#


def _ext_prime_show_tim(df_raw, tYs, suff, fig='_TD'):
    if fig.endswith('TDs'):
        scat_X = df_raw[tYs[2]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[8]].values.astype(DTY_FLT)
        annotX = r'T_{\mathbf{D}_\mathbf{a}(S)}'
        annotY = r'T_{\hat{\mathbf{D}}_\mathbf{a}(S)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S)} }{ T_{\mathbf{D}_\mathbf{a}(S)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S)} }{ T_{\mathbf{D}_\mathbf{a}(S)} })',
            # r'T_{\hat{\mathbf{D}}_\mathbf{a}} = T_{\mathbf{D}_\mathbf{a}}']
            r'T_{\hat{\mathbf{D}}_\mathbf{a}(S)} = T_{\mathbf{D}_\mathbf{a}(S)}']
    elif fig.endswith('TDf'):
        scat_X = df_raw[tYs[3]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[9]].values.astype(DTY_FLT)
        annotX = r'T_{\mathbf{D}_{f,\mathbf{a}}(S)}'
        annotY = r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{f,\mathbf{a}}(S)} })',
            r'T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} = T_{\mathbf{D}_{f,\mathbf{a}}(S)}']
    elif fig.endswith('Tcdot'):  # 'TDdot'
        scat_X = np.concatenate([df_raw[tYs[2]].values.astype(DTY_FLT),
                                 df_raw[tYs[3]].values.astype(DTY_FLT)], axis=0)
        scat_Y = np.concatenate([df_raw[tYs[8]].values.astype(DTY_FLT),
                                 df_raw[tYs[9]].values.astype(DTY_FLT)], axis=0)
        annotX = r'T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)}'
        annotY = r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)} })',
            r'T_{\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)} = T_{\mathbf{D}_{\cdot,\mathbf{a}}(S)}']
        fig_nm = '_'.join([suff, 'T_cs'])  # 'cs'])
    elif fig.endswith('Talt'):  # 'TDalt'
        scat_X = (df_raw[tYs[2]].values.astype(DTY_FLT) +
                  df_raw[tYs[3]].values.astype(DTY_FLT))
        scat_Y = (df_raw[tYs[8]].values.astype(DTY_FLT) +
                  df_raw[tYs[9]].values.astype(DTY_FLT))
        fig_nm = '_'.join([suff, 'T_as'])  # 'as'])
    else:  # elif fig.endswith('TD'):
        scat_X = df_raw[tYs[12]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[13]].values.astype(DTY_FLT)
        fig_nm = '_'.join([suff, fig])
    if fig.endswith('TDs') or fig.endswith('TDf'):
        fig_nm = '_'.join([suff, fig])
    if fig.endswith('Talt') or fig.endswith('TD'):
        annotX = r'T_{\mathbf{D}_\mathbf{a}(S)} + T_{\mathbf{D}_{f,\mathbf{a}}(S)}'
        annotY = r'T_{\hat{\mathbf{D}}_\mathbf{a}(S)} + T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)}'
        annotZ = [
            r'\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S)}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_\mathbf{a}(S)}+T_{\mathbf{D}_{f,\mathbf{a}}(S)} }-1',
            r'\lg(\frac{ T_{\hat{\mathbf{D}}_\mathbf{a}(S)} + T_{\hat{\mathbf{D}}_{f,\mathbf{a}}(S)} }{ T_{\mathbf{D}_\mathbf{a}(S)} + T_{\mathbf{D}_{f,\mathbf{a}}(S)} })',
            r'T_{\hat{\mathbf{D}}_\mathbf{a}}+T_{\hat{\mathbf{D}}_{f,\mathbf{a}}} = T_{\mathbf{D}_\mathbf{a}}+T_{\mathbf{D}_{f,\mathbf{a}}}']

    annots = ['${}$ (sec)'.format(annotX), '${}$'.format(annotZ[1]),
              '${}$'.format(annotZ[2])]  # '${}={}$'.format(annotY, annotX)]
    scat_Z = np.log10(scat_Y / scat_X)
    if not (fig.endswith('TD') or fig.endswith('Tcdot')):
        single_line_reg_with_distr(scat_X, scat_Z, annots, fig_nm,
                                   linreg=True, snspec='sty6')
        return
    single_line_reg_with_distr(scat_X, scat_Z, annots, fig_nm + '_6',
                               linreg=True, snspec='sty6')
    annots[1] = '${}$  (sec)'.format(annotY)
    single_line_reg_with_distr(scat_X, scat_Y, annots, fig_nm + '_4',
                               linreg=True, snspec='sty4')
    if not fig.endswith('TD'):
        return
    scat_Z = scat_Y / scat_X - 1
    annots[1] = '${}$'.format(annotZ[0])
    single_line_reg_with_distr(scat_X, scat_Z, annots, fig_nm + '_7',
                               linreg=True, snspec='sty6')


def _ext_prime_show_scat(df_raw, tYs, suff, fig, abbr=False):
    if fig.endswith('Ds_max'):
        scat_X = df_raw[tYs[0]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[6]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_\mathbf{a}(S)'
        annotY = r'\hat{\mathbf{D}}_\mathbf{a}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_\mathbf{a}(S) }{ \mathbf{D}_\mathbf{a}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_\mathbf{a} = \mathbf{D}_\mathbf{a}'
    elif fig.endswith('Df_max'):
        scat_X = df_raw[tYs[1]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[7]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_{f,\mathbf{a}}(S)'
        annotY = r'\hat{\mathbf{D}}_{f,\mathbf{a}}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}(S) }{ \mathbf{D}_{f,\mathbf{a}}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_{f,\mathbf{a}} = \mathbf{D}_{f,\mathbf{a}}'
    elif fig.endswith('Ds_avg'):
        scat_X = df_raw[tYs[4]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[10]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_\mathbf{a}^{avg}(S)'
        annotY = r'\hat{\mathbf{D}}_\mathbf{a}^{avg}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_\mathbf{a}^{avg}(S) }{ \mathbf{D}_\mathbf{a}^{avg}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_\mathbf{a}^{avg} = \mathbf{D}_\mathbf{a}^{avg}'
    elif fig.endswith('Df_avg'):
        scat_X = df_raw[tYs[5]].values.astype(DTY_FLT)
        scat_Y = df_raw[tYs[11]].values.astype(DTY_FLT)
        annotX = r'\mathbf{D}_{f,\mathbf{a}}^{avg}(S)'
        annotY = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^{avg}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{f,\mathbf{a}}^{avg}(S) }{ \mathbf{D}_{f,\mathbf{a}}^{avg}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_{f,\mathbf{a}}^{avg} = \mathbf{D}_{f,\mathbf{a}}^{avg}'
    elif fig.endswith('D_max'):  # 'Dmax'
        scat_X = np.concatenate([df_raw[tYs[0]].values.astype(DTY_FLT),
                                 df_raw[tYs[1]].values.astype(DTY_FLT)], axis=0)
        scat_Y = np.concatenate([df_raw[tYs[6]].values.astype(DTY_FLT),
                                 df_raw[tYs[7]].values.astype(DTY_FLT)], axis=0)
        annotX = r'\mathbf{D}_{\cdot,\mathbf{a}}(S)'
        annotY = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}} = \mathbf{D}_{\cdot,\mathbf{a}}'
    elif fig.endswith('D_avg'):  # 'Davg'
        scat_X = np.concatenate([df_raw[tYs[4]].values.astype(DTY_FLT),
                                 df_raw[tYs[5]].values.astype(DTY_FLT)], axis=0)
        scat_Y = np.concatenate([df_raw[tYs[10]].values.astype(DTY_FLT),
                                 df_raw[tYs[11]].values.astype(DTY_FLT)], axis=0)
        annotX = r'\mathbf{D}_{\cdot,\mathbf{a}}^{avg}(S)'
        annotY = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg}(S)'
        annotZ = r'\frac{ \hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg}(S) }{ \mathbf{D}_{\cdot,\mathbf{a}}^{avg}(S) }-1'
        annotZp_abbr = r'\hat{\mathbf{D}}_{\cdot,\mathbf{a}}^{avg} = \mathbf{D}_{\cdot,\mathbf{a}}^{avg}'

    annots = ['${}$'.format(annotX), '${}$'.format(annotY),
              '${}={}$'.format(annotY, annotX) if not abbr else '${}$'.format(annotZp_abbr)]
    single_line_reg_with_distr(scat_X, scat_Y, annots, suff + fig, linreg=True, snspec='sty3b')
    scat_Z = scat_Y / scat_X - 1
    annots[1] = '${}$'.format(annotZ)
    single_line_reg_with_distr(
        scat_X, scat_Z, annots, suff + '_diff' + fig, linreg=True,
        snspec='sty6')


# -------------------------------
# RQ1.
#   compared with sota fairness
#


# class Plot2_comparison:  # Plot4_
class Plot3_comparison(GraphSetup):
    def __init__(self, nb_iter, nb_cls, m1=25, m2=11,
                 n_e=2, figname='exp2_'):
        self._nb_cls = nb_cls
        gen, rep = None, None  # False, False
        super().__init__(gen, rep, m1, m2, n_e, figname)
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter

        # self._picked_keys = ['DP', 'EO', 'PQP'] + [
        #     'DR', r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        self._picked_keys = BLFAIR + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        self._pick_metric = [
            'Accuracy', 'Precision', 'Recall', r'$f_1$ score',
            # 'FPR rate', 'FNR rate', 'Sensitivity', 'Specificity']
            'Sensitivity', 'Specificity', 'G_mean', 'dp']
        self._cmap_name = 'bright'

    def prepare_graph(self, omitted=True):
        tmp_p = 207 if omitted else 236
        csv_row_1 = unique_column(11 + 1 + tmp_p * 2)

        tmp_norm = 8 if omitted else 13  # ??
        tmp_vote = 5 if omitted else 11  # ?*4
        tmp_manf = 29      # 29*4   =116
        tmp_manf_ext = 10  # 10*6+3 =63
        # tmp_p = normal + vote*4 + manf*4 + manf_ext

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + tmp_p]
        tag_tst = csv_row_1[12 + tmp_p:]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0, omitted=True):
        tmp_norm = 8 if omitted else 13  # ??
        tmp_vote = 5 if omitted else 11  # ?*4
        tmp_manf = 29      # 29*4   =116
        tmp_manf_ext = 10  # 10*6+3 =63

        # vote_minus = tmp_norm * 3 + (tmp_vote - 2)
        # vote_minus = tmp_norm * 3 + (0 if omitted else 3 * 2)
        # tag_f_vot = tag[vote_minus: vote_minus + 5]
        # tag_f_man = tag[vote_minus + 5: vote_minus + 5 + 29 * 4]
        norm_minus = tmp_norm   # * 3
        vote_minus = norm_minus + tmp_vote * 4
        manf_minus = vote_minus + 29 * 4

        tag_acc = tag[: tmp_norm]
        tag_f_vot = tag[norm_minus: norm_minus + tmp_vote * 4]
        tag_f_man = tag[vote_minus: vote_minus + 29 * 4]
        tag_f_ext = tag[manf_minus: manf_minus + 63]

        tag_f_vot = [tag_f_vot[: tmp_vote],
                     tag_f_vot[tmp_vote: tmp_vote * 2],
                     tag_f_vot[tmp_vote * 2: tmp_vote * 3],
                     tag_f_vot[tmp_vote * 3:]]
        tag_f_man = [tag_f_man[:29], tag_f_man[29: 58],
                     tag_f_man[58: 29 * 3], tag_f_man[87:]]
        tag_f_ext = [tag_f_ext[:10], tag_f_ext[10: 20], tag_f_ext[20: 30],
                     tag_f_ext[30: 40], tag_f_ext[40: 50],
                     tag_f_ext[50: 60], tag_f_ext[60:] + [''] * 7]

        del norm_minus, vote_minus, manf_minus
        col_X = tag_acc[ind]  # aka. ↑ tmp_f_vm
        # tYs_k1, tYs_k2 = [1, 2, 3, 5, ], [2, 9]  # [2, 5, ] df
        # tYs_k2: df_ecai (direct vs. approx vs. approx ext), df_nips *3, df_nips_avg *2
        tYs_k1, tYs_k2 = [0, 1, 2, 3, ], [6, 14, 22, 7, 15, 23, 9, 25, ]
        if not omitted:
            tYs_k1 = [k + 6 for k in tYs_k1]  # k + 3 * 2
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tYs_k3 = [6, 7, 9, ]   # tYs_k3 = [6, 7, 9, 8]
        tmp_f_ext = [[t[k] for k in tYs_k3] for t in tag_f_ext[:-1]]
        tmp_f_ext = [
            tmp_f_ext[1] + tmp_f_ext[4], tmp_f_ext[2] + tmp_f_ext[5],
            tmp_f_ext[0] + tmp_f_ext[3], tmp_f_ext[0] + tmp_f_ext[3]]
        tmp_f_vm = [[tag[
            ind], col_X] + t1 + t2 + t3 for t1, t2, t3 in zip(
            tmp_f_vot, tmp_f_man, tmp_f_ext)]
        # tmp_f_vm = [[tag[ind], col_X
        #              ] + t1 + t2 for t1, t2 in zip(tmp_f_vot, tmp_f_man)]
        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2, tYs_k3, tmp_f_ext
        return tag_acc, tag_f_vot, tag_f_man, tag_f_ext, col_X, tmp_f_vm

    # DRAWING PREPARATION

    def drawing_fig2_alt(self, dframe, tag, ind, nb_set, id_set,
                         each_gen, each_att, joint='none', fig='tst',
                         pre='minmax'):
        _, _, tag_f_man, tag_f_ext, col_X, \
            tmp_f_vm = self.picking_fig_tags(tag, ind=0)  # ,tag_f_vot,
        # part3: 10 direct + 6 manf calculation + 10 manf_ext calculation + 3 =29
        #        [Ds_01, Df_01, t_Ds, t_Df, Ds_avg, Df_avg, ]
        tYs_k2_direct = [0, 3, 2, 5, 1, 4, ]        # DirectDist vs.
        tYs_k2_approx = [10, 12, 11, 13, ]          # vs. ApproxDist
        tYs_k3_approx = [16, 19, 18, 21, 17, 20, ]  # vs. ApproxDist
        # tYs_k3_approx = [i + 16 for i in [0, 3, 2, 5, 1, 4, ]]
        # tYs_k2_approx = [i + 10 for i in [0, 2, 1, 3, ]]
        _, suffix = self.draw_sub2_jt(joint)  # tmp,
        tYs_k2 = [0, 10, 3, 12, 26, 27, ] + [2, 11, 5, 13, ]
        tmp_f_vm = [[t[k] for k in tYs_k2] for t in tag_f_man]
        del col_X, tYs_k2_direct, tYs_k2_approx, tYs_k3_approx

        # if ind is None:
        suff_4 = '_'.join([self._figname[:-1], pre, suffix, fig, 'scat'])
        suff_5 = '_'.join([self._figname[:-1], pre, suffix, fig, 'tim'])
        df_raw = self.draw_sub2_dat2(
            dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)
        tYs = tmp_f_vm[0]
        # _sub_depict_scat(df_raw, tYs, suff_4)
        # _sub_depict_tim(df_raw, tYs, suff_5)
        _sub_depict_scat(df_raw, tYs, suff_4, diff=True)
        _sub_depict_tim(df_raw, tYs, suff_5, diff=True, log_taken=True)

        suff_6 = '_'.join([self._figname[:-1], pre, suffix, fig, 'dsep'])
        _sub_depict_sep(df_raw, tYs, suff_6, '_Ds')  # fig=,diff=True)
        _sub_depict_sep(df_raw, tYs, suff_6, '_Df')  # fig=,diff=True)
        suff_6 = suff_6.replace('dsep', 'dalt')
        _sub_depict_sep_alt(df_raw, tYs, suff_6, '_TDs', double_sep=False)
        _sub_depict_sep_alt(df_raw, tYs, suff_6, '_TDf', double_sep=False)
        return

    def drawing_fig1_alt(self, dframe, tag, ind, nb_set, id_set,
                         each_gen, each_att, joint='none', fig='tst',
                         pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        fig_nm = self._figname.replace('iter5_cls7_', '')[:-1]
        suff_1 = fig_nm + '_{}_pc1_{}_mat{}'.format(suffix, fig, ind)
        suff_2 = fig_nm + '_{}_pc2_{}_mat{}'.format(suffix, fig, ind)
        suff_3 = fig_nm + '_{}_pc3_{}_mat{}'.format(suffix, fig, ind)
        del fig_nm

        col_Y, annotY = 'Fairness', 'Fairness measure'  # ' Measure'
        # annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(self._pick_metric[ind])
        annotXpz = r'Performance ({})'.format(self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        _, _, _, _, col_X, tmp_f_vm = self.picking_fig_tags(tag, ind=ind)
        tmp_f_vm = [t[1: -12] for t in tmp_f_vm]  # t[1: -4]
        tag_Ys = tmp_f_vm[0][1:7]  # tmp_f_vm[0][2:8]  # tmp_f_vm[0][2:]

        '''
        df_raw = self.draw_sub1_dat2(
            # dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm, tmp)
            dframe, nb_set, id_set, tmp_f_vm, tmp)
        '''
        df_raw = self.draw_sub2_dat2(
            dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)

        kws = {'cmap_name': self._cmap_name}  # kwargs
        # scatter_with_marginal_distrib(
        #     df_raw, col_X, col_Y, tag_Ys, self._picked_keys,
        #     annotX, annotY, figname=suff_1 + '_s', **kws)
        scatter_with_marginal_distrib(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            annotXpz, annotY, figname=suff_3 + '_s', **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, self._picked_keys,
            figname=suff_2 + '_b4', alpha_loc='b4', alpha_rev=True,
            annotY=annotZ, cmap_name='coolwarm_r', alpha_clarity=.15)
        return

    def draw_sub2_dat2(self, dframe, nb_set, id_set, each_gen, each_att,
                       tmp_f_vm):
        i = 0  # i, k = 0, 0
        df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1: id_set[i + 1]]
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
            columns = {t2: t1 for t1, t2 in zip(tmp_f_vm[0], tmp_f_vm[1])}
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)
        return df_raw

    def draw_sub1_dat2(self, dframe, nb_set, id_set,  # each_gen, each_att,
                       tmp_f_vm, tmp):
        i, j = 0, 0
        df_raw = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                          id_set[i + 1]]
        for i in range(1, nb_set):
            df_tmp = dframe[tmp_f_vm[0]].iloc[id_set[i] + 1:
                                              id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
            for j in tmp[1:]:
                df_tmp = dframe[tmp_f_vm[j]].iloc[id_set[i] + 1:
                                                  id_set[i + 1]]
                columns = {t2: t1 for t1, t2 in zip(
                    tmp_f_vm[0], tmp_f_vm[j])}
                df_tmp = df_tmp.rename(columns=columns)
                df_raw = pd.concat([df_raw, df_tmp], axis=0)
        df_raw = df_raw.reset_index(drop=True)
        return df_raw

    # NEW EXTENSION

    def drawing_fig3(self, dframe, tag, ind, nb_set, id_set,
                     each_gen, each_att, joint='none', fig='tst',
                     pre='minmax'):
        # part3: 10 direct + 6 manf calc + 10 manf_ext calc + 3 =29
        #       [Ds_01, Df_01, t_Ds, t_Df, Ds_avg, Df_avg, ]
        tYs_k2_direct = [0, 3, 2, 5, 1, 4, ]        # DirectDist vs.
        tYs_k2_approx = [10, 12, 11, 13, ]          # vs. ApproxDist
        tYs_k3_approx = [16, 19, 18, 21, 17, 20, ]  # vs. ApproxDist
        # part4: 10 direct.(both sen_att) + 10*2 direct.(sing sen_att) +
        #        10 approx.(both sen_att) + 10*2 approx.(sing sen_att) +
        #        t(DistDirect_multivar) + t(DistExtend) + t(..)
        tYs_k2 = [0, 10, 3, 12, 26, 27, ] + [2, 11, 5, 13, ]
        tYs_k3 = [0, 16, 3, 19, 26, 27, ] + [2, 18, 5, 21, ]  # default:max
        tYs_k3_avg = [1, 17, 4, 20, 26, 27, ] + [2, 18, 5, 21, ]
        tYs_k4 = [0, 30, 3, 33, 60, 61] + [2, 32, 5, 35, ]  # DistDirect/Approx
        tYs_k4_avg = [1, 31, 4, 34, 60, 61] + [2, 32, 5, 35, ]
        del tYs_k2_approx, tYs_k2_direct, tYs_k3_approx
        del tYs_k2, tYs_k3, tYs_k4, tYs_k3_avg, tYs_k4_avg

        _, _, tag_f_man, tag_f_ext, _, _ = self.picking_fig_tags(
            tag, 0)  # ,tag_f_vot,
        tmp, suffix = self.draw_sub2_jt(joint)
        # tYs_ext_tim*: 6 direct (=2 both, 2 sing#1, 2 sing#2) + 6 approx + 2
        tYs_ext_max = [0, 3, 10, 13, 20, 23, ] + [30, 33, 40, 43, 50, 53, ]
        tYs_ext_avg = [1, 4, 11, 14, 21, 24, ] + [31, 34, 41, 44, 51, 54, ]
        tim_ext = [2, 5, 12, 15, 22, 25, ] + [32, 35, 42, 45, 52, 55, ] + [60, 61]
        del tYs_ext_max, tYs_ext_avg, tim_ext  # Ds,Df,t(Ds),t(Df),Ds_avg,Df_avg
        tYs_k4 = [[t[k] for k in [
            0, 3, 2, 5, 1, 4, ]] for t in tag_f_ext[:-1]]
        tYs_k3 = [[t[k] for k in [
            0, 3, 2, 5, 1, 4,                 # direct
            10, 12, 11, 13,                   # manf approx
            16, 19, 18, 21, 17, 20,           # mext approx
            26, 27, 28]] for t in tag_f_man]  # t()
        tim_ext = tag_f_ext[-1][: 3]
        tYs_k3_att01 = tYs_k3[: 2]  # 6 direct + (4 manf + 6 mext) approx + 3 tc
        tYs_k4_att01 = [tYs_k4[1] + tYs_k4[4] + tim_ext[: 2],
                        tYs_k4[2] + tYs_k4[5] + tim_ext[: 2]]
        tYs_k4_multivar = tYs_k4[0] + tYs_k4[3] + tim_ext[: 2]  # DistExtend
        tmp_f_vm = [t1 + t2 for t1, t2 in zip(tYs_k3_att01, tYs_k4_att01)]
        del tYs_k4, tYs_k3, tim_ext, tYs_k3_att01, tYs_k4_att01

        suff_4 = '_'.join([self._figname[:-1], pre, fig, suffix, 'scat'])
        suff_5 = '_'.join([self._figname[:-1], pre, fig, suffix, 'tim'])
        # df_raw = self.draw_sub2_dat2(
        #     dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)
        # df_raw = self.draw_sub3_dat(
        df_raw = self.draw_sub1_dat2(
            dframe, nb_set, id_set, tmp_f_vm, tmp)  # , each_gen, each_att
        tYs = tmp_f_vm[0]  # 19 manf (=6+4+6+3) + 14 manf_ext (=6+6+2)

        _ext_sub_show_scat(df_raw, tYs, suff_4, '_max', diff=True)
        _ext_sub_show_scat(df_raw, tYs, suff_4, '_avg', diff=True)
        # _ext_sub_show_tim(df_raw, tYs, suff_5, diff=True)
        suff_6 = '_'.join([self._figname[:-1], pre, fig, suffix, 'dsep'])
        '''
        _ext_sub_show_sep(df_raw, tYs, suff_6, '_Ds_max')  # ,diff=True)
        _ext_sub_show_sep(df_raw, tYs, suff_6, '_Df_max')  # ,diff=True)
        _ext_sub_show_sep(df_raw, tYs, suff_6, '_Ds_avg')  # ,diff=True)
        _ext_sub_show_sep(df_raw, tYs, suff_6, '_Df_avg')  # ,diff=True)
        '''
        suff_6 = suff_6.replace('dsep', 'dalt')
        _ext_sub_show_sep_alt(df_raw, tYs, suff_6, '_TDdot', double_sep=False)
        '''
        _ext_sub_show_sep_alt(df_raw, tYs, suff_6, '_TDdot', diff=False)
        # _ext_sub_show_sep_alt(df_raw, tYs, suff_6, '_TD', double_sep=False)
        _ext_sub_show_sep_alt(df_raw, tYs, suff_6, '_TDs', diff=False)
        _ext_sub_show_sep_alt(df_raw, tYs, suff_6, '_TDf', diff=False)
        '''

        df_raw = self.draw_sub3_dat(dframe, nb_set, id_set, tYs_k4_multivar)
        # suff_7 = '_'.join([self._figname[:-1], pre, suffix, 'ext'])
        suff_7 = '_'.join([self._figname[:-1], pre, 'ext', suffix])
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_D_max', abbr=True)
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_D_avg', abbr=True)
        _ext_prime_show_tim(df_raw, tYs_k4_multivar, suff_7 + '_tim', fig='Tcdot')
        '''
        _ext_prime_show_tim(df_raw, tYs_k4_multivar, suff_7 + '_tim', fig='TD')
        _ext_prime_show_tim(df_raw, tYs_k4_multivar, suff_7, fig='TDs')
        _ext_prime_show_tim(df_raw, tYs_k4_multivar, suff_7, fig='TDf')
        _ext_prime_show_tim(df_raw, tYs_k4_multivar, suff_7, fig='Talt')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Dmax')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Davg')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Ds_max')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Df_max')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Ds_avg')
        _ext_prime_show_scat(df_raw, tYs_k4_multivar, suff_7 + '_scat', '_Df_avg')
        '''

    def draw_sub3_dat(self, dframe, nb_set, id_set, tYs_k4_multivar):
        i = 0  # i, k = 0, 0
        df_raw = dframe[tYs_k4_multivar].iloc[id_set[i] + 1: id_set[i + 1]]
        for i in range(1, nb_set):
            df_tmp = dframe[tYs_k4_multivar].iloc[id_set[i] + 1: id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw.reset_index(drop=True)

    def drawing_fig4(self, dframe, tag, ind, nb_set, id_set,
                     each_gen, each_att,
                     joint='none', fig='tst', pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        suff_pre = '_'.join([
            self._figname.replace('iter5_cls7_', '')[:-1],
            # pre, fig, suffix, 'mat{}'.format(ind)])
            pre, suffix, 'mat{}'.format(ind)])
        col_Y, annotY = 'Fairness', 'Fairness measure'
        annotXpz = r'Performance ({})'.format(self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        # _, _, _, _, col_X, tmp_f_vm = self.picking_fig_tags(tag, ind=ind)
        # tmp_f_vm = [t[1: -4] for t in tmp_f_vm]

        # _, tag_f_vot, tag_f_man, tag_f_ext, col_X, tmp_f_vm = self.picking_fig_tags(tag, ind)
        _, _, tag_f_man, tag_f_ext, col_X, tmp_f_vm = self.picking_fig_tags(tag, ind)
        # tmp_f_vm: 2 (delta,normal) +4 vote (=3+1) +8 manf (=3+3+2) +6 mext (=3+3)
        # tmp_f_vm: delta=abs(adversarial-normal),normal, GF*3(DP/EO/PQP),DR,
        #           df_ecai|hat|hat (DistDirect_bin,ApproxDist_bin,DistApprox)
        #           df_nips|hat|hat (DistDirect_bin,ApproxDist_bin,DistApprox)
        #           df_nips_avg|hat (DistDirect_bin,DistApprox)
        #           df_ecai,df_nips,df_nips_avg(DistDirect_multivar)
        #           hat_df_ecai,hat_df_nips,hat_df_nips_avg(DistApprox|DistExtend)
        tYs_k4 = [1, 2, 3, 4, 5, ] + [6, 7, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
        tmp_f_vm = [[t[k] for k in tYs_k4] for t in tmp_f_vm]
        # tmp_f_vm: 1 +4 vote +7 manf (=2 ecai+3 nips+2 avg) +6 mext (=3 direct+3 approx)
        # tmp_f_vm: normal, GF*3(DP/EO/PQP),DR, df_ecai|hat, df_nips|hat|hat'
        #           df_nips_avg|hat', df_ecai|nips|nips_avg,hat_df_ecai|nips|nips_avg
        tag_Ys_direct = [tmp_f_vm[0][k] for k in [1, 2, 3, 4] + [5, 7, 10, 13, 14]]  # 0,,12
        tag_Ys_approx = [tmp_f_vm[0][k] for k in [1, 2, 3, 4] + [6, 9, 11, 16, 17]]  # 0,,15
        # tag_Ys_*: 1+4+3*2, 3*= df_ecai|nips|nips_avg
        # tag_Ys_*: normal, GF*3,DR, *(DistDirect_bin), *(DistDirect_multivar)
        #           normal, GF*3,DR, *(ApproxDist_bin|DistApprox*2), *(DistApprox)
        picked_direct = [r'$\mathbf{df}$ ',  # r'$\mathbf{df}$ prev',
                         r'$\mathbf{df}$      bin-val', r'$\mathbf{df}^{avg}$ bin-val',
                         r'$\mathbf{df}$      multival', r'$\mathbf{df}^{avg}$ multival']
        picked_approx = [r'$\hat{\mathbf{df}}$ ',  # r'$\hat{\mathbf{df}}$ prev',
                         r'$\hat{\mathbf{df}}$      bin-val',
                         r'$\hat{\mathbf{df}}^{avg}$ bin-val',
                         r'$\hat{\mathbf{df}}$      multival',
                         r'$\hat{\mathbf{df}}^{avg}$ multival']
        pikced_keys = self._picked_keys[: 4]

        # df_raw = self.draw_sub2_dat2(dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm[:2])
        df_raw = self.draw_sub1_dat2(dframe, nb_set, id_set, tmp_f_vm[:2], tmp)
        kws = {'cmap_name': self._cmap_name, 'snspec': 'sty1', 'identity': 'identity'}
        '''
        scatter_with_marginal_distrib(df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct[1:], pikced_keys + picked_direct, annotXpz, annotY, figname=suff_pre + '_pc3s', **kws)
        scatter_with_marginal_distrib(df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx[1:], pikced_keys + picked_approx, annotXpz, annotY, figname=suff_pre + '_pc3t', **kws)
        '''
        line_reg_with_marginal_distr(df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
                                     pikced_keys + picked_direct, annotXpz, annotY,
                                     figname=suff_pre + '_pc3s', **kws)
        line_reg_with_marginal_distr(df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,
                                     pikced_keys + picked_approx, annotXpz, annotY,
                                     figname=suff_pre + '_pc3t', **kws)
        kws = {'alpha_loc': 'b4', 'alpha_rev': True, 'annotY': annotZ,
               'cmap_name': 'viridis_r', 'alpha_clarity': .15}  # 'coolwarm_r'
        picked_direct[0] = r'$\mathbf{df}$      prev'
        picked_approx[0] = r'$\hat{\mathbf{df}}$      prev'
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
            pikced_keys + picked_direct, figname=suff_pre + '_lc2s',
            **kws)  # figname=suff_pre + '_pc2s', **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,
            pikced_keys + picked_approx, figname=suff_pre + '_lc2t',
            **kws)  # figname=suff_pre + '_pc2t', **kws)

        del tYs_k4, tag_Ys_direct, tag_Ys_approx, picked_direct, picked_approx
        tag_Ys_direct = [[t[k] for k in [4, 13, 14, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]
        tag_Ys_approx = [[t[k] for k in [4, 16, 17, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]
        # tag_Ys_direct = tag_Ys_direct[0] + tag_Ys_direct[1] + tag_Ys_direct[2]
        # tag_Ys_approx = tag_Ys_approx[0] + tag_Ys_approx[1] + tag_Ys_approx[2]
        # df_raw_direct = self.draw_sub1_dat2(dframe, nb_set, id_set, tag_Ys_direct[:2], tmp)
        # df_raw_approx = self.draw_sub1_dat2(dframe, nb_set, id_set, tag_Ys_approx[:2], tmp)
        tag_Ys_merge = [t_dir + t_app[1:] + [  # or tmp_f_vm[0][0]
            col_X] for t_dir, t_app in zip(tag_Ys_direct, tag_Ys_approx)]
        # tag_Ys_merge: DR,df_nips,df_nips_avg,hat_df_nips,hat_df_nips_avg, normal
        df_raw = self.draw_sub1_dat2(dframe, nb_set, id_set, tag_Ys_merge[:2], tmp)
        pikced_keys = ['DR', r'$\mathbf{df}$      multival', r'$\mathbf{df}^{avg}$ multival',
                       r'$\hat{\mathbf{df}}$      multival', r'$\hat{\mathbf{df}}^{avg}$ multival']
        kws = {'cmap_name': 'Spectral', 'snspec': 'sty1', 'identity': 'identity'}
        kw_alt = {'alpha_loc': 'b4', 'alpha_rev': True, 'alpha_clarity': .15, 'cmap_name': 'cool'}
        line_reg_with_marginal_distr(df_raw, col_X, col_Y, tag_Ys_merge[0][:-1],
                                     pikced_keys, annotXpz, annotY,
                                     figname=suff_pre + '_pc4a',
                                     curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tag_Ys_merge[0][:-1],
            pikced_keys, figname=suff_pre + '_lc4c',
            annotY=annotZ, **kw_alt)
        df_raw, tmp_Ys_merge = self.draw_sub4_dat(dframe, nb_set, id_set, tag_Ys_merge)
        pikced_keys = ['DR', r'$\mathbf{df}$      extend',
                       r'$\mathbf{df}^{avg}$ extend',
                       r'$\hat{\mathbf{df}}$      extend', r'$\hat{\mathbf{df}}^{avg}$ extend']
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tmp_Ys_merge[:-1], pikced_keys,
            annotXpz, annotY, figname =suff_pre + '_pc5a',
            curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, col_X, col_Y, tmp_Ys_merge[:-1], pikced_keys,
            # figname=suff_pre + '_pc5c', annotY=annotZ, **kw_alt)
            figname=suff_pre + '_lc5c', annotY=annotZ, **kw_alt)
        del kws, kw_alt, tmp_Ys_merge, pikced_keys, tag_Ys_merge, tag_Ys_direct, tag_Ys_approx
        return

    def draw_sub4_dat(self, dframe, nb_set, id_set, tag_Ys_merge):
        # tmp_Ys_merge = tag_Ys_merge[0][:-5] + tag_Ys_merge[-1][1:]
        tmp_Ys_merge = tag_Ys_merge[0][:1] + tag_Ys_merge[-1][1:]
        i, k = 0, 0
        # df_raw = dframe[tag_Ys_merge[0]].iloc[id_set[i] + 1: id_set[i + 1]]
        df_raw = dframe[tmp_Ys_merge].iloc[id_set[i] + 1: id_set[i + 1]]
        # df_raw = df_raw.rename(columns={tmp_Ys_merge[0]: tag_Ys_merge[0][0]})
        columns = {tag_Ys_merge[-1][0]: tmp_Ys_merge[0]}
        for i in range(1, nb_set):
            df_tmp = dframe[tag_Ys_merge[-1]].iloc[id_set[i] + 1: id_set[i + 1]]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw.reset_index(drop=True), tmp_Ys_merge


class CurrPlot3B_comparison(Plot3_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp3b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, each_att, each_gen = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=3 + 1)  # each_set,
        _, _, tag_tst = self.prepare_graph()  # tag_pm,tag_trn,

        # fairmanf plotting
        '''
        ind = None  # 0 for 'Accuracy', 3 for 'f1_score'
        self.drawing_fig2_alt(raw_dframe, tag_tst, ind,
                              nb_set, id_set, each_gen, each_att, pre=pre)
        for ind in [0, 3, 1, 2, 5]:
          self.drawing_fig1_alt(raw_dframe, tag_tst, ind, nb_set, id_set,
                            each_gen, each_att, pre=pre)
        '''

        # fairmanf_ext plotting
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        self.drawing_fig4(raw_dframe, tag_tst, 0, nb_set, id_set,
                          each_gen, each_att, pre=pre)


class CurrPlot3D_comparison(Plot3_comparison):  # CurrPlot3C
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp3d_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        nb_set, id_set, _, each_att, each_gen = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=11 + 3, nc_sens=0)  # each_set,
        _, _, tag_tst = self.prepare_graph()  # tag_pm,tag_trn,

        # fairmanf plotting
        '''
        ind = None  # 0 for 'Accuracy', 3 for 'f1_score'
        self.drawing_fig2_alt(raw_dframe, tag_tst, ind,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        for ind in [0, 3, 1, 2, 5]:
            self.drawing_fig1_alt(raw_dframe, tag_tst, ind, nb_set, id_set,
                            each_gen, each_att, pre=pre)
        '''

        # fairmanf_ext plotting
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        self.drawing_fig4(raw_dframe, tag_tst, 0, nb_set, id_set,
                          each_gen, each_att, pre=pre)


class CurrPlot3C_comparison(Plot3_comparison):  # CurrPlot3B
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp3c_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        (nb_set, id_set, _, each_att, each_gen) = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=11 + 3, nc_sens=3 + 1)  # each_set,
        tag_pm, tag_trn, tag_tst = self.prepare_graph()

        # fairmanf plotting
        # fairmanf_ext plotting
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        self.drawing_fig4(raw_dframe, tag_tst, 0, nb_set, id_set,
                          each_gen, each_att, pre=pre)


# -------------------------------
# RQ1.
#   compared with sota fairness
#


class Plot4_comparison(Plot3_comparison):
    def prepare_graph(self, omitted=True):
        tmp_p = 247 if omitted else 286
        csv_row_1 = unique_column(11 + 1 + tmp_p * 2)

        tmp_norm = 8 if omitted else 13   # ??
        tmp_vote = 11 if omitted else 17  # ?*4
        tmp_manf = 29      # 29*4   =116
        tmp_manf_ext = 10  # 10*6+3 =63

        params = csv_row_1[: 11 + 1]  # last: Ensem/ut
        tag_trn = csv_row_1[12: 12 + tmp_p]
        tag_tst = csv_row_1[12 + tmp_p:]
        return params, tag_trn, tag_tst

    def picking_fig_tags(self, tag, ind=0, omitted=True):
        tmp_norm = 8 if omitted else 13   # ??
        tmp_vote = 11 if omitted else 17  # ?*4
        tmp_manf = 29      # 29*4   =116
        tmp_manf_ext = 10  # 10*6+3 =63

        norm_minus = tmp_norm * 3
        vote_minus = norm_minus + tmp_vote * 4
        manf_minus = vote_minus + 29 * 4

        tag_acc = tag[norm_minus - tmp_norm: norm_minus]
        tag_f_vot = tag[norm_minus: norm_minus + tmp_vote * 4]
        tag_f_man = tag[vote_minus: vote_minus + 29 * 4]
        tag_f_ext = tag[manf_minus: manf_minus + 63]

        tag_f_vot = [tag_f_vot[: tmp_vote],
                     tag_f_vot[tmp_vote: tmp_vote * 2],
                     tag_f_vot[tmp_vote * 2: tmp_vote * 3],
                     tag_f_vot[tmp_vote * 3:]]
        tag_f_man = [tag_f_man[:29], tag_f_man[29: 58],
                     tag_f_man[58: 29 * 3], tag_f_man[87:]]
        tag_f_ext = [tag_f_ext[:10], tag_f_ext[10: 20], tag_f_ext[20: 30],
                     tag_f_ext[30: 40], tag_f_ext[40: 50],
                     tag_f_ext[50: 60], tag_f_ext[60:] + [''] * 7]

        del norm_minus, vote_minus, manf_minus
        col_X = tag_acc[ind]
        tYs_k1, tYs_k2 = [0, 1, 2, 4, ], [6, 14, 22, 7, 15, 23, 9, 25, ]
        tYs_k1 = [k + 6 for k in tYs_k1]  # k + 3 * 2
        if not omitted:
            tYs_k1 = [k + 10 for k in [2, 3, 4, 6]]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        tYs_k3 = [6, 7, 9, ]  # tYs_k3 = [6, 7, 9, 8]
        tmp_f_ext = [[t[k] for k in tYs_k3] for t in tag_f_ext[:-1]]
        tmp_f_ext = [
            tmp_f_ext[1] + tmp_f_ext[4], tmp_f_ext[2] + tmp_f_ext[5],
            tmp_f_ext[0] + tmp_f_ext[3], tmp_f_ext[0] + tmp_f_ext[3]]
        tmp_f_vm = [[tag[ind], col_X  # col_X is \Delta(?)
                     ] + t1 + t2 + t3 for t1, t2, t3 in zip(
            tmp_f_vot, tmp_f_man, tmp_f_ext)]
        del tmp_f_vot, tmp_f_man, tYs_k1, tYs_k2, tYs_k3, tmp_f_ext
        return tag_acc, tag_f_vot, tag_f_man, tag_f_ext, col_X, tmp_f_vm

    def drawing_fig1_alt(self, dframe, tag, ind, nb_set, id_set,
                         each_gen, each_att, joint='none', fig='tst',
                         pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        fig_nm = self._figname.replace('iter5_cls7_', '')[: -1]
        # suff_1 = fig_nm + '_{}_pc1_{}_mat{}'.format(suffix, fig, ind)
        # suff_2 = fig_nm + '_{}_pc2_{}_mat{}'.format(suffix, fig, ind)
        # suff_3 = fig_nm + '_{}_pc3_{}_mat{}'.format(suffix, fig, ind)
        suff_1 = fig_nm + '_{}_{}_mat{}_pc1'.format(suffix, fig, ind)
        suff_2 = fig_nm + '_{}_{}_mat{}_pc2'.format(suffix, fig, ind)
        suff_3 = fig_nm + '_{}_{}_mat{}_pc3'.format(suffix, fig, ind)
        del fig_nm

        col_Y, annotY = 'Fairness', 'Fairness measure'  # 'Measure'
        annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(self._pick_metric[ind])
        annotXpz = r'Performance ({})'.format(self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        _, _, _, _, col_X, tmp_f_vm = self.picking_fig_tags(tag, ind=ind)
        tmp_f_vm = [t[: -12] for t in tmp_f_vm]  # [t[: -4] for t in tmp_f_vm]
        tag_Ys = tmp_f_vm[0][2: 8]

        df_raw = self.draw_sub2_dat2(
            dframe, nb_set, id_set, each_gen, each_att, tmp_f_vm)
        kws = {'cmap_name': self._cmap_name}
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

        suff_1p = suff_1.replace('pc1', 'sic1')  # 'palt1'
        suff_3p = suff_3.replace('pc3', 'sic3')  # depict,paint,show,present
        kws = {'cmap_name': self._cmap_name, 'snspec': 'sty5b',
               'distrib': False}
        # kws['identity'] = '{:3s}'.format('') + r'$\hat{\mathbf{D}}=\mathbf{D}$'

        # picked_keys = self._picked_keys[:4] + [
        #     self._picked_keys[-2] + '{:3s}'.format(''),
        #     self._picked_keys[-1] + '{:3s}'.format('')]
        picked_keys = ['{:4s}'.format(i) for i in self._picked_keys[:4]
                       ] + ['{}{:3s}'.format(self._picked_keys[-2], ''),
                            '{}{:3s}'.format(self._picked_keys[-1], '')]

        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys, picked_keys,
            annotX, annotY, figname=suff_1p + '_x', invt_a=False, **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys, picked_keys,
            annotXpz, annotY, figname=suff_3p + '_x', invt_a=False, **kws)
        return

    def drawing_fig4(self, dframe, tag, ind, nb_set, id_set, each_gen,
                     each_att, joint='none', fig='tst', pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        suff_pre = '_'.join([
            self._figname.replace('iter5_cls7_', '')[:-1],
            pre, suffix, 'mat{}'.format(ind)])  # fig,
        col_Y, annotY = 'Fairness', 'Fairness measure'
        annotXpz = r'Performance ({})'.format(self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(self._pick_metric[ind])

        _, _, tag_f_man, tag_f_ext, col_X, tmp_f_vm = self.picking_fig_tags(
            tag, ind)  # ,tag_f_vot,
        tYs_k4 = [0, 1, 2, 3, 4, 5] + [6, 7, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
        tmp_f_vm = [[t[k] for k in tYs_k4] for t in tmp_f_vm]
        tag_Ys_direct = [tmp_f_vm[0][k] for k in [2, 3, 4, 5] + [6, 8, 11, 14, 15]]  # 13,
        tag_Ys_approx = [tmp_f_vm[0][k] for k in [2, 3, 4, 5] + [7, 10, 12, 17, 18]]  # 16,
        picked_direct = [r'$\mathbf{df}$ ',  # r'$\mathbf{df}$ prev',
                         r'$\mathbf{df}$      bin-val', r'$\mathbf{df}^{avg}$ bin-val',
                         r'$\mathbf{df}$      multival', r'$\mathbf{df}^{avg}$ multival']
        picked_approx = [r'$\hat{\mathbf{df}}$ ',  # r'$\hat{\mathbf{df}}$ prev',
                         r'$\hat{\mathbf{df}}$      bin-val',
                         r'$\hat{\mathbf{df}}^{avg}$ bin-val',
                         r'$\hat{\mathbf{df}}$      multival',
                         r'$\hat{\mathbf{df}}^{avg}$ multival']
        pikced_keys = self._picked_keys[: 4]

        df_raw = self.draw_sub1_dat2(
            dframe, nb_set, id_set, tmp_f_vm[:2], tmp)
        kws = {'cmap_name': self._cmap_name, 'snspec': 'sty1',
               'identity': 'identity'}
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
            pikced_keys + picked_direct, annotXpz, annotY,
            figname=suff_pre + '_pc3s', **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,

            pikced_keys + picked_approx, annotXpz, annotY,
            figname=suff_pre + '_pc3t', **kws)
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_direct,
            pikced_keys + picked_direct, annotX, annotY,
            figname=suff_pre + '_pc1s', **kws)
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_approx,
            pikced_keys + picked_approx, annotX, annotY,
            figname=suff_pre + '_pc1t', **kws)
        kws = {'alpha_loc': 'b4', 'alpha_rev': True, 'annotY': annotZ,
               'cmap_name': 'viridis_r', 'alpha_clarity': .15}  # 'coolwarm_r'
        picked_direct[0] = r'$\mathbf{df}$      prev'
        picked_approx[0] = r'$\hat{\mathbf{df}}$      prev'
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
            pikced_keys + picked_direct,
            figname=suff_pre + '_lc2s', **kws)  # '_pc2s'
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,
            pikced_keys + picked_approx,
            figname=suff_pre + '_lc2t', **kws)  # '_pc2t'
        del tYs_k4, tag_Ys_direct, tag_Ys_approx, picked_direct, picked_approx

        tag_Ys_direct = [[t[k] for k in [5, 14, 15, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]  # tmp_f_vm]
        tag_Ys_approx = [[t[k] for k in [5, 17, 18, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]  # tmp_f_vm]
        tag_Ys_merge = [t_dir + t_app[1:] + [
            tmp_f_vm[0][0], col_X] for t_dir, t_app in zip(tag_Ys_direct, tag_Ys_approx)]
        df_raw = self.draw_sub1_dat2(dframe, nb_set, id_set, tag_Ys_merge[:2], tmp)
        pikced_keys = ['DR', r'$\mathbf{df}$      multival',
                       r'$\mathbf{df}^{avg}$ multival',
                       r'$\hat{\mathbf{df}}$      multival',
                       r'$\hat{\mathbf{df}}^{avg}$ multival']
        kws = {'cmap_name': 'Spectral', 'snspec': 'sty1',
               'identity': 'identity'}
        kw_alt = {'alpha_loc': 'b4', 'alpha_rev': True,
                  'alpha_clarity': .15, 'cmap_name': 'cool'}
        kws['cmap_name'] = 'Paired_r'  # 'bone, winter'
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_merge[0][:-2], pikced_keys,
            annotX, annotY, figname=suff_pre + '_pc4b',
            curr_legend_nb_split=6, **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_merge[0][:-2],
            pikced_keys, annotXpz, annotY, figname=suff_pre + '_pc4a',
            curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_merge[0][:-2],
            pikced_keys, figname=suff_pre + '_lc4c', annotY=annotZ,
            **kw_alt)

        df_raw, tmp_Ys_merge = self.draw_sub4_dat(dframe, nb_set, id_set, tag_Ys_merge)
        pikced_keys = ['DR', r'$\mathbf{df}$      extend',
                       r'$\mathbf{df}^{avg}$ extend',
                       r'$\hat{\mathbf{df}}$      extend',
                       r'$\hat{\mathbf{df}}^{avg}$ extend']
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tmp_Ys_merge[:-2], pikced_keys,
            annotX, annotY, figname =suff_pre + '_pc5b',
            curr_legend_nb_split=6, **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tmp_Ys_merge[:-2], pikced_keys,
            annotXpz, annotY, figname =suff_pre + '_pc5a',
            curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tmp_Ys_merge[:-2], pikced_keys,
            figname=suff_pre + '_lc5c', annotY=annotZ, **kw_alt)
        del kws, kw_alt, tmp_Ys_merge, pikced_keys
        del tag_Ys_merge, tag_Ys_direct, tag_Ys_approx
        return

    def drawing_fig5_alt4(self, dframe, tag, ind, nb_set, id_set,
                          each_gen, each_att, joint='none', fig='tst',
                          pre='minmax'):
        tmp, suffix = self.draw_sub2_jt(joint)
        suff_pre = '_'.join([
            self._figname.replace('iter5_cls7_', '')[:-1],
            pre, suffix, 'mat{}'.format(ind)])
        col_Y, annotY = 'Fairness', 'Fairness measure'
        annotXpz = r'Performance ({})'.format(self._pick_metric[ind])
        annotZ = ' error rate' if ind == 0 else r'$($1$-$ performance$)$'
        annotX = r'$\Delta$ Performance ($\Delta$ {})'.format(
            self._pick_metric[ind])
        (_, _, _, tag_f_ext,  # ,tag_f_vot,tag_f_man,
         col_X, tmp_f_vm) = self.picking_fig_tags(tag, ind)

        tYs_k4 = [0, 1, 2, 3, 4, 5] + [
            6, 7, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
        tmp_f_vm = [[t[k] for k in tYs_k4] for t in tmp_f_vm]
        tag_Ys_direct = [tmp_f_vm[0][k] for k in [
            2, 3, 4, 5] + [6, 8, 11, 14, 15]]
        tag_Ys_approx = [tmp_f_vm[0][k] for k in [
            2, 3, 4, 5] + [7, 10, 12, 17, 18]]
        picked_direct = [r'$\mathbf{df}$ ',  # r'$\mathbf{df}$ prev',
                         r'$\mathbf{df}$      bin-val',
                         r'$\mathbf{df}^{avg}$ bin-val',
                         r'$\mathbf{df}$      multival',
                         r'$\mathbf{df}^{avg}$ multival']
        picked_approx = [r'$\hat{\mathbf{df}}$ ',  # r'$\hat{\mathbf{df}}$ prev',
                         r'$\hat{\mathbf{df}}$      bin-val',
                         r'$\hat{\mathbf{df}}^{avg}$ bin-val',
                         r'$\hat{\mathbf{df}}$      multival',
                         r'$\hat{\mathbf{df}}^{avg}$ multival']
        pikced_keys = self._picked_keys[: 4]

        df_raw = self.draw_sub1_dat2(
            dframe, nb_set, id_set, tmp_f_vm[:2], tmp)
        kws = {'cmap_name': self._cmap_name, 'snspec': 'sty1',
               'identity': 'identity'}
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
            pikced_keys + picked_direct, annotXpz, annotY,
            figname=suff_pre + '_pc3s', **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,
            pikced_keys + picked_approx, annotXpz, annotY,
            figname=suff_pre + '_pc3t', **kws)
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_direct,
            pikced_keys + picked_direct, annotX, annotY,
            figname=suff_pre + '_pc1s', **kws)
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_approx,
            pikced_keys + picked_approx, annotX, annotY,
            figname=suff_pre + '_pc1t', **kws)

        kws = {'alpha_loc': 'b4', 'alpha_rev': True, 'annotY': annotZ,
               'cmap_name': 'viridis_r', 'alpha_clarity': .15}
        picked_direct[0] = r'$\mathbf{df}$      prev'
        picked_approx[0] = r'$\hat{\mathbf{df}}$      prev'
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_direct,
            pikced_keys + picked_direct,
            figname=suff_pre + '_lc2s', **kws)  # '_pc2s'
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_approx,
            pikced_keys + picked_approx,
            figname=suff_pre + '_lc2t', **kws)  # '_pc2t'
        del tYs_k4, tag_Ys_direct, tag_Ys_approx, picked_direct, picked_approx

        tag_Ys_direct = [[t[k] for k in [5, 14, 15, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]
        tag_Ys_approx = [[t[k] for k in [5, 17, 18, ]] for t in [
            tmp_f_vm[0], tmp_f_vm[1], tmp_f_vm[3]]]
        tag_Ys_merge = [t_dir + t_app[1:] + [
            tmp_f_vm[0][0], col_X] for t_dir, t_app in zip(
                tag_Ys_direct, tag_Ys_approx)]
        df_raw = self.draw_sub1_dat2(
            dframe, nb_set, id_set, tag_Ys_merge[:2], tmp)
        pikced_keys = ['DR', r'$\mathbf{df}$      multival',
                       r'$\mathbf{df}^{avg}$ multival',
                       r'$\hat{\mathbf{df}}$      multival',
                       r'$\hat{\mathbf{df}}^{avg}$ multival']
        kws = {'cmap_name': 'Spectral', 'snspec': 'sty1',
               'identity': 'identity'}
        kw_alt = {'alpha_loc': 'b4', 'alpha_rev': True,
                  'alpha_clarity': .15, 'cmap_name': 'cool'}
        kws['cmap_name'] = 'Paired_r'
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tag_Ys_merge[0][:-2], pikced_keys,
            annotX, annotY, figname=suff_pre + '_pc4b',
            curr_legend_nb_split=6, **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_merge[0][:-2],
            pikced_keys, annotXpz, annotY, figname=suff_pre + '_pc4a',
            curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tag_Ys_merge[0][:-2],
            pikced_keys, figname=suff_pre + '_lc4c', annotY=annotZ,
            **kw_alt)

        df_raw, tmp_Ys_merge = self.draw_sub4_dat(
            dframe, nb_set, id_set, tag_Ys_merge)
        pikced_keys = ['DR', r'$\mathbf{df}$      extend',
                       r'$\mathbf{df}^{avg}$ extend',
                       r'$\hat{\mathbf{df}}$      extend',
                       r'$\hat{\mathbf{df}}^{avg}$ extend']
        line_reg_with_marginal_distr(
            df_raw, col_X, col_Y, tmp_Ys_merge[:-2], pikced_keys,
            annotX, annotY, figname =suff_pre + '_pc5b',
            curr_legend_nb_split=6, **kws)
        line_reg_with_marginal_distr(
            df_raw, tmp_f_vm[0][0], col_Y, tmp_Ys_merge[:-2], pikced_keys,
            annotXpz, annotY, figname =suff_pre + '_pc5a',
            curr_legend_nb_split=6, **kws)
        lineplot_with_uncertainty(
            df_raw, tmp_f_vm[0][0], col_Y, tmp_Ys_merge[:-2], pikced_keys,
            figname=suff_pre + '_lc5c', annotY=annotZ, **kw_alt)

        del kws, kw_alt, tmp_Ys_merge, pikced_keys, tag_Ys_merge, tag_Ys_direct, tag_Ys_approx
        return


# cont.

class CurrPlot4B_comparison(Plot4_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp4b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        (nb_set, id_set, each_set,
         each_att, each_gen) = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=3 + 1)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        # self.drawing_fig5_alt4(raw_dframe, tag_tst, 0, nb_set, id_set,
        #                        each_gen, each_att, pre=pre)
        return


class CurrPlot4D_comparison(Plot4_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp4b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        (nb_set, id_set, each_set,
         each_att, each_gen) = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=11 + 3, nc_sens=0)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        self.drawing_fig4(raw_dframe, tag_tst, 0, nb_set, id_set,
                          each_gen, each_att, pre=pre)


class CurrPlot4C_comparison(Plot4_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp4b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']

    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        (nb_set, id_set, each_set,
         each_att, each_gen) = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=11 + 3, nc_sens=3 + 1)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()
        self.drawing_fig3(raw_dframe, tag_tst, None,
                          nb_set, id_set, each_gen, each_att, pre=pre)
        self.drawing_fig4(raw_dframe, tag_tst, 0, nb_set, id_set,
                          each_gen, each_att, pre=pre)


# -------------------------------
# RQ7.
#   comparison between parallel computing and w/o
#


class Plot7_parallel_computing(GraphSetup):
    def __init__(self, nb_iter, mp_cores=3,
                 m1=20, m2=8, n_e=2, figname='exp7a_'):
        self._nb_iter = 1 if nb_iter <= 0 else nb_iter
        self._mp_cores = mp_cores
        gen = rep = None  # False
        super().__init__(gen, rep, m1, m2, n_e, figname)

    def recap_sub_data(self, dframe, nb_row=4):
        nb_set = len(dframe) - nb_row + 1
        nb_set = nb_set // (self._nb_iter + 1)
        id_set = [i * (self._nb_iter + 1) for i in range(nb_set + 1)]
        id_set = [i + nb_row - 1 for i in id_set]
        return nb_set, id_set

    def prepare_graph(self):
        csv_row_1 = unique_column(11 + 58)
        params = csv_row_1[: 11]
        tag_trn = csv_row_1[11:]
        return params, tag_trn  # ,tag

    def picking_fig_tags(self, tag):
        # Ds,Ds_avg,t(Ds), 3*2+2+3*2 =14
        tag_bin_sa1 = tag[: 14]
        tag_bin_sa2 = tag[14: 28]
        tag_multivar = [tag[28: 28 + 9],
                        tag[37: 37 + 9],
                        tag[46: 46 + 9]]
        tag_ut = tag[-3:]
        return tag_bin_sa1, tag_bin_sa2, tag_multivar, tag_ut

    # aka. def draw_sub3_dat():
    def gathering_whole_dat(self, dframe, nb_set, id_set, tag):
        i, k = 0, 0
        df_raw = dframe[tag].iloc[id_set[i] + 1: id_set[i + 1]]
        for i in range(1, nb_set):
            df_tmp = dframe[tag].iloc[id_set[i] + 1: id_set[i + 1]]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw.reset_index(drop=True)

    def gathering_sen_att_both(self, dframe, tag_sa1, tag_sa2):
        df_no_sa1 = dframe[tag_sa1]
        df_no_sa2 = dframe[tag_sa2]
        df_no_sa2 = df_no_sa2.iloc[self._nb_iter:]
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_no_sa2 = df_no_sa2.rename(columns=columns)
        df_raw = pd.concat([df_no_sa1, df_no_sa2], axis=0)
        return df_raw.reset_index(drop=True)


def _mp_present_tim_bin(df_bin, tag, suff, remark='tim'):
    if remark == 'tim':
        suff += '_binval_tim'
        scat_X = df_bin[tag[1]].values.astype(DTY_FLT)
        scat_Ys = [df_bin[tag[2]].values.astype(DTY_FLT),
                   df_bin[tag[4]].values.astype(DTY_FLT),
                   df_bin[tag[3]].values.astype(DTY_FLT)]

        annotX = r'T_{\mathbf{D}(S_1,\bar{S}_1)}'
        annotY = r'T_{\mathbf{D}_{\mathbf{a}}(S,a_i)}'
        annotY_hat = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)}'
        annotYs = [r'$T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)}$',
                   '${}$'.format(annotY_hat),
                   '${}$ par'.format(annotY_hat)] 
        annots = ['${}$ aka. ${}$'.format(annotY, annotX),
                  '${}$ aka. {}'.format(annotY_hat, annotYs[0]),
                  '${} = {}$'.format(annotY_hat, annotY)]
        multi_lin_reg_without_distr(scat_X, scat_Ys, annotYs, annots,
                                    suff, snspec='sty4')

        scatter_parl_chart_renew(scat_Ys[1], scat_Ys[2],
                                 figname=suff, identity=True)
        Ys = [np.log10(y / scat_X) for y in scat_Ys]
        annots[1] = r'$\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{\mathbf{a}}(S,a_i)} })$'
        multi_lin_reg_without_distr(scat_X, Ys, annotYs, annots,  # suff+'_log',
                                    suff + '_alt', snspec='sty6')
        return

    if remark in ['scat_max', 'Ds_max']:
        suff += '_binval_scat_max'  # '_Ds_max'
        scat_X_max = df_bin[tag[1]].values.astype(DTY_FLT)
        scat_Ys_max = [df_bin[tag[2]].values.astype(DTY_FLT),
                       df_bin[tag[4]].values.astype(DTY_FLT),
                       df_bin[tag[3]].values.astype(DTY_FLT)]

        annotX = r'\mathbf{D}(S_1,\bar{S}_1)'
        annotY = r'\mathbf{D}_{\mathbf{a}}(S,a_i)'
        annotY_hat = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
        annotYs = [r'$\hat{\mathbf{D}}(S_1,\bar{S}_1)$',
                   '${}$'.format(annotY_hat), '${}$ par'.format(annotY_hat)]
        annots = ['${}$ aka. ${}$'.format(annotY, annotX),
                  '${}$ aka. {}'.format(annotY_hat, annotYs[0]),
                  '${} = {}$'.format(annotY_hat, annotY)]
        scat_X, scat_Ys = scat_X_max, scat_Ys_max
        del scat_X_max, scat_Ys_max, annotX, annotY, annotY_hat

    elif remark in ['scat_avg', 'Ds_avg']:
        suff += '_binval_scat_avg'  # '_Ds_avg'
        scat_X_avg = df_bin[tag[6]].values.astype(DTY_FLT)
        scat_Ys_avg = [df_bin[tag[-1]].values.astype(DTY_FLT),
                       df_bin[tag[-2]].values.astype(DTY_FLT)]

        annotY = r'\mathbf{D}_{\mathbf{a}}^{avg}(S,a_i)'
        annotY_hat = r'\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i)'
        annotYs = ['${}$'.format(annotY_hat), '${}$ par'.format(annotY_hat)]
        annots = ['${}$'.format(annotY), '${}$'.format(annotY_hat),
                  r'$\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i) = \mathbf{D}_{\mathbf{a}}^{avg}(S,a_i)$']
        scat_X, scat_Ys = scat_X_avg, scat_Ys_avg
        del scat_X_avg, scat_Ys_avg, annotY, annotY_hat

    multi_lin_reg_without_distr(scat_X, scat_Ys, annotYs, annots,
                                suff, snspec='sty3b')  # 'sty3a')
    Ys = [y / scat_X - 1 for y in scat_Ys]
    if remark in ['scat_max', 'Ds_max']:
        annots[1] = r'$\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S,a_i) }{ \mathbf{D}_{\mathbf{a}}(S,a_i) }-1$'
    elif remark in ['scat_avg', 'Ds_avg']:
        annots[1] = r'$\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i) }{ \mathbf{D}_{\mathbf{a}}^{avg}(S,a_i) }-1$'
    multi_lin_reg_without_distr(scat_X, Ys, annotYs, annots,  # suff+'_log',
                                suff + '_alt', snspec='sty6')
    return


def _mp_present_tim_multivar(df_multivar, tag, suff, remark='tim'):
    if remark.endswith('tim'):  # remark == 'tim':
        # suff += '_multival_tim'
        annotX = r'T_{\mathbf{D}_{\mathbf{a}}(S)}'
        annotY = r'T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)}'

    elif remark.endswith('max'):
        # suff += '_multival_Dsmax'
        annotX = r'\mathbf{D}_{\mathbf{a}}(S)'
        annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
    elif remark.endswith('avg'):
        # suff += '_multival_Dsavg'
        annotX = r'\mathbf{D}_{\mathbf{a}}^{avg}(S)'
        annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S)'
    annotYs = ['${}$'.format(annotY), '${}$ par'.format(annotY)]
    annots = ['${}$'.format(annotX), annotYs[0],
              '${}={}$'.format(annotY, annotX)
              ]  # '${}$ (sec)'.format(annotX),
    if remark.endswith('tim'):
        annots[0] += ' (sec)'
    kws = {'snspec': 'sty4' if remark == 'tim' else 'sty3b'}
    suff = '{}_multival_{}'.format(suff, remark)

    scat_X = df_multivar[tag[0]].values.astype(DTY_FLT)
    scat_Y_seq = df_multivar[tag[2]].values.astype(DTY_FLT)
    scat_Y_par = df_multivar[tag[1]].values.astype(DTY_FLT)
    scat_Ys = [scat_Y_seq, scat_Y_par]
    if remark.endswith('tim'):
        scatter_parl_chart_renew(  # + '_mp3c'
            scat_Y_seq, scat_Y_par, figname=suff, identity=True)
    del scat_Y_seq, scat_Y_par
    multi_lin_reg_without_distr(scat_X, scat_Ys, annotYs, annots, suff,
                                **kws)
    if remark == 'tim':
        Ys = [np.log10(y / scat_X) for y in scat_Ys]
        annots[1] = r'$\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })$'
    else:
        Ys = [y / scat_X - 1 for y in scat_Ys]
    if remark.endswith('max'):
        annots[1] = r'$\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1$'
    elif remark.endswith('avg'):
        annots[1] = r'$\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^{avg}(S) }-1$'
    multi_lin_reg_without_distr(scat_X, Ys, annotYs, annots,
                                suff + '_alt', snspec='sty6')  # '_log'
    del annotX, annotY
    return


class Distributed_GA_mp(Plot7_parallel_computing):
    def schedule_mspaint(self, raw_dframe, mp_cores=3, pre='minmax',
                         verbose=False):
        nb_set, id_set = self.recap_sub_data(raw_dframe, nb_row=4)
        _, tag_trn = self.prepare_graph()  # tag_pm, /tag_pms
        (tag_bin_sa1, tag_bin_sa2, tag_multivar,
         tag_ut) = self.picking_fig_tags(tag_trn)

        df_raw = self.gathering_whole_dat(raw_dframe, nb_set, id_set, tag_trn)
        suff = '{}{}_mp{}c'.format(self._figname, pre, mp_cores)
        if verbose:
            self.draw_distapprox(df_raw, nb_set, id_set, tag_bin_sa1,
                                 tag_bin_sa2, suff)
        self.draw_distextend(df_raw, nb_set, id_set, tag_multivar,
                             tag_ut, suff, verbose)
        return

    def draw_distapprox(self, dframe, nb_set, id_set,
                        tag_bin_sa1, tag_bin_sa2, suff):
        # Time Cost
        tag_tim = [2, 5, 7, 10, 13]
        tag_tim = [[tag_bin_sa1[i] for i in tag_tim],
                   [tag_bin_sa2[i] for i in tag_tim]]

        df_bin = self.gathering_sen_att_both(dframe,
                                             tag_tim[0], tag_tim[1])
        tag_tim = tag_tim[0]
        suff += '_approx'
        _mp_present_tim_bin(df_bin, tag_tim, suff, remark='tim')

        tag_Ds_max = [0, 3, 6, 8, 11]
        tag_Ds_avg = [1, 4, 9, 12]
        tag_Ds = tag_Ds_max + tag_Ds_avg
        tag_Ds = [[tag_bin_sa1[i] for i in tag_Ds],
                  [tag_bin_sa2[i] for i in tag_Ds]]

        df_bin = self.gathering_sen_att_both(dframe, tag_Ds[0], tag_Ds[1])
        tag_Ds = tag_Ds[0]
        _mp_present_tim_bin(df_bin, tag_Ds, suff, remark='scat_max')
        _mp_present_tim_bin(df_bin, tag_Ds, suff, remark='scat_avg')
        return

    def draw_distextend(self, dframe, nb_set, id_set,
                        tag_multivar, tag_ut, suff, verbose=False):
        # Time Cost, `DistExtend` _multivar
        tag_extend = [0, 1, 2]
        tag_extend = [[t[i] for i in tag_extend] for t in tag_multivar]
        tag_extend = tag_extend[0] + tag_extend[1] + tag_extend[2]
        df_multivar = dframe[tag_extend]

        suff += '_extend'
        tag = [tag_extend[2], tag_extend[5], tag_extend[8]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'tim')
        tag = [tag_extend[i] for i in [0, 3, 6]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'Dsmax')  # 'Ds_max')
        tag = [tag_extend[i] for i in [1, 4, 7]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'Dsavg')  # 'Ds_avg')

        if not verbose:
            return
        # each `DistApprox` in DistExtend
        tag_sa1 = tag_multivar[0][3: 6] + tag_multivar[1][
            3: 6] + tag_multivar[2][3: 6]
        tag_sa2 = tag_multivar[0][6:] + tag_multivar[
            1][6:] + tag_multivar[2][6:]
        df_multivar = self.gathering_sen_att_both(dframe,
                                                  tag_sa1, tag_sa2)
        suff = suff.replace('extend', 'approx')
        tag = [tag_sa1[i] for i in [2, 5, 8]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'tim')
        tag = [tag_sa1[i] for i in [0, 3, 6]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'Dsmax')
        tag = [tag_sa1[i] for i in [1, 4, 7]]
        _mp_present_tim_multivar(df_multivar, tag, suff, 'Dsavg')

        return


# -------------------------------
#


class Plot5_hyperparameter_renew(Plot7_parallel_computing):
    def __init__(self, nb_iter, mp_cores=3, omitted=True,
                 m1=25, m2=11, n_e=2, figname='exp5_'):
        super().__init__(nb_iter, mp_cores, m1, m2, n_e, figname)
        self._omit = omitted
        # Hyper-parameters
        self._m2_set = list(range(2, 23, 1))    # HyperEA,
        self._m1_set = list(range(3, 50, 2))    # ,HyperEB
        if omitted:
            self._m2_set = list(range(2, 14, 1))  # len= 21->12
            self._m1_set = list(range(3, 34, 2))  # len= 24->16
        # end of Hyper-parameters

    def prepare_graph(self, n_l):
        n_ell = 3 * n_l + 3
        csv_row_1 = unique_column(11 + n_ell)
        params = csv_row_1[: 11]
        direct = csv_row_1[11: 11 + 3]
        approx = csv_row_1[14: 14 + n_l]
        approx_avg = csv_row_1[14 + n_l: -n_l]
        approx_tim = csv_row_1[-n_l:]
        return params, direct, approx, approx_avg, approx_tim

    def recap_sub_data(self, dframe, nb_row=3):
        each_att = 3 * self._nb_iter
        each_gen = (1 + 2) * self._nb_iter
        each_set = each_att * 2 + each_gen + 1
        nb_set = len(dframe) - nb_row + 1
        nb_set = nb_set // each_set
        id_set = [i * each_set for i in range(nb_set + 1)]
        id_set = [i + nb_row - 1 for i in id_set]
        return nb_set, id_set, each_set, each_att, each_gen

    def recap_curr_dat_split(self, curr_df, curr_id, tag_direct,
                             tag_approx, tag_ap_avg, tag_ap_tim,
                             n_l, n_k=5, n_a=2, remark='Ds_avg'):
        # na_prime = n_a  # 2
        D_drt_bin = np.zeros((2, n_a, n_k))        # DistDirect_bin
        D_est_bin = np.zeros((n_a, n_l, n_k))      # ApproxDist_bin
        D_new_binval = np.zeros((n_a, n_l, n_k))   # dist_eap_bin_val
        D_ext_drt = np.zeros((1 + n_a, n_k))       # dist_extend_drt
        D_ext_est = np.zeros((1 + n_a, n_l, n_k))  # dist_extend_est
        # k = 2 if remark == 'tim' else (1 if remark.endswith('avg') else 0)
        if remark.endswith('tim'):
            k, tag_k = 2, tag_ap_tim
        elif remark.endswith('avg'):
            k, tag_k = 1, tag_ap_avg
        else:
            k, tag_k = 0, tag_approx
        # remark in ['Ds/Ds_max', 'Ds_avg', 'tim']

        for i in range(n_a):
            start = curr_id[0] + 1 + i * n_k * 3
            now_end = start + n_k - 1
            # for k in range(3):
            # DistDirect_bin'_10
            D_drt_bin[0, i] = curr_df[
                tag_direct[k]].loc[start: now_end].astype(DTY_FLT)
            # DistDirect_bin
            D_drt_bin[1, i] = curr_df[
                tag_direct[k]].loc[start + n_k: now_end + n_k].astype(DTY_FLT)

            # Initial implementation: ApproxDist_bin
            start = curr_id[0] + 1 + i * n_k * 3 + n_k
            now_end = start + n_k - 1
            for j in range(n_l):
                D_est_bin[i, j] = curr_df[tag_k[j]].loc[start: now_end].astype(DTY_FLT)

            # Extension implementation: DistApprox (bin-val)
            start = curr_id[0] + 1 + i * n_k * 3 + n_k * 2
            now_end = start + n_k - 1
            for j in range(n_l):
                D_new_binval[i, j] = curr_df[tag_k[j]].loc[start: now_end].astype(DTY_FLT)

        # Extension implementation: DistExtend +DistApprox*2
        for i in range(1 + n_a):
            start = curr_id[0] + 1 + 3 * n_k * 2 + i * n_k
            now_end = start + n_k - 1
            # for k in range(3):
            D_ext_drt[i] = curr_df[tag_direct[k]].loc[start: now_end].astype(DTY_FLT)
            for j in range(n_l):
                D_ext_est[i, j] = curr_df[tag_k[j]].loc[start: now_end].astype(DTY_FLT)

        return D_drt_bin, D_est_bin, D_new_binval, D_ext_drt, D_ext_est

    def recap_curr_dat_merge(self, dframe, id_set, tag_direct, tag_approx,
                             tag_ap_avg, tag_ap_tim, n_l, n_k=5, remark='tim'):
        if remark.endswith('tim'):
            k, tag_k = 2, tag_ap_tim
        elif remark.endswith('avg'):
            k, tag_k = 1, tag_ap_avg
        else:
            k, tag_k = 0, tag_approx
        nb_set = len(id_set) - 1
        na_prime = 2 * nb_set - 1  # 1 + 2 * 4
        Dbin_drt = np.zeros((2, na_prime, n_k))       # D_drt_bin
        Dbin_est = np.zeros((n_l, na_prime, n_k))     # D_est_bin
        Dnew_binval = np.zeros((n_l, na_prime, n_k))  # D_new_binval
        Dext_drt_both = np.zeros((nb_set, n_k))
        Dext_est_both = np.zeros((n_l, nb_set, n_k))
        Dext_drt_each = np.zeros((na_prime, n_k))
        Dext_est_each = np.zeros((n_l, na_prime, n_k))

        i = 0
        curr_df = dframe.iloc[id_set[i]: id_set[i + 1]]
        tmp_drt, tmp_est, tmp_bin, t_ext_drt, t_ext_est = self.recap_curr_dat_split(
            curr_df, [id_set[i], id_set[i + 1]], tag_direct, tag_approx,
            tag_ap_avg, tag_ap_tim, n_l, n_k, n_a=1, remark=remark)
        idx_p = idx = 0  # start, index = 0, 0
        Dbin_drt[0, idx_p] = tmp_drt[0, idx]
        Dbin_drt[1, idx_p] = tmp_drt[1, idx]
        for j in range(n_l):
            Dbin_est[j, idx_p] = tmp_est[idx, j]
            Dnew_binval[j, idx_p] = tmp_bin[idx, j]
        Dext_drt_both[i] = t_ext_drt[0]
        Dext_drt_each[idx_p] = t_ext_drt[1 + idx]
        for j in range(n_l):
            Dext_est_both[j, i] = t_ext_est[0, j]
            Dext_est_each[j, idx_p] = t_ext_est[1 + idx, j]

        for i in range(1, nb_set):
            curr_df = dframe.iloc[id_set[i]: id_set[i + 1]]
            tmp_drt, tmp_est, tmp_bin, t_ext_drt, t_ext_est = self.recap_curr_dat_split(
                curr_df, [id_set[i], id_set[i + 1]], tag_direct,
                tag_approx, tag_ap_avg, tag_ap_tim, n_l, n_k,
                n_a=2, remark=remark)
            for idx in range(2):
                idx_p += 1
                Dbin_drt[0, idx_p] = tmp_drt[0, idx]
                Dbin_drt[1, idx_p] = tmp_drt[1, idx]
                for j in range(n_l):
                    Dbin_est[j, idx_p] = tmp_est[idx, j]
                    Dnew_binval[j, idx_p] = tmp_bin[idx, j]
                Dext_drt_each[idx_p] = t_ext_drt[1 + idx]
                for j in range(n_l):
                    Dext_est_each[j, idx_p] = t_ext_est[1 + idx, j]
            Dext_drt_both[i] = t_ext_drt[0]
            for j in range(n_l):
                Dext_est_both[j, i] = t_ext_est[0, j]

        return (Dbin_drt, Dbin_est, Dnew_binval,
                Dext_drt_both, Dext_est_both,
                Dext_drt_each, Dext_est_each)

    def present_ext_multivar(self, DExt_drt_pl, DExt_est_pl,
                             n_l,  # DExt_drt_sing, DExt_est_sing, n_l,
                             tag_ms_set, picked_m, suff, remark='tim',
                             alternative=True):  # default:False
        # DExt_drt/est_pl  .shape= (nb_set  , n_k) | (n_l, nb_set  , n_k)
        # DExt_drt/est_sing.shape= (na_prime, n_k) | (n_l, na_prime, n_k)
        DExt_drt_pl = DExt_drt_pl.reshape(-1)
        DExt_est_pl = DExt_est_pl.reshape(n_l, -1)
        # DExt_drt_sing = DExt_drt_sing.reshape(-1)
        # DExt_est_sing = DExt_est_sing.reshape(n_l, -1)

        if remark.endswith('tim'):
            annotX = r'T_{ \mathbf{D}_{\mathbf{a}}(S) }'
            annotY = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }'
            anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S)} }{ T_{\mathbf{D}_{\mathbf{a}}(S)} })'
        elif remark.endswith('avg'):
            annotX = r'\mathbf{D}_{\mathbf{a}}^{avg}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^{avg}(S) }-1'
        else:
            annotX = r'\mathbf{D}_{\mathbf{a}}(S)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}(S)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }-1'

        annots = ['${}$'.format(annotX), '${}$'.format(annotY),
                  '${}={}$'.format(annotY, annotX)]
        kw = {'snspec': 'sty5b', 'corr': False}
        if remark.endswith('tim'):
            annots[0] += ' (sec)'
            annots[1] += ' (sec)'
            DExt_est_pl_alt = np.array([np.log10(
                DExt_est_pl[i] / DExt_drt_pl) for i in range(n_l)])
        else:
            kw['corr'] = True
            kw['curr_legend_nb_split'] = 6
            DExt_est_pl_alt = np.array([
                DExt_est_pl[i] / DExt_drt_pl - 1 for i in range(n_l)])

        hyper_params_lin_reg(DExt_drt_pl, DExt_est_pl, tag_ms_set, picked_m,
                             annots, suff, **kw)  # corr=False,snspec='sty5b')
        if not alternative:
            return
        annots[1] = '${}$'.format(anotAP)
        kw['snspec'] = 'sty6'
        hyper_params_lin_reg(DExt_drt_pl, DExt_est_pl_alt, tag_ms_set,
                             picked_m, annots, suff + '_alt', **kw)
        return

    def present_ext_midterm(self, DExt_drt_sing, DExt_est_sing, n_l,
                            tag_ms, picked_m, suff, remark='tim', alternative=True):
        # DExt_drt/est_sing.shape= (na_prime, n_k) | (n_l, na_prime, n_k)
        DExt_drt_sing = DExt_drt_sing.reshape(-1)
        DExt_est_sing = DExt_est_sing.reshape(n_l, -1)

        if remark.endswith('tim'):
            annotX = r'T_{ \mathbf{D}_{\mathbf{a}}(S,a_i) }'
            annotY = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}}(S,a_i) }'
            anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}_{\mathbf{a}}(S,a_i)} })'
        elif remark.endswith('avg'):
            annotX = r'\mathbf{D}_{\mathbf{a}}^{avg}(S,a_i)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i) }{ \mathbf{D}_{\mathbf{a}}^{avg}(S,a_i) }-1'
        else:
            annotX = r'\mathbf{D}_{\mathbf{a}}(S,a_i)'
            annotY = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
            anotAP = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S,a_i) }{ \mathbf{D}_{\mathbf{a}}(S,a_i) }-1'

        annots = ['${}$'.format(annotX), '${}$'.format(annotY),
                  '${}={}$'.format(annotY, annotX)]
        kw = {'snspec': 'sty5b', 'corr': False}
        if remark.endswith('tim'):
            annots[0] += ' (sec)'
            annots[1] += ' (sec)'
            DExt_est_sing_alt = np.array([np.log10(
                DExt_est_sing[i] / DExt_drt_sing) for i in range(n_l)])
        else:
            kw['corr'] = True
            kw['curr_legend_nb_split'] = 6
            DExt_est_sing_alt = np.array([
                DExt_est_sing[i] / DExt_drt_sing - 1 for i in range(n_l)])

        hyper_params_lin_reg(DExt_drt_sing, DExt_est_sing, tag_ms,
                             picked_m, annots, suff, **kw)
        if not alternative:
            return
        annots[1] = '${}$'.format(anotAP)
        kw['snspec'] = 'sty6'
        hyper_params_lin_reg(DExt_drt_sing, DExt_est_sing_alt, tag_ms,
                             picked_m, annots, suff + '_alt', **kw)
        return

    def present_ext_binval(self, D_drt, D_est, DExt_binval, n_l,
                           tag_ms, picked_m, suff, remark='tim',
                           alternative=True):
        # (bin) D_drt.shape= (2,   na_prime, n_k)
        # (bin) D_est.shape= (n_l, na_prime, n_k)
        # DExt_binval.shape= (n_l, na_prime, n_k)
        D_drt = D_drt.reshape(2, -1)
        D_est = D_est.reshape(n_l, -1)
        DExt_binval = DExt_binval.reshape(n_l, -1)

        if remark.endswith('tim'):
            annotX = r'T_{ \mathbf{D}(S_1,\bar{S}_1) }'
            annotY = r'T_{ \hat{\mathbf{D}}(S_1,\bar{S}_1) }'
            annotY_ext = r'T_{ \hat{\mathbf{D}}_{\mathbf{a}}(S,a_i) }'
        elif remark.endswith('avg'):
            annotX = r''
            annotY = r''
            annotY_ext = r'\hat{\mathbf{D}}_{\mathbf{a}}^{avg}(S,a_i)'
        else:
            annotX = r'\mathbf{D}(S_1,\bar{S}_1)'
            annotY = r'\hat{\mathbf{D}}(S_1,\bar{S}_1)'
            annotY_ext = r'\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)'
        if remark.endswith('avg'):
            return

        annots = ['${}$'.format(annotX), '${}$'.format(annotY),
                  '${} = {}$'.format(annotY, annotX)]
        annots_ext = ['${}$'.format(annotX), '${}$'.format(annotY_ext),
                      '${} = {}$'.format(annotY_ext, annotX)]
        # annots_ext = annots.copy()
        # annots_ext[1] = '${}$'.format(annotY_ext)
        kw = {'snspec': 'sty5b', 'corr': False}
        if remark.endswith('tim'):
            annots[0] += ' (sec)'
            annots[1] += ' (sec)'
            annots_ext[0] += ' (sec)'
            annots_ext[1] += ' (sec)'
            D_est_alt = np.array([np.log10(
                D_est[i] / D_drt[1]) for i in range(n_l)])
            DExt_bin_alt = np.array([np.log10(
                DExt_binval[i] / D_drt[1]) for i in range(n_l)])
        else:
            kw['corr'] = True
            kw['curr_legend_nb_split'] = 6
            D_est_alt = np.array([
                D_est[i] / D_drt[1] - 1 for i in range(n_l)])
            DExt_bin_alt = np.array([
                DExt_binval[i] / D_drt[1] - 1 for i in range(n_l)])
        # suff_1 = suff.replace(remark, 'ecai_{}'.format(remark))
        # suff_2 = suff.replace(remark, 'nips_{}'.format(remark))
        suff_1 = suff.replace(remark, 'ver1_{}'.format(remark))
        suff_2 = suff.replace(remark, 'ver2_{}'.format(remark))
        hyper_params_lin_reg(
            D_drt[1], D_est, tag_ms, picked_m, annots, suff_1, **kw)
        hyper_params_lin_reg(
            D_drt[1], DExt_binval, tag_ms, picked_m, annots_ext, suff_2,
            **kw)
        if not alternative:
            return

        if remark.endswith('tim'):
            anotAP = r'\lg(\frac{ T_{\hat{\mathbf{D}}(S_1,\bar{S}_1)} }{ T_{\mathbf{D}(S_1,\bar{S}_1)} })'
            anotAP_ext = r'\lg(\frac{ T_{\hat{\mathbf{D}}_{\mathbf{a}}(S,a_i)} }{ T_{\mathbf{D}(S_1,\bar{S}_1)} })'
        else:
            anotAP = r'\frac{ \hat{\mathbf{D}}(S_1,\bar{S}_1) }{ \mathbf{D}(S_1,\bar{S}_1) }-1'
            anotAP_ext = r'\frac{ \hat{\mathbf{D}}_{\mathbf{a}}(S,a_i) }{ \mathbf{D}(S_1,\bar{S}_1) }-1'
        annots[1] = '${}$'.format(anotAP)
        annots_ext[1] = '${}$'.format(anotAP_ext)
        # suff_3 = suff.replace(remark, 'alt_ecai_{}'.format(remark))
        # suff_4 = suff.replace(remark, 'alt_nips_{}'.format(remark))
        suff_3 = suff.replace(remark, 'alt_ver1_{}'.format(remark))
        suff_4 = suff.replace(remark, 'alt_ver2_{}'.format(remark))
        kw['snspec'] = 'sty6'
        hyper_params_lin_reg(D_drt[1], D_est_alt, tag_ms, picked_m,
                             annots, suff_3, **kw)
        hyper_params_lin_reg(D_drt[1], DExt_bin_alt, tag_ms, picked_m,
                             annots_ext, suff_4, **kw)
        return


class HyperEA_renew_m1fix(Plot5_hyperparameter_renew):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m2_set)
        _, tag_drt, tag_apx, tag_avg, tag_ut = self.prepare_graph(
            n_l)  # tag_pm,
        _, id_set, _, _, _ = self.recap_sub_data(raw_dframe, 3)  # nb_set,
        n_k = self._nb_iter
        ms_set = [r'$m_2$={}'.format(i) for i in self._m2_set]
        picked_m = [2, 4, 6, 8, 10]  # [4,6,8,10,12]

        this_suf = '{}{}_mp_m2fix'.format(self._figname, pre)
        for this_rmk in ['tim', 'Dsmax', 'Dsavg']:
            (D_drt, D_est, DExt_bin, DExt_drt_pl, DExt_est_pl,
             DExt_drt_sing, DExt_est_sing) = self.recap_curr_dat_merge(
                raw_dframe, id_set, tag_drt, tag_apx, tag_avg, tag_ut, n_l, n_k, this_rmk)
            self.present_ext_multivar(
                DExt_drt_pl, DExt_est_pl, n_l, ms_set, picked_m,
                this_suf + '_ext_' + this_rmk,
                remark=this_rmk, alternative=False)
            self.present_ext_midterm(
                DExt_drt_sing, DExt_est_sing, n_l, ms_set, picked_m,
                this_suf + '_mid_' + this_rmk,
                remark=this_rmk, alternative=False)
            self.present_ext_binval(
                D_drt, D_est, DExt_bin, n_l, ms_set, picked_m,
                this_suf + '_prev_' + this_rmk,
                remark=this_rmk, alternative=False)  # True)
        return


class HyperEB_renew_m2fix(Plot5_hyperparameter_renew):
    def schedule_mspaint(self, raw_dframe, pre='minmax'):
        n_l = len(self._m1_set)  # tag_pm,
        _, tag_drt, tag_apx, tag_avg, tag_ut = self.prepare_graph(n_l)
        nb_set, id_set, _, _, _ = self.recap_sub_data(raw_dframe, 3)
        n_k = self._nb_iter
        ms_set = [r'$m_1$={}'.format(i) for i in self._m1_set]
        picked_m = [0, 3, 6, 9, 12]  # [3,9,15,21,27]

        this_suf = '{}{}_mp_m1fix'.format(self._figname, pre)
        for this_rmk in ['tim', 'Dsmax', 'Dsavg']:
            (D_drt, D_est, DExt_bin, DExt_drt_pl, DExt_est_pl,
             DExt_drt_sing, DExt_est_sing) = self.recap_curr_dat_merge(
                raw_dframe, id_set, tag_drt, tag_apx, tag_avg, tag_ut, n_l, n_k, this_rmk)
            self.present_ext_multivar(
                DExt_drt_pl, DExt_est_pl, n_l, ms_set, picked_m,
                this_suf + '_ext_' + this_rmk,
                remark=this_rmk, alternative=False)
            self.present_ext_midterm(
                DExt_drt_sing, DExt_est_sing, n_l, ms_set, picked_m,
                this_suf + '_mid_' + this_rmk,
                remark=this_rmk, alternative=False)
            self.present_ext_binval(
                D_drt, D_est, DExt_bin, n_l, ms_set, picked_m,
                this_suf + '_prev_' + this_rmk,
                remark=this_rmk, alternative=False)  # True)
        return


# -------------------------------
# RQ5.
#   hyper-parameter sensitivity
#


# -------------------------------
#


# ===============================
# Table 2C


# -------------------------------
# RQ1.
#   compared with sota fairness
#


class Table4_comparison(Plot4_comparison):
    def __init__(self, nb_iter, nb_cls, m1=25, m2=11,
                 n_e=2, figname='tab2_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)
        self._abbr_clfs = ['DT', 'NB', 'LR1', 'LR2', 'LM1', 'LM2',
                           'kNNu', 'kNNd', 'MLP', 'linSVM', 'SVM']
        self._abbr_ensf = ['bagging', 'AdaBoost', 'LightGBM',
                           'FairGBM (fpr)', 'FairGBM (fnr)',
                           'FairGBM (fpr,fnr)', 'AdaFair']

    def picking_tab_tags(self, tag, ind=[0, 1, 2, 3, 5], dist_df='both',
                         omitted=True):  # ind=[0,1,2,3,7]
        '''
        acc_orgin = tag[: 13 - 1]  # origin 13-1
        acc_delta = tag[13 * 2: 13 * 3 - 1]
        tag_f_vot = tag[39: 39 + 11 * 4]  # +7*4
        tag_f_man = tag[83: 83 + 29 * 4]  # 67+16*4
        tag_f_ext = tag[199: 199 + 10 * 3 * 2 + 3]
        '''

        tmp_norm = 8 if omitted else 13
        tmp_vote = 11 if omitted else 17
        tmp_manf = 29      # 29*4     =116
        tmp_manf_ext = 10  # 10*3*2+6 =63
        norm_minus = tmp_norm * 3
        vote_minus = norm_minus + tmp_vote * 4
        manf_minus = vote_minus + 29 * 4

        acc_orgin = tag[: tmp_norm]
        acc_delta = tag[tmp_norm * 2: norm_minus]
        tag_f_vot = tag[norm_minus: norm_minus + tmp_vote * 4]
        tag_f_man = tag[vote_minus: vote_minus + 29 * 4]
        tag_f_ext = tag[manf_minus: manf_minus + 63]
        del norm_minus, vote_minus, manf_minus

        tag_f_vot = [tag_f_vot[: tmp_vote],
                     tag_f_vot[tmp_vote: tmp_vote * 2],
                     tag_f_vot[tmp_vote * 2: tmp_vote * 3],
                     tag_f_vot[tmp_vote * 3:]]
        tag_f_man = [tag_f_man[:29], tag_f_man[29: 58],
                     tag_f_man[58: 29 * 3], tag_f_man[87:]]
        tag_f_ext = [tag_f_ext[:10], tag_f_ext[10: 20], tag_f_ext[20: 30],
                     tag_f_ext[30: 40], tag_f_ext[40: 50],
                     tag_f_ext[50: 60], tag_f_ext[60:] + [''] * 7]
        del tmp_norm, tmp_vote, tmp_manf, tmp_manf_ext

        # tYs_k2: `6,14,22`: df.ecai (DistDirect_bin, ApproxDist_bin, DistApprox bin-val)
        #         `7,15,23`: df.nips (DistDirect_bin, ApproxDist_bin, DistApprox bin-val)
        #         `9,   25`: df.nips.avg (DistDirect_bin, DistApprox bin-val)
        # tYs_k3: `6,7,9`: df_ecai, df.nips, df.nips.avg
        tYs_k3 = [6, 7, 9, ]  # tYs_k3 = [6, 7, 9, 8]
        tYs_k1, tYs_k2 = [0, 1, 2, 4], [6, 14, 22, 7, 15, 23, 9, 25, ]
        tYs_k1 = [k + 6 for k in tYs_k1]  # k+3*2
        if not omitted:
            tYs_k1 = [k + 10 for k in [2, 3, 4, 6]]
        if dist_df == 'direct':
            tYs_k2 = [6, 7, 9, ]
        elif dist_df == 'approx':
            tYs_k2 = [14, 15, 23, 25, ]
        else:
            tYs_k2 = [6, 7, 9, 14, 15, 22, 23, 25]  # [6, 14, 7, 15, 23, 9, 25]
        # if dist_df == 'direct':
        #   tYs_k2, tYs_k3 = [6, 7, 9, ], [6, ]
        # elif dist_df == 'approx':
        #   tYs_k2, tYs_k3 = [14, 15, ], [7, ]
        # elif dist_df == 'extend':
        #   tYs_k2, tYs_k3 = [22, 23, 25, ], [9, ]
        tmp_f_vot = [[t[k] for k in tYs_k1] for t in tag_f_vot]
        tmp_f_man = [[t[k] for k in tYs_k2] for t in tag_f_man]
        # # tYs_k3: `6,7,9`: df_ecai, df.nips, df.nips.avg
        # tYs_k3 = [6, 7, 9, ]  # tYs_k3 = [6, 7, 9, 8]
        tmp_f_ext = [[t[k] for k in tYs_k3] for t in tag_f_ext[:-1]
                     ] + [tag_f_ext[-1][: 3]]  # tim_elapsed
        # tmp_f_ext = [tmp_f_ext[0] + tmp_f_ext[3],
        #              tmp_f_ext[1] + tmp_f_ext[4],
        #              tmp_f_ext[2] + tmp_f_ext[5],
        #              tmp_f_ext[6] + [''] * 3]
        if dist_df == 'direct':
            tmp_f_ext = [tmp_f_ext[1], tmp_f_ext[2], tmp_f_ext[0], tmp_f_ext[6]]
        elif dist_df == 'approx':
            tmp_f_ext = [tmp_f_ext[4], tmp_f_ext[5], tmp_f_ext[3], tmp_f_ext[6]]
        else:
            tmp_f_ext = [tmp_f_ext[1] + tmp_f_ext[4],
                         tmp_f_ext[2] + tmp_f_ext[5],
                         tmp_f_ext[0] + tmp_f_ext[3],
                         tmp_f_ext[6] + [''] * 3]

        tmp_f_vm = [t1 + t2 for t1, t2 in zip(tmp_f_vot, tmp_f_man)]
        del tmp_f_vot, tmp_f_man
        tmp_a1 = [acc_orgin[k] for k in ind]
        tmp_a2 = [acc_delta[k] for k in ind]
        return tmp_a1, tmp_a2, tmp_f_vm, tmp_f_ext

    def tabulating_fifth_alt(self, dframe, tag_trn, tag_tst, nb_set, id_set,
                             each_gen, each_att, ind=[0, 1, 2, 3, 5], ddof=1,
                             dist_df='both', picked_clf=0, csv_w=None,
                             suffix='suff', alpha=.5):  # suffix,prefix
        tYs_k2_k3 = [0, 1, 2, 3, ] + [4, 7, ]  # [4, 5, 6, 7, 8]
        if dist_df == 'direct':
            tYs_k2_k3 = [0, 1, 2, 3, ] + [4, ]
        elif dist_df == 'approx':
            tYs_k2_k3 = [0, 1, 2, 3, ] + [7, ]
        ta_org, ta_dta, tf_vm, tf_ext = self.picking_tab_tags(
            tag_trn, ind, dist_df)
        tf_vm = [[t[k] for k in tYs_k2_k3] for t in tf_vm]
        tag_trn_a_f = (ta_org + ta_dta + tf_vm[0] + tf_ext[0],
                       ta_org + ta_dta + tf_vm[1] + tf_ext[1])
        ta_org, ta_dta, tf_vm, tf_ext = self.picking_tab_tags(
            tag_tst, ind, dist_df)
        tf_vm = [[t[k] for k in tYs_k2_k3] for t in tf_vm]
        tag_tst_a_f = (ta_org + ta_dta + tf_vm[0] + tf_ext[0],
                       ta_org + ta_dta + tf_vm[1] + tf_ext[1])
        del ta_org, ta_dta, tf_vm, tf_ext, tYs_k2_k3
        # tag_trn/tst_a_f, 22 columns: 5+5 (orgin+delta) +4 (GF*3+DR) +2 (manf
        #                              direct/approx) +3*2 (manf_ext 3*drt+3*apx)
        # nb_col = 22 if dist_df == 'both' else 18  # 5*2+4+4*1|2
        nb_col = len(ind) * 2 + 4 + 4 * (2 if dist_df == 'both' else 1)
        nb_att = nb_set * 2 - 1   # aka. # (nb_set - 1) * 2 + 1
        nb_clf = (each_gen + each_att) // self._nb_iter
        U_f1_raw = np.zeros((nb_att, nb_col, self._nb_iter))
        # U_cp_raw = np.zeros((nb_att, nb_col, self._nb_iter))

        i, k = 0, 0
        df_trn = dframe[tag_trn_a_f[0]].iloc[id_set[i] + 1: id_set[i + 1]]
        df_tst = dframe[tag_tst_a_f[0]].iloc[id_set[i] + 1: id_set[i + 1]]
        # U_trn_raw, U_tst_raw = self.tabulate_five_sub1_dat(
        _, U_tst_raw = self.tabulate_five_sub1_dat(
            df_trn, df_trn, nb_clf, nb_ind=len(ind))
        U_f1_raw[k] = U_tst_raw[picked_clf]
        k += 1

        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(curr_set,
                                  curr_set + each_gen + each_att))
            df_trn = dframe[tag_trn_a_f[0]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[0]].iloc[curr_loc]
            _, U_tst_raw = self.tabulate_five_sub1_dat(
                df_trn, df_trn, nb_clf, nb_ind=len(ind))
            U_f1_raw[k] = U_tst_raw[picked_clf]
            k += 1

            curr_loc = list(range(curr_set, curr_set + each_gen)) + list(range(
                curr_set + each_gen + each_att, curr_set + each_gen + each_att * 2))
            df_trn = dframe[tag_trn_a_f[1]].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f[1]].iloc[curr_loc]
            _, U_tst_raw = self.tabulate_five_sub1_dat(
                df_trn, df_trn, nb_clf, nb_ind=len(ind))
            U_f1_raw[k] = U_tst_raw[picked_clf]
            k += 1

        suff = suffix + '_tst_each_confusion'  # suff = self._figname.replace()
        # U_cp_raw, Uq1, Uq2 = self.tabulate_five_sub3_plo(U_f1_raw, ind,
        #                                                  suff)
        U_cp_raw, _, Uq2 = self.tabulate_five_sub3_plo(U_f1_raw, ind,
                                                       suff)
        if 0 in ind:
            self.tabulate_five_sub4_plo(
                U_cp_raw, nb_ind=len(ind), loc=0,
                suff=suffix + '_tst_tradeoff_ind0',
                alpha=alpha, num_gap=100, ddof=1)
            self.tabulate_five_sub4_plo(
                U_cp_raw, nb_ind=len(ind), loc=1,
                suff=suffix + '_tst_tradeoff_ind3',
                alpha=alpha, num_gap=100, ddof=1)
        ans_tex, cmp_tex = self.tabulate_five_sub2_out(U_f1_raw, ddof,
                                                       rez=4)
        tmp_a = [self._pick_metric[k] for k in ind] + [''] * len(ind)
        tmp_b = self._picked_keys + ['direct', '', '', 'approx', '', '']
        tmp_f = 'picked_clf = {} {}'.format(i, self._abbr_ensf[picked_clf])
        csv_w.writerow([tmp_f] + [''] * 21 + ['[END]'])
        csv_w.writerow([''] + tmp_a + tmp_b)
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerows(cmp_tex)
        csv_w.writerow([''])
        del tmp_a, tmp_b, tmp_f, ans_tex, cmp_tex
        return

    def tabulate_five_sub1_dat(self, df_trn, df_tst, nb_clf,
                               nb_ind=5):  # ,picked_clf):
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
        # # U_trn/tst_raw .shape= (14+4, 8, 5)
        U_trn_raw[:, :nb_ind * 2, :] = U_trn_raw[:, :nb_ind * 2, :] * 100
        U_tst_raw[:, :nb_ind * 2, :] = U_tst_raw[:, :nb_ind * 2, :] * 100
        return U_trn_raw, U_tst_raw

    def tabulate_five_sub2_out(self, U_f1_raw, ddof=1, rez=4):
        nb_att, nb_col, k = U_f1_raw.shape  # (9,22,5)
        U_avg = U_f1_raw.mean(axis=2)
        U_std = U_f1_raw.std(axis=2, ddof=ddof)
        ans_tex, cmp_tex = [], []  # 对照组
        for i in range(nb_att):
            tmp_tex, tmp_my = [''], ['']
            for j in range(nb_col):
                tmp_tex.append(_encode_sign(U_avg[i, j], U_std[i, j],
                                            rez))
                mu, _, sigma = _avg_and_stdev(
                    U_f1_raw[i, j].tolist(), k)  # _,sigma2,_=
                tmp_my.append(_encode_sign(mu, sigma, rez))
            ans_tex.append(tmp_tex)
            cmp_tex.append(tmp_my)
        return ans_tex, cmp_tex

    def tabulate_five_sub3_plo(self, U_f1_raw, ind, figname='_confusion',
                               cmap_name='PuBu', rotate=0):  # 'Blurs_r'
        #                        cmap_name='GnBu'  # jet,Blues,
        # U_f1_raw .shape= (9,16|22,5) =(nb_att,nb_col,nb_iter)
        # 22 columns: 5|2*2 (normal+delta metric) +4 (3*GF+DR) +
        nb_att, nb_col, k = U_f1_raw.shape
        U_cp_raw = np.zeros((nb_att, nb_col - 4, k))
        for j in range(nb_col - 8 + 1):
            U_cp_raw[:, j, :] = U_f1_raw[:, j, :]
        for j in range(nb_col - 6, nb_col - 3):
            U_cp_raw[:, j - 1, :] = U_f1_raw[:, j, :]
        assert np.all(U_cp_raw[:, -4, :] == U_f1_raw[:, -8, :])
        assert np.all(U_cp_raw[:, -3, :] == U_f1_raw[:, -6, :])
        assert np.all(U_cp_raw[:, -2, :] == U_f1_raw[:, -5, :])
        assert np.all(U_cp_raw[:, -1, :] == U_f1_raw[:, -4, :])
        # U_cp_raw .shape= (9,12|18,5) =(nb_att,nb_col-4,nb_iter)

        # tmp_a = [self._pick_metric[k] for k in ind] + [''] * len(ind)
        # tmp_b = self._picked_keys + ['direct', '', '', 'approx', '', '']
        tmp_a = [self._pick_metric[k] for k in ind] + [
            r'$\Delta$' + self._pick_metric[k] for k in ind]
        tmp_b = self._picked_keys[: -1] + ['direct', '', '[END]']
        nb_col, nb_ind = nb_col - 4, len(ind)  # 2, 16-4=12
        U_cp_part1 = np.zeros((nb_att, nb_col - nb_ind, k))
        U_cp_part2 = np.zeros((nb_att, nb_col - nb_ind, k))
        for j in range(nb_ind):
            U_cp_part1[:, j, :] = U_cp_raw[:, j, :]
            U_cp_part2[:, j, :] = U_cp_raw[:, j + nb_ind, :]
        # for j in np.arange(nb_col - nb_ind - 1, nb_ind - 1, -1):
        for j in np.arange(nb_col - 1, 2 * nb_ind - 1, -1):
            U_cp_part1[:, j - nb_ind, :] = U_cp_raw[:, j, :]
            U_cp_part2[:, j - nb_ind, :] = U_cp_raw[:, j, :]
        # (Pdb) np.arange(12-1, 4-1, -1)  array([11,10,9,8, 7,6,5,4])
        # for j in range(nb_ind, nb_col - nb_ind):  # not j+2*nb_ind
        #   U_cp_part1[:, j, :] = U_cp_raw[:, j + nb_ind, :]
        #   U_cp_part2[:, j, :] = U_cp_raw[:, j + nb_ind, :]
        # 0,1, 4,5,6,7, 8,9,10,11  # 0,1 # 2->10, 2+4
        # 2,3, 4,5,6,7, 8,9,10,11  # 2,3 # 2->10, 2+2
        assert np.all(U_cp_part1[:, :nb_ind, :] == U_cp_raw[:, :nb_ind, :])
        assert np.all(U_cp_part2[:, :nb_ind, :] == U_cp_raw[:, nb_ind:2 * nb_ind, :])
        assert np.all(U_cp_part1[:, nb_ind:, :] == U_cp_raw[:, -8:, :])
        assert np.all(U_cp_part2[:, nb_ind:, :] == U_cp_raw[:, -8:, :])

        # U_f1_raw    .shape= (9,16|22,5) =(nb_att,nb_col,nb_iter)
        # U_cp_raw    .shape= (9,12|18,5) =(nb_att,nb_col-4=*,nb_iter)
        # U_cp_part1/2.shape= (9,10|13,5) =(nb_att,*-len(ind),nb_iter)
        nb_col = nb_col - nb_ind  # 12-2=10 or 18-5=13
        U_cp_part1 = U_cp_part1.transpose((1, 0, 2)).reshape(nb_col, -1)
        U_cp_part2 = U_cp_part2.transpose((1, 0, 2)).reshape(nb_col, -1)
        tmp_b = self._picked_keys[: -2] + [
            r'$\mathbf{df}_{prev}$' + '\n bin-val  ',
            r'  $\mathbf{df}_{prev}$' + '\n \t multival',
            r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']  # 4+4=8
        U_cp_tmp = U_cp_raw.transpose((1, 0, 2)).reshape(nb_col + nb_ind, -1)
        analogous_confusion_extended(
            U_cp_tmp[: nb_ind * 2], U_cp_tmp[nb_ind * 2:], tmp_a, tmp_b,
            figname, cmap_name=cmap_name, rotate=rotate)
        return U_cp_raw, U_cp_part1, U_cp_part2

    def tabulate_five_sub4_plo(self, U_cp_raw, nb_ind, loc=0,
                               suff='suff_ind',
                               alpha=.5, num_gap=100, ddof=0):
        # U_cp_tmp .shape= (12|18, 45) =(nb_col=?*2+4+4, nb_att*nb_iter)
        # nb_col = U_cp_tmp.shape[0] - 2 * nb_ind
        # U_cp_raw .shape= (9,12|18,5) =(nb_att,nb_col-4=*,nb_iter)
        tmp_b = self._picked_keys[: -2] + [
            r'$\mathbf{df}_{prev}$ bin-val',
            r'$\mathbf{df}_{prev}$ multival',
            r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']  # 4+4=8 # key_fair
        annotY = ' error rate' if loc == 0 else r'$($1$-$ performance$)$'  # ind==0
        kw_alt = {'alpha_loc': 'b4', 'alpha_clarity': .15,
                  'cmap_name': 'RdPu'}  # OrRd,cool,BuPu,YlGnBu ,RdPu
        X = np.linspace(0, 1, num_gap)
        nb_att, nb_col, k = U_cp_raw.shape

        # U_cp_raw .shape= (9,12|18,5) =(nb_att,nb_col-4=*,nb_iter)
        # U_cp_tmp .shape= (12|18, 45) =(nb_col=?*2+4+4, nb_att*nb_iter)
        U_cp_tmp = U_cp_raw.transpose((1, 0, 2)).reshape(nb_col, -1)
        nb_col = nb_col - 2 * nb_ind  # =12-2*2=8  # .shape= (12,45)
        U_cp_tmp_norm = U_cp_tmp[: nb_ind]     # .shape= (2,45)
        U_cp_tmp_fair = U_cp_tmp[nb_ind * 2:]  # .shape= (8,45)
        baseline_Ys = np.zeros((nb_col, num_gap, k * nb_att))  # (8,100,45)
        for j in range(nb_col):
            for kv_a, alpha in enumerate(X):
                baseline_Ys[j][kv_a] = (1 - U_cp_tmp_norm[loc] / 100
                                        ) * alpha + (1. - alpha) * U_cp_tmp_fair[j]
        fnm = '{}_each_att_gather'.format(suff)

        f_tmp = [0, 1, 2, 3, 4, 6, 7]
        tmp_b_shrink = [tmp_b[k] for k in f_tmp]
        _uncertainty_plotting(X, baseline_Ys[f_tmp], tmp_b_shrink,
                              annotY, ddof, figname=fnm, **kw_alt)
        return

    def tabulating_sixth_alt(self, dframe, tag_trn, tag_tst, nb_set, id_set,
                             each_gen, each_att, ind=[0, 1, 2, 3, 5], ddof=1,
                             dist_df='both', picked_clf=2, csv_w=None,
                             suffix='suff', alpha=.5):
        tYs_k2_k3 = [3, 4, 7, ]  # [0, 1, 2, 3, ] + [4, 7, ]  # DR,df.ecai,hat_df.ecai
        if dist_df == 'direct':
            tYs_k2_k3 = [3, ] + [4, ]
        elif dist_df == 'approx':
            tYs_k2_k3 = [3, ] + [7, ]
        ta_org, ta_dlt, tf_vm, tf_ext = self.picking_tab_tags(tag_trn, ind, dist_df)
        tf_vm = [[t[k] for k in tYs_k2_k3] for t in tf_vm]
        sen_att_1 = tf_vm[0][: -1] + tf_ext[0][: 3]  # 2+3=5
        sen_att_2 = tf_vm[1][: -1] + tf_ext[1][: 3]  # 2+3=5
        tag_trn_a_f = ta_org + ta_dlt + sen_att_1 + sen_att_2 + tf_ext[2][: 3]  # 23
        ta_org, ta_dlt, tf_vm, tf_ext = self.picking_tab_tags(tag_tst, ind, dist_df)
        tf_vm = [[t[k] for k in tYs_k2_k3] for t in tf_vm]
        sen_att_1 = tf_vm[0][: -1] + tf_ext[0][: 3]
        sen_att_2 = tf_vm[1][: -1] + tf_ext[1][: 3]
        tag_tst_a_f = ta_org + ta_dlt + sen_att_1 + sen_att_2 + tf_ext[2][: 3]
        del ta_org, ta_dlt, tf_vm, tf_ext, tYs_k2_k3
        # tag_trn/tst_a_f, 23 columns:
        #       5+5 (orgin+delta) +5*2 (DR, df_prev_bin-val, df_prev_multival,
        #       df, df_avg) each sen-att +3 (df_prev, df, df_avg) both sen-att
        nb_col = len(ind) * 2 + 5 * 2 + 3  # 23 or (2|3)*2+13 =17|19
        # nb_att = nb_set * 2 - 1
        nb_clf = (each_gen + each_att * 2) // self._nb_iter
        U_f1_raw = np.zeros((nb_set, nb_col + 1, self._nb_iter))

        i, k = 0, 0  # .fillna(0.)
        df_trn = dframe[tag_trn_a_f].iloc[id_set[i] + 1: id_set[i + 1]].fillna(0)
        df_tst = dframe[tag_tst_a_f].iloc[id_set[i] + 1: id_set[i + 1]].fillna(0)
        U_trn_raw, U_tst_raw = self.tabulate_five_sub1_dat(
            df_trn, df_tst, nb_clf - 4, nb_ind=len(ind))
        # U_f1_raw[k] = U_tst_raw[picked_clf]
        loc_a, loc_b = nb_col - 13, nb_col - 8
        U_f1_raw[k][:-4] = U_tst_raw[picked_clf][:-3]  # nb_col-3
        U_f1_raw[k][-4] = (U_tst_raw[picked_clf][loc_a] + U_tst_raw[picked_clf][loc_b]) / 2
        U_f1_raw[k][-3:] = U_tst_raw[picked_clf][-3:]
        k += 1

        for i in range(1, nb_set):
            curr_set = id_set[i] + 1
            curr_loc = list(range(curr_set, curr_set + each_gen + 2 * each_att))
            df_trn = dframe[tag_trn_a_f].iloc[curr_loc]
            df_tst = dframe[tag_tst_a_f].iloc[curr_loc]
            U_trn_raw, U_tst_raw = self.tabulate_five_sub1_dat(
                df_trn, df_trn, nb_clf, nb_ind=len(ind))
            # U_f1_raw[k] = U_tst_raw[picked_clf]
            U_f1_raw[k][:-4] = U_tst_raw[picked_clf][:-3]  # nb_col-3
            U_f1_raw[k][-4] = (U_tst_raw[picked_clf][loc_a] +
                               U_tst_raw[picked_clf][loc_b]) / 2.
            U_f1_raw[k][-3:] = U_tst_raw[picked_clf][-3:]
            k += 1

        # U_cp_raw = self.tabulate_six_sub3_plo(U_f1_raw, ind, suff + '_confusion')
        suff = suffix + '_tst_both_confusion'  # sing/pl # suffix + '_confusion'
        U_cp_raw = self.tabulate_six_sub3_plo(U_f1_raw, ind, suff)
        if 0 in ind:
            self.tabulate_six_sub4_plo(U_cp_raw, nb_ind=len(ind), loc=0,
                                       suff =suffix + '_tst_tradeoff_ind0',
                                       alpha=alpha, num_gap=100, ddof=1)
            self.tabulate_six_sub4_plo(U_cp_raw, nb_ind=len(ind), loc=1,
                                       suff =suffix + '_tst_tradeoff_ind3',
                                       alpha=alpha, num_gap=100, ddof=1)
        ans_tex, cmp_tex = self.tabulate_five_sub2_out(U_f1_raw, ddof, rez=4)
        tmp_a = [self._pick_metric[k] for k in ind] + [
            r'$\Delta$' + self._pick_metric[k] for k in ind]
        tmp_b = ['DR', 'df_prev bin-val', ' (multival)', '', ''] * 2 + [
            '[DR.mean]', 'df_prev', 'df', 'df_avg']
        tmp_f = 'picked_clf = {} {}'.format(i, self._abbr_ensf[picked_clf])
        csv_w.writerow([tmp_f] + [''] * nb_col + ['[END]'])  # (nb_col-1)
        csv_w.writerow([''] + tmp_a + tmp_b)
        csv_w.writerows(ans_tex)
        csv_w.writerow([''])
        csv_w.writerows(cmp_tex)
        csv_w.writerow([''])
        del tmp_a, tmp_b, ans_tex, cmp_tex, suff
        return

    def tabulate_six_sub3_plo(self, U_f1_raw, ind, figname='_confusion',
                              cmap_name='BuGn', rotate=15):  # 0): # PuBu
        # U_f1_raw .shape= (5,17|23,5) =(nb_set,nb_col,nb_iter)
        # 23 columns: 5|2*2 (normal+delta metric) +5*2 (DR, df_prev bin-val,
        #             df_prev multival, df, df_avg) +3 (df_prev, df, df_avg)
        nb_set, nb_col, k = U_f1_raw.shape
        U_cp_raw = np.zeros((nb_set, nb_col - 2 * 2 - 1, k))  # 17-5=12
        nb_ind = len(ind)  # [0,..,3, 4,.,8,  9,.,13, 14,15,16]
        # updated: [0,..,3, 4,.,7,8, 9,.,12,13, 14,.,16,17] --> [.., 4,.,7,.,10,11,12]
        for j in range(nb_ind * 2 + 1):                       # array([0,1,2,3]) #0,1,2,3
            U_cp_raw[:, j, :] = U_f1_raw[:, j, :]      # < 0,1,2,3, 4 >> ?
        for j in range(nb_ind * 2 + 3, nb_ind * 2 + 5 + 1):   # array([6,7,8])   #4,5,6,
            U_cp_raw[:, j - 2, :] = U_f1_raw[:, j, :]  # < 7,8, 9     >> 5,6,7  =?-2
        for j in range(nb_ind * 2 + 8, nb_ind * 2 + 10 + 1):  # array([11,12,13])#7,8,9,
            U_cp_raw[:, j - 4, :] = U_f1_raw[:, j, :]  # < 12,13, 14  >> 8,9,10 =?-4
        for j in range(nb_ind * 2 + 12, nb_ind * 2 + 14):
            U_cp_raw[:, j - 5, :] = U_f1_raw[:, j, :]  # <     16,17  >> 11,12  =?-5
        # for j in range(nb_ind * 2 + 11, nb_ind * 2 + 13):     # array([15,16])   #10,11 #14,
        #   U_cp_raw[:, j - 5, :] = U_f1_raw[:, j, :]  # < 15,16 >> 11,12 
        nb_ind = nb_ind * 2  # 4,5,6, 7,8,9, 10,11 <-- 4,7,8, 9,12,13, 15,16
        # updated:     4,5,6, 7,8,9, 10, 11,12 <-- 4,7,8, 9,12,13, 14, 16,17
        assert np.all(U_cp_raw[:, :nb_ind, :] == U_f1_raw[:, :nb_ind, :])
        assert np.all(U_cp_raw[:, nb_ind, :] == U_f1_raw[:, nb_ind, :])
        assert np.all(U_cp_raw[:, nb_ind + 1, :] == U_f1_raw[:, nb_ind + 3, :])
        assert np.all(U_cp_raw[:, nb_ind + 2, :] == U_f1_raw[:, nb_ind + 4, :])
        assert np.all(U_cp_raw[:, nb_ind + 3, :] == U_f1_raw[:, nb_ind + 5, :])
        assert np.all(U_cp_raw[:, nb_ind + 4, :] == U_f1_raw[:, nb_ind + 8, :])
        assert np.all(U_cp_raw[:, nb_ind + 5, :] == U_f1_raw[:, nb_ind + 9, :])
        # assert np.all(U_cp_raw[:, nb_ind + 6, :] == U_f1_raw[:, nb_ind + 11, :])
        # assert np.all(U_cp_raw[:, nb_ind + 7, :] == U_f1_raw[:, nb_ind + 12, :])
        assert np.all(U_cp_raw[:, nb_ind + 6, :] == U_f1_raw[:, nb_ind + 10, :])
        assert np.all(U_cp_raw[:, nb_ind + 7, :] == U_f1_raw[:, nb_ind + 12, :])
        assert np.all(U_cp_raw[:, nb_ind + 8, :] == U_f1_raw[:, nb_ind + 13, :])
        loc_a, loc_b = nb_ind + 0, nb_ind + 3
        assert np.all(U_cp_raw[:, -3, :] * 2 == U_cp_raw[:, loc_a, :] + U_cp_raw[:, loc_b, :])
        # U_cp_raw .shape= (nb_set=5, 13|19, 5=nb_iter)  # 4+3*2+3
        U_cp_raw[0][-3] = U_cp_raw[0][-9]  # .shape= (5,12|18+1,5)  # 4,7 < 13-3,13-9

        tmp_a = [self._pick_metric[k] for k in ind] + [
            r'$\Delta$' + self._pick_metric[k] for k in ind]
        tmp_b = ['DR', r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']
        tmp_c = ['sa#1\n' + t for t in tmp_b] + ['sa#2\n' + t for t in tmp_b] + [
            '' + t for t in [r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']]
        tmp_d = [r'DR$_1$', r'$\mathbf{df}_1$', r'$\mathbf{df}_1^{avg}$',
                 r'DR$_2$', r'$\mathbf{df}_2$', r'$\mathbf{df}_2^{avg}$',
                 r'  DR$_{avg}$', r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']
        # U_cp_raw .shape= (5,12|18,5) =(nb_set,nb_col-5,nb_iter)
        nb_col = nb_col - 2 * 2 - 1

        U_cp_tmp = U_cp_raw.transpose((1, 0, 2))  # .shape= (13|18,5=nb_set,5)
        Mat_A, Mat_B = U_cp_tmp[:nb_ind], U_cp_tmp[nb_ind:]  # (4|9=3*3, 5,5)
        cm = np.zeros((nb_ind, nb_col - nb_ind))
        for i in range(nb_ind):
            for j in range(nb_col - nb_ind):
                if j not in [3, 4, 5]:
                    ct = np.corrcoef(Mat_B[j].reshape(-1),
                                     Mat_A[i].reshape(-1))[1, 0]
                else:
                    ct = np.corrcoef(Mat_B[j][1:].reshape(-1),
                                     Mat_A[i][1:].reshape(-1))[1, 0]
                # if i >= 3 and j >= 7:
                #   pdb.set_trace()
                cm[i, j] = ct
        # pdb.set_trace()
        analogous_confusion_extended(
            Mat_A.reshape(nb_ind, -1), Mat_B.reshape(nb_col - nb_ind, -1),
            tmp_a, tmp_d, figname, cm, cmap_name=cmap_name, rotate=rotate)
        return U_cp_raw

    def tabulate_six_sub4_plo(self, U_cp_raw, nb_ind, loc=0,
                              suff='suff_ind',
                              alpha=.5, num_gap=100, ddof=1):
        # U_cp_raw .shape= (5,12|18+1,5) =(nb_set,nb_col-5,nb_iter)
        tmp_d = [r'DR$_{avg}$', r'$\mathbf{df}$', r'$\mathbf{df}^{avg}$']
        annotY = ' error rate' if loc == 0 else r'$($1$-$ performance$)$'  # ind==0
        kw_alt = {'alpha_loc': 'b4', 'alpha_clarity': .15, 'cmap_name': 'GnBu_r'}
        X = np.linspace(0, 1, num_gap)
        nb_set, nb_col, k = U_cp_raw.shape  # nb_col= 4+3*2+3

        # U_cp_raw .shape= (5,12|18+1,5) =(nb_set,nb_col-5,nb_iter)
        # U_cp_tmp .shape= (12|18+1, 25) =(nb_col', nb_set*nb_iter)
        U_cp_tmp = U_cp_raw.transpose((1, 0, 2)).reshape(nb_col, -1)
        U_cp_tmp_norm = U_cp_tmp[: nb_ind]  # .shape= (2,25)
        U_cp_tmp_fair = U_cp_tmp[-3:]       # .shape= (3,25)
        baseline_Ys = np.zeros((3, num_gap, k * nb_set))  # (3,100,25)
        for j in range(3):
            for kv_a, alpha in enumerate(X):
                baseline_Ys[j][kv_a] = (
                    1 - U_cp_tmp_norm[loc] / 100
                ) * alpha + (1. - alpha) * U_cp_tmp_fair[j]
        fnm = '{}_both_set_gather'.format(suff)
        _uncertainty_plotting(X, baseline_Ys, tmp_d, annotY, ddof,
                              figname=fnm, **kw_alt)
        return

    def tabulating_third(self, dframe, tag_trn, tag_tst, nb_set, id_set,
                         each_gen, each_att, ind=[0, 3], ddof=0,
                         alpha=.7, dist_df='both', csv_w=None):
        tmp_a_org, _, tmp_f_vm, tmp_ext = self.picking_tab_tags(
            tag_trn, ind, dist_df)
        tag_trn_a_f = (
            tmp_a_org + tmp_f_vm[0] + tmp_ext[0],  # tmp_ext[1],
            tmp_a_org + tmp_f_vm[1] + tmp_ext[1])  # tmp_ext[2])
        tmp_a_org, _, tmp_f_vm, tmp_ext = self.picking_tab_tags(
            tag_tst, ind, dist_df)
        tag_tst_a_f = (
            tmp_a_org + tmp_f_vm[0] + tmp_ext[0],  # tmp_ext[1],
            tmp_a_org + tmp_f_vm[1] + tmp_ext[1])  # tmp_ext[2])
        nb_col = 7 if dist_df == 'both' else 5  # 6 or 5 where 6=2+4
        nb_att = (nb_set - 1) * 2 + 1
        # U_f1_raw = np.zeros((nb_att, 1 + nb_col, self._nb_iter))
        U_cp_raw = np.zeros((nb_att, 1 + nb_col, self._nb_iter))

        i, k = 0, 0
        df_trn = dframe[tag_trn_a_f[0]].iloc[id_set[i] + 1: id_set[i + 1]]
        df_tst = dframe[tag_tst_a_f[0]].iloc[id_set[i] + 1: id_set[i + 1]]
        pdb.set_trace()
        return

    def tabulating_third_sub1(self, df_trn, df_tst, each_gen, each_att,
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
        # # U_trn/tst_raw .shape= (14+4, 8, 5)

        k = nb_col - (7 if dist_df == 'both' else 5)
        U_trn_tmp = np.zeros((nb_clf, nb_col - k, self._nb_iter))
        U_tst_tmp = np.zeros((nb_clf, nb_col - k, self._nb_iter))
        for j in range(k, nb_col):
            U_trn_tmp[:, j - k, :] = (1 - U_trn_raw[
                :, 0, :]) * alpha + (1 - alpha) * U_trn_raw[:, j, :]
            U_tst_tmp[:, j - k, :] = (1 - U_tst_raw[
                :, 0, :]) * alpha + (1 - alpha) * U_tst_raw[:, j, :]
        # # U_trn/tst_tmp .shape= (14+4, 6, 5)
        # the smaller the better: alpha* error rate +(1-alpha)* fairness

        return U_trn_raw, U_tst_raw, U_trn_tmp, U_tst_tmp


# -------------------------------
# RQ1.
#   compared with sota fairness (cont.)
#


class CurrTab4B_comparison(Table4_comparison):
    def __init__(self, nb_iter, nb_cls, m1, m2, n_e, figname='exp4b_'):
        super().__init__(nb_iter, nb_cls, m1, m2, n_e, figname)  # tab4b_

    def schedule_spreadsheet(self, raw_dframe, pre='minmax', logger=None):
        (nb_set, id_set, each_set,
         each_att, each_gen) = self.recap_sub_data(
            raw_dframe, nb_row=4, nc_norm=3, nc_sens=3 + 1)
        tag_pm, tag_trn, tag_tst = self.prepare_graph()

        suff = self._figname.replace('exp4b_', 'tab4b_') + pre
        # log_document = suff + '_table_third.csv'
        log_document = suff + '_table_fifth.csv'
        csv_t = open(log_document, 'w')
        csv_w = csv.writer(csv_t)

        i = 2  # for i in range(3 + 4):
        self.tabulating_fifth_alt(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set, each_gen,
            each_att, ind=[0, 3], ddof=1, picked_clf=i, csv_w=csv_w,
            suffix=suff)
        self.tabulating_fifth_alt(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set, each_gen,
            each_att, ind=[1, 2, 5], ddof=1, picked_clf=i, csv_w=csv_w,
            suffix=suff + '_alt')
        # for i in [2, 0, 1, 3, 4, 5, 6]:
        self.tabulating_sixth_alt(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set, each_gen,
            each_att, ind=[0, 3], ddof=1, picked_clf=i, csv_w=csv_w,
            suffix=suff)
        self.tabulating_sixth_alt(
            raw_dframe, tag_trn, tag_tst, nb_set, id_set, each_gen,
            each_att, ind=[1, 2, 5], ddof=1, picked_clf=i, csv_w=csv_w,
            suffix=suff + '_alt')

        csv_t.close()
        del csv_t, csv_w, suff, log_document
        return


# -------------------------------
#


# ===============================
#


# -------------------------------
#

# -------------------------------
#
