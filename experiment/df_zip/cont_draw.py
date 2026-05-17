# coding: utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpecFromSubplotSpec
# from mpl_toolkits.axes_grid1 import ImageGrid
from hfm.hfm_df import differentiate_tim, differentiate_val

from pyfair.facil.utils_const import DTY_FLT, subfig_ind
from pyfair.facil.draw_prelim import _style_set_axis, _setup_figshow
from pyfair.granite.draw_graph import _sns_line_err_bars
from pyfair.granite.draw_addtl import (
    _subproc_pl_lin_reg, _subproc_pl_lin_reg_alt, _subproc_pl_identity,
    _add_on_DR_plot)


# plt.rcParams["text.usetex"] = True
_pl_myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_navy = 'navy'


def _hp_subproc_linreg(ax, X, Ys, picked_keys, annotZ, snspec, myclr,
                       corr=False, curr_legend_nb_split=4, identity=True,
                       differentiate=False):
    _curr_mk = ['D', '^', 'v', 'o', '*', 's', 'd', '<', '>']
    _curr_sz = [10, 15, 15, 12, 29, 11, 15, 15, 15]
    if isinstance(picked_keys, str):
        _subproc_pl_lin_reg(ax, X, Ys, picked_keys, annotZ, snspec, _navy,
                            False, corr, 's', 11)
        _subproc_pl_lin_reg_alt(ax, X, Ys, snspec, _navy)
    else:
        for i, key in enumerate(picked_keys):
            _subproc_pl_lin_reg(ax, X, Ys[i], key, annotZ, snspec, myclr[i],
                                True, corr, _curr_sz[i], _curr_mk[i])
        for i, key in enumerate(picked_keys):
            _subproc_pl_lin_reg_alt(ax, X, Ys[i], snspec, myclr[i])
    if identity:
        _subproc_pl_identity(ax, [X, X], annotZ, snspec)

    if snspec in ['sty5a', 'sty5b', 'sty6']:
        _curr_nc = 1 if len(picked_keys) <= curr_legend_nb_split else 2
        _curr_lo = 'upper right' if snspec == 'sty6' else 'upper left'
        _curr_fram = dict(frameon=True, loc=_curr_lo, framealpha=.5,
                          handleheight=1.2, handletextpad=0,
                          ncol=_curr_nc, columnspacing=.8)
        del _curr_nc, _curr_lo
    elif snspec in ['sty3a', 'sty3b', 'sty8a', 'sty8b']:
        _curr_fram = dict(frameon=False, loc='upper left')
    elif snspec in ['sty4', ]:  # 'sty6', 'sty4']:
        _curr_fram = dict(frameon=True, framealpha=.5, loc='best')
        #                   # loc='upper right' if snspec == 'sty6' else 'best')
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = dict(family=_curr_ft, size=8)
    ax.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    del _curr_fram, _curr_ft, _curr_sz, _curr_mk

    ax = _style_set_axis(ax, invt=False, dirc='in')
    # ax.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    # ax.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    return


def hyper_pms_lin_reg_gather(X, Ys, picked_keys, annots=('X', 'Y', 'Y=X'),
                             figname='smd', sharey=True, strt=0, snspec='sty',
                             identity=True, differentiate=False):
    num_c = len(X)  # =len(Ys)=len(picked_keys)  # figsize=(14, 2.8)
    fig = plt.figure(figsize=(9.1, 2.4) if num_c > 2 else (4.34, 2.4),
                     dpi=300, constrained_layout=True)
    outer = fig.add_gridspec(1, num_c, wspace=.07, hspace=.35)
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    corr = not ('tim' in figname)
    ref_ax = None
    for ik, ax_spec in enumerate(outer):
        if sharey:
            inner = fig.add_subplot(ax_spec, sharey=ref_ax)  # if sharey else None)
            if ref_ax is None:
                ref_ax = inner
        else:
            inner = fig.add_subplot(ax_spec)
        curr_X, curr_Ys = X[ik], Ys[ik]
        if differentiate and corr:
            curr_Ys = [differentiate_val(curr_X, curY) for curY in curr_Ys]
        elif differentiate:
            curr_Ys = [differentiate_tim(curr_X, curY) for curY in curr_Ys]
        _hp_subproc_linreg(inner, curr_X, curr_Ys, picked_keys[ik], annotZ, snspec,
                           _pl_myclr, corr, curr_legend_nb_split=(6 if corr else 4),
                           identity=identity, differentiate=differentiate)
        # 只让最左边显示 y 轴
        if ik > 0:
            inner.tick_params(labelleft=False)   # 隐藏 y tick label
            inner.set_ylabel("")                 # 隐藏 y label
        else:
            inner.set_ylabel(annots[1], fontsize=9)  # ,y=.45)
            tmp_y = inner.get_ylim()
            if tmp_y[1] // np.max(Ys) > 4:
                inner.set_ylim((min(tmp_y[0] + 10, 0.), tmp_y[1] / 4))
            del tmp_y
        inner.set_xlabel(annots[0] + f'\n{subfig_ind(ik+strt)}', fontsize=9)  # ,x=.45)
    _setup_figshow(fig, figname)
    return


def _lr_subproc_myclr(snspec, Ys):
    myclr = _pl_myclr
    if snspec.startswith('sty7'):
        myclr = [_navy, ] + myclr[:6]
        del myclr[2:5]
        snspec = snspec.replace('y7', 'y6')
    elif snspec in ('sty3c', 'sty6c', 'sty4c',):
        myclr = _pl_myclr[1:]
    elif snspec in ('sty3d', 'sty6d', 'sty4d',):
        myclr = _pl_myclr[2:]
    elif snspec in ('sty3e', 'sty4e', 'sty6e',):
        myclr = _pl_myclr[1:3] + _pl_myclr[5:]
        if len(Ys) == 1:
            myclr = _pl_myclr[:1] + myclr
    elif snspec == 'sty4' and len(Ys) == 2:
        myclr = _pl_myclr[:1] + myclr

    if snspec in ['sty3a', 'sty3b', 'sty3c', 'sty3d', 'sty3e']:
        _curr_fram = {'frameon': snspec in ('sty3b', 'sty3e', 'sty3c'),
                      'loc': 'upper left' if snspec != 'sty3e' else 'best',
                      'fontsize': 6, 'framealpha': .5}
    elif snspec == 'sty3bf':
        _curr_fram = {'frameon': 'lower right', 'fontsize': 6, 'framealpha': .5}
    elif snspec in ['sty6', 'sty6c', 'sty6d', 'sty6e', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5, 'loc': 'best'}  # 'upper right'
    elif snspec in ['sty4', 'sty8a', 'sty8b', 'sty4d', 'sty4c', 'sty4e', ]:
        _curr_fram = {'loc': 'upper left' if snspec == 'sty4e' else 'best',
                      'frameon': not snspec.startswith('sty8'), 'framealpha': .5}
    return myclr, _curr_fram


def multi_lin_reg_wo_distr_gather(X, Ys, Zs, annots=(('X', 'Y', 'Y=X'),),
                                  figname='smd', sharey=True, strt=0, snspec='sty'):
    num_c = len(X)  # =len(Ys)
    fig = plt.figure(figsize=(9.1, 2.4), dpi=300, constrained_layout=True)
    outer = fig.add_gridspec(1, num_c, wspace=0.07, hspace=.35)
    # annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}
    ref_ax = None

    for ik, ax_spec in enumerate(outer):
        myclr, _curr_fram = _lr_subproc_myclr(snspec[ik], Ys[ik])
        if sharey and ik >= num_c - 2:
            inner = fig.add_subplot(ax_spec, sharey=ref_ax)
            if ref_ax is None:
                ref_ax = inner
        else:
            inner = fig.add_subplot(ax_spec)
        annotZ = annots[ik][2] if len(annots[ik]) > 2 else r'$f(x)=x$'
        _add_on_DR_plot(inner, X[ik], Ys[ik], Zs[ik], annotZ, snspec[ik], myclr)
        inner.legend(prop=legend_font, labelspacing=.14, handletextpad=.21, **_curr_fram)
        _style_set_axis(inner)
        inner.set_xlabel(annots[ik][0] + f'\n{subfig_ind(strt+ik)}',
                         fontsize=9, family=_curr_ft, x=.55)
        inner.set_ylabel(annots[ik][1], fontsize=9, family=_curr_ft, y=.55)
        if ik > num_c - 2:
            inner.tick_params(labelleft=False)
            inner.set_ylabel("")
    _setup_figshow(fig, figname=figname)
    return
