# coding: utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpecFromSubplotSpec
# from mpl_toolkits.axes_grid1 import ImageGrid

from pyfair.facil.utils_const import DTY_FLT, subfig_ind
from pyfair.facil.draw_prelim import _style_set_axis, _setup_figshow
from pyfair.granite.draw_graph import _sns_line_err_bars


_pl_myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_navy = 'navy'


def hyper_pms_lin_reg_gather():
    fig = plt.figure(figsize=(14, 7.8), dpi=300, constrained_layout=True)
    return
