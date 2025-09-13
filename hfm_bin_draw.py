# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#

import argparse
import time
import sys

from hfm.utils.recorders import elegant_print
from hfm.utils.decorators import elegant_durat, elegant_dated

from experiment.df_bin.manf_plt import (
    Plot5A_hyperpm, Plot5B_hyperpm,
    Plot2A_comparison, Plot2B_comparison,  # Plot2C_comparison,
    Replot2A_comparison, Replot2B_comparison, Replot2C_comparison)
from experiment.df_bin.manf_tab import Table2C_comparison

from experiment.df_bin.rev_manf_anal import (
    Ver2_PlotA_fair_ens, Ver2_PlotA_norm_cls)


# ===============================
# Empirical results


class ManfDrawing(object):
    def __init__(self, trial_type, nb_iter=5, m1=30, m2=10,
                 gen=False, rep=False, prep=False,
                 abbr_cls='DT', nb_cls=1, constraint_type='FPR,FNR',
                 prefix='', ratio=.75, screen=True, logged=False):
        # super().__init__(data_type)
        self._data_set = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        self._prefix = prefix
        self._ratio = ratio
        self.preparing_iterator(
            trial_type, nb_iter, m1, m2, gen, rep, prep,
            abbr_cls, nb_cls, constraint_type, screen, logged)

    def preparing_iterator(self,
                           trial_type, nb_iter, m1, m2, gen, rep,
                           prep, abbr_cls, nb_cls, constraint_type,
                           screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter
        self._gen_iter = gen
        self._rep_iter = rep  # cross-validation split
        self._prep = prep     # pre-processing data
        self._m1, self._m2 = m1, m2
        self._screen, self._logged = screen, logged

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        # self.saIndex = [-1] if self._data_type=='ricci' else [-2,-1]

        self._log_document = "_".join([
            trial_type,
            "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
            str(self._prep), 'pms'])

        return

    def trial_one_process(self):
        since = time.time()
        logger = None
        elegant_print(
            "[BEGAN {}]".format(elegant_dated(since)), logger)
        # START

        if self._trial_type[-6:] in ['expt5a', 'expt5b']:
            if self._trial_type.endswith('expt5a'):
                self._iterator = Plot5A_hyperpm(
                    self._nb_iter, None, None, self._m1, self._m2)
            elif self._trial_type.endswith('expt5b'):
                self._iterator = Plot5B_hyperpm(
                    self._nb_iter, None, None, self._m1, self._m2)

            self.drawing_expt5(self._prefix)

        elif self._trial_type[-6:] in ['expt2a', 'expt2b', 'expt2c']:
            figname = 'exp{}_iter{}_cls{}_{}'.format(
                self._trial_type[-2:], self._nb_iter, self._nb_cls,
                self._prep)

            if self._trial_type.endswith('expt2a'):
                self._iterator = Plot2A_comparison(
                    self._nb_iter, self._nb_cls, self._m1, self._m2,
                    figname)
            elif self._trial_type.endswith('expt2b'):
                self._iterator = Plot2B_comparison(
                    self._nb_iter, self._nb_cls, self._m1, self._m2,
                    figname)
            elif self._trial_type.endswith('expt2c'):
                self._iterator = Table2C_comparison(
                    self._nb_iter, self._nb_cls, self._m1, self._m2,
                    figname)

            self.drawing_expt2(self._ratio, self._prefix)

        # END
        tim_elapsed = time.time() - since
        elegant_print(["Duration /TimeCost: {}".format(
            elegant_durat(tim_elapsed)),
            "[ENDED {}]".format(
                elegant_dated(time.time()))], logger)
        return

    def drawing_expt5(self, prefix=''):
        xlsx_name = 'rept_expt5_{}_pms'.format(self._prep)
        if prefix != '':
            xlsx_name = prefix + ') ' + xlsx_name

        sheet_name = 'exp{}_{}_'.format(
            self._trial_type[-2:], 'iter{}'.format(
                self._nb_iter) if self._nb_iter > 0 else 'sing')
        if self._trial_type.endswith('expt5a'):
            sheet_name += 'ma{}'.format(self._m1)
        elif self._trial_type.endswith('expt5b'):
            sheet_name += 'mb{}'.format(self._m2)

        raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_dframe=raw_df)

    def drawing_expt2(self, ratio=.75, prefix=''):
        xlsx_name = '{}_iter{}_cls{}_pms_ratio{}_rep'.format(
            self._trial_type[: -1],
            self._nb_iter, self._nb_cls, int(ratio * 100))
        if prefix != '':
            xlsx_name = '{}) '.format(prefix) + xlsx_name
        sheet_name = 'exp{}_{}'.format(self._trial_type[-2:],
                                       self._prep)
        raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_dframe=raw_df)

        if self._trial_type.endswith('expt2c'):
            self._iterator.schedule_spreadsheet(raw_dframe=raw_df)


class Replot_ManfDrawing(ManfDrawing):
    def trial_one_process(self):
        since = time.time()
        logger = None
        elegant_print(
            "[BEGAN {}]".format(elegant_dated(since)), logger)

        if self._trial_type[-6:] in ['expt2a', 'expt2b', 'expt2c']:
            figname = 'exp{}_iter{}_cls{}_'.format(
                self._trial_type[-2:], self._nb_iter, self._nb_cls)
            figname = 'exp{}_'.format(self._trial_type[-2:])
            pre = self._prep if self._prep != 'min_max' else 'minmax'
            self.drawing_expt2(figname, pre, self._ratio, self._prefix)

        tim_elapsed = time.time() - since
        elegant_print(["Duration /TimeCost: {}".format(
            elegant_durat(tim_elapsed)),
            "[ENDED {}]".format(elegant_dated(time.time()))], logger)
        return

    def drawing_expt2(self, figname, pre, ratio=.75, prefix=''):
        if self._trial_type.endswith('expt2a'):
            self._iterator = Replot2A_comparison(
                self._nb_iter, self._nb_cls, self._m1, self._m2,
                figname)
        elif self._trial_type.endswith('expt2b'):
            self._iterator = Replot2B_comparison(
                self._nb_iter, self._nb_cls, self._m1, self._m2,
                figname)
        elif self._trial_type.endswith('expt2c'):
            self._iterator = Replot2C_comparison(
                self._nb_iter, self._nb_cls, self._m1, self._m2,
                figname)

        xlsx_name = '{}_iter{}_cls{}_pms_ratio{}_rep'.format(
            self._trial_type[:-1], self._nb_iter, self._nb_cls,
            int(ratio * 100))
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)
        sheet_name = 'exp{}_{}'.format(self._trial_type[-2:],
                                       self._prep)
        raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_dframe=raw_df, pre=pre)
        return


# -------------------------------
# Revision


class FairManfRevision:
    def __init__(self, trial_type, prep=False, nb_iter=5,
                 ratio=.97, m1=20, m2=8, fix_m2=False,
                 nb_cls=7, screen=True, logged=False):
        self._screen, self._logged = screen, logged
        self._dataset = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        self._ratio = ratio

        self._trial_type = trial_type
        self._prep, self._nb_cv = prep, nb_iter
        self._m1, self._m2 = m1, m2
        self._fix_m2 = fix_m2

        self._log_document = "_".join([
            trial_type, prep.replace('_', ''),
            "cv{}".format(nb_iter) if nb_iter > 0 else 'sing',
            'pms'])
        self._nb_cls = nb_cls

    def trial_one_process(self):
        since = time.time()
        elegant_print("[BEGAN {}]".format(elegant_dated(since)))
        # START

        xlsx_name = '{}_iter{}_pms'.format(
            self._trial_type[:-1], self._nb_cv)
        xlsx_name = f'{xlsx_name}_fair_int'
        sheet_name = 'exp{}_{}'.format(
            self._trial_type[-2:], self._prep.replace('_', ''))

        if self._trial_type.endswith('exp1b'):
            self._iterator = Ver2_PlotA_fair_ens()
        elif self._trial_type.endswith('exp1c'):
            self._iterator = Ver2_PlotA_norm_cls()

        raw_df = self._iterator.load_raw_dataset(
            xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_df, sheet_name)
        del xlsx_name, sheet_name, raw_df, self._iterator

        # END
        tim_elapsed = time.time() - since
        elegant_print(["Duration /TimeCost: {}".format(
            elegant_durat(tim_elapsed, False)),
            "[ENDED] {}".format(elegant_dated(time.time()))])
        return


# -------------------------------
#


# ===============================
# Plotting


# -------------------------------
# hyper-parameters


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp", "--expt-id", type=str, default="rept_expt5",
        help="Type of trial: experiment id")
    parser.add_argument(
        "-pre", "--data-preprocessing", type=str,
        default="min_max", choices=[
            'none', 'standard', 'min_max', 'normalize'])
    parser.add_argument('-re', '--replot', action='store_true')
    parser.add_argument('-v', '--round', type=str, default='ver1')

    parser.add_argument('-nk', "--nb-iter", type=int, default=0,
                        help="Cross validation")
    parser.add_argument(
        '--nb-cls', type=int, default=7, help='For ensemble methods')
    parser.add_argument(
        '--nb-pru', type=int, default=3, help='For pruning methods')
    parser.add_argument(
        '--gen-iter', type=bool, default=False, help="param: gen")
    parser.add_argument(
        '--rep-iter', type=bool, default=False, help="param: cvs")
    parser.add_argument(
        '-m1', '--m1-chosen', type=int, default=20, help="m1")
    parser.add_argument(
        '-m2', '--m2-chosen', type=int, default=8, help="m2")

    parser.add_argument(
        "--screen", action="store_false", help="Where to output")
    parser.add_argument(
        "--logged", action="store_true", help="Where to output")
    parser.add_argument(
        '--prefix', default='', help='prefix and suffix')
    parser.add_argument(
        '--ratio', type=float, default=.95,
        help='Previously for ICML / NeurIPS 23')
    return parser


# -------------------------------
# hyper-parameters


screen = logged = None
parser = default_parameters()
args = parser.parse_args()

trial_type = args.expt_id
screen = args.screen
logged = args.logged


kwargs = {}
kwargs['nb_iter'] = args.nb_iter
kwargs['prep'] = args.data_preprocessing


if args.round == 'ver2':
    kwargs['nb_iter'] = 5  # '-nk 5'
    case = FairManfRevision(trial_type, **kwargs)
    case.trial_one_process()
    sys.exit()


if trial_type[-6:] in ['expt5a', 'expt5b']:

    if trial_type[-6:] in ('expt5a',):
        kwargs['m1'] = args.m1_chosen
    elif trial_type[-6:] in ('expt5b',):
        kwargs['m2'] = args.m2_chosen

    kwargs['gen'] = args.gen_iter
    # kwargs['prefix'] = 'manfRW'
    case = ManfDrawing(trial_type, prefix=args.prefix, **kwargs)
    case.trial_one_process()

if trial_type[-6:] in ['expt2a', 'expt2b', 'expt2c']:
    kwargs['nb_cls'] = args.nb_cls
    kwargs['rep'] = True
    kwargs['ratio'] = .75
    # kwargs['prefix'] = 'manfRW_TDbug'
    kwargs['nb_iter'] = 5

    if not args.replot:
        case = ManfDrawing(trial_type, **kwargs)
    else:
        case = Replot_ManfDrawing(trial_type, **kwargs)
    case.trial_one_process()


# -------------------------------
# Empirical plotting
"""
python hfm_bin_draw.py -nk 5 -exp rept_expt5a -m1 20
python hfm_bin_draw.py -nk 5 -exp rept_expt5b -m2 8
python hfm_bin_draw.py -exp mCV_expt2c -pre min_max
python hfm_bin_draw.py -exp mCV_expt2a -pre min_max -re

# python hfm_bin_draw.py -exp mCV_expt2b -pre min_max
python hfm_bin_draw.py -v ver2 -exp mCV_exp1b -pre min_max
python hfm_bin_draw.py -v ver2 -exp mCV_exp1c -pre min_max
"""
