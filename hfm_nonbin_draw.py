# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds, extension
#


import argparse
import time
import sys
import pdb

from hfm.utils.decorators import elegant_dated, fantasy_durat
from hfm.utils.recorders import elegant_print

from experiment.df_nonbin.mext_plt import (
    CurrPlot3B_comparison, CurrPlot3C_comparison, CurrPlot3D_comparison,
    CurrPlot4B_comparison, CurrPlot4C_comparison, CurrPlot4D_comparison,
    Distributed_GA_mp, HyperEA_renew_m1fix, HyperEB_renew_m2fix,
    CurrTab4B_comparison)

from experiment.df_nonbin.rev_mext_plt import (
    RevP_ZA_efficient, RevP_ZB_efficient, RevP_ZC_efficient,
    RevP_YA_embedding, RevP_YB_embedding, RevP_YC_embedding,
    RevP_XC_statsParity, RevP_XD_statsParity,
    RevP_XE_statsParity, RevP_XF_statsParity)

from experiment.df_nonbin.rev_mext_plt import (
    ConvFig_5C_exact, ConvFig_5D_exact, ConvFig_5E_exact,
    ConvFig_5B_exact,  # ConvFig_4B_exact,
    # ConvFig_4C_exact, ConvFig_4D_exact, ConvFig_4E_exact)
    ConvFig_4E_exact, ConvFig_5F_exact,  # ConvFig_5H_exact,
)   # ConvFig_5I_exact, ConvFig_5Isimpl)
from experiment.df_nonbin.rev_mext_plt_cor import (
    ConvFig_5H_exact, ConvFig_5I_exact, ConvFig_5Isimpl)
from experiment.df_nonbin.rev_mext_plt_cor import HPEA_m1fix, HPEB_m2fix

from experiment.df_zip.mcvg_plt import (  # DistPerf_draw
    cvgPlt1C_take, cvgPlt1A_anal, cvgPlt1B_anal, cvgPlt1_anal_gather)


# ===============================
# Empirical results

# from prgm.fairmanf_ext.mext_sim import ManfExtEmpirical
# from prgm.fairmanf_ext.mext_sim import ManfExtPrime_Empirical


class ManfExtDrawing(object):
    def __init__(self, trial_type, nb_iter=5, m1=25, m2=11,
                 n_e=2, ratio=.76, prep=False, gen=False, rep=False,
                 nb_cls=7, abbr_cls='DT', constraint_type='FPR,FNR',
                 mp_cores=3, m2_fixed=False,
                 omitted=True, prefix='', screen=True, logged=False):
        # super().__init__(data_type)
        self._data_set = ['ricci', 'german', 'adult', 'ppr', 'ppvr',
                          'tmp-simulative']
        # self._prefix = prefixs
        self._prefix = prefix
        self._omitted = omitted
        # self._n_e = n_e
        self._ratio = ratio
        self._mp_cores = mp_cores  # number of machines in parallel comput.
        self._m2_fixed = m2_fixed
        self.preparing_iterator(
            trial_type, nb_iter, m1, m2, n_e, prep,
            gen, rep, nb_cls, abbr_cls, constraint_type,
            screen, logged)

    def preparing_iterator(self, trial_type, nb_iter, m1, m2, n_e, prep,
                           gen, rep, nb_cls, abbr_cls, constraint_type,
                           screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter
        self._gen_iter = gen
        self._rep_iter = rep  # cv split
        self._prep = prep     # pre-processing data
        self._m1, self._m2 = m1, m2
        self._n_e = n_e
        self._screen, self._logged = screen, logged

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type
        # self.saIndex = [-1] if self._data_type=='ricci' else [-2,-1]

        self._log_document = "_".join([
            trial_type,
            "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
            str(self._prep), 'pms'])

    # def preparing_iterator_core(self):
    #   pass

    def trial_one_process(self):
        since = time.time()
        logger = None
        elegant_print("[BEGAN {}]".format(elegant_dated(since)), logger)

        # START
        self.preparing_iterator_core(self._trial_type, self._prefix)
        # END

        tim_elapsed = time.time() - since
        elegant_print([
            "Duration /TimeCost: {}".format(fantasy_durat(tim_elapsed)),
            "[ENDED {}]".format(elegant_dated(time.time()))], logger)
        return

    def preparing_iterator_core(self, trial_type, prefix=''):

        pre = self._prep.replace('_', '')
        # figname = 'exp{}_{}_'.format(trial_type[-2:], pre)
        pms = {'nb_iter': self._nb_iter,  # 'nb_cls': self._nb_cls,
               'm1': self._m1, 'm2': self._m2, 'n_e': self._n_e}
        figname = 'exp{}_'.format(trial_type[-2:])

        if trial_type[-6:] in ['expt3b', 'expt3c', 'expt3d']:
            pms['nb_cls'] = self._nb_cls
            if trial_type.endswith('expt3b'):
                self._iterator = CurrPlot3B_comparison(figname=figname,
                                                       **pms)
            elif trial_type.endswith('expt3c'):
                self._iterator = CurrPlot3C_comparison(figname=figname,
                                                       **pms)
            elif trial_type.endswith('expt3d'):
                self._iterator = CurrPlot3D_comparison(figname=figname,
                                                       **pms)
            self.drawing_whole_exp3_exp4(pre, prefix)
        elif trial_type[-6:] in ['expt4b', 'expt4c', 'expt4d']:
            pms['nb_cls'] = self._nb_cls
            if trial_type.endswith('expt4b'):
                self._iterator = CurrPlot4B_comparison(figname=figname,
                                                       **pms)
                self._tabulater = CurrTab4B_comparison(figname=figname,
                                                       **pms)
            elif trial_type.endswith('expt4c'):
                self._iterator = CurrPlot4C_comparison(figname=figname,
                                                       **pms)
            elif trial_type.endswith('expt4d'):
                self._iterator = CurrPlot4D_comparison(figname=figname,
                                                       **pms)
            self.drawing_whole_exp3_exp4(pre, prefix)

        elif trial_type[-6:] in ['expt7a']:
            # del pms['nb_cls']
            pms['mp_cores'] = self._mp_cores
            self._iterator = Distributed_GA_mp(figname=figname, **pms)
            xlsx_name = '{}_iter{}_pms_ma{}_mb{}'.format(
                trial_type, self._nb_iter, self._m1, self._m2)
            if prefix != '':
                xlsx_name = '{}) {}'.format(prefix, xlsx_name)
            sheet_name = 'exp{}_{}'.format(trial_type[-2:], self._prep)
            raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
            self._iterator.schedule_mspaint(raw_df, self._mp_cores, pre)
            del xlsx_name, sheet_name, raw_df

        elif trial_type[-6:] in ['expt5a', 'expt5b']:
            # del pms['nb_cls']
            pms['mp_cores'] = self._mp_cores
            pms['omitted'] = self._omitted
            if trial_type.endswith('expt5a'):
                self._iterator = HyperEA_renew_m1fix(figname=figname,
                                                     **pms)
                xlsx_name = '{}_iter{}_pms_ma{}_alt'.format(
                    trial_type, self._nb_iter, self._m1)
            elif trial_type.endswith('expt5b'):
                self._iterator = HyperEB_renew_m2fix(figname=figname,
                                                     **pms)
                xlsx_name = '{}_iter{}_pms_mb{}_alt'.format(
                    trial_type, self._nb_iter, self._m2)
            self.drawing_whole_exp5(pre, xlsx_name, prefix)
            del xlsx_name
        return

    def drawing_whole_exp5(self, pre, xlsx_name, prefix=''):
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)
        sheet_name = 'exp{}_{}'.format(self._trial_type[-2:], self._prep)
        raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_df, pre=pre)

    def drawing_whole_exp3_exp4(self, pre, prefix=''):
        xlsx_name = '{}_iter{}_cls{}_pms_ratio{}_rep'.format(
            self._trial_type[: -1], self._nb_iter, self._nb_cls,
            int(self._ratio * 100))
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)

        sheet_name = 'exp{}_{}'.format(self._trial_type[-2:], self._prep)
        raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(raw_df, pre=pre)

        if self._trial_type[-6:] in ['expt4b']:
            self._tabulater.schedule_spreadsheet(raw_df, pre)
        return


# -------------------------------
# Revision


class Rev_ManfExtDrawing(object):
    def __init__(self, trial_type, nb_iter=5, m1=25, m2=11, n_e=2, n_p=3,
                 ratio=.97, prep='min_max', gen=False, rep=True,
                 nb_cls=7, mp_cores=3, m2_fixed=False, omitted=True,
                 prefix='', screen=True, logged=False):
        self._data_set = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        self._prefix = prefix
        self._omitted = omitted
        self._ratio = ratio
        self._mp_cores = mp_cores
        self._m2_fixed = m2_fixed
        self.preparing_iterator(trial_type, nb_iter, m1, m2, n_e, prep,
                                gen, rep, nb_cls, screen, logged)
        self._n_p = n_p

    def preparing_iterator(self, trial_type, nb_iter, m1, m2, n_e, prep,
                           gen, rep, nb_cls, screen=True, logged=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter
        self._gen_iter = gen
        self._rep_iter = rep  # cv split
        self._prep = prep     # pre-processing data
        self._m1, self._m2 = m1, m2
        self._n_e = n_e
        self._screen, self._logged = screen, logged

        # self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._log_document = "_".join([trial_type, 'it{}'.format(
            nb_iter) if nb_iter > 0 else 'sing',
            str(prep.replace('_', '')), 'pms'])

    def trial_one_process(self, logger=None):
        since = time.time()  # logger = None
        elegant_print("[BEGAN {}]".format(elegant_dated(since)), logger)
        # START
        self.preparing_iterator_core(self._trial_type, self._prefix)
        # END
        tim_elapsed = time.time() - since
        elegant_print([
            "Duration /TimeCost: {}".format(fantasy_durat(tim_elapsed)),
            "[ENDED {}]".format(elegant_dated(time.time()))], logger)
        return

    def preparing_iterator_core(self, trial_type, prefix=''):
        # pre = self._prep.replace('_', '')
        pms = {'m1': self._m1, 'm2': self._m2, 'n_e': self._n_e}
        figname = 'exp{}_'.format(trial_type[-2:])

        if trial_type[-6: -1] == 'rexp1':
            self.drawing_whole_rexp1(trial_type, prefix, figname, pms)
        elif trial_type[-6: -1] == 'rexp2':
            self.drawing_whole_rexp2(trial_type, prefix, figname, pms)
        elif trial_type[-6: -1] == 'rexp3':
            self.drawing_whole_rexp3(trial_type, prefix, figname, pms)

        elif trial_type[-6: -1] == 'rexp9':  # 'rexp5':  # prefix,
            self.drawing_whole_expr5(trial_type, figname, pms)
        elif trial_type[-6: -1] == 'rexp8':  # 'rexp4':  # prefix,
            self.drawing_whole_expr4(trial_type, figname, pms)
        elif trial_type[-6:] in ('exhp5a', 'exhp5b',
                                 'exph5a', 'exph5b'):
            self.drawing_rept_exhp6(trial_type, figname, pms)
        return

    def drawing_rept_exhp6(self, trial_type, figname, pms):
        xlsx_name = '{}_nk{}_r{}_pms_alt'.format(
            trial_type[:-1], self._nb_iter, int(self._ratio * 100))
        pre = self._prep.replace('_', '')
        sheet_name = 'exp{}_{}'.format(trial_type[-2:], pre)
        # pms['figname'] = figname
        pms['mp_cores'] = self._mp_cores
        pms['omitted'] = self._omitted
        pms['nk'] = self._nb_iter
        if trial_type[-6:] in ('exhp5a', 'exph5a'):
            iterator = HPEA_m1fix(figname=figname, **pms)
        elif trial_type[-6:] in ('exhp5b', 'exph5b'):
            iterator = HPEB_m2fix(figname=figname, **pms)
        df = iterator.load_raw_dataset(xlsx_name, sheet_name)
        iterator.schedule_mspaint(df, pre)
        del xlsx_name, sheet_name, df, iterator
        return

    def drawing_whole_expr5(self, trial_type, figname, pms):
        # trial = if trial_type.endswith('5b') else trial_type[:-1]
        # xlsx_name = '{}_nk{}_r{}_pms'.format(  # '5b'
        #     trial_type if trial_type.endswith('9b') else trial_type[
        #         :-1], self._nb_iter, int(self._ratio * 100))
        xlsx_name = '{}_nk{}_r{}_pms'.format(  # '5b'
            trial_type if trial_type[
                # -2:] in ('9b', '9f',) else trial_type[:-1],
                -2:] in ('9b',) else trial_type[:-1],
            self._nb_iter, int(self._ratio * 100))
        if trial_type[-2:] in ('9b', '9f', '9g',
                               '9h', '9i'):  # .endswith('9b','5b'):
            xlsx_name += '_cf{}_rep'.format(self._nb_cls)
        pre = self._prep.replace('_', '')
        sheet_name = '{}_{}'.format(trial_type[-5:], pre)
        if trial_type[-2:] not in (
            '9f', '9g',  # not trial_type.endswith('9f'):
                '9h', '9i',):
            sheet_name = sheet_name.replace('exp9', 'exp5')
        # sheet_name = sheet_name.replace('exp8', 'exp4')
        pms['figname'] = figname

        if trial_type.endswith('rexp9c'):  # 'rexp5c'):
            # self._iterator = ConvP_5C_exact(figname, **pms)
            self._iterator = ConvFig_5C_exact(figname, **pms)
        elif trial_type.endswith('rexp9d'):
            self._iterator = ConvFig_5D_exact(figname, **pms)
        elif trial_type.endswith('rexp9e'):
            xlsx_name = xlsx_name.replace('exp9', 'exp8')
            self._iterator = ConvFig_5E_exact(figname, **pms)
        elif trial_type.endswith('rexp9b'):
            self._iterator = ConvFig_5B_exact(figname, **pms)
        elif trial_type.endswith('rexp9f'):
            self._iterator = ConvFig_5F_exact(**pms)  # figname,
        elif trial_type.endswith('rexp9g'):
            self._iterator = ConvFig_5H_exact(**pms)  # figname,
        elif trial_type.endswith('rexp9h'):
            self._iterator = ConvFig_5I_exact(**pms)
        elif trial_type.endswith('rexp9i'):
            self._iterator = ConvFig_5Isimpl(**pms)
        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, pre)
        return

    def drawing_whole_expr4(self, trial_type, figname, pms):
        pre = self._prep.replace('_', '')
        xlsx_name = '{}_nk{}_r{}_pms'.format(  # '4b'
            trial_type if trial_type.endswith('8b') else trial_type[
                :-1], self._nb_iter, int(self._ratio * 100))
        if trial_type.endswith('8b'):  # '4b'):
            xlsx_name += '_cf{}_rep'.format(self._nb_cls)
        sheet_name = '{}_{}'.format(trial_type[-5:], pre)
        sheet_name = sheet_name.replace('exp8', 'exp4')

        pms['figname'] = figname
        if trial_type.endswith('rexp8e'):
            self._iterator = ConvFig_4E_exact(**pms)  # figname,
        # elif trial_type.endswith('rexp8c'):  # 'rexp4c'):
        #     self._iterator = ConvFig_4C_exact(figname, **pms)
        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, pre)
        return

    def drawing_whole_rexp3(self, trial_type, prefix, figname, pms):
        xlsx_name = '{}2_nk{}_r{}_pms_rep'.format(
            trial_type[:-2], self._nb_iter, int(self._ratio * 100))
        if trial_type[-2:] not in ['3c', '3d', '3b']:
            xlsx_name = xlsx_name.replace('rexp2', 'rexp3')  # 'exp'
            xlsx_name += '_cf{}'.format(self._nb_cls)
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)
        sheet_name = '{}_{}'.format(
            trial_type[-5:], self._prep.replace('_', ''))

        if trial_type.endswith('rexp3c') or trial_type.endswith('3b'):
            self._iterator = RevP_XC_statsParity(figname=figname, **pms)
        elif trial_type.endswith('rexp3d'):
            self._iterator = RevP_XD_statsParity(figname=figname, **pms)
        elif trial_type.endswith('rexp3e'):
            self._iterator = RevP_XE_statsParity(figname=figname, **pms)
        elif trial_type.endswith('rexp3f'):
            self._iterator = RevP_XF_statsParity(figname=figname, **pms)

        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, self._prep.replace('_', ''))
        return

    def drawing_whole_rexp2(self, trial_type, prefix, figname, pms):
        xlsx_name = '{}1_nk{}_r{}_pms'.format(
            trial_type[:-2], self._nb_iter, int(self._ratio * 100))
        if not trial_type.endswith('2a'):
            xlsx_name = xlsx_name.replace('rexp1', 'rexp2')  # 'rexp'
            xlsx_name += '_rep'
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)
        sheet_name = '{}_{}'.format(
            trial_type[-5:], self._prep.replace('_', ''))

        if trial_type.endswith('rexp2a'):
            self._iterator = RevP_YA_embedding(figname=figname, **pms)
        elif trial_type.endswith('rexp2b'):
            self._iterator = RevP_YB_embedding(figname=figname, **pms)
        elif trial_type.endswith('rexp2c'):
            self._iterator = RevP_YC_embedding(figname=figname, **pms)

        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, self._prep.replace('_', ''))
        return

    def drawing_whole_rexp1(self, trial_type, prefix, figname, pms):
        xlsx_name = '{}_nk{}_r{}_pms'.format(
            trial_type[:-1], self._nb_iter, int(self._ratio * 100))
        if prefix != '':
            xlsx_name = '{}) {}'.format(prefix, xlsx_name)
        sheet_name = '{}_{}'.format(
            trial_type[-5:], self._prep.replace('_', ''))

        if trial_type.endswith('rexp1a'):
            self._iterator = RevP_ZA_efficient(figname=figname, **pms)
        elif trial_type.endswith('rexp1b'):
            self._iterator = RevP_ZB_efficient(figname=figname, **pms)
        elif trial_type.endswith('rexp1c'):
            self._iterator = RevP_ZC_efficient(figname=figname, **pms)

        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, self._prep.replace('_', ''))
        return

    # 4convergence

    def trial_one_process_cvg(self, logger=None):
        since = time.time()
        elegant_print("[BEGAN {}]".format(elegant_dated(since)), logger)
        # START

        pre = self._prep.replace('_', '')
        fgn = self._trial_type.split('_')[-1]
        self.subproc_cvg1(self._trial_type, pre, fgn)

        # END
        tim_elapsed = time.time() - since
        elegant_print([
            "Duration /TimeCost: {}".format(fantasy_durat(tim_elapsed)),
            "[ENDED {}]".format(elegant_dated(time.time()))], logger)
        return

    def subproc_cvg1(self, trial_type, pre, fgn):
        xlsx_name = '{}_nk{}_r{}_pms'.format(
            trial_type[:-1], self._nb_iter, int(self._ratio * 100))
        # if trial_type.endswith('cvg1c'):
        xlsx_name += f'_ne{self._n_e}p{self._n_p}'
        # xlsx_name += f'_ma{self._m1}' * trial_type.endswith(
        #     'cvg1a') + f'_mb{self._m2}' * trial_type.endswith('cvg1b')
        xlsx_name += ('_rep' * self._rep_iter + '_gen' * self._gen_iter)
        sheet_name = 'exp{}_{}'.format(trial_type[-2:], pre)

        kws = dict(m1=self._m1, m2=self._m2, n_e=self._n_e,
                   n_p=self._n_p, figname=fgn)
        # self._iterator = DistPerf_draw(self._nb_iter, **kws)
        if trial_type.endswith('cvg1g'):
            self._iterator = cvgPlt1_anal_gather(self._nb_iter, **kws)
            df_a = self._iterator.load_raw_dataset(xlsx_name, f'exp1a_{pre}')
            df_b = self._iterator.load_raw_dataset(xlsx_name, f'exp1b_{pre}')
            self._iterator.schedule_mspaint(df_a, df_b, pre)
            return

        if trial_type.endswith('cvg1c'):
            self._iterator = cvgPlt1C_take(self._nb_iter, **kws)
        elif trial_type.endswith('cvg1a'):
            self._iterator = cvgPlt1A_anal(self._nb_iter, **kws)
        elif trial_type.endswith('cvg1b'):
            self._iterator = cvgPlt1B_anal(self._nb_iter, **kws)

        df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
        self._iterator.schedule_mspaint(df, pre)
        return


# -------------------------------
#


# ===============================
# Plotting


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-exp', "--expt-id", type=str, default='mCV_expt2c',
        help="Type of trial: experiment id")
    parser.add_argument(
        '-pre', "--data-preprocessing", type=str, default="min_max",
        choices=["none", "standard", "min_max", "normalize"])
    parser.add_argument('--omit', action='store_false', help='--omitted')
    parser.add_argument('-rev', '--revision-plt', action='store_true')

    parser.add_argument('-cvg', '--converge', type=str, default='')
    parser.add_argument('-np', '--n_p_chosen', type=int, default=3)

    parser.add_argument('-m1', '--m1-chosen', type=int, default=25)
    parser.add_argument('-m2', '--m2-chosen', type=int, default=11)
    parser.add_argument('-ne', '--n_e_chosen', type=int, default=2)
    parser.add_argument('-fix', '--m2-fixed', action='store_true')
    parser.add_argument('-mp', "--mp-cores", type=int, default=3,
                        help="multiprocessing cores")
    parser.add_argument('-nk', "--nb-iter", type=int, default=5,
                        help="Cross validation")  # pm: --gen|cvs-iter
    parser.add_argument("-ratio", type=float, default=.76,
                        help="Perturbing (disturbing) ratio")

    parser.add_argument('-gen', action='store_true', help='')
    parser.add_argument('-rep', action='store_false', help='')
    parser.add_argument(
        '--nb-cls', type=int, default=7, help='#classifier')
    parser.add_argument(
        "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM",  # "LR",
            "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
            "MLP", "lmSGD", "NN", "LM",
            'bagging', 'AdaBoost', 'LightGBM', 'FairGBM', 'AdaFair',
        ], help='Individual classifier options')
    parser.add_argument(
        '-constr', '--constraint-type', type=str, default='FPR,FNR')

    parser.add_argument(
        "--screen", action="store_false", help="Where to output")
    parser.add_argument(
        "--logged", action="store_true", help="Where to output")
    parser.add_argument('--prefix', default='', help='prefix and suffix')
    return parser


# -------------------------------
# hyper-parameters


screen = logged = None
parser = default_parameters()
args = parser.parse_args()

trial_type = args.expt_id
# prep = args.data_preprocessing
screen = args.screen
logged = args.logged


if args.converge:
    kwargs = dict(prep=args.data_preprocessing,
                  screen=screen, logged=logged)
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['n_p'] = args.n_p_chosen
    kwargs['ratio'] = .97

    kwargs['nb_iter'] = args.nb_iter  # nb_cv
    if trial_type[-5:] in ('cvg1a', 'cvg1b', 'cvg1g'):
        kwargs['m2_fixed'] = True
    if trial_type.endswith('cvg1g'):
        kwargs['m1'] = 20
        kwargs['m2'] = 8
    if 'cvg1' in trial_type:
        kwargs['rep'] = args.rep
        kwargs['gen'] = args.gen
    case = Rev_ManfExtDrawing(trial_type, **kwargs)
    case.trial_one_process_cvg()
    sys.exit()


kwargs = {}
kwargs['nb_iter'] = args.nb_iter
kwargs['prep'] = args.data_preprocessing
kwargs['omitted'] = args.omit  # =args.omitted
kwargs['prefix'] = args.prefix
case = None


if args.revision_plt:
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['m2_fixed'] = args.m2_fixed  # args.fix
    kwargs['mp_cores'] = args.mp_cores  # args.mp
    kwargs['ratio'] = .97  # args.ratio

    kwargs['gen'] = args.gen  # args.gen_iter
    kwargs['rep'] = args.rep  # args.rep_iter
    kwargs['screen'] = args.screen
    kwargs['logged'] = args.logged
    # kwargs['prefix'] = 'RW_May3'

    case = Rev_ManfExtDrawing(trial_type, **kwargs)
    case.trial_one_process()
    del case, screen, logged, kwargs
    del parser, args, trial_type
    sys.exit()


# if 'expt3' in trial_type:    # ['expt3b','expt3c','expt3d']
#   pass
# elif 'expt4' in trial_type:  # ['expt4b','expt4c','expt4d']
#   pass
if ('expt3' in trial_type) or ('expt4' in trial_type):
    kwargs['rep'] = True
    kwargs['ratio'] = .76
    kwargs['nb_iter'] = 5
    kwargs['nb_cls'] = args.nb_cls

    # if 'expt3' in trial_type:
    #     kwargs['prefix'] = 'RWrou3'
    # elif 'expt4' in trial_type:
    #     kwargs['prefix'] = 'RWrou4'
    # kwargs['prefix'] = 'RWrou7'

elif 'expt7' in trial_type:
    kwargs['nb_iter'] = args.nb_iter  # default:5
    kwargs['mp_cores'] = args.mp_cores
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['gen'] = args.gen  # default:False
    kwargs['rep'] = args.rep  # default:False
    # kwargs['prefix'] = 'RWrou8'

elif ('expt5' in trial_type) or ('expt6' in trial_type):
    kwargs['nb_iter'] = args.nb_iter  # default:5
    kwargs['omitted'] = args.omit
    kwargs['mp_cores'] = args.mp_cores
    # kwargs['prefix'] = 'RWrou7'

    if trial_type[-2:] in ['7a', '5a', '6a']:
        kwargs['m1'] = args.m1_chosen
    elif trial_type[-2:] in ['7a', '5b', '6b']:
        kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['gen'] = args.gen  # default:False
    kwargs['rep'] = args.rep  # default:False

case = ManfExtDrawing(trial_type, **kwargs)
case.trial_one_process()

del case, screen, logged
del parser, args, trial_type, kwargs


# -------------------------------
# Empirical plotting
"""
python hfm_nonbin_draw.py -exp rept_expt7a -pre min_max -m1 20 -m2 8
python hfm_nonbin_draw.py -exp rept_expt5a -pre min_max -m1 20
python hfm_nonbin_draw.py -exp rept_expt5b -pre min_max -m2 8
python hfm_nonbin_draw.py -exp mCV_expt4b -pre min_max

python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp1b -pre min_max
python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp2c -pre min_max
python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp3e -pre min_max

python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9i
python hfm_nonbin_draw.py -rev -ratio .97 -exp rept_exhp5a|b -pre min_max
"""

"""
python hfm_nonbin_draw.py -cvg may12 -exp mCV_cvg1c -pre min_max
python hfm_nonbin_draw.py -cvg may12 -exp mCV_cvg1a -m1 20 -pre min_max
python hfm_nonbin_draw.py -cvg may12 -exp mCV_cvg1b -m2 8  -pre min_max
python hfm_nonbin_draw.py -cvg may12 -exp mCV_cvg1g -pre min_max
"""
