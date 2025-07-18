# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds, extension
#


import argparse
import time

from hfm.utils.decorators import (
    elegant_dated, fantasy_durat, fantasy_timer)
from hfm.utils.recorders import elegant_print

from experiment.ver2.mext_plt import (
    CurrPlot3B_comparison, CurrPlot3C_comparison, CurrPlot3D_comparison,
    CurrPlot4B_comparison, CurrPlot4C_comparison, CurrPlot4D_comparison,
    Distributed_GA_mp, HyperEA_renew_m1fix, HyperEB_renew_m2fix,
    CurrTab4B_comparison)


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
        pms = {'nb_iter': self._nb_iter, 'nb_cls': self._nb_cls,
               'm1': self._m1, 'm2': self._m2, 'n_e': self._n_e}
        figname = 'exp{}_'.format(trial_type[-2:])

        if trial_type[-6:] in ['expt3b', 'expt3c', 'expt3d']:
            if trial_type.endswith('expt3b'):
                self._iterator = CurrPlot3B_comparison(figname=figname, **pms)
            elif trial_type.endswith('expt3c'):
                self._iterator = CurrPlot3C_comparison(figname=figname, **pms)
            elif trial_type.endswith('expt3d'):
                self._iterator = CurrPlot3D_comparison(figname=figname, **pms)
            self.drawing_whole_exp3_exp4(pre, prefix)
        elif trial_type[-6:] in ['expt4b', 'expt4c', 'expt4d']:
            if trial_type.endswith('expt4b'):
                self._iterator = CurrPlot4B_comparison(figname=figname, **pms)
                self._tabulater = CurrTab4B_comparison(figname=figname, **pms)
            elif trial_type.endswith('expt4c'):
                self._iterator = CurrPlot4C_comparison(figname=figname, **pms)
            elif trial_type.endswith('expt4d'):
                self._iterator = CurrPlot4D_comparison(figname=figname, **pms)
            self.drawing_whole_exp3_exp4(pre, prefix)

        elif trial_type[-6:] in ['expt7a']:
            del pms['nb_cls']
            pms['mp_cores'] = self._mp_cores
            self._iterator = Distributed_GA_mp(figname=figname, **pms)
            xlsx_name = '{}_iter{}_pms_ma{}_mb{}'.format(
                trial_type, self._nb_iter, self._m1, self._m2)
            if prefix != '':
                xlsx_name = '{}) {}'.format(prefix, xlsx_name)
            sheet_name = 'exp{}_{}'.format(trial_type[-2:], self._prep)
            raw_df = self._iterator.load_raw_dataset(xlsx_name, sheet_name)
            self._iterator.schedule_mspaint(raw_df, self._mp_cores, pre=pre)
            del xlsx_name, sheet_name, raw_df

        elif trial_type[-6:] in ['expt5a', 'expt5b']:
            del pms['nb_cls']
            pms['mp_cores'] = self._mp_cores
            pms['omitted'] = self._omitted
            if trial_type.endswith('expt5a'):
                self._iterator = HyperEA_renew_m1fix(figname=figname, **pms)
                xlsx_name = '{}_iter{}_pms_ma{}_alt'.format(
                    trial_type, self._nb_iter, self._m1)
            elif trial_type.endswith('expt5b'):
                self._iterator = HyperEB_renew_m2fix(figname=figname, **pms)
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


# ===============================
# Plotting


def default_parameters():
    parser = argparse.ArgumentParser()  # -expt,-data,-prep
    parser.add_argument(
        '-exp', "--expt-id", type=str, default='mCV_expt2c',
        help="Type of trial: experiment id")
    parser.add_argument(
        '-dat', "--dataset", type=str, default="ricci",
        choices=["ricci", "german", "adult", "ppr", "ppvr",
                 "tmp_simulative", "tmp", "simulative"])
    parser.add_argument(
        '-pre', "--data-preprocessing", type=str, default="min_max",
        choices=["none", "standard", "min_max", "normalize"])
    parser.add_argument(
        '--omit', action='store_false', help='--omitted')

    parser.add_argument(
        '-m1', '--m1-chosen', type=int, default=25, help='m1')
    parser.add_argument(
        '-m2', '--m2-chosen', type=int, default=11, help='m2')
    parser.add_argument(
        '-ne', '--n_e_chosen', type=int, default=2, help='n_e')
    parser.add_argument(
        "--ratio", type=float, default=.76, help="Disturbing ratio")
    parser.add_argument(
        '--gen', action='store_true', help='param: --gen-iter')
    parser.add_argument(
        '--rep', action='store_true', help='param: --cvs-iter')

    parser.add_argument(
        '-nk', "--nb-iter", type=int, default=5, help="Cross validation")
    parser.add_argument(
        '-mp', "--mp-cores", type=int, default=3, help="multiprocessing")
    parser.add_argument(
        '--fix', action='store_true', help='-fix, --m2-fixed')

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
        "--screen", action="store_true", help="Where to output")
    parser.add_argument(
        "--logged", action="store_false", help="Where to output")
    return parser


# -------------------------------
# hyper-parameters


screen = logged = None
parser = default_parameters()
args = parser.parse_args()

trial_type = args.expt_id
data_type = args.dataset
nb_iter = args.nb_iter
screen = args.screen
logged = args.logged


kwargs = {}
kwargs['prep'] = args.data_preprocessing

if ('expt2' in trial_type) or (
        'expt3' in trial_type) or ('expt4' in trial_type):
    kwargs['rep'] = True
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['ratio'] = args.ratio
    kwargs['nb_cls'] = args.nb_cls
    # kwargs['nb_iter'] = 5
    kwargs['omitted'] = args.omit
    kwargs['n_e'] = args.n_e_chosen

elif ('expt5' in trial_type) or ('expt6' in trial_type):
    if trial_type[-2:] in ['5a', '6a']:
        kwargs['m1'] = args.m1_chosen
    elif trial_type[-2:] in ['5b', '6b']:
        kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['gen'] = args.gen  # .gen_iter  default:False
    kwargs['rep'] = args.rep  # .rep_iter  default:False
    kwargs['alternative'] = True
    if trial_type[-2:] in ['6a', '6b']:
        kwargs['alternative'] = False
        kwargs['nb_cls'] = args.nb_cls
        kwargs['abbr_cls'] = args.abbr_cls
        kwargs['constraint_type'] = args.constraint_type
    kwargs['omitted'] = args.omit

elif 'expt7' in trial_type:
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['gen'] = args.gen  # .gen_iter  default:False
    kwargs['rep'] = args.rep  # .rep_iter  default:False
    kwargs['m2_fixed'] = args.fix  # default:False
    kwargs['abbr_cls'] = args.abbr_cls
    kwargs['mp_cores'] = args.mp_cores


if data_type.endswith('simulative') or data_type.startswith('tmp'):
    pass
elif 'expt2' in trial_type:
    case = ManfExtEmpirical(trial_type, data_type, nb_iter,
                            screen=screen, logged=logged, **kwargs)

elif ('expt3' in trial_type) or ('expt4' in trial_type):
    kwargs['mp_cores'] = args.mp_cores
    case = ManfExtPrime_Empirical(trial_type, data_type, nb_iter,
                                  screen=screen, logged=logged, **kwargs)
elif ('expt5' in trial_type) or ('expt6' in trial_type):
    kwargs['mp_cores'] = args.mp_cores
    case = ManfExtPrime_Empirical(trial_type, data_type, nb_iter,
                                  screen=screen, logged=logged,
                                  **kwargs)
elif 'expt7' in trial_type:
    case = ManfExtPrime_Empirical(trial_type, data_type, nb_iter,
                                  screen=screen, logged=logged,
                                  **kwargs)

mode = "a" if data_type == 'adult' else "w"
case.trial_one_process(mode=mode)


del screen, logged, kwargs
del trial_type, data_type, nb_iter
del parser, args, case, mode


# -------------------------------
# Empirical plotting
"""
python hfm_ver2_draw.py -exp mCV_expt3b -pre min_max
python hfm_ver2_draw.py -exp mCV_expt4b -pre min_max

python fmext_draw.py -exp rept_expt7a -pre min_max -m1 20 -m2 8
python fmext_draw.py -exp rept_expt5a -pre min_max -m1 20
python fmext_draw.py -exp rept_expt5b -pre min_max -m2 8
"""
