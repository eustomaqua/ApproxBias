# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds, extension
#


import argparse
import sys
from experiment.df_nonbin.mext_sim import (
    ManfExtEmpirical, ManfExtPrime_Empirical)
from experiment.df_nonbin.rev_mext_sim import Rev_ManfExtPrime_Empir


def default_parameters():
    parser = argparse.ArgumentParser()  # -expt,-data,-prep
    parser.add_argument(
        '-exp', "--expt-id", type=str, default='mCV_expt2c',
        help="Type of trial: experiment id")
    parser.add_argument(
        '-dat', "--dataset", type=str, default="ricci",
        choices=["ricci", "german", "adult", "ppr", "ppvr"])
    parser.add_argument(
        '-pre', "--data-preprocessing", type=str, default="min_max",
        choices=["none", "standard", "min_max", "normalize"])
    parser.add_argument(
        '--omit', action='store_false', help='--omitted')
    parser.add_argument('-rev', '--revision', action='store_true')

    parser.add_argument('-m1', '--m1-chosen', type=int, default=25)
    parser.add_argument('-m2', '--m2-chosen', type=int, default=11)
    parser.add_argument('-ne', '--n_e_chosen', type=int, default=2)
    parser.add_argument("--ratio", type=float, default=.76,
                        help="Perturbing (disturbing) ratio")
    parser.add_argument('--fix', action='store_true', help='--m2-fixed')

    parser.add_argument('-nk', "--nb-iter", type=int, default=5,
                        help="Cross validation")
    parser.add_argument('-mp', "--mp-cores", type=int, default=3,
                        help="multiprocessing cores")
    parser.add_argument(
        '-gen', action='store_true', help='param: --gen-iter')
    parser.add_argument(
        '-rep', action='store_true', help='param: --cvs-iter')

    parser.add_argument(
        '--nb-cls', type=int, default=7, help='#classifier')
    parser.add_argument(
        "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM",  # "LR",
            "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
            "MLP", "lmSGD", "NN", "LM",
            'bagging', 'AdaBoost', 'LightGBM', 'FairGBM', 'AdaFair',
        ], help='Individual classifier options or ensemble')
    parser.add_argument(
        '-constr', '--constraint-type', type=str, default='FPR,FNR')

    parser.add_argument(
        "--screen", action="store_true", help="Where to output")
    parser.add_argument(
        "--logged", action="store_false", help="Where to output")
    return parser


screen = logged = None
parser = default_parameters()
args = parser.parse_args()

trial_type = args.expt_id
data_type = args.dataset
# nb_iter = args.nb_iter
# screen = args.screen
# logged = args.logged


kwargs = {}
kwargs['prep'] = args.data_preprocessing


if args.revision:
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['m2_fixed'] = args.fix
    kwargs['mp_cores'] = args.mp_cores
    kwargs['ratio'] = .97  # args.ratio

    kwargs['nb_iter'] = args.nb_iter
    kwargs['gen'] = args.gen
    kwargs['rep'] = args.rep
    kwargs['screen'] = args.screen
    kwargs['logged'] = args.logged

    if trial_type[-6: -1] in ('rexp1', 'rexp2'):
        pass
    elif trial_type[-6:] in ('rexp3c', 'rexp3b'):
        kwargs['abbr_cls'] = 'BaseNet'
    elif trial_type.endswith('rexp3d'):
        kwargs['abbr_cls'] = args.abbr_cls
    elif trial_type[-6:] in ('rexp3f', 'rexp3e'):
        kwargs['abbr_cls'] = ''
        if trial_type.endswith('rexp3e'):
            kwargs['nb_cls'] = args.nb_cls

    case = Rev_ManfExtPrime_Empir(trial_type, data_type, **kwargs)
    # mode = 'a' if data_type == 'adult' else 'w'
    case.trial_one_process(mode='w')
    del kwargs, trial_type, data_type, args, parser, case
    sys.exit()


# Experiments
nb_iter = args.nb_iter
screen = args.screen
logged = args.logged

if ('expt2' in trial_type) or (
        'expt3' in trial_type) or ('expt4' in trial_type):
    kwargs['rep'] = True
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['ratio'] = args.ratio
    kwargs['nb_cls'] = args.nb_cls
    kwargs['omitted'] = args.omit

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


# if data_type.endswith('simulative') or data_type.startswith('tmp'):
#     pass
if 'expt2' in trial_type:
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


# Experiments
"""
python hfm_ver2_exec.py -exp mCV_expt4* -dat * --nb-cls 3 -nk 2
python hfm_ver2_exec.py -exp rept_expt5a -dat * -nk 5 -m1 20
python hfm_ver2_exec.py -exp rept_expt5b -dat * -nk 5 -m2 8
python hfm_ver2_exec.py -exp rept_expt7a -dat * -nk 5 -m1 20 -m2 8 --fix

python hfm_ver2_exec.py -rev -exp mCV_rexp1b -dat german -nk 1
python hfm_ver2_exec.py -rev -exp mCV_rexp2c -dat ricci -nk 2 -rep
python hfm_ver2_exec.py -rev -exp mCV_rexp3e -dat ricci -nk 2 -rep
python hfm_ver2_exec.py -rev -exp mCV_rexp3e -dat german -nk 2 -rep
"""
