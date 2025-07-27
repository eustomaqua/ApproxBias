# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#


import argparse
from experiment.df_bin.manf_sim import (
    ManfPrime_Empirical, ManfSimulative)  # ManfEmpirical


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp", "--expt-id", type=str, default="mCV_expt1",
        help="Type of trial: experiment id")
    parser.add_argument(
        "-dat", "--dataset", type=str, default="ricci",
        choices=["ricci", "german", "adult", "ppr", "ppvr", 'tmp'])
    parser.add_argument(
        "-prep", "--data-preprocessing", type=str,
        default='min_max', choices=[
            'none', 'standard', 'min_max', 'normalize'])

    parser.add_argument('-nk', "--nb-iter", type=int, default=5,
                        help="Cross validation")
    parser.add_argument(
        '--gen-iter', type=bool, default=False, help="param: gen")
    parser.add_argument(
        '--rep-iter', type=bool, default=False, help="param: cvs")
    parser.add_argument(
        '-m1', '--m1-chosen', type=int, default=20, help="m1")
    parser.add_argument(
        '-m2', '--m2-chosen', type=int, default=10, help="m2")
    parser.add_argument(
        "--ratio", type=float, default=.75, help="Disturbing ratio")

    parser.add_argument(
        "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM",  # "LR",
            "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
            "MLP", "lmSGD", "NN", "LM",
            'bagging', 'AdaBoost', 'LightGBM', 'FairGBM',
            'AdaFair', ], help="Individual classifiers")
    parser.add_argument(
        '--nb-cls', type=int, default=1, help='#classifiers')
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
abbr_cls = args.abbr_cls
nb_iter = args.nb_iter
screen = args.screen
logged = args.logged


kwargs = {}
kwargs['nb_iter'] = nb_iter
kwargs['prep'] = args.data_preprocessing
if trial_type[-6:] in ('expt5a', 'expt6a'):
    kwargs['m1'] = args.m1_chosen
elif trial_type[-6:] in ('expt5b', 'expt6b'):
    kwargs['m2'] = args.m2_chosen

if trial_type[-6:] in ['expt5a', 'expt5b', 'expt5c']:
    if trial_type.startswith(
            'repetit') or trial_type.startswith('rept'):
        kwargs['gen'] = args.gen_iter
    elif trial_type.startswith('KF') or trial_type[:3] == 'mCV':
        kwargs['rep'] = args.rep_iter
elif trial_type[-6:] in ['expt6a', 'expt6b']:
    kwargs['rep'] = True
    kwargs['nb_cls'] = args.nb_cls
    kwargs['constraint_type'] = args.constraint_type

elif 'expt2' in trial_type:
    kwargs['rep'] = True
    kwargs['nb_iter'] = args.nb_iter
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['nb_cls'] = args.nb_cls
    kwargs['ratio'] = args.ratio


if data_type.endswith(
        'simulative') or data_type.startswith('tmp'):
    case = ManfSimulative(
        trial_type, 'simulative', abbr_cls,
        screen=screen, logged=logged, **kwargs)
else:
    # case = ManfEmpirical(
    case = ManfPrime_Empirical(
        trial_type, data_type, abbr_cls,
        screen=screen, logged=logged, **kwargs)

mode = "a" if data_type == 'adult' else "w"
case.trial_one_process(mode=mode)


del screen, logged, kwargs
del trial_type, data_type, abbr_cls, nb_iter
del parser, args, case


# Experiments
"""
python hfm_ver1_exec.py -exp rept_expt5a -dat tmp|ricci -nk 0|2 -m1 5
python hfm_ver1_exec.py -exp rept_expt5b -dat tmp|ricci -nk 0|2 -m2 5
python hfm_ver1_exec.py -exp rept_expt5c -dat tmp|ricci -nk 1

python hfm_ver1_exec.py -exp mCV_expt6a -dat ricci --abbr-cls bagging/FairGBM/AdaFair --nb-cls 3
python hfm_ver1_exec.py -exp mCV_expt6b -dat ricci --abbr-cls bagging/FairGBM/AdaFair --nb-cls 3

python hfm_ver1_exec.py -exp mCV_expt2a|2b|2c|2d|2e -dat ricci --nb-cls 3 -m1 25 -m2 11
"""
