# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds, extension
#


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
        '-pre', "--data-preprocessing", type=str, default="none",
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
    parser.add_argument('-nk', "--nb-iter", type=int, default=5,
                        help="Cross validation")
    parser.add_argument('-mp', "--mp-cores", type=int, default=3,
                        help="multiprocessing")
    parser.add_argument(
        '--fix', action='store_true', help='-fix, --m2-fixed')

    parser.add_argument(
        '--nb-cls', type=int, default=7, help='#classifier')
    parser.add_argument(
        "--abbr-cls", type=str, default="DT",
        choices=[
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
