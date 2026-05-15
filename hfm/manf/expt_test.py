# coding: utf-8

import pdb
import numpy as np


def test_case():
    from experiment.df_zip.mcvg_sim import ManfCvgPrime, ManfCvgEmpir
    logger = None
    kw = dict(prep='min_max', nb_cv=1)  # 2)
    # kw['prep'] = 'none'
    trial_type = 'mCV_may12_cvg1a'  # 1c|d
    data_type = 'ricci'             # 'german'

    # cs = ManfCvgEmpir(trial_type, data_type, **kw)
    cs = ManfCvgPrime(trial_type, data_type, **kw)
    # cs.preparing_curr_dat(logger)
    cs.coding_per_dataset(logger)
    # pdb.set_trace()
    return


def excl_test_expt():
    from experiment.df_zip.mcvg_exp import DistPerformance
    from hfm.manf.dist_internal import Direct_bin
    cs = DistPerformance([1, 1])

    n, nd = 324, 17
    nc = na = 2
    nai = 3
    X = np.random.rand(n, nd) * 10
    y = np.random.randint(nc, size=n)  # binary classification
    A = np.random.randint(nai, size=(n, na)) + 1
    y_XnA = np.concatenate([y.reshape(-1, 1), X], axis=1)
    indices = [[A[:, i] == j + 1 for j in range(
        nai)] for i in range(na)]

    m1, m2, n_e, n_p = 20, 8, 3, 3
    Direct_bin(y_XnA, A[:, 1], 1, indices[1][0], 'euclidean', n_p)
    cs.subproc_core_sup(y_XnA, A, indices, m1, m2, n_e, func='euclidean')
    cs.subproc_core_alt(y_XnA, A[:, 1], indices[1], m1, m2, n_e, func='euclidean')
    return
