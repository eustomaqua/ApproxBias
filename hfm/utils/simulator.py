# coding: utf-8


import numpy as np
from hfm.utils.verifiers import check_zero


# -----------------------
# Synthetic data
#
#   nb_lbl: number of labels/classes
#   nb_spl: number of instances
#   nb_ftr: number of features
#   nb_clf: number of classifiers


def synthetic_lbl(nb_lbl, nb_spl, prng=None):
    y_inst = np.repeat(range(nb_lbl), nb_spl / nb_lbl + 1)
    y_inst = y_inst.reshape(nb_lbl, -1).T.reshape(-1)
    y_inst = y_inst[: nb_spl]
    if not prng:
        np.random.shuffle(y_inst)
    else:
        prng.shuffle(y_inst)
    return y_inst.tolist()


def synthetic_dat(nb_lbl, nb_spl, nb_ftr, prng=None):
    if not prng:
        X_inst = np.random.rand(nb_spl, nb_ftr)
        y_inst = np.random.randint(nb_lbl, size=nb_spl)
    else:
        X_inst = prng.rand(nb_spl, nb_ftr)
        y_inst = prng.randint(nb_lbl, size=nb_spl)
    return X_inst.tolist(), y_inst.tolist()


def synthetic_set(nb_lbl, nb_spl, nb_clf, prng=None):
    if not prng:
        prng = np.random
    y_inst = prng.randint(nb_lbl, size=nb_spl)
    yt_cls = prng.randint(nb_lbl, size=(nb_clf, nb_spl))
    coef = prng.rand(nb_clf)
    coef /= check_zero(np.sum(coef))
    return y_inst.tolist(), yt_cls.tolist(), coef.tolist()


def synthetic_clf(y_inst, nb_clf, err=.1, prng=None):
    if not prng:
        prng = np.random
    nb_spl, nb_lbl = len(y_inst), len(set(y_inst))
    yt_clf = np.repeat(y_inst, repeats=nb_clf, axis=0)
    yt_clf = yt_clf.reshape(-1, nb_clf).T
    num = int(nb_spl * err)
    for k in range(nb_clf):
        for _ in range(num):
            i = prng.randint(nb_spl)
            yt_clf[k][i] = nb_lbl - 1 - yt_clf[k][i]
    return yt_clf.tolist()
