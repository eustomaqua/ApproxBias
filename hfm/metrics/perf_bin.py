# coding: utf-8
# performance.py, confusion_mat.py
#
# Aim to provide:
#   Confusion matrices
#   Performance-based metrics
#

import numpy as np
import numba
from hfm.utils.verifiers import check_zero


def comp_accuracy(y, hx):
    t = np.mean(np.equal(y, hx))
    return float(t)


def comp_error_rate(y, hx):
    t = np.mean(np.not_equal(y, hx))
    return float(t)


# ---------------------
# Contingency table (binary)
#
# |True label `y`| Prediction `f(x)`   |
# |              | Positive | Negative |
# | Positive (1) |    TP    |    FN    |
# | Negative (0) |    FP    |    TN    |
#


@numba.jit(nopython=True)
def contingency_tab_bi(y, y_hat, pos=1):
    # For one single classifier
    tp = np.sum((y == pos) & (y_hat == pos))
    fn = np.sum((y == pos) & (y_hat != pos))
    fp = np.sum((y != pos) & (y_hat == pos))
    tn = np.sum((y != pos) & (y_hat != pos))
    # return tp, fp, fn, tn
    return tp, fn, fp, tn


def contingency_tab_mu(y, y_hat, vY):
    dY = len(vY)
    Cij = np.zeros(shape=(dY, dY), dtype='int')
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum(
                (y == vY[i]) & (y_hat == vY[j]))
    return Cij  # np.ndarray


# ---------------------
# After having the confusion matrix,
# for one single classifier, (binary)


# Accuracy
def calc_accuracy(tp, fn, fp, tn):
    n = float(tp + fp + fn + tn)
    tmp = (tp + tn) / check_zero(n)
    return float(tmp)


# Error rate = 1 - accuracy
def calc_error_rate(tp, fn, fp, tn):
    n = float(tp + fp + fn + tn)
    tmp = (fp + fn) / check_zero(n)
    return float(tmp)


# Precision
def calc_precision(tp, fn, fp, tn):
    denominator = float(tp + fp)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# Recall
def calc_recall(tp, fn, fp, tn):
    denominator = float(tp + fn)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# F1 measure
def calc_f1_score(tp, fn, fp, tn):
    n = float(tp + fp + fn + tn)
    denominator = n + tp - tn
    tmp = 2 * tp / check_zero(denominator)
    return float(tmp)


# F1
def calc_f_beta(p, r, beta=1):
    denominator = beta ** 2 * p + r
    numerator = (1. + beta**2) * p * r
    tmp = numerator / check_zero(denominator)
    return float(tmp)


# ---------------------
#
