# coding: utf-8
#
# Aim to provide:
#   Performance-based metrics
#

import numpy as np
from hfm.utils.verifiers import check_zero


def comp_accuracy(y, hx):
    t = np.mean(np.equal(y, hx))
    return float(t)


def comp_error_rate(y, hx):
    t = np.mean(np.not_equal(y, hx))
    return float(t)


# ---------------------
# Confusion matrix
#
#  label | prediction
# |      | f(x) positive | negative |
# | y= 1 |       TP      |    FN    |
# | y=-1 |       FP      |    TN    |
#


# ---------------------
# After having the confusion matrix,
# for one single classifier,


# Accuracy
def calc_accuracy(tp, fp, fn, tn):
    n = float(tp + fp + fn + tn)
    tmp = (tp + tn) / check_zero(n)
    return float(tmp)


# Error rate = 1 - accuracy
def calc_error_rate(tp, fp, fn, tn):
    n = float(tp + fp + fn + tn)
    tmp = (fp + fn) / check_zero(n)
    return float(tmp)


# Precision
def calc_precision(tp, fp, fn, tn):
    denominator = float(tp + fp)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# Recall
def calc_recall(tp, fp, fn, tn):
    denominator = float(tp + fn)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# F1 measure
def calc_f1_score(tp, fp, fn, tn):
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


# ---------------------
#


# ---------------------
#
