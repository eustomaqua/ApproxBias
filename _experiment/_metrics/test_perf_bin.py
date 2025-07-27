# coding: utf-8
# test_performance.py

# import pdb
import numpy as np
from sklearn import metrics

# from hfm.metrics.confusion_mat import *
# from hfm.metrics.performance import *
from hfm.utils.verifiers import check_equal
# from hfm.metrics.perf_bin import *
from hfm.metrics.excl_perf_bin import (
    contingency_tab_bi, contingency_tab_mu,
    calc_accuracy, calc_precision, calc_recall,
    calc_f1_score, calc_f_beta)


n = 201  # n, nc = 110, 3
y = np.random.randint(2, size=n)
y_hat = np.random.randint(2, size=n)

# z = np.random.randint(nc, size=n)
# z_hat = np.random.randint(nc, size=n)
vY = [1, 0]


def test_contingency_bi():
    tmp = metrics.confusion_matrix(y, y_hat)
    ans = metrics.confusion_matrix(y, y_hat, labels=vY)

    assert tmp[0, 0] == ans[1, 1]
    assert tmp[1, 1] == ans[0, 0]
    assert tmp[0, 1] == ans[1, 0]
    assert tmp[1, 0] == ans[0, 1]

    res = contingency_tab_bi(y, y_hat, pos=vY[0])
    assert np.equal(ans.ravel(), res).all()

    res = contingency_tab_mu(y, y_hat, vY)
    assert np.equal(ans, res).all()
    return


def test_perf_bin():
    # pos = vY[0]
    cm_bin = contingency_tab_bi(y, y_hat, pos=vY[0])
    cm_non = contingency_tab_mu(y, y_hat, vY)
    cm = metrics.confusion_matrix(y, y_hat, labels=vY)
    assert np.equal(cm_bin, cm.ravel()).all()
    assert np.equal(cm_non, cm).all()

    ans_1 = metrics.accuracy_score(y, y_hat)
    ans_2 = metrics.precision_score(y, y_hat)
    ans_3 = metrics.recall_score(y, y_hat)
    ans_4 = metrics.f1_score(y, y_hat)
    ans_5 = metrics.fbeta_score(y, y_hat, beta=2)

    res_1 = calc_accuracy(*cm_bin)
    p = calc_precision(*cm_bin)
    r = calc_recall(*cm_bin)
    res_4 = calc_f1_score(*cm_bin)
    res_5 = calc_f_beta(p, r, beta=2)

    assert ans_1 == res_1
    assert ans_2 == p
    assert ans_3 == r
    # assert ans_4 == res_4
    # assert ans_5 == res_5
    assert check_equal(ans_4, res_4, 10**8)
    assert check_equal(ans_5, res_5, 10**8)
    return
