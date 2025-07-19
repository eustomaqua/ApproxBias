# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


import numpy as np
import numba

from hfm.utils.verifiers import (
    DTY_INT, judge_transform_need, check_zero)


# =====================================
# Metrics
# =====================================


# contingency_table
# -------------------------------------
# input: lists, not np.ndarray


'''
contingency_table_binary
    |         | hi = +1 | hi = -1 |
    | hj = +1 |    a    |    c    |
    | hj = -1 |    b    |    d    |
'''


def contingency_table_binary(ha, hb):
    if len(ha) != len(hb):
        raise AssertionError(  # number of instances/samples
            "The shapes of two individual classifiers are different.")

    tem = np.concatenate([ha, hb]).tolist()
    vY, dY = judge_transform_need(tem)
    if dY > 2:
        raise AssertionError(
            "`contingency_table` works for binary classification only.")
    elif dY == 2:
        ha = [i * 2 - 1 for i in ha]
        hb = [i * 2 - 1 for i in hb]

    hi = np.array(ha, dtype=DTY_INT)
    hj = np.array(hb, dtype=DTY_INT)
    a = np.sum((hi == 1) & (hj == 1))
    b = np.sum((hi == 1) & (hj == -1))
    c = np.sum((hi == -1) & (hj == 1))
    d = np.sum((hi == -1) & (hj == -1))
    return a, b, c, d


'''
contingency_table_binary
    |         | hj = +1 | hj = -1 |
    | hi = +1 |    a    |    b    |
    | hi = -1 |    c    |    d    |

contingency_table_multiclass
    |         | hb == y | hb != y |
    | ha == y |    a    |    b    |
    | ha != y |    c    |    d    |
'''


def contingency_table_multiclass(ha, hb, y):
    # Do NOT use this function to calcuate!
    a = np.sum(np.logical_and(np.equal(ha, y), np.equal(hb, y)))
    c = np.sum(np.logical_and(np.not_equal(ha, y), np.equal(hb, y)))
    b = np.sum(np.logical_and(np.equal(ha, y), np.not_equal(hb, y)))
    d = np.sum(np.logical_and(np.not_equal(ha, y), np.not_equal(hb, y)))
    return int(a), int(b), int(c), int(d)


'''
contingency_table_{?}
    |              | hb!=y, hb=-1 | hb==y, hb=1 |
    | ha!=y, ha=-1 |   d          |   c         |
    | ha==y, ha= 1 |   b          |   a         |

contingency_table_multi
    |               | hb= c_0 | hb= c_1 | hb= c_{n_c-1} |
    | ha= c_0       |  C_{00} |  C_{01} |  C_{0?}       |
    | ha= c_1       |  C_{10} |  C_{11} |  C_{1?}       |
    | ha= c_{n_c-1} |  C_{?0} |  C_{?1} |  C_{??}       |
'''


def contingency_table_multi(hi, hj, y=list()):
    tem = np.concatenate([hi, hj, y]).tolist()
    vY, dY = judge_transform_need(tem)
    if dY == 1:
        dY = 2

    ha, hb = np.array(hi), np.array(hj)
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)

    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    return Cij.copy()  # np.ndarray


# -------------------------------------
# TP, FP, FN, TN
# -------------------------------------
# refs:
# https://blog.csdn.net/zyq11223/article/details/79085711
# https://blog.csdn.net/AlexFaker/article/details/108286805
# https://betterbench.blog.csdn.net/article/details/126667066
'''
|                      | predicted label positive  |  negative |
|true label is positive| true positive (TP)|false negative (FN)|
|true label is negative|false positive (FP)| true negative (TN)|
'''
# y, hx: list of scalars (as elements)


def calc_confusion(y, hx, pos=1):
    TP = np.logical_and(np.equal(y, pos), np.equal(hx, pos))
    FP = np.logical_and(np.not_equal(y, pos), np.equal(hx, pos))
    FN = np.logical_and(np.equal(y, pos), np.not_equal(hx, pos))
    TN = np.logical_and(np.not_equal(y, pos), np.not_equal(hx, pos))
    TP = np.sum(TP).tolist()  # TP = float(np.sum(TP))
    FP = np.sum(FP).tolist()  # FP = float(np.sum(FP))
    FN = np.sum(FN).tolist()  # FN = float(np.sum(FN))
    TN = np.sum(TN).tolist()  # TN = float(np.sum(TN))
    return TP, FP, FN, TN


def calc_accuracy(y, hx):
    n = len(y)
    t = np.sum(np.equal(y, hx)).tolist()
    # n = float(len(y))
    # t = float(np.sum(np.equal(y, hx)))
    return t / n  # == (TP+TN)/N 准确率


@numba.jit(nopython=True)
def calc_Acc(TP, FP, FN, TN):
    N = TP + FP + FN + TN
    accuracy_ = (TP + TN) / N
    return accuracy_, N


def calc_PR(TP, FP, FN):
    '''
    precision = TP / (TP + FP)  # 查准率,精确率
    recall = TP / (TP + FN)     # 查全率,召回率

    # N = len(y)
    # TN = N - TP - FP - FN
    # F1 = 2 * TP / (N + TP - TN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    '''

    precision = TP / check_zero(TP + FP)
    recall = TP / check_zero(TP + FN)
    F1 = 2 * TP / check_zero(2 * TP + FP + FN)
    return precision, recall, F1


def calc_ROC(TP, FP, FN, TN):
    '''
    TPR = TP / (TP + FN)  # 真正率,召回率,命中率 hit rate
    FPR = FP / (TN + FP)  # 假正率=1-特异度,误报/虚警/误检率 false alarm
    FNR = FN / (TP + FN)  # 漏报率 miss rate，也称为漏警率、漏检率
    TNR = TN / (TN + FP)  # 特异度 specificity
    # expect FPR,FNR smaller, TNR larger
    '''

    TPR = TP / check_zero(TP + FN)
    FPR = FP / check_zero(TN + FP)
    FNR = FN / check_zero(TP + FN)
    TNR = TN / check_zero(TN + FP)
    return TPR, FPR, FNR, TNR


'''
# def calc_F1(y, hx, pos=1, beta=1):
def calc_F1(P, R):
  # F1 = 2 * P * R / (P + R)
  F1 = 2 * P * R / robust_zero(P + R)
  return F1

def calc_Fbeta(P, R, beta=1):
  # numerator = (1 + beta**2) * P * R
  # denominator = (beta**2 * P) + R
  # F_beta = numerator / denominator

  beta2 = beta ** 2
  denom = robust_zero(beta2 * P + R)
  fbeta = (1 + beta2) * P * R / denom
  # return F1, fbeta
  return fbeta
'''


def calc_F1(P, R, beta=1):
    if beta == 1:
        F1 = 2 * P * R / check_zero(P + R)
        return F1

    beta2 = beta ** 2
    denom = check_zero(beta2 * P + R)
    fbeta = (1 + beta2) * P * R / denom
    return fbeta


# def calc_PRF1_multi(y, hx):
def calc_PRF1_multi_lists(y, hx):
    vY, _ = judge_transform_need(y + hx)  # dY = len(vY)

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []  # TN_list = []
    FN_list = []  # F1_list = []

    for pos in vY:
        # TP, FP, FN, TN = calc_confusion(y, hx, pos)
        # P, R, F1 = calc_PR(TP, FP, FN)
        TP, FP, FN, _ = calc_confusion(y, hx, pos)
        P, R, _ = calc_PR(TP, FP, FN)

        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        # TN_list.append(TN)
        P_list.append(P)
        R_list.append(R)
        # F1_list.append(F1)

    N = len(y)
    # return macro_P, macro_R, macro_F1, micro_P, micro_R, micro_F1
    return vY, N, P_list, R_list, TP_list, FP_list, FN_list


def calc_PRF1_multi_macro(P_list, R_list, beta=1):
    # BUG: macro_P = np.sum(P_list).tolist() / N
    # BUG: macro_R = np.sum(R_list).tolist() / N
    macro_P = np.mean(P_list).tolist()
    macro_R = np.mean(R_list).tolist()
    denom = check_zero(macro_P + macro_R)
    macro_F1 = 2 * macro_P * macro_R / denom

    beta2 = beta ** 2
    denom = beta2 * macro_P + macro_R
    fbeta = (1 + beta2) * macro_P * macro_R / check_zero(denom)
    # Notice there is a difference between mine and sklearn.
    # sklearn uses np.average([f1/fbeta])

    # return macro_P, macro_R, macro_F1
    return macro_P, macro_R, macro_F1, fbeta


def calc_PRF1_multi_micro(TP_list, FP_list, FN_list, beta=1):
    TP_avg = np.mean(TP_list).tolist()
    FP_avg = np.mean(FP_list).tolist()
    FN_avg = np.mean(FN_list).tolist()
    # TN_avg = np.mean(TN_list).tolist()

    micro_P = TP_avg / check_zero(TP_avg + FP_avg)
    micro_R = TP_avg / check_zero(TP_avg + FN_avg)
    # micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R)
    denom = check_zero(micro_P + micro_R)
    micro_F1 = 2 * micro_P * micro_R / denom

    beta2 = beta ** 2
    denom = (beta**2 * micro_P) + micro_R
    fbeta = (1 + beta2) * micro_P * micro_R / check_zero(denom)

    # return micro_P, micro_R, micro_F1
    return micro_P, micro_R, micro_F1, fbeta


# =====================================
# Oracle bounds for fairness
# =====================================
'''
marginalised groups
|      | h(xneg,gzero)=1 | h(xneg,gzero)=0 |
| y= 1 |    TP_{gzero}   |    FN_{gzero}   |
| y= 0 |    FP_{gzero}   |    TN_{gzero}   |
privileged group
|      | h(xneg,gones)=1 | h(xneg,gones)=0 |
| y= 1 |    TP_{gones}   |    FN_{gones}   |
| y= 0 |    FP_{gones}   |    TN_{gones}   |

instance (xneg,xpos) --> (xneg,xqtb)
        xpos might be `gzero` or `gones`

C_{ij}
|     | hx=0 | hx=1 | ... | hx=? |
| y=0 | C_00 | C_01 | ... | C_0* |
| y=1 | C_10 | C_11 |     | C_1* |
| ... | ...  | ...  |     | ...  |
| y=? | C_*0 | C_*1 | ... | C_*? |
'''
# y, hx: list of scalars (as elements)


def marginalised_contingency(y, hx, vY, dY):
    assert len(y) == len(hx), "Shapes do not match."
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
    for i in range(dY):
        for j in range(dY):
            tmp = np.logical_and(
                np.equal(y, vY[i]), np.equal(hx, vY[j]))
            Cij[i, j] = np.sum(tmp).tolist()
            # Cij[i, j] = int(np.sum(tmp))
    return Cij  # np.ndarray


@numba.jit(nopython=True)
def marginalised_confusion(Cij, loc=1):
    Cm = np.zeros((2, 2), dtype=DTY_INT)
    # loca = vY.index(pos)  # [[TP,FN],[FP,TN]]

    Cm[0, 0] = Cij[loc, loc]
    Cm[0, 1] = np.sum(Cij[loc]) - Cij[loc, loc]
    Cm[1, 0] = np.sum(Cij[:, loc]) - Cij[loc, loc]

    # Cm[1, 1] = np.sum(Cij[:loca, :, loca])
    Cm[1, 1] = (np.sum(Cij) + Cij[loc, loc] 
                - np.sum(Cij[loc]) - np.sum(Cij[:, loc]))
    return Cm  # np.ndarray


def marginalised_split_up(y, hx, priv=1, sens=list()):
    gones_y_ = [i for i, j in zip(y, sens) if j == priv]
    gzero_y_ = [i for i, j in zip(y, sens) if j != priv]
    gones_hx = [i for i, j in zip(hx, sens) if j == priv]
    gzero_hx = [i for i, j in zip(hx, sens) if j != priv]
    # return gones_y_, gones_hx, gzero_y_, gzero_hx
    return gones_y_, gzero_y_, gones_hx, gzero_hx


def marginalised_matrixes(y, hx, pos=1, priv=1, sens=list()):
    """ y, hx: list, shape=(N,), true label and prediction
    pos : which label is viewed as positive, might be multi-class.
    sens: which group these instances are from, including one priv-
          ileged group and one/multiple marginalised group.
          or list of boolean (as elements)
    priv: which one indicates the privileged group.
    """
    vY, _ = judge_transform_need(y + hx)
    dY = len(vY)

    gones_y_, gzero_y_, gones_hx, gzero_hx \
        = marginalised_split_up(y, hx, priv, sens)
    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)

    loca = vY.index(pos)  # [[TP,FN],[FP,TN]]
    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


def marginalised_pd_mat(y, hx, pos=1, idx_priv=list()):
    # y : not pd.DataFrame, is pd.core.series.Series
    # hx: not pd.DataFrame, is np.ndarray
    # tmp = y.to_numpy().tolist() + hx.tolist()

    if isinstance(y, list) or isinstance(hx, list):
        y, hx = np.array(y), np.array(hx)

    # y : np.ndarray, =pd.DataFrame.to_numpy().reshape(-1)
    # hx: np.ndarray
    tmp = y.tolist() + hx.tolist()
    vY, _ = judge_transform_need(tmp)
    dY = len(vY)

    gones_y_ = y[idx_priv].tolist()
    gzero_y_ = y[np.logical_not(idx_priv)].tolist()
    gones_hx = hx[idx_priv].tolist()
    gzero_hx = hx[np.logical_not(idx_priv)].tolist()

    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)
    loca = vY.index(pos)

    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    # gones_Cm:  for privileged group
    # gzero_Cm:  for marginalised groups
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


# -------------------------------------
# Unconscious/unawareness fairness
# -------------------------------------
''' Cm
|        | hx= pos | hx= neg |
| y= pos |    TP   |    FN   |
| y= neg |    FP   |    TN   |
'''


# 假设不同群体成员具有同样的工作潜能
# aka. (TP+FN)/N = P[y=1]

def unpriv_unaware(gones_Cm, gzero_Cm):
    # aka. prerequisite
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[0, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[0, 1]) / N0
    return float(g1), float(g0)


# 在无意识前提下，分类
# aka. (TP+FP)/N = P[h(x)=1]
# def unpriv_prereq(gones_Cm, gzero_Cm):


# -------------------------------------
# Group fairness (measures)
# -------------------------------------


# 1) demographic parity
# 人口统计均等
# aka. (TP+FP)/N = P[h(x)=1]

def unpriv_group_one(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 0]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 0]) / N0
    return float(g1), float(g0)


# 2) equality of opportunity
# 胜率均等
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]

def unpriv_group_two(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[0, 1]
    t0 = gzero_Cm[0, 0] + gzero_Cm[0, 1]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# 3) predictive quality parity
# 预测概率均等
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]

def unpriv_group_thr(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[1, 0]
    t0 = gzero_Cm[0, 0] + gzero_Cm[1, 0]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# 自定义 = accuracy 准确度
# aka. (TP+TN)/N = P[h(x)=y]

def unpriv_manual(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 1]) / N0
    return float(g1), float(g0)


# -------------------------------------
# Individual fairness
# -------------------------------------


# -------------------------------------
# Procedural fairness (measures)
# -------------------------------------
