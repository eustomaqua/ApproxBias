# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#   fairness in manifolds and its extension
#


import numpy as np
# import numba
# import pdb
from hfm.utils.decorators import fantasy_timer
from hfm.utils.verifiers import check_zero
from hfm.metrics.contingency_mat import contingency_tab_bi


class _elem:
    @staticmethod
    def _indices(vA, idx, ex):
        tmp = list(range(len(vA)))
        tmp.remove(idx)
        n = sum(ex) - ex[idx]
        return tmp, n, sum(ex)  # =nt


def zero_division(dividend, divisor):
    # divided_by_zero
    if divisor == 0 and dividend == 0:
        return 0.
    elif divisor == 0:
        return 10.  # return 1.
    return dividend / divisor


# =====================================
# Fairness research & oracle bounds
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


def marginalised_np_mat(y, y_hat, pos_label=1,
                        priv_idx=list()):
    if isinstance(y, list) or isinstance(y_hat, list):
        y, y_hat = np.array(y), np.array(y_hat)

    g1_y = y[priv_idx]  # idx_priv]
    g0_y = y[~priv_idx]
    g1_hx = y_hat[priv_idx]
    g0_hx = y_hat[~priv_idx]

    g1_Cm = contingency_tab_bi(g1_y, g1_hx, pos_label)
    g0_Cm = contingency_tab_bi(g0_y, g0_hx, pos_label)
    # g1_Cm: for the privileged group
    # g0_Cm: for marginalised group(s)
    return g1_Cm, g0_Cm


def marginalised_np_gen(y, y_hat, A, priv_val=1,
                        pos_label=1):
    if 0 in A and len(set(A)) == 2:
        vA = list(set(A))[:: -1]
        idx = vA.index(priv_val)
        g_y = [y[A == i] for i in vA]
        g_hx = [y_hat[A == i] for i in vA]
        g_Cm = [contingency_tab_bi(
            i, j, pos_label) for i, j in zip(g_y, g_hx)]
        ex = [sum(A == i) for i in vA]
        return g_Cm, vA, idx, ex

    vA = sorted(set(A))
    idx = vA.index(priv_val)
    g_y = [y[A == i] for i in vA]
    g_hx = [y_hat[A == i] for i in vA]
    gs_Cm = [contingency_tab_bi(
        i, j, pos_label) for i, j in zip(g_y, g_hx)]
    ex = [sum(A == i) for i in vA]
    return gs_Cm, vA, idx, ex


# -------------------------------------
# Group fairness (measures)
# -------------------------------------
''' Cm
|        | hx= pos | hx= neg |
| y= pos |    TP   |    FN   |
| y= neg |    FP   |    TN   |
'''
# return tp, fp, fn, tn


# 1) demographic parity
# 人口统计均等
# aka. (TP+FP)/N = P[h(x)=1]

def unpriv_group_one(g1_Cm, g0_Cm):
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[1]) / n1
    g0 = (g0_Cm[0] + g0_Cm[1]) / n0
    return float(g1), float(g0)


# 2) equality of opportunity
# 胜率均等
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]

def unpriv_group_two(g1_Cm, g0_Cm):
    t1 = g1_Cm[0] + g1_Cm[2]
    t0 = g0_Cm[0] + g0_Cm[2]
    g1 = g1_Cm[0] / check_zero(t1)
    g0 = g0_Cm[0] / check_zero(t0)
    return float(g1), float(g0)


# 3) predictive quality parity
# 预测概率均等
# 3) predictive parity
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]

def unpriv_group_thr(g1_Cm, g0_Cm):
    t1 = g1_Cm[0] + g1_Cm[1]
    t0 = g0_Cm[0] + g0_Cm[1]
    g1 = g1_Cm[0] / check_zero(t1)
    g0 = g0_Cm[0] / check_zero(t0)
    return float(g1), float(g0)


def calc_fair_group(g1, g0):
    # aka. def group_fair()
    return abs(g1 - g0)


# 假设不同群体成员具有同样的工作潜能
# aka. (TP+FN)/N = P[y=1]

def unpriv_unaware(g1_Cm, g0_Cm):
    # aka. prerequisite
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[2]) / n1
    g0 = (g0_Cm[0] + g0_Cm[2]) / n0
    return float(g1), float(g0)


# 自定义 = accuracy 准确度
# aka. (TP+TN)/N = P[h(x)=y]
def unpriv_manual(g1_Cm, g0_Cm):
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[3]) / n1
    g0 = (g0_Cm[0] + g0_Cm[3]) / n0
    return float(g1), float(g0)


# =====================================
# Metrics
# =====================================


# =====================================
# Oracle bounds for fairness
# =====================================
# Note that y|hx is np.ndarray


@fantasy_timer
def StatsParity_sing(hx, idx_Sjs, pos=1):
    total = np.mean(hx == pos)
    item = [np.mean(hx[idx] == pos) for idx in idx_Sjs]
    elements = [float(np.abs(i - total)) for i in item]
    n_ai = len(idx_Sjs)
    return max(elements), sum(elements) / n_ai


@fantasy_timer
def StatsParity_mult(hx, idx_Ai_Sjs, pos=1):
    half_mid = [StatsParity_sing(
        hx, idx_Sjs) for idx_Sjs in idx_Ai_Sjs]
    half_mid, half_ut = zip(*half_mid)
    half_pl_max, half_pl_avg = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    return max(half_pl_max), sum(half_pl_avg) / float(n_a), (
        half_pl_max, half_pl_avg, half_ut)


# =====================================
# Fairness manifold + extension
# =====================================


@fantasy_timer
def extGrp1_DP_sing(y, hx, idx_Sjs, pos=1):
    total = np.mean(hx == pos)
    alternative = [np.mean(hx[idx] == pos) for idx in idx_Sjs]

    if np.isnan(alternative).any():
        alternative = np.nan_to_num(alternative).tolist()
    if np.isnan(total):
        total = float(np.nan_to_num(total))

    elements = [float(np.abs(i - total)) for i in alternative]
    # if np.isnan(elements).any():
    #     pdb.set_trace()
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def extGrp2_EO_sing(y, hx, idx_Sjs, pos=1):
    idx = y == pos  # renew_y = y[idx]
    renew_hx = hx[idx]
    renew_Sj = [Sj[idx] for Sj in idx_Sjs]
    total = np.mean(renew_hx == pos)
    alternative = [np.mean(renew_hx[Sj] == pos) for Sj in renew_Sj]

    if np.isnan(alternative).any():
        alternative = np.nan_to_num(alternative).tolist()
    if np.isnan(total):
        total = float(np.nan_to_num(total))

    elements = [float(np.abs(i - total)) for i in alternative]
    # if np.isnan(elements).any():
    #     pdb.set_trace()
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def extGrp3_PQP_sing(y, hx, idx_Sjs, pos=1):
    idx = hx == pos
    renew_y = y[idx]
    renew_Sj = [Sj[idx] for Sj in idx_Sjs]
    total = np.mean(renew_y == pos)
    alternative = [np.mean(renew_y[Sj] == pos) for Sj in renew_Sj]

    if np.isnan(alternative).any():
        alternative = np.nan_to_num(alternative).tolist()
    if np.isnan(total):  # or total!=total
        total = float(np.nan_to_num(total))

    elements = [float(np.abs(i - total)) for i in alternative]
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def alterGrps_sing(alternative, idx_Sjs):
    ele = list(map(float, alternative))
    n_ai = len(idx_Sjs)
    renewal = [abs(ele[i] - ele[j]) for i in range(
        n_ai) for j in range(i + 1, n_ai)]
    n_aj_prime = len(renewal)
    return max(renewal), sum(renewal) / n_aj_prime


def alterGroups_pl(half_alter, idx_Ai_Sjs, n_a):
    half_alter = [alterGrps_sing(alt, idx_Sjs)[
        0] for alt, idx_Sjs in zip(half_alter, idx_Ai_Sjs)]
    half_pl_max, half_pl_avg = zip(*half_alter)
    return max(half_pl_max), sum(half_pl_avg) / n_a


@fantasy_timer
def extGrp1_DP_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
    half_mid = [extGrp1_DP_sing(
        y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
    half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    if not alter:
        return max(half_pl_max), sum(half_pl_avg) / n_a
    return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)


@fantasy_timer
def extGrp2_EO_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
    half_mid = [extGrp2_EO_sing(
        y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
    half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    if not alter:
        return max(half_pl_max), sum(half_pl_avg) / n_a
    return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)


@fantasy_timer
def extGrp3_PQP_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
    half_mid = [extGrp3_PQP_sing(
        y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
    half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    if not alter:
        return max(half_pl_max), sum(half_pl_avg) / n_a
    return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)


# =====================================
#
# =====================================
