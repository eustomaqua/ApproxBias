# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


import numpy as np
import numba


# =====================================
# Preliminaries
# =====================================
# original instance : (xneg, xpos, y )
# slightly disturbed: (xneg, xqtb, y')
#
# X, X': list, shape (nb_inst, nb_feat)
# y, y': list, shape (nb_inst,)
# sensitive attributes: list, (nb_feat,)
#       \in {0,1}^nb_feat
#       represent: if it is a sensitive attribute
#


def disturb_slightly(X, sens=None, ratio=.4):
    if (not sens) or (not isinstance(sens, list)):
        return X
    dim = np.shape(X)
    # noise = np.random.randint(2, size=dim)
    # noise = np.multiply(noise, sens)
    # return np.subtract(X, noise).tolist()

    X = np.array(X)
    ''' # ratio=?
    for k in range(dim[1]):
        if not sens[k]:
            continue
        Tk = X[:, k]  # xpos
        Tq = 1 - Tk   # xqtb
        Ti = np.random.rand(dim[0])
        T = [q if i <= ratio else j for j, q, i in zip(Tk, Tq, Ti)]
        X[:, k] = T
    '''  # ratio=?

    for i in range(dim[0]):
        Ti = X[i]    # xpos
        Tq = 1 - Ti  # xqtb
        # T = [q if j else k for k, q, j in zip(Ti, Tq, sens)]
        Tk = np.random.rand(dim[1])
        Tk = np.logical_and(Tk, sens)
        T = [q if k else j for j, q, k in zip(Ti, Tq, Tk)]
        X[i] = T

    return X.tolist()


# def disturb_predicts(X, sens, clf, mu=0, sigma=.01):
#   Xp = disturb_slightly(X, sens, mu=mu, sigma=sigma)
#   return yt, ys
def disturb_predicts(X, sens, clf, ratio=.4):
    Xp = disturb_slightly(X, sens, ratio)
    yt = clf.predict(X).tolist()
    yp = clf.predict(Xp).tolist()
    return yt, yp


# -------------------------------------
# Ensemble methods / Majority vote
# -------------------------------------
# weights of individual classifiers / coefficients
#   $\rho = [w_1,w_2,...,w_m]^\mathsf{T}$
#
# \begin{equation}
#   MV_\rho(bmx) =
#     \argmax_{y\in\mathcal{y}}
#       \sum_{j=1}^m w_j \mathbb{I}(f_j(x) = y)
# \end{equation}
#
# NB. Ties are resolved arbitrarily.
#

# def majority_voting_subscript_rho(y, yt, weight):
def majority_vote_subscript_rho(y, yt, weight):
    # that is, weighted_voting()

    # yt: list, shape (nb_cls, nb_inst)
    # y : list, shape (nb_inst,)
    # weight: list, shape (nb_cls,)
    vY = np.unique(np.concatenate([[y], yt]))

    coef = np.array([weight]).T
    weig = [  # weighted
        np.sum(coef * np.equal(yt, i), axis=0) for i in vY]
    loca = np.array(weig).argmax(axis=0)  # location

    # TODO: ties? or what?
    return [vY[i] for i in loca]  # i.e., fens


# For one single individual or the ensemble classifier,
#
# 0-1 loss:
# \begin{align}
#   \ell(h(X),Y) = \mathbb{I}(h(X) \neq Y)
#   \hat{L}(h,S) = 1/n \sum_{i=1}^n \ell(h(X_i),Y_i)
#   L(h) = \mathbb{E}_{(X,Y)\sim D}[\ell(h(X),Y)]
# \end{align}
#
# fairness quality:
# \begin{align}
#   ell l_{loss}(f,bmx) = \mathbb{I}(
#       f(xneg,xpos) \neq y )
#     = \mathbb{I}(f(bmx) \neq y)
#
#   ell l_{fair}(f,bmx) = \mathbb{I}(
#       f(xneg,xpos) \neq f(xneg,xqtb) )
#
#   hat L_{loss}(f,S)
#     = \frac{1}{n} \sum_{1<=i<=n} ell l_{loss}(f,bmx_i)
#   hat L_{fair}(f,S)
#     = \frac{1}{n} \sum_{1<=i<=n} ell l_{fair}(f,bmx_i)
#
#   cal L_{loss}(f)
#     = \mathbb{E}_{(bmx,y)\sim\mathcal{D}}[ ell l_{loss}(f,bmx)]
#   cal L_{fair}(f)
#     = \mathbb{E}_{(bmx,y)\sim\mathcal{D}}[ ell l_{fair}(f,bmx)]
# \end{align}


def ell_fair_x(fxp, fxq):
    # both are: list, shape (nb_inst,)
    # function is symmetrical
    # value belongs to {0,1}, set
    return np.not_equal(fxp, fxq).tolist()


def ell_loss_x(fxp, y):
    # both are: list, shape (nb_inst,)
    return np.not_equal(fxp, y).tolist()


def hat_L_fair(fxp, fxq):
    # both: list, shape (nb_inst,)
    # function is symmetrical
    # value belongs to [0,1], interval
    Lfair = ell_fair_x(fxp, fxq)
    return np.mean(Lfair).tolist()  # float


def hat_L_loss(fxp, y):
    # both: list, shape (nb_inst,)
    Lloss = ell_loss_x(fxp, y)
    return np.mean(Lloss).tolist()  # float


# For two different individual members,
#
# expected tandem loss:
# \begin{equation}
#   L(h,h') = \mathbb{E}_D[
#     \mathbb{I}(
#       h(X) \neq Y  \land  h'(X) \neq Y
#     )]
# \end{equation}
#
# tandem fairness quality:
# \begin{align}
#   ell l_{loss}(f,f',bmx) = \mathbb{I}(
#       f(xneg,xpos) \neq y  \land
#       f'(xneg,xpos) \neq y )
#     = \mathbb{I}(f(bmx) \neq y  \land  f'(bmx) \neq y)
#
#   ell l_{fair}(f,f',bmx) = \mathbb{I}(
#       f(xneg,xpos) \neq f(xneg,xqtb)  \land
#       f'(xneg,xpos) \neq f'(xneg,xqtb) )
#
#   hat L_{loss}(f,f',S)
#     = 1/n \sum_{1<=i<=n} ell l_{loss}(f,f',bmx_i)
#   hat L_{fair}(f,f',S)
#     = 1/n \sum_{1<=i<=n} ell l_{fair}(f,f',bmx_i)
#
#   cal L_{loss}(f,f')
#     = \mathbb{E}_{(bmx,y)\sim\mathcal{D}}[ell l_{loss}(f,f',bmx)]
#   cal L_{fair}(f,f')
#     = \mathbb{E}_{(bmx,y)\sim\mathcal{D}}[ell l_{fair}(f,f',bmx)]
# \end{align}


def tandem_fair(fa_p, fa_q, fb_p, fb_q):
    # whole: list, shape (nb_inst,)
    ha = np.not_equal(fa_p, fa_q)
    hb = np.not_equal(fb_p, fb_q)
    tmp = np.logical_and(ha, hb)
    return np.mean(tmp).tolist()  # float


def tandem_loss(fa, fb, y):
    # whole: list, shape (nb_inst,)
    ha = np.not_equal(fa, y)
    hb = np.not_equal(fb, y)
    tmp = np.logical_and(ha, hb)
    return np.mean(tmp).tolist()  # float


# Objective
#   L(f,f) = L(f) = lam*L_fair+(1-lam)*L_loss

def hat_L_objt(fxp, fxq, y, lam):
    l_fair = hat_L_fair(fxp, fxq)
    l_acc_p = hat_L_loss(fxp, y)
    return lam * l_fair + (1. - lam) * l_acc_p


def tandem_objt(fa, fa_q, fb, fb_q, y, lam):
    l_fair = tandem_fair(fa, fa_q, fb, fb_q)
    l_acc_p = hat_L_loss(fa, y)
    l_acc_q = hat_L_loss(fb, y)
    # l_acc = (1. - lam) * (l_acc_p + l_acc_q) / 2.
    # return lam * l_fair + l_acc
    l_acc = (l_acc_p + l_acc_q) / 2.
    return lam * l_fair + (1. - lam) * l_acc


def cal_L_obj_v1(yt, yq, y, wgt, lam=.5):
    # aka. def hat_L_objective()
    # aka. def hat_L_objt()

    L_fair = Erho_sup_L_fair(yt, yq, wgt)
    L_acc = E_rho_L_loss_f(yt, y, wgt)
    return lam * L_fair + (1. - lam) * L_acc


def cal_L_obj_v2(yt, yq, y, wgt, lam=.5):
    nb_cls = len(wgt)

    ans = []
    for i in range(nb_cls):
        tmp = [tandem_objt(yt[i], yq[i],
                           yt[j], yq[j],
                           y, lam) for j in range(nb_cls)]
        ans.append(tmp)

    res = np.sum(np.multiply(ans, wgt), axis=1)
    res = np.sum(np.multiply(res, wgt), axis=0)
    return res.tolist()


# def ell_fair(yt, ys)
# def tandom_fair(yt_p, ys_p, yt_q, ys_q)
# def hat_L_fair(yp, yq)
#
# def tandem_fair(ya_p, ya_q, yb_p, yb_q)
# def tandem_loss(ya, yb, y)
#
# def ell_fair_fc():
#     pass
# def ell_loss_fc():
#     pass


# -------------------------------------
# Losses
# -------------------------------------
# L*_S Empir: for dataset
# L*_Exp ect: expectation
#


# -------------------------------------
# Manage diversity
# -------------------------------------


# =====================================
# Theorems
# =====================================

# MVrho = majority_vote_subscript_rho(y, yt, wgt)
# MVpmo = majority_vote_subscript_rho(y, yq, wgt)


def L_fair_MV_rho(MVrho, MVpmo):
    return hat_L_fair(MVrho, MVpmo)


def L_loss_MV_rho(MVrho, y):
    return hat_L_loss(MVrho, y)


# -------------------------------------
# Theorem 3.1.
# First-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 2Exp_\rho[ L(h)] $
#
# \begin{align}
#   cal L_{loss}(MV_\rho) <= 2Exp_\rho[cal L_{loss}(f)]
#   cal L_{fair}(MV_\rho) <= 2Exp_\rho[cal L_{fair}(f)]
# \end{align}


def E_rho_L_fair_f(yt, yq, wgt):
    E_rho = [hat_L_fair(
        p, q) for p, q in zip(yt, yq)
    ]  # list, shape (nb_cls,)
    # E_rho = np.mean(np.not_equal(yt, yq), axis=1)
    # ans = np.dot(wgt, E_rho).tolist()
    tmp = np.sum(np.multiply(wgt, E_rho))
    return tmp.tolist()  # float


def E_rho_L_loss_f(yt, y, wgt):
    E_rho = [hat_L_loss(p, y) for p in yt]
    # return np.mean(E_rho).tolist()
    tmp = np.sum(np.multiply(wgt, E_rho))
    return tmp.tolist()  # float


# -------------------------------------
# Theorem 3.3.
# Second-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 4Exp_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   L_{loss}(MV_\rho) <= 4Exp_{\rho^2}[ L_{loss}( f,f')]
#   L_{fair}(MV_\rho) <= 4Exp_{\rho^2}[ L_{fair}( f,f')]
# \end{align}


def Erho_sup_L_fair(yt, yq, wgt, nb_cls=None):
    if not nb_cls:
        nb_cls = len(wgt)  # number of weights

    L_f_fp = []
    for p in range(nb_cls):
        tmp = [tandem_fair(
            yt[p], yq[p],
            yt[i], yq[i]) for i in range(nb_cls)]
        L_f_fp.append(tmp)
    # L_f_fp: list, shape (nb_cls, nb_cls)

    E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
    E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
    # E_rho2: list, shape (nb_cls,)
    # E_rho2: dtype('float64'), shape ()

    return E_rho2.tolist()  # float


def Erho_sup_L_loss(yt, y, wgt, nb_cls=None):
    if not nb_cls:
        nb_cls = len(wgt)  # length of weights

    L_f_fp = []
    for p in range(nb_cls):
        tmp = [tandem_loss(
            yt[p],
            yt[i], y) for i in range(nb_cls)]
        L_f_fp.append(tmp)
    # L_f_fp: list, shape (nb_cls, nb_cls)

    E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
    E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
    return E_rho2.tolist()  # float


# -------------------------------------
# Lemma 3.2.
# -------------------------------------
# $ E_D[E_\rho[
#         \mathbb{I}(h(X) \neq y)
#     ]^2] = E_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(bmx) \neq y)
#   ]^2] = E_{\rho^2}[ L_{loss}( f,f')]
#
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(xneg,xpos) \neq f(xneg,xqtb))
#   ]^2] = E_{\rho^2}[ L_{fair}( f,f')]
# \end{align}


def ED_Erho_I_fair(yt, yq, wgt):
    wt = np.array([wgt]).T      # .shape=(#cls)
    I_f = np.not_equal(yt, yq)  # .shape=(#cls,#inst)
    Erho = np.sum(wt * I_f, axis=0)  # siz=(#inst,)
    # E_rho = np.mean(np.multiply(wt, If), axis=0)
    ED = np.mean(Erho * Erho)  # siz=(),dtype('float64')
    return ED.tolist()  # float


def ED_Erho_I_loss(yt, y, wgt):
    wt = np.array([wgt]).T
    I_f = np.not_equal(yt, y)
    Erho = np.sum(wt * I_f, axis=0)
    ED = np.mean(Erho * Erho)
    return ED.tolist()  # float


# def E_rho_L_fair_f(yq, yt):
# def E_rho_L_loss_f(yo, yt):
# def E_rho_sup2_L_fair(yq, yt, wgt, nb_cls=None):
# def E_rho_sup2_L_loss(y, yt, wgt, nb_cls=None):
# def ED_Erho_I_fair(yq, yt, wgt):
# def ED_Erho_I_loss(y, yt, wgt):


# -------------------------------------
# Theorem 3.4.
# C-tandem oracle bound
# -------------------------------------
# If $E_\rho[ L(h)] < 1/2$, then
# \begin{equation}
#   L(MV_\rho) \leq
#   \frac{
#     E_{\rho^2}[ L(h,h')] - E_\rho[ L(h)]^2
#   }{
#     E_{\rho^2}[ L(h,h')] - E_\rho[ L(h)] +1/4
#   }
# \end{equation}
#

# If $Exp_\rho[ L_{loss}] <1/2$, then
# \begin{equation}
#   L_{loss}(MV_\rho) \leqslant \frac{
#       Exp_{\rho^2}[ L_{loss}(f,f')]
#     - Exp_\rho[ L_{loss}(f)]^2
#   }{
#       Exp_{\rho^2}[ L_{loss}(f,f')]
#     - Exp_\rho[ L_{loss}(f)]
#     + 1/4
#   }
# \end{equation}
#
# If $Exp_\rho[ L_{fair}] <1/2$, then
# \begin{equation}
#   L_{fair}(MV_\rho) \leqslant \frac{
#       Exp_{\rho^2}[ L_{fair}(f,f')]
#     - Exp_\rho[ L_{fair}(f)]^2
#   }{
#       Exp_{\rho^2}[ L_{fair}(f,f')]
#     - Exp_\rho[ L_{fair}(f)]
#     + 1/4
#   }
# \end{equation}
#


# -------------------------------------
# Theorem 3.5.
# -------------------------------------


# =====================================
# Theorems
# =====================================


# -------------------------------------
# Theorem 3.1.
# First-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 2Exp_\rho[ L(h)] $
#
# \begin{align}
#   cal L_{loss}(MV_\rho) <= 2Exp_\rho[cal L_{loss}(f)]
#   cal L_{fair}(MV_\rho) <= 2Exp_\rho[cal L_{fair}(f)]
# \end{align}
#


# -------------------------------------
# Lemma 3.2.
# -------------------------------------
# $ E_D[E_\rho[
#         \mathbb{I}(h(X) \neq y)
#     ]^2] = E_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(bmx) \neq y)
#   ]^2] = E_{\rho^2}[ L_{loss}( f,f')]
#
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(xneg,xpos) \neq f(xneg,xqtb))
#   ]^2] = E_{\rho^2}[ L_{fair}( f,f')]
# \end{align}
#


# -------------------------------------
# Theorem 3.3.
# Second-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 4Exp_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   L_{loss}(MV_\rho) <= 4Exp_{\rho^2}[ L_{loss}( f,f')]
#   L_{fair}(MV_\rho) <= 4Exp_{\rho^2}[ L_{fair}( f,f')]
# \end{align}
#


# -------------------------------------
# Theorem 3.4.
# C-tandem oracle bound
# -------------------------------------


# -------------------------------------
# Theorem 3.5.
# -------------------------------------
