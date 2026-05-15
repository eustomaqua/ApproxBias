# coding: utf-8

import pdb
import numpy as np
from hfm.manf.dist_internal import (
    Direct_bin, Direct_nonbin, Direct_multiver, curr_intermediate)
from hfm.manf.dist_external import (
    Approx_nonbin, StratES_nonbin, StratRA_nonbin, EffExact_multiver)
from hfm.utils.verifiers import check_equal, poset_nolessthan


n, nd = 324, 17
nc = na = nai = 2
nai = 3
loc = 1

X = np.random.rand(n, nd) * 10
y = np.random.randint(nc, size=n)  # binary classification
A = np.random.randint(nai, size=(n, na)) + 1

X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
indices = [[A[:, i] == j + 1 for j in range(
    nai)] for i in range(na)]

priv = 1
n_p = 3
A_i = A[:, loc]
B_i = (A[:, loc] == priv).astype('int')
idx_Sjs = [indices[loc][0], ~indices[loc][0]]
m1, m2, n_e = 10, 4, 2


def naive_bin(X_nA_y, A_i, idx_Sjs, func='euclidean'):
    from hfm.manf.earlybreak import Naive_nonbin, EffHD_nonbin
    from hfm.manf.earlybreak import Naive_bin, EffHD_bin
    from hfm.manf.dist_internal import Approx_bin

    ot1 = Naive_bin(X_nA_y, idx_Sjs[0], func, n_p)
    ot2 = EffHD_bin(X_nA_y, idx_Sjs[0], func, n_p)
    my1 = Direct_bin(X_nA_y, A_i, priv, idx_Sjs[0], func, n_p)
    my2 = Approx_bin(X_nA_y, A_i, m1, m2, func, n_p)
    my3 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, func, n_p)
    my4 = Direct_nonbin(X_nA_y, A_i, priv, idx_Sjs, func, n_p)

    assert ot1[1] > ot2[1] and ot1[0] == ot2[0]
    assert check_equal(my1[0][0], [ot1[0], ot2[0]])
    assert poset_nolessthan(my2[0], my1[0])
    if len(set(A_i)) != 2:
        assert poset_nolessthan(my3[0], my4[0])
        return
    assert poset_nolessthan(my3[0], my1[0])
    assert check_equal(my4[0], my1[0])
    assert check_equal(my4[0][0], ot1[0])
    # assert all([check_equal(i, j) for i, j in zip(my1[0], my4[0])])
    return


def naive_nonbin(X_nA_y, A_i, idx_Sjs, func='euclidean'):
    from hfm.manf.earlybreak import Naive_nonbin, EffHD_nonbin
    # from hfm.manf.earlybreak import Naive_bin, EffHD_bin

    ot1 = Naive_nonbin(X_nA_y, idx_Sjs, func, n_p)
    ot2 = EffHD_nonbin(X_nA_y, idx_Sjs, func, n_p)
    my1 = Direct_nonbin(X_nA_y, A_i, priv, idx_Sjs, func, n_p)
    my3 = Approx_nonbin(X_nA_y, A_i, m1, m2, n_e, func, n_p)
    my4 = StratES_nonbin(X_nA_y, A_i, n_e, func, n_p)
    my5 = StratRA_nonbin(X_nA_y, A_i, m1, m2, n_e, func, n_p)

    assert ot1[1] > ot2[1] and ot1[0] == ot2[0]
    assert check_equal(my1[0][0], [ot1[0], ot2[0]])
    assert poset_nolessthan(my3[0], my1[0])
    assert poset_nolessthan(my4[0], my1[0])
    assert poset_nolessthan(my5[0], my1[0])
    return


def Naive_multiver(X_nA_y, A, indices, func='euclidean'):
    from hfm.manf.earlybreak import Naive_multiver, EffHD_multiver

    ot1 = Naive_multiver(X_nA_y, indices, func, n_p)
    ot2 = EffHD_multiver(X_nA_y, indices, func, n_p)
    my1 = Direct_multiver(X_nA_y, A, priv, indices, func, n_p)
    my3 = EffExact_multiver(X_nA_y, A, 'Vacant', m1, m2, n_e, func, n_p)
    my4 = EffExact_multiver(X_nA_y, A, 'ES', m1, m2, n_e, func, n_p)
    my5 = EffExact_multiver(X_nA_y, A, 'RA', m1, m2, n_e, func, n_p)

    # pdb.set_trace()
    assert ot1[-1] > ot2[-1] and ot1[0][0] == ot2[0][0]
    assert check_equal(my1[0][0], [ot1[0][0], ot2[0][0]])
    assert check_equal(my4[0][:2], my1[0][:2])
    assert poset_nolessthan(my3[0][:2], my1[0][:2])
    assert poset_nolessthan(my5[0][:2], my1[0][:2])
    return


def test_naive():
    naive_bin(X_nA_y, B_i, idx_Sjs)
    naive_bin(X_nA_y, A_i, indices[loc])
    naive_nonbin(X_nA_y, B_i, idx_Sjs)
    naive_nonbin(X_nA_y, A_i, indices[loc])
    Naive_multiver(X_nA_y, A, indices)
    return
