# coding: utf-8

import pdb
import time
from numba import set_num_threads
import numpy as np
from hfm.utils.verifiers import poset_nolessthan

from hfm.manf.renew_drt import Direct_bin as re_bin
from hfm.manf.renew_app import Approx_bin as re_Approx_bin
from hfm.manf.renew_drt import Direct_nonbin as re_nonbin
from hfm.manf.renew_app import Approx_nonbin as re_Approx_nonbin
from hfm.manf.renew_cvg import StratES_nonbin as re_StratES
from hfm.manf.renew_cvg import StratRA_nonbin as re_StratRA


from hfm.manf.dist_internal import Direct_bin, Direct_nonbin
from hfm.manf.dist_internal import Approx_bin
from hfm.manf.dist_external import (
    Approx_nonbin, StratES_nonbin, StratRA_nonbin)
from hfm.dist_drt import DirectDist_bin as prev_bin
from hfm.dist_drt import DirectDist_nonbin as prev_nonbin
from hfm.dist_est_bin import ApproxDist_bin as prev_Approx_bin
from hfm.dist_est_nonbin import ApproxDist_nonbin as prev_Approx_nonbin
from hfm.dist_cvg_nonbin import StratEarlyStop as prev_StratES
from hfm.dist_cvg_nonbin import StratRearrange as prev_StratRA
from hfm.dist_cvg_nonbin import StratVacant as prev_Vacant


n, nd = 6324, 17
nc = na = n_e = 2
nai = 3
m1, m2 = 20, 8
priv = 1

X = np.random.rand(n, nd) * 10
y = np.random.randint(nc, size=n)  # binary classification
A_i = np.random.randint(nai, size=n) + 1

X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
X_nA_y = np.ascontiguousarray(X_nA_y)  # ,dtype=np.float64)
A_i = np.asarray(A_i)
indices = [A_i == j + 1 for j in range(nai)]

curr_p = [7, 2, 1, float('inf'), 3, ]
curr_d = ['cos_sim', 'correla', 'euclidean',
          'manhattan', 'chebyshev', 'minkowski']
ele_i, ele_ic = X_nA_y[:2]
B_i = (A_i == priv).astype('int')
B_ind = [indices[0], ~indices[0]]


# def this_case1(curr, ind):
#     nb_v4_d1 = [re_bin(X_nA_y, curr, p, priv) for p in curr_p][1:]
#     nb_v4_a1 = [re_Approx_bin(X_nA_y, curr, p, m1, m2) for p in curr_p][1:]
#     pdb.set_trace()
#     return


# def this_case2(curr, ind):
#     nb_v4_d2 = [re_nonbin(X_nA_y, curr, p, priv) for p in curr_p][1:]
#     nb_v4_a2 = [re_Approx_nonbin(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
#     nb_v4_es = [re_StratES(X_nA_y, curr, p) for p in curr_p][1:]
#     nb_v4_ra = [re_StratRA(X_nA_y, curr, p, m1, m2, n_e) for p in curr_p][1:]
#     pdb.set_trace()
#     return


# def test_renew():
#     B_i = (A_i == priv).astype('int')
#     B_ind = [indices[0], ~indices[0]]
#     this_case1(B_i, B_ind)
#     # this_case1(A_i, indices)
#     return


# 预热：让 JIT 编译/缓存加载完成，这次不计时
N = 12
curr, ind, p = B_i, B_ind, curr_p[1]
# curr, ind = A_i, indices
re_bin(X_nA_y, curr, p, priv)
re_Approx_bin(X_nA_y, curr, p, m1, m2)  # priv)

re_nonbin(X_nA_y, curr, p, priv)
re_Approx_nonbin(X_nA_y, curr, p, m1, m2, n_e)
re_StratES(X_nA_y, curr, p)
re_StratRA(X_nA_y, curr, p, m1, m2, n_e)


t0 = time.perf_counter()
for _ in range(N):
    # _Approx_bin_sub(X_nA_y, B_i, 2.0, m1, m2)
    re_Approx_bin(X_nA_y, curr, p, m1, m2)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)

t2 = time.perf_counter()
for _ in range(N):
    re_bin(X_nA_y, curr, p, priv)
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
print('')

t2 = time.perf_counter()
for _ in range(N):
    re_nonbin(X_nA_y, curr, p, priv)
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
t0 = time.perf_counter()
for _ in range(N):
    re_Approx_nonbin(X_nA_y, curr, p, m1, m2, n_e)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)

t0 = time.perf_counter()
for _ in range(N):
    re_StratES(X_nA_y, curr, p)
t1 = time.perf_counter()
print("   strat es :", (t1 - t0) / N)
t0 = time.perf_counter()
for _ in range(N):
    re_StratRA(X_nA_y, curr, p, m1, m2, n_e)
t1 = time.perf_counter()
print("   strat ra :", (t1 - t0) / N)


print('\n')
p = curr_d[2]
Direct_bin(X_nA_y, curr, priv, ind[0], p)
Approx_bin(X_nA_y, curr, m1, m2, p)
Direct_nonbin(X_nA_y, curr, priv, ind, p)
Approx_nonbin(X_nA_y, curr, m1, m2, n_e, p)
StratES_nonbin(X_nA_y, curr, n_e, p)
StratRA_nonbin(X_nA_y, curr, m1, m2, n_e, p)

prev_bin(X_nA_y, ind[0])
prev_Approx_bin(X_nA_y, curr, ind[0], m1, m2)
prev_nonbin(X_nA_y, ind)
prev_Approx_nonbin(X_nA_y, curr, m1, m2, n_e)
prev_StratES(X_nA_y, curr, n_e)
prev_StratRA(X_nA_y, curr, m1, m2, n_e)
prev_Vacant(X_nA_y, curr, m1, m2, n_e)


# def mypar_cnter():
#     return


t0 = time.perf_counter()
for _ in range(N):
    Approx_bin(X_nA_y, curr, m1, m2, p)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)
t2 = time.perf_counter()
for _ in range(N):
    Direct_bin(X_nA_y, curr, priv, ind[0], p)
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
print('')
t2 = time.perf_counter()
for _ in range(N):
    Direct_nonbin(X_nA_y, curr, priv, ind, p)
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
t0 = time.perf_counter()
for _ in range(N):
    Approx_nonbin(X_nA_y, curr, m1, m2, n_e, p)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)
t0 = time.perf_counter()
for _ in range(N):
    StratES_nonbin(X_nA_y, curr, n_e, p)
t1 = time.perf_counter()
print("   strat es :", (t1 - t0) / N)
t0 = time.perf_counter()
for _ in range(N):
    StratRA_nonbin(X_nA_y, curr, m1, m2, n_e, p)
t1 = time.perf_counter()
print("   strat ra :", (t1 - t0) / N)
print('\n')

t0 = time.perf_counter()
for _ in range(N):
    prev_Approx_bin(X_nA_y, curr, ind[0], m1, m2)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)
t2 = time.perf_counter()
for _ in range(N):
    prev_bin(X_nA_y, ind[0])
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
print('')
t2 = time.perf_counter()
for _ in range(N):
    prev_nonbin(X_nA_y, ind)
t3 = time.perf_counter()
print(' direct cost:', (t3 - t2) / N)
t0 = time.perf_counter()
for _ in range(N):
    prev_Approx_nonbin(X_nA_y, curr, m1, m2, n_e)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)
t0 = time.perf_counter()
for _ in range(N):
    prev_StratES(X_nA_y, curr, n_e)
t1 = time.perf_counter()
print("   strat es :", (t1 - t0) / N)
t0 = time.perf_counter()
for _ in range(N):
    prev_StratRA(X_nA_y, curr, m1, m2, n_e)
t1 = time.perf_counter()
print("   strat ra :", (t1 - t0) / N)
for _ in range(N):
    prev_Vacant(X_nA_y, curr, m1, m2, n_e)
t1 = time.perf_counter()
print("avg per call:", (t1 - t0) / N)


print('')
pdb.set_trace()
