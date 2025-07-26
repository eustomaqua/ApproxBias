.. quickstart.rst


================
Getting started
================


We provide the evaluation of extra discrimination introduced in the learning procedure for three cases: 1) only one bi-valued sensitive attribute (sen-att); 2) one multi-valued sen-att; and 3) more than one sen-att. Among them, case 1 comes from [P1]_, and two others come from [P2]_.

.. [P1] Does Machine Bring in Extra Bias in Learning? Approximating Fairness in Models Promptly https://arxiv.org/abs/2405.09251
.. [P2] Approximating Discrimination Within Models When Faced With Several Non-Binary Sensitive Attributes https://arxiv.org/abs/2408.06099


.. This is a short tutorial covering all the aforementioned cases and methods; Note to check your configuration please before running the example. 
.. (see `Examples`_)

Here is a short tutorial covering all the aforementioned cases and methods (see `Examples`_); Note to check your configuration please before running the example (see `Requirements`_).



Requirements
=============

We developed `ApproxBias <https://github.com/eustomaqua/ApproxBias>`_ with ``Python 3.8``, and also tested it with ``Python 3.11`` at the time of release. Remember to choose the ``requirements.txt`` accordingly. 

.. code-block:: console
  :linenos:

  $ # Install Anaconda/miniconda if you didn't
  $
  $ # Create a virtual environment
  $ conda create -n test python=3.11 # or 3.8
  $ source activate test
  $
  $ # Install packages
  $ pip install --upgrade pip
  $ pip install -r requirements.txt  # Python 3.11
  $ # pip install -r reqs_dev.txt    # Python 3.8
  $ # python -m pytest
  $
  $ # Delete the virtual environment
  $ source deactivate
  $ conda remove -n test --all


Examples
=============

You may need to adjust the forms of the data you use as follows.

.. code-block:: python
  :linenos:

  # Load data: X, A, y, f(x)
  #   X: non-sen-att, shape=(#, #non-sen-att)
  #   A: sen-att, shape=(#, #sen-att)
  #   y: label, shape=(#,)
  #   f(x): prediction, shape=(#,)
  import numpy as np

  # param priv_val: indicating the privileged group
  #   Note that it may vary for different sen-att-s; in that case, modify
  #   `sa_val` accordingly.
  X_nA_y = np.concatenate([y.reshape(-1, 1).astype('float'), X], axis=1)
  sa_val = [set(A[:, i]) for i in range(A.shape[1])]
  sa_val = [[priv_val]+list(i - set({priv_val})) for i in sa_val]
  sa_idx = [[A[:, i] == k for k in j]  for i, j in enumerate(sa_val)]
  X_nA_fx = np.concatenate([fx.reshape(-1, 1).astype('float'), X], axis=1)

  # How to modify `sa_val`, for example, if we have a list of privileged
  # values to indicate their members, that is,
  # param priv_val: a list of priv_vals, shape=(#sen-att,)
  sa_val = [set(A[:, i]) for i in range(A.shape[1])]
  sa_val = [[j]+list(i - set({j})) for i,j in zip(sa_val, priv_val)]
  sa_idx = [[A[:, i] == k for k in j]  for i, j in enumerate(sa_val)]


Here are examples of three aforementioned cases respectively.

**Case 1**, *bi-valued*

.. """""""""""""""""

.. code-block:: python
  :linenos:

  # Case 1: one bi-valued sen-att, take the k-th sen-att for example

  from hfm.dist_drt import DirectDist_bin
  from hfm.hfm_df import bias_degree_bin
  (D, _), _ = DirectDist_bin(X_nA_y, sa_idx[k][0])
  (Df, _), _ = DirectDist_bin(X_nA_fx, sa_idx[k][0])
  df_prev, _ = bias_degree_bin(D, Df)

  # If you'd like to compute the distances quicker
  from hfm.dist_est_bin import ApproxDist_bin
  # param m1: designated number for repetition
  # param m2: designated number for comparison
  hat_D, _ = ApproxDist_bin(X_nA_y, A[:, k], sa_idx[k][0], m1, m2)
  hat_Df, _ = ApproxDist_bin(X_nA_fx, A[:, k],sa_idx[k][0], m1, m2)
  hat_df_prev, _ = bias_degree_bin(hat_D, hat_Df)

**Case 2**, *multi-valued*

.. """"""""""""""""""""

.. code-block:: python
  :linenos:

  # Case 2: one multi-valued sen-att, take the k-th sen-att for example

  from hfm.dist_drt import DirectDist_nonbin
  from hfm.hfm_df import bias_degree_nonbin
  D, _ = DirectDist_nonbin(X_nA_y, sa_idx[k])
  Df, _ = DirectDist_nonbin(X_nA_fx, sa_idx[k])
  df_max, _ = bias_degree_nonbin(D[0], Df[0])
  df_avg, _ = bias_degree_nonbin(D[1], Df[1])

  # If you'd like to compute the distances quicker
  from hfm.dist_est_nonbin import ApproxDist_nonbin
  hat_D, _ = ApproxDist_nonbin(X_nA_y, A[:, k], m1, m2)
  hat_Df, _ = ApproxDist_nonbin(X_nA_fx, A[:, k], m1, m2)
  # compute hat_Df, hat_df_{max, avg} analogously
  hat_df_max, _ = bias_degree_nonbin(hat_D[0], hat_Df[0])
  hat_df_avg, _ = bias_degree_nonbin(hat_D[1], hat_Df[1])

**Case 3**, *more than one*

.. """""""""""""""""""""

.. code-block:: python
  :linenos:

  # Case 3: more than one sen-att

  from hfm.dist_drt import DirectDist_multiver
  D = DirectDist_multiver(X_nA_y, sa_idx)[0][:-1]
  Df = DirectDist_multiver(X_nA_fx, sa_idx)[0][:-1]
  df_max, _ = bias_degree_nonbin(D[0], Df[0])
  df_avg, _ = bias_degree_nonbin(D[1], Df[1])

  # If you'd like to compute the distances quicker
  from hfm.dist_est_nonbin import ExtendDist_multiver_mp
  hat_D = ExtendDist_multiver_mp(X_nA_y, A, m1, m2)[0][:-1]
  hat_Df = ExtendDist_multiver_mp(X_nA_fx, A, m1, m2)[0][:-1]
  # compute hat_Df, hat_df_{max, avg} analogously
  hat_df_max, _ = bias_degree_nonbin(hat_D[0], hat_Df[0])
  hat_df_avg, _ = bias_degree_nonbin(hat_D[1], hat_Df[1])


You're welcome to adjust the parameters (except ``priv_val``, which depends on the data you use) as needed or to explore potential improvements. Please note that this version may contain typos or errors; If you find any, feel free to contact us or `raise an issue <https://github.com/eustomaqua/ApproxBias/issues>`_.

.. `submit a pull request <https://github.com/eustomaqua/ApproxBias/pulls>`_.
.. easily use the `_`
.. If you would like to observe the time 
.. You can easily observe the time that each operation would consume, just use the `_` that we omitted earlier.



.. .. tip: :

.. hint::

  To observe the consumed time of each operation, just use the ``_`` that we omitted earlier.


For example,

.. code-block:: python
  :linenos:

  # Case 1, bi-valued
  (D, _), tim_elapsed = DirectDist_bin(X_nA_y, sa_idx[k][0])
  df_prev, tim_elapsed = bias_degree_bin(D, Df)
  hat_D, tim_consumed = ApproxDist_bin(X_nA_y, A[:, k], sa_idx[k][0], m1, m2)

  # Case 2, multi-valued
  D, tim_elapsed = DirectDist_nonbin(X_nA_y, sa_idx[k])
  df_max, tim_elapsed = bias_degree_nonbin(D[0], Df[0])
  hat_D, tim_consumed = ApproxDist_nonbin(X_nA_y, A[:, k], m1, m2)

  # Case 3, more than one
  D, tim_elapsed = DirectDist_multiver(X_nA_y, sa_idx)
  D = D[:-1]
  df_max, tim_elapsed = bias_degree_nonbin(D[0], Df[0])
  hat_D, tim_consumed = ExtendDist_multiver_mp(X_nA_y, A, m1, m2)
  hat_D = hat_D[:-1]

.. To understand these distances and HFM, see :doc:`methodology <methodology>`

To understand these distances and HFM in more detail, see :doc:`methodology <methodology>`.
