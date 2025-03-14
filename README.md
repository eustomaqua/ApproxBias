# ApproxBias

![CircleCI](https://img.shields.io/circleci/build/github/eustomaqua/ApproxBias/master)
[![Documentation Status](https://readthedocs.org/projects/approxbias/badge/?version=latest)](https://approxbias.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/d0f9d3235ebf4454b3f43beb137bb2c7)](https://app.codacy.com/gh/eustomaqua/ApproxBias/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d0f9d3235ebf4454b3f43beb137bb2c7)](https://app.codacy.com/gh/eustomaqua/ApproxBias/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) 

<!--
Does machine bring in extra bias in learning? Approximating discrimination within models quickly 
sa_val = [set(A[:, i]) for i in range(A.shape[1])]
sa_idx = [[A[:,i]==j for j in sa_val[i]] for i in range(A.shape[1])]
D, _ = DirectDist_bin(X_nA_y, A[:, 0]==priv_val)
Df, _ = DirectDist_bin(X_nA_fx, A[:, 0]==priv_val)
D, _ = ApproxDist_bin(X_nA_y, A[:, 0], A[:, 0]==priv_val, m1, m2)
Df, _ = ApproxDist_bin(X_nA_fx, A[:, 0], A[:, 0]==priv_val, m1, m2)

- (non-archival summary) 
The proposed fairness measure is called *harmonic fairness measure via manifolds (HFM)* with three 
We provide the evaluation of extra discrimination for three cases: 1) only one bi-valued sensitive attribute (sen-att); 2) one multi-valued sen-att; and 3) more than one sen-att. Among them, case 1 comes from \[**P1**\], and two others come from \[**P2**\]. 
Note that the computation of HFM  # a brief example , quick tutorial
-->

Here we release the code of proposed methods from our following papers
- \[**P1**\] *Does Machine Bring in Extra Bias in Learning? Approximating Fairness in Models Promptly* [[arXiv]](https://arxiv.org/abs/2405.09251)
- \[**P2**\] *Approximating Discrimination Within Models When Faced With Several Non-Binary Sensitive Attributes* [[arXiv]](https://arxiv.org/abs/2408.06099)


## Getting started

<!--
### Instructions


# Load data: X, A, y
# Then get X_nA_y, sa_val, sa_idx, and X_nA_fx

# Load data: X, A, y; etc.
# Load data: X, A, y; etc.


### Citing this work
-->

We proposed a fairness measure named *harmonic fairness measure via manifolds (HFM)* with three optional versions, which deals with a fine-grained discrimination evaluation for one or more sensitive attributes (sen-att-s). HFM relies on the Euclidean Hausdorff distance, of which the direct computation is rather heavy. To accelerate the distance computation, we further proposed a few approximation algorithms for efficient bias evaluation.

In other words, we provide the evaluation of extra discrimination for three cases: 1) only one bi-valued sensitive attribute (sen-att); 2) one multi-valued sen-att; and 3) more than one sen-att. Among them, case 1 comes from \[**P1**\], and two others come from \[**P2**\]. Here is a [short tutorial](#Examples) covering all the aforementioned cases and methods.



### Requirements

We developed it with **Python=3.8** and also tested it with **Python=3.11** at the time of release. Note to choose the `requirements.txt` accordingly. 


```shell
# Create a virtual environment
conda create -n test python=3.11  # or 3.8
source activate test

# Install packages
pip install --upgrade pip
pip install -r requirements.txt   # Python 3.11
# pip install -r reqs_dev.txt     # Python 3.8
# python -m pytest

# Delete the virtual environment
source deactivate
conda remove -n test --all
```



### Examples

<!--
# pip install carbontracker==1.2.5
# python -m pytest .

D, _ = DirectDist_bin(X_nA_y, sa_idx[k][0])
Df, _ = DirectDist_bin(X_nA_fx, sa_idx[k][0])
df_max, _ = bias_degree_bin(D[0], Df[0])
df_avg, _ = bias_degree_bin(D[1], Df[1])  # not in [P1], added later

D, _ = ApproxDist_bin(X_nA_y, A[:, k], sa_idx[k][0], m1, m2)
Df, _ = ApproxDist_bin(X_nA_fx, A[:, k],sa_idx[k][0], m1, m2)
from hfm.dist_est_nonbin import ApproxDist_nonbin_mpver
-->

You may need to adjust the forms of the data you use as follows:

```python
# Load data: X, A, y, f(x)
#   X: non-sen-att, shape=(#, #non-sen-att)
#   A: sen-att, shape=(#, #sen-att)
#   y: label, shape=(#,)
#   f(x): prediction, shape=(#,)

# param priv_val: indicating the privileged group
X_nA_y = np.concatenate([y.reshape(-1, 1).astype('float'), X], axis=1)
sa_val = [set(A[:, i]) for i in range(A.shape[1])]
sa_val = [[priv_val]+list(i - set({priv_val})) for i in sa_val]
sa_idx = [[A[:, i] == k for k in j]  for i, j in enumerate(sa_val)]
X_nA_fx = np.concatenate([fx.reshape(-1, 1).astype('float'), X], axis=1)
```

Here are examples of three aforementioned cases respectively:

```python
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
```

```python
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
```

```python
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
```

<!--
You're free to adjust the parameters if needed or for potential better results. 

PS. Mistakes may exist in the current version. Feel free to contact us or submit a pull request please if you find any. 
-->

You're welcome to adjust the parameters (except `priv_val`, which depends on the data you use) as needed or to explore potential improvements. Please note that this version may contain typos or errors; If you find any, feel free to contact us or submit a pull request.



## Additional information

Kindly cite our work please if you find this repository useful.

```bibtex
@article{bian2024does,
  author  = {Bian, Yijun and Luo, Yujie},
  title   = {Does Machine Bring in Extra Bias in Learning? Approximating Fairness in Models Promptly},
  journal = {arXiv preprint arXiv:2405.09251},
  year    = {2024},
}

@article{bian2024approximating,
  author  = {Bian, Yijun and Luo, Yujie and Xu, Ping},
  title   = {Approximating Discrimination Within Models When Faced With Several Non-Binary Sensitive Attributes},
  journal = {arXiv preprint arXiv:2408.06099},
  year    = {2024},
}
```

### Licence

*ApproxBias* is released under the [MIT Licence](./LICENSE).

