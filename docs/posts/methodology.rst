.. methodology.rst


Methodology
================


.. This work is about how to evaluate the discrimination level of ML models from a manifold perspective, that is, *if the instances (with the same sensitive attribute values) are viewed as data points on certain manifolds, the manifold(s) representing members from the marginalized/unprivileged group(s) are supposed to be as close as possible to that representing members from the privileged group*. 

We introduced two fairness measures to deal with three cases: 1) when facing one bi-valued sensitive attribute (sen-att); 2) when facing one multi-valued sen-att; and 3) when facing several sensitive attributes (sen-att-s). The proposed fairness measures are built upon the Euclidean Hausdorff distance of sets, to describe the distance between manifolds. 

.. They, introduced in mathematics.
.. https://web.stanford.edu/class/cs273/scribing/2004/class8/scribe8.pdf
.. https://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html





.. A dataset is denoted by :math:`S` where the 
.. drawn from an feature-label space :math:`\mathcal{X\times Y}` based on an unknown distribution. 
.. that is, finite :math:`\mathcal{Y}=\{1,2,...,n_c\} (n_c\geqslant 2)`, where :math:`n_c` is the number of labels.
.. The feature/
.. , depending on the number of labels :math:`n_c`.

.. note::
  
  **Notations:** 
  Given a dataset :math:`S=\{(\mathbf{x}_i,\mathbf{a}_i,y_i)\mid i=1,2,...,n\}` composed of i.i.d. instances including sen-att-s, we denote one instance by :math:`(\mathbf{x,a})=[x_1,...,x_{n_d},a_1,...,a_{n_a}]^\mathsf{T}` for clarity.

Therein, :math:`n_a\geqslant 1` is the number of sensitive/protected attributes and :math:`n_d` is that of unprotected attributes in the instance. 
Note that :math:`a_i=1` means the corresponding instance is a member from the privileged group, and :math:`i\in[n]` is used to represent :math:`i\in\{1,2,...,n\}` for brevity. 
The instances in :math:`S` are i.i.d. (independent and identically distributed); the task is binary or multi-class classification. 

.. [#footnote]_ 
.. . [#footnote] (i.i.d.) independent and identically distributed
.. Therein, 
.. We use :math:`a_i=1` to denote that the corresponding instance is a member from the privileged group. 
.. Note that :math:`i\in[n]` is used to represent :math:`i\in\{1,2,...,n\}` for brevity. 


Distance between sets
^^^^^^^^^^^^^^^^^^^^^

1. For one bi-valued sen-att :math:`\mathbf{a}=[a_i]^\mathsf{T}` and :math:`a_i\in\mathcal{A}_i=\{0,1\}`,

  - The dataset :math:`S` can be divided into two subsets :math:`S_1=\{(\mathbf{x},a_i,y)\in S\mid a_i=1\}` and :math:`\bar{S}_1=S\setminus S_1=\{(\mathbf{x},a_i,y)\in S\mid a_i\neq 1\}`
  - Given a specific distance metric :math:`\mathbf{d}(\cdot,\cdot)` (e.g. the standard Euclidean metric) on the feature space, the distance between these two subsets (that is, :math:`S_1` and :math:`\bar{S}_1`) is

  .. math::
     \begin{align}
     \mathbf{D}(S_1,\bar{S}_1)= \max\Big\{ 
     & \max_{(\mathbf{x,a},y)\in S_1} \min_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \mathbf{d}\big( (\mathbf{x},y),(\mathbf{x}',y') \big) ,\\
     & \max_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \min_{(\mathbf{x,a},y)\in S_1} \mathbf{d}\big( (\mathbf{x},y),(\mathbf{x}',y') \big) 
     \Big\}
     \end{align}

  - For a trained classifier :math:`f(\cdot)` where :math:`\hat{y}=f(\mathbf{x,a})`,

  .. math::
     \begin{align}
     \mathbf{D}_f(S_1,\bar{S}_1)= \max\Big\{ 
     & \max_{(\mathbf{x,a},y)\in S_1} \min_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \mathbf{d}\big( (\mathbf{x},\hat{y}),(\mathbf{x}',\hat{y}') \big) ,\\
     & \max_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \min_{(\mathbf{x,a},y)\in S_1} \mathbf{d}\big( (\mathbf{x},\hat{y}),(\mathbf{x}',\hat{y}') \big) 
     \Big\}
     \end{align}

  - By recording the true label :math:`y` and the prediction :math:`\hat{y}` as one denotation (say :math:`\ddot{y}`) for simplification, we can rewrite them as

  .. math::
     \begin{align}
     \mathbf{D}_{\cdot}(S_1,\bar{S}_1)= \max\Big\{ 
     & \max_{(\mathbf{x,a},y)\in S_1} \min_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \mathbf{d}\big( (\mathbf{x},\ddot{y}),(\mathbf{x}',\ddot{y}') \big) ,\\
     & \max_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_1} \min_{(\mathbf{x,a},y)\in S_1} \mathbf{d}\big( (\mathbf{x},\ddot{y}),(\mathbf{x}',\ddot{y}') \big) 
     \Big\}
     \end{align}


2. For one multi-valued sen-att :math:`a_i\in\mathcal{A}_i=\{1,2,...,n_{a_i}\}`, where :math:`n_{a_i}\geqslant 3`

  - The dataset :math:`S` can be divided into a few disjoint sets according to the value of this attribute :math:`a_i`, that is, :math:`S_j=\{(\mathbf{x},a_i,y)\in S\mid a_i=j\} ,\forall j\in\mathcal{A}_i`
  - The distance above can be extended to: i) *maximal distance measure for one sen-att*, and ii) *average distance measure for one sen-att*

  .. math::
     \begin{align}
     \mathbf{D}_{\cdot,\mathbf{a}}(S,a_i) &= 
     \max_{1\leqslant j\leqslant n_{a_i}}\Big\{
     \max_{(\mathbf{x,a},y)\in S_j} \min_{(\mathbf{x}',\mathbf{a}',y')\in \bar{S}_j} \mathbf{d}\big(
     (\mathbf{x},\ddot{y}), (\mathbf{x}',\ddot{y}')
     \big) \Big\} \\
     \mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S,a_i) &=
     \frac{1}{n} \sum_{j=1}^{n_{a_i}} \sum_{(\mathbf{x,a},y)\in S_j} \min_{(\mathbf{x}',\mathbf{a}',y')\in\bar{S}_j}
     \mathbf{d}\big(
     (\mathbf{x},\ddot{y}), (\mathbf{x}',\ddot{y}')
     \big)
     \end{align}

  - Notice that :math:`\bar{S}_j=S\setminus S_j`, and :math:`\mathbf{D}_{\cdot,\mathbf{a}}(S,a_i)= \mathbf{D}_\cdot(S_1,\bar{S}_1)` when :math:`\mathcal{A}_i=\{0,1\}`


3. For more than one sen-att :math:`\mathbf{a}=[a_1,...,a_{n_a}]^\mathsf{T}`, where :math:`n_a\geqslant 2` and :math:`n_{a_i}\geqslant 2`

  - The generalised distance measures include: i) *maximal distance measure for sen-att-s*, and ii) *average distance measure for sen-att-s*

  .. math::
     \begin{align}
     \mathbf{D}_{\cdot,\mathbf{a}}(S) &=
     \max_{1\leqslant i\leqslant n_a}
     \mathbf{D}_{\cdot,\mathbf{a}}(S,a_i) \\
     \mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S) &=
     \frac{1}{n_a} \sum_{i=1}^{n_a}
     \mathbf{D}_{\cdot,\mathbf{a}}^\text{avg}(S,a_i)
     \end{align}



Model fairness assessment
^^^^^^^^^^^^^^^^^^^^^^^^^

For one bi-valued sen-att [#P1]_, we remark :math:`\mathbf{D}(S_1,\bar{S}_1)` reflects the biases from the data and :math:`\mathbf{D}_f(S_1,\bar{S}_1)` reflects the biases from the algorithm. Then the following value could be used to reflect the fairness degree of this classifier, that is,

.. math::
  \mathbf{df}_\text{prev}(f)= \frac{ \mathbf{D}_f(S_1,\bar{S}_1) }{ \mathbf{D}(S_1,\bar{S}_1) }-1

For multi-valued sen-att-s [#P2]_, we remark that :math:`\mathbf{D}_{\mathbf{a}}(S), \mathbf{D}_{\mathbf{a}}^\text{avg}(S)` reflect the biases from the data and :math:`\mathbf{D}_{f,\mathbf{a}}(S), \mathbf{D}_{f,\mathbf{a}}^\text{avg}(S)` reflect the biases in the learning algorithm. Then the following value could be used to reflect the fairness degree of this classifier, that is,

.. math::
  \begin{align}
  \mathbf{df}(f) &= \log\left(
  \frac{ \mathbf{D}_{f,\mathbf{a}}(S) }{ \mathbf{D}_{\mathbf{a}}(S) }
  \right) \\
  \mathbf{df}^\text{avg}(f) &= \log\left(
  \frac{ \mathbf{D}_{f,\mathbf{a}}^\text{avg}(S) }{ \mathbf{D}_{\mathbf{a}}^\text{avg}(S) }
  \right)
  \end{align}

.. [#P1] Does Machine Bring in Extra Bias in Learning? Approximating Fairness in Models Promptly https://arxiv.org/abs/2405.09251
.. [#P2] Approximating Discrimination Within Models When Faced With Several Non-Binary Sensitive Attributes https://arxiv.org/abs/2408.06099


.. Quick approximation of distances
.. --------------------------------

Approximation of distances
^^^^^^^^^^^^^^^^^^^^^^^^^^







.. hint::
   
  hello

