.. documentation.rst



Documentation
===============
.. toctree::
   :maxdepth: 2


.. Proposed HFM
.. ------------



Performance-based metrics
-------------------------


**The most commonly used performance measures in classification tasks**

.. 1. Error rate :math:`\mathrm{err}= \frac{1}{n}\sum_{i=1}^n \mathbb{I}(f(\mathbf{x}_i) \neq y_i)`
.. 2. Accuracy :math:`\mathrm{acc}= \frac{1}{n}\sum_{i=1}^n \mathbb{I}(f(\mathbf{x}_i) =y_i) =1-\mathrm{err}`
.. This is the confusion matrix of binary classification

1. Error rate

   .. math::
      \mathrm{err}\triangleq
      \frac{1}{N}\sum_{i=1}^N 
      \mathbb{I}(f(\mathbf{x}_i) \neq y_i)

2. Accuracy

   .. math::
      \begin{align}
      \mathrm{acc}&\triangleq \frac{1}{N}\sum_{i=1}^N \mathbb{I}(f(\mathbf{x}_i) =y_i)
      \\ &= 1-\mathrm{err}
      \end{align}


.. note::
   - :math:`N` is the total number of samples.


**In practice, people also care about**

3. Precision 

   .. math::
      P\triangleq \frac{TP}{TP+FP}

4. Recall 

   .. math::
      R\triangleq \frac{TP}{TP+FN}

5. F1-measure

   .. math::
      \begin{align}
      F1 &\triangleq \frac{2\times P\times R}{P+R}
      \\ &= \frac{2\times TP}{N+TP-TN}
      \end{align}

.. note::
   - The **confusion matrix** of binary classification

   +--------------------+----------+----------+
   | Ground-truth class | Predicted class     |
   +                    +----------+----------+
   | (aka. label)       | Positive | Negative |
   +====================+==========+==========+
   | Positive           |  TP      |  FN      |
   +--------------------+----------+----------+
   | Negative           |  FP      |  TN      |
   +--------------------+----------+----------+


6. The general form of :math:`F1`-measure is :math:`F_\beta`

   .. math::
      F_\beta= \frac{
         (1+\beta^2)\times P\times R
      }{
         (\beta^2\times P)+R
      }


.. seealso::
   F1 is the harmonic mean of precision and recall,

   .. math::
      \frac{1}{F1} =\frac{1}{2}
      \cdot\left(
         \frac{1}{P} +\frac{1}{R}
      \right) \,

   :math:`F_\beta` is the weighted harmonic mean,

   .. math::
      \frac{1}{F_\beta}= \frac{1}{1+\beta^2}
      \cdot\left(
         \frac{1}{P}+ \frac{\beta^2}{R}
      \right) \,


**If data imbalance is involved, people may care about**

7. *Sensitivity* (the larger the better), aka. *recall*, true positive rate (TPR), or hit rate
   
   .. math::
      \begin{align}
      \mathrm{TPR} &= \mathrm{sen}
      = R \\& \triangleq
      \frac{TP}{TP+FN}
      \end{align}

8. *Specificity* (the larger the better)
   
   .. math::
      \begin{align}
      \mathrm{spe} &
      =  1- \mathrm{FPR}
      \end{align}


8. False positive rate (FPR), aka. false alarm (the smaller the better)
   
   .. math::
      \mathrm{FPR} \triangleq \frac{FP}{TN+FP}

9. miss rate (the smaller the better)
   
   .. math::
      \mathrm{mis}= 1- \mathrm{TPR}

..       \begin{align}
..       \mathrm{} &= \frac{FN}{TP+FN}
..       \\ &= 1-\mathrm{TPR}
..       \end{align}


Fairness-relevant measures
--------------------------

