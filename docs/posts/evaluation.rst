.. evaluation.rst


.. Proposed HFM
.. ------------


Model evaluation
================



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
      \\&= R \\& \triangleq
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

.. ; These criteria are rarely satisfied all at once.
.. \leqslant\epsilon
.. \leqslant \tau=0.8
.. f(\mathbf{x}_i)=1 \mid y_i=1 , \mathbf{x}_i\in S
.. f(\mathbf{x})=1 \mid \mathbf{a}=0

There are three statistical non-discrimination criteria [#B1]_: independence, separation, and sufficiency. We choose some commonly used group fairness measures listed here.

- To meet the *independence* criterion,

  1. Demographic/statistical parity (DP) [#R1c]_:sup:`,` [#R1b]_

  .. math::
     \begin{align}
     \mathrm{Pr}\{
      \hat{y}=1 \mid \mathbf{a}=0
     \} =\mathrm{Pr}\{
      \hat{y}=1 \mid \mathbf{a}=1
     \}
     \end{align}

  2. Disparate impact (i.e. "80% rule") [#R1a]_

  .. math::
     \frac{
      \mathrm{Pr}\{ \hat{y}=1 \mid \mathbf{a}=0 \}
     }{
      \mathrm{Pr}\{ \hat{y}=1 \mid \mathbf{a}=1 \}
     } \geqslant \tau=0.8

- To meet the *separation* criterion,

  3. Equality of opportunity (EO) [#R2]_:sup:`,` [#R1b]_

  .. math::
     \mathrm{Pr}\{
     \hat{y}=1 \mid y=1, \mathbf{a}=0
     \} =
     \mathrm{Pr}\{
     \hat{y}=1 \mid y=1, \mathbf{a}=1
     \}

  4. Equalised odds [#R2]_

  .. math::
     \mathrm{Pr}\{
      \hat{y}=1 \mid y, \mathbf{a}=0
     \} =\mathrm{Pr}\{
      \hat{y}=1 \mid y, \mathbf{a}=1
     \} ,\, \forall y\in\{0,1\}

- To meet the *sufficiency* criterion,

  5. Predictive parity (PP) [#R3a]_:sup:`,` [#R3b]_

  .. math::
     \mathrm{Pr}\{
      y=1 \mid \hat{y}=1 , \mathbf{a}=0
     \} =\mathrm{Pr}\{
      y=1 \mid \hat{y}=1 , \mathrm{a}=1
     \}


.. note::
    The above measures are used for one single sen-att cases in binary classification. 

    Here we may have a dataset :math:`S=\{(\mathbf{x}_i,y_i)\}_{i=1}^n` where :math:`\mathbf{x}` includes one sen-att :math:`\mathbf{a}`, and :math:`\mathbf{a}=1` represents the privileged group. Note that :math:`\hat{y}` denotes the classification predictions of :math:`f(\cdot)`.

.. Note that :math:`\mathbf{a}=1` and :math:`0` represents the privileged and marginalised group(s) respectively.
.. . . [#R1b] Gajane, Pratik, and Mykola Pechenizkiy. "On formalizing fairness in prediction with machine learning." arXiv preprint arXiv:1710.03184, 2017.
.. . . [#R1b] Gajane, Pratik, and Mykola Pechenizkiy. "On formalizing fairness in prediction with machine learning." In the 5th Workshop on Fairness, Accountability, and Transparency in Machine Learning (FAT/ML), 2018.

.. [#B1] Barocas, S., Hardt, M., and Narayanan, A. Fairness and machine learning. fairmlbook.org, 2019. (Chapters 3.4--3.10, cf. https://fairmlbook.org/)
.. [#R1c] Dwork C, Hardt M, Pitassi T, Reingold O, & Zemel R (2012, January). Fairness through awareness. In proceedings of the 3rd innovations in theoretical computer science conference (pp. 214-226).
.. [#R1b] Gajane P, & Pechenizkiy M (2018). On formalizing fairness in prediction with machine learning. In the 5th workshop on fairness, accountability, and transparency in machine learning (FAT/ML).
.. [#R1a] Feldman M, Friedler SA, Moeller J, Scheidegger C, & Venkatasubramanian S (2015, August). Certifying and removing disparate impact. In proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining (pp. 259-268).
.. [#R2] Hardt M, Price E, & Srebro N (2016). Equality of opportunity in supervised learning. Advances in neural information processing systems, 29.
.. [#R3a] Chouldechova A (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big data, 5(2), 153-163.
.. [#R3b] Verma S, & Rubin J (2018, May). Fairness definitions explained. In proceedings of the international workshop on software fairness (pp. 1-7).
