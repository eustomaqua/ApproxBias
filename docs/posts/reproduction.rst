.. reproduction.rst



Reproduction
===============
.. toctree::
   :maxdepth: 1


The `experimental data <https://github.com/eustomaqua/ApproxBias/tree/master/findings>`_ are released with ApproxBias.


To reproduce our empirical results for **binary cases** [#P1]_, you may collect `data <https://doi.org/10.5281/zenodo.21263075>`_ and do the following.

.. code-block:: console
  :linenos:

  $ python hfm_bin_draw.py -nk 5 -exp rept_expt5a -m1 20     # Fig. 7
  $ python hfm_bin_draw.py -nk 5 -exp rept_expt5b -m2 8      # Fig. 7
  $ python hfm_bin_draw.py -exp mCV_expt2c -pre min_max      # Fig. 2, 6, 9
  $ python hfm_bin_draw.py -exp mCV_expt2a -pre min_max -re  # Fig. 5
  $ python hfm_bin_draw.py -v ver2 -exp mCV_exp1b            # Fig. 3, 4
  $ python hfm_bin_draw.py -v ver4 -exp mCV_expt8h           # Fig. 8
  $ python hfm_bin_draw.py -v ver4 -exp mCV_expt8g -pre standard
  $ python hfm_bin_draw.py -v ver4 -exp mCV_expt8g -pre normalize
  $ python hfm_bin_draw.py -v ver4 -exp mCV_expt8g -pre min_max
  $ python hfm_bin_draw.py -v ver4 -exp mCV_expt8g -pre none # Fig. 10

.. # Fig.5; Fig. 2,4,6, Tables 2-3; Fig.1; Fig.3
.. To reproduce our empirical results for **binary cases** [#P1]_, you may do as follows.


To reproduce our empirical results for **non-binary cases** [#P2]_, you may do as follows.

.. $ # cd ~/ApproxBias
.. $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp9i   # Fig. 1, 2, 4 & Table 2
.. $ python hfm_nonbin_draw.py -rev -ratio .97 -exp rept_exhp5a  # Fig. 3(a-f), 5(a-f)
.. $ python hfm_nonbin_draw.py -rev -ratio .97 -exp rept_exhp5b  # Fig. 3(g-l), 5(g-l)
.. $ python hfm_nonbin_draw.py -exp mCV_expt4b                   # Fig. 7
.. $ python hfm_nonbin_draw.py -exp rept_expt7a -m1 20 -m2 8     # Fig. 9
.. $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp1b   # Fig. 6
.. $ python hfm_nonbin_draw.py -exp rept_expt5a -m1 20           # Fig. 8(a-d)
.. $ python hfm_nonbin_draw.py -exp rept_expt5b -m2 8            # Fig. 8(e-h)
.. $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp2c   # Fig. 10


.. code-block:: console
  :linenos:

  $ # cd ~/ApproxBias
  $ python hfm_nonbin_draw.py -exp mCV_expt4b            # Fig. 1, 2, 5 & 4; Tables 3 & 4
  $ python hfm_nonbin_draw.py -exp rept_expt5a -m1 20          # Fig. 6 & 7
  $ python hfm_nonbin_draw.py -exp rept_expt5b -m2 8           # Fig. 6 & 7
  $ python hfm_nonbin_draw.py -exp rept_expt7a -m1 20 -m2 8    # Fig. 8
  $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp1b  # Fig. 9 & 4
  $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp2c  # Fig. 10
  $ python hfm_nonbin_draw.py -rev -ratio .97 -exp mCV_rexp3e  # Fig. 3

.. # To get Fig. 1, 2, 5 & 4, as well as Table 3



.. [#P2] Fast discrimination assessment for multiple non-binary sensitive attributes (arXiv preprint `2408.06099 <https://arxiv.org/pdf/2408.06099>`_)
.. [#P1] Measuring model-induced discrimination via efficient fairness approximation (arXiv preprint `2405.09251 <https://arxiv.org/pdf/2405.09251>`_, doi `10.1109/TNNLS.2026.3706648 <https://doi.org/10.1109/TNNLS.2026.3706648>`_, data `10.5281/zenodo.21263075 <https://doi.org/10.5281/zenodo.21263075>`_)


.. [P2 Approximating discrimination within models when faced with several non-binary sensitive attributes (arXiv preprint https://arxiv.org/pdf/2408.06099)
.. [P1 Does machine bring in extra bias in learning? Approximating fairness in models promptly (arXiv preprint https://arxiv.org/pdf/2405.09251)
