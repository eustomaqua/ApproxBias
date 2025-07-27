.. reproduction.rst



Reproduction
===============
.. toctree::
   :maxdepth: 1


The `experimental data <https://github.com/eustomaqua/ApproxBias/tree/master/findings>`_ are released with ApproxBias.

To reproduce our empirical results for **non-binary cases** [#P2]_, you may do as follows.

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



.. [#P2] Approximating discrimination within models when faced with several non-binary sensitive attributes (arXiv preprint https://arxiv.org/pdf/2408.06099)
.. [#P1] Does machine bring in extra bias in learning? Approximating fairness in models promptly (arXiv preprint https://arxiv.org/pdf/2405.09251)

