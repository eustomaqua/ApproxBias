.. ApproxBias documentation master file, created by
   sphinx-quickstart on Fri Mar 14 15:07:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ApproxBias documentation
========================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.
..
..

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Welcome to the ApproxBias's documentation. This is to help you reproduce our work from

#. Does Machine Bring in Extra Bias in Learning? Approximating Fairness in Models Promptly `[arXiv 2405.09251] <https://arxiv.org/abs/2405.09251>`_
#. Approximating Discrimination Within Models When Faced With Several Non-Binary Sensitive Attributes `[arXiv 2408.06099] <https://arxiv.org/abs/2408.06099>`_

We proposed a fairness measure named *harmonic fairness measure via manifolds (HFM)* with three optional versions, which deals with a fine-grained discrimination evaluation for one or more sensitive attributes (sen-att-s). *HFM* relies on the Euclidean Hausdorff distance, of which the direct computation is rather heavy. To accelerate the distance computation, we further proposed a few approximation algorithms for efficient bias evaluation.

- To get started quickly, see :doc:`an example <posts/quickstart>` here
- To understand the methodology, see :doc:`methodology <posts/methodology>`
- To reproduce the empirical results, see :doc:`instructions <posts/reproduction>`
.. - To learn the model evaluation we use, see :doc:`evaluation <posts/evaluation>`


.. - To reproduce our results, see :doc:`reproduction <posts/reproduction>`
.. - To dive into more details, see :doc:`documentation <posts/documentation>`

.. :doc:`methodology.rst <static/methodology>`
.. Welcome to the ApproxBias's documentation. This is to help you reproduce our work in


--------

.. Taking a short cut
.. Shortcuts
.. ==================

SHORTCUTS
^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   posts/quickstart.rst
   posts/methodology.rst
   posts/reproduction.rst

..    posts/evaluation.rst

..    posts/documentation.rst
.. static/methodology.rst



.. .. toctree: :
..    :hidden:
..    :caption: ApproxBias
..    :caption: Reproduction
..
..    posts/reproduction.rst
.. Sphinx Theme Gallery <https://sphinx-themes.org>

