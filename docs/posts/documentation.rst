.. documentation.rst



Documentation
===============
.. toctree::
   :maxdepth: 2


**Configuration**

.. code-block:: console
  :linenos:

  $ conda env list
  $ conda create -n py3e python=3.11
  $ source activate py3e
  $ pip uninstall wheel packaging
  $
  $ pip install -U sphinx
  $ # pip install sphinx_rtd_theme
  $ # pip install recommonmark
  $ pip install renku-sphinx-theme
  $
  $ cd docs
  $ sphinx-quickstart
  $ make html
  $
  $ conda deactivate
  $ conda remove -n py3e --all


Reference: `reStructuredText basic`_

.. _reStructuredText basic: https://www.sphinx-doc.org/zh-cn/master/usage/restructuredtext/basics.html
.. https://docs.renkulab.io/en/0.12.7/how-to-guides/index.html

