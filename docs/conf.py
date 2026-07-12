# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# eustomaqua, Yijun

project = 'ApproxBias'
copyright = '2025, Yj'
author = 'Yj'
# author = 'Yijun Bian'
# version = '0.2.1'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# extensions = [
#     'recommonmark',
#     'sphinx.ext.todo',
#     'sphinx.ext.githubpages',
# ]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'piccolo_theme'
# html_theme = 'furo'
html_theme = 'renku'
# html_theme = "sphinx_rtd_theme"
# extensions = ['recommonmark', 'sphinx_markdown_tables']
# html_theme = 'alabaster'
html_static_path = ['_static']
html_show_sourcelink = False
