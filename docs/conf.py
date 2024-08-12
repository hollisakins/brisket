# Configuration file for the Sphinx documentation builder.

# -- Project information
import sys, os
sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file

project = 'BRISKET'
copyright = '2024, Hollis Akins'
author = 'Hollis Akins'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.autodoc',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.intersphinx',
]
# autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autodoc_typehints = 'description'
# autoapi_dirs = ['../brisket']
# autoapi_add_toctree_entry = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
