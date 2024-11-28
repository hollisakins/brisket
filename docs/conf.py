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
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
]


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

automodapi_inheritance_diagram = False
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

source_suffix = '.rst'

master_doc = 'index'

# -- Options for HTML output

html_theme = 'furo'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['.DS_Store', '**.ipynb_checkpoints']

edit_on_github_project = 'hollisakins/brisket'
edit_on_github_branch = 'master'
# edit_on_github_src = 'docs/'
