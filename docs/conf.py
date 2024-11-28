# -- Path
import sys, os
# sys.path.insert(0, os.path.abspath("../../brisket"))
sys.path.insert(0, os.path.abspath("../../"))
# sys.path.insert(0, os.path.abspath("../brisket")) 
# sys.path.insert(0, os.path.abspath("../"))

import brisket 

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'BRISKET'
copyright = '2024, Hollis Akins'
author = 'Hollis Akins'

release = '0.1'
version = '0.1.0'



# -- General configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.napoleon',
    "sphinx.ext.autodoc",  # core library for html generation from docstrings
    "sphinx.ext.autosummary",  # create neat summary tables
    # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.viewcode",
    # Automatically document param types (less noise in class signature)
    "sphinx_autodoc_typehints",
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
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
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autodoc_typehints = 'description'
# autoapi_dirs = ['../brisket']
# autoapi_add_toctree_entry = False

html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
set_type_checking_flag = (
    True  # Enable 'expensive' imports for sphinx_autodoc_typehints
)
nbsphinx_allow_errors = True  # Continue through Jupyter errors

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


html_title = 'BRISKET'