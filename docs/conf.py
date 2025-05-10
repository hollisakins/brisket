import os, sys

project = "brisket"
release = '0.1.0'

sys.path.insert(0, os.path.abspath("../brisket"))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

extensions = [
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",  # core library for html generation from docstrings
    "sphinx.ext.autosummary",  # create neat summary tables
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    # "sphinx_gallery.gen_gallery",
    "sphinx_toolbox.collapse",
    "sphinx_copybutton",  # Add a copy button to code blocks
]


autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_member_order = 'bysource'

templates_path = ["_templates"]

master_doc = "index"

html_theme = "furo"
html_title = "BRISKET"
