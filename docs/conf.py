import os
import sys


sys.path.extend([os.path.abspath('mock'),     # import torch, numpy
                 os.path.abspath('..'),       # import metadata
                 os.path.abspath('../src')])  # import signatory

import metadata


project = metadata.project.title()
copyright = metadata.copyright
author = metadata.author
version = release = metadata.version

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.autosummary']

napoleon_use_admonition_for_examples = True
autodoc_mock_imports = ['{}._impl'.format(metadata.project)]
autodoc_member_order = 'bysource'
intersphinx_mapping = {'torch': ('https://pytorch.org/docs/stable/', None)}

master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


def setup(app):
    app.add_css_file("theme_overrides.css")
