import os
import sys
import types

sys.path.extend([os.path.abspath('..'),       # import metadata
                 os.path.abspath('../src')])  # import signatory
torch_module = types.ModuleType('torch')
torch_module.Tensor = None
sys.modules['torch'] = torch_module

import metadata


project = metadata.project.title()
copyright = metadata.copyright
author = metadata.author
release = metadata.version

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc']

autodoc_mock_imports = ['{}._impl'.format(metadata.project)]
napoleon_use_admonition_for_examples = True

master_doc = 'index'

# if necessary
# pip install sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
