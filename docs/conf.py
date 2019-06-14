import os
import sys


# I've tried a bunch of other approaches and none of them build docs properly... to investigate.
class torch:
    class Tensor:
        pass

    class autograd:
        class Function:
            pass

    class nn:
        class Module:
            pass

        class functional:
            class relu:
                pass


sys.path.extend([os.path.abspath('..'),       # import metadata
                 os.path.abspath('../src')])  # import signatory


sys.modules['torch'] = torch
sys.modules['torch.autograd'] = torch.autograd
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional

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
