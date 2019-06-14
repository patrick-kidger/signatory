import os
import sys
import unittest.mock as mock


# Just adding to autodoc_mock_imports results in mocks in the documentation
class Mock(mock.MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return mock.MagicMock()


class Tensor:
    __qualname__ = 'torch.Tensor'


sys.path.extend([os.path.abspath('..'),       # import metadata
                 os.path.abspath('../src')])  # import signatory
torch_mock = Mock()
torch_mock.Tensor = Tensor
sys.modules['torch'] = torch_mock
sys.modules['torch.autograd'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.nn.functional'] = Mock()

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
