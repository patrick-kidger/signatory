import os
import sys
import types


# I don't think any other approach can work in general: there's no way for something asking for e.g. torch.Tensor to
# know if that's a class, module, function...
class torch(types.ModuleType):
    class Tensor:
        pass

    class autograd(types.ModuleType):
        class Function:
            pass

    class nn(types.ModuleType):
        class Module:
            pass

        class functional(types.ModuleType):
            def relu():
                pass


sys.modules['torch'] = torch
sys.modules['torch.autograd'] = torch.autograd
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional


sys.path.extend([os.path.abspath('..'),       # import metadata
                 os.path.abspath('../src')])  # import signatory
import metadata


project = metadata.project.title()
copyright = metadata.copyright
author = metadata.author
release = metadata.version

# must have installed py2annotate in order generate docs
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'py2annotate', 'sphinx.ext.intersphinx']

napoleon_use_admonition_for_examples = True
autodoc_mock_imports = ['{}._impl'.format(metadata.project)]
intersphinx_mapping = {'torch': ('https://pytorch.org/docs/stable/', None)}

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
