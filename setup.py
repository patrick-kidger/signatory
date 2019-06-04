import os
import setuptools
import sys

name = 'torchtest'
description = 'Testing build an extension to deep learning frameworks'

# We assert that the {name} folder and {name}/__init__.py file exist. We have to keep the name in sync with
# whatever we call the package.
top_python_package = os.path.join(os.path.dirname(__file__), name)
if not os.path.isdir(top_python_package):
    raise SystemExit('Misconfigured name.')
if not os.path.isfile(os.path.join(top_python_package, '__init__.py')):
    raise SystemExit('Misconfigured name.')

# Find out what framework we're building for: we expect one or more of these on the command line when running setup.py
# e.g. `python setup.py install -t` will install for pytorch
type_flags = {'--torch':      'torch',
              '-t':           'torch',
              '--pytorch':    'torch',
              '-pt':          'torch',
              '--tensorflow': 'tensorflow',
              '-tf':          'tensorflow',
              '--numpy':      'numpy',
              '-np':          'numpy'}

selected_type_flags = set()
for i, arg in enumerate(sys.argv):
    try:
        _type_flag = type_flags[arg]
    except KeyError:
        pass
    else:
        selected_type_flags.add(_type_flag)
        del sys.argv[i]

if len(selected_type_flags) == 0:
    raise SystemExit('Must specify a type flag')

with open('VERSION', 'r') as f:
    version = f.read()

with open('README.rst', 'r') as f:
    readme = f.read()

packages = [name]
ext_modules = []
kwargs = {}
install_requires = []
setup_requires = []

for type_flag in selected_type_flags:
    if type_flag == 'torch':
        import torch.utils.cpp_extension as cpp
        include_dirs = cpp.include_paths()
        kwargs['cmdclass'] = {'build_ext': cpp.BuildExtension}
    elif type_flag == 'tensorflow':
        raise SystemExit('TensorFlow not yet supported.')
    elif type_flag == 'numpy':
        import numpy as np
        include_dirs = [np.get_include()]
        setup_requires.append('pybind11')
    else:
        raise SystemExit('Uh oh, not supposed to be able to get here...')
    packages.append('{name}.{type_flag}'.format(name=name, type_flag=type_flag))
    ext_modules.append(setuptools.Extension(name='{name}.{type_flag}._impl'.format(name=name, type_flag=type_flag),
                                            sources=['src/bind.cpp'],
                                            language='c++',
                                            include_dirs=include_dirs,
                                            define_macros=[('TYPEFLAG', type_flag),
                                                           ('EXTENSION_NAME', '_impl')]))

setuptools.setup(name=name,
                 version=version,
                 description=description,
                 long_description=readme,
                 packages=packages,
                 ext_modules=ext_modules,
                 install_requires=install_requires,
                 setup_requires=setup_requires,
                 **kwargs)
