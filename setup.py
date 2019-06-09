import os
import setuptools
import torch.utils.cpp_extension as cpp

name = 'signatory'

with open('VERSION', 'r') as f:
    version = f.read()

with open('README.rst', 'r') as f:
    readme = f.read()

# We assert that the src/{name} folder and src/{name}/__init__.py file exist. We have to keep the name in sync with
# whatever we call the package.
top_python_package = os.path.join(os.path.dirname(__file__), 'src', name)
if not os.path.isdir(top_python_package):
    raise SystemExit('Top python package does not match given name.')
if not os.path.isfile(os.path.join(top_python_package, '__init__.py')):
    raise SystemExit('Top python package does not have an __init__.py file.')
if readme.split('\n', maxsplit=1)[0].lower() != name:
    raise SystemExit('Name in Readme does not match given name.')

description = readme.split('\n', maxsplit=3)[2]

setuptools.setup(name=name,
                 version=version,
                 description=description,
                 long_description=readme,
                 packages=[name],
                 ext_package=name,
                 package_dir={'': 'src'},
                 ext_modules=[cpp.CppExtension(name='_impl',
                                               sources=['src/pytorchbind.cpp',
                                                        'src/signature.cpp'])],
                 cmdclass={'build_ext': cpp.BuildExtension})
