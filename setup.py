import setuptools
import torch.utils.cpp_extension as cpp

with open('VERSION', 'r') as f:
    version = f.read().strip()

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(name='torchtest',
                 version=version,
                 description='Testing build an extension to PyTorch',
                 long_description=readme,
                 ext_modules=[cpp.CppExtension('torchtest', ['pybind.cpp'])],
                 cmdclass={'build_ext': cpp.BuildExtension})
