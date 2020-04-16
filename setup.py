# Copyright 2019 Patrick Kidger. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""setup.py - hopefully you know what this does without me telling you..."""


import setuptools
import sys
try:
    import torch.utils.cpp_extension as cpp
except ImportError:
    raise ImportError("PyTorch is not installed, and must be installed prior to installing Signatory.")
    
import metadata

extra_compile_args = []

# fvisibility flag because of https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
if not sys.platform.startswith('win'):  # linux or mac
    extra_compile_args.append('-fvisibility=hidden')

if sys.platform.startswith('win'):  # windows
    extra_compile_args.append('/openmp')
else:  # linux or mac
    extra_compile_args.append('-fopenmp')

ext_modules = [cpp.CppExtension(name='_impl',
                                sources=['src/logsignature.cpp',
                                         'src/lyndon.cpp',
                                         'src/misc.cpp',
                                         'src/pytorchbind.cpp',
                                         'src/signature.cpp',
                                         'src/tensor_algebra_ops.cpp'],
                                depends=['src/logsignature.hpp',
                                         'src/lyndon.hpp',
                                         'src/misc.hpp',
                                         'src/signature.hpp',
                                         'src/tensor_algebra_ops.hpp'],
                                extra_compile_args=extra_compile_args)]


setuptools.setup(name=metadata.project,
                 version=metadata.version,
                 author=metadata.author,
                 author_email=metadata.author_email,
                 maintainer=metadata.author,
                 maintainer_email=metadata.author_email,
                 description=metadata.description,
                 long_description=metadata.readme,
                 url=metadata.url,
                 license=metadata.license,
                 keywords=metadata.keywords,
                 classifiers=metadata.classifiers,
                 zip_safe=False,
                 python_requires=metadata.python_requires,
                 packages=[metadata.project],
                 ext_package=metadata.project,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
