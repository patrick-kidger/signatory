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


if not (sys.platform.startswith('darwin') or sys.platform.startswith('linux')):
    # To be able to build on Windows we need to be able to use CL.
    # So we have to prefix every call to run or check_output with something to run the appropriate vcvars file.
    # This is pretty fragile. If the implementation of torch.utils.cpp_extension changes then this may well break, but
    # it doesn't look like there are better options.
    # Even the if statement above is just chosen to mimic the checks made in torch.utils.cpp_extension.

    class subprocess_proxy(object):
        def __init__(self):
            import os
            import subprocess
            try:
                # Python 2
                self.stringtype = basestring
            except NameError:
                # Python 3
                self.stringtype = str

    def modify_args(self, args):
        try:
            vcvars_location = os.environ['SIGNATORY_VCVARS']
        except KeyError:
            return args
        if vcvars_location[0] != '"':
            vcvars_location = '"' + vcvars_location
        if vcvars_location[-1] != '"':
            vcvars_location = vcvars_location + '"'
        if isinstance(args, self.stringtype):
            args = '{} && '.format(vcvars_location) + args
        elif isinstance(args, (tuple, list)):
            args.insert(0, vcvars_location)
        else:
            raise ValueError("args must be a string or list.")
        return args

    def check_output(self, args, **kwargs):
        args = self.modify_args(args)
        return subprocess.check_output(args, **kwargs)

    def run(self, args, **kwargs):
        args = self.modify_args(args)
        return subprocess.run(args, **kwargs)

    def __getattr__(self, item):
        if item == 'run':
            return self.run
        elif item == 'check_output':
            return self.check_output
        else:
            return getattr(subprocess, item)

    cpp.subprocess = subprocess_proxy()


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
                                extra_compile_args=['-fvisibility=hidden'])]
# fvisibility flag because of https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes


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
