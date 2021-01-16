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
"""The metadata for the project."""


import io
import os
import re
#### DO NOT IMPORT NON-(STANDARD LIBRARY) MODULES HERE


project = 'signatory'
author = "Patrick Kidger"
copyright = "2019, {}".format(author)
author_email = "contact@kidger.site"
url = "https://github.com/patrick-kidger/signatory"
license = "Apache-2.0"
python_requires = "~=3.6"
keywords = "signature"
classifiers = ["Development Status :: 5 - Production/Stable",
               "Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: 3.8",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

description = 'Differentiable computations of the signature and logsignature transforms, on both CPU and GPU.'

here = os.path.realpath(os.path.dirname(__file__))

# for simplicity we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, 'src', project, '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    readme = f.read()
