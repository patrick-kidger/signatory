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
"""This installs Signatory.

See the README.rst for an explanation.
"""


import io
import os
import re
import setuptools
import subprocess
import sys

import metadata
import version


installer_name = metadata.project + '_installer'
installer_description = 'Installs the Signatory project'
here = os.path.realpath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    installer_long_description = f.read()

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is not installed, and must be installed prior to installing Signatory.")


signatory_version_to_install = version.version
if sys.platform in ('win32', 'darwin'):
    tag = '-torch' + torch.__version__
    signatory_version_to_install += tag

    # So we've found the tag corresponding to the version of Signatory we'd like to install.
    # But let's do our best to be forwards compatible, and check whether that version exists and given a friendly error
    # message if it doesn't.

    def check_signatory_version_exists():
        # Crazily, this is one of the best ways to get a list of all possible versions of Signatory
        error_message = subprocess.Popen('pip install signatory==blork', stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, shell=True).stderr.readlines()
        version_re = re.compile(rb'from versions: ([-0-9\. ,torch]*)\)')
        for line in error_message:
            match = version_re.search(line)
            if match:
                break
        else:
            # Something's gone wrong with how we're parsing the message about the versions of Signatory available.
            # We won't attempt anything further, and hope that any of the normal error messages will be informative.
            return

        # Parse versions of Signatory to get the one that we want to install
        available_signatory_versions = [sig_version.decode() for sig_version in match.group(1).split(b', ')]
        if signatory_version_to_install not in available_signatory_versions:
            raise ImportError("\n"
                              "Could not locate suitable Signatory version. This means that precompiled binaries of "
                              "Signatory are not available for the version of PyTorch that you are using.\n"
                              "Detected PyTorch version {torch_version}, and we are seeking Signatory version "
                              "{signatory_version}, so the Signatory version we are seeking should be tagged as "
                              "'{signatory_version_to_install}'. However available Signatory versions are only:\n "
                              "{available_signatory_versions}.\n"
                              "To solve this issue you will probably have to install Signatory from source, by "
                              "running:\n"
                              "    pip install signatory --no-binary signatory\n"
                              "Note that you must have a C++ compiler installed and known to pip."
                              .format(torch_version=torch.__version__,
                                      signatory_version=version.version,
                                      signatory_version_to_install=signatory_version_to_install,
                                      available_signatory_versions=available_signatory_versions))
    check_signatory_version_exists()


setuptools.setup(name=installer_name,
                 version=version.version,
                 author=metadata.author,
                 author_email=metadata.author_email,
                 maintainer=metadata.author,
                 maintainer_email=metadata.author_email,
                 description=installer_description,
                 long_description=installer_long_description,
                 url=metadata.url,
                 license=metadata.license,
                 keywords=metadata.keywords,
                 classifiers=metadata.classifiers,
                 zip_safe=False,
                 python_requires=metadata.python_requires,
                 install_requires=['signatory=={}'.format(signatory_version_to_install)])
