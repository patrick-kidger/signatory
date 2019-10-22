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
"""Defines set up for building the installer."""


import io
import os
import shutil
import sys

here = os.path.realpath(os.path.dirname(__file__))


def set_up():
    sys.path.append(os.path.realpath(os.path.join(here, '..')))
    import metadata

    with io.open(os.path.join(here, 'version.py'), 'w', encoding='utf-8') as f:
        f.write('version = "' + str(metadata.version) + '"\n')

    shutil.copy2(os.path.realpath(os.path.join(here, '..', 'metadata.py')),
                 os.path.realpath(os.path.join(here, 'metadata.py')))
    shutil.copy2(os.path.realpath(os.path.join(here, '..', 'LICENSE')),
                 os.path.realpath(os.path.join(here, 'LICENSE')))


def clean():
    os.remove(os.path.realpath(os.path.join(here, 'metadata.py')))
    os.remove(os.path.realpath(os.path.join(here, 'version.py')))
    os.remove(os.path.realpath(os.path.join(here, 'LICENSE')))
    try:
        shutil.rmtree(os.path.realpath(os.path.join(here, 'signatory_installer.egg-info')))
    except FileNotFoundError:
        pass
