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


import iisignature
import pytest
import signatory
import sys


# Replace the weak dictionary with a regular dictionary for speed
if not hasattr(signatory.SignatureToLogSignature, '_lyndon_info_capsule_cache'):
    raise RuntimeError('Expected signatory.SignatureToLogSignature to have a cache for lyndon info capsules')
signatory.SignatureToLogSignature._lyndon_info_capsule_cache = {}

pytest.register_assert_rewrite('helpers')


@pytest.fixture
def no_parallelism():
    """Disable parallelism in this test."""
    current_parallelism = signatory.max_parallelism()
    signatory.max_parallelism(1)
    yield
    signatory.max_parallelism(current_parallelism)


@pytest.fixture(scope='session')
def iisignature_prepare():
    """Caches the results of iisignature's prepare() function, which is often quite time consuming."""
    _iisignature_prepare_cache = {}

    def _iisignature_prepare(channels, depth):
        try:
            return _iisignature_prepare_cache[(channels, depth)]
        except KeyError:
            prepared = iisignature.prepare(channels, depth)
            _iisignature_prepare_cache[(channels, depth)] = prepared
            return prepared

    return _iisignature_prepare


@pytest.fixture(scope='module')
def path_hack(request):
    """Hacks the PYTHONPATH to be able to import other things."""
    original_path = sys.path.copy()
    add_to_path = getattr(request.module, 'add_to_path')
    if isinstance(add_to_path, (tuple, list)):
        sys.path.extend(add_to_path)
    else:
        sys.path.append(add_to_path)
    yield
    sys.path = original_path
