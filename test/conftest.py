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
"""Configures the tests."""


import copy
import pytest
import signatory
import sys

from helpers import validation as v


# Replace the weak dictionary with a regular dictionary for speed
if not hasattr(signatory.SignatureToLogSignature, '_lyndon_info_capsule_cache'):
    raise RuntimeError('Expected signatory.SignatureToLogSignature to have a cache for lyndon info capsules')
signatory.SignatureToLogSignature._lyndon_info_capsule_cache = {}


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true', dest='slow', default=False, help="Run slow tests as well.")


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: mark a test as being slow and excluded from default test runs')
    if not config.option.slow:
        if hasattr(config.option, 'markexpr') and len(config.option.markexpr) > 0:
            config.option.markexpr = '(' + config.option.markexpr + ') and not slow'
        else:
            config.option.markexpr = 'not slow'


def pytest_collection_finish(session):
    cycle = v.signatory_functionality_graph.get_cycle()
    if cycle is not None:
        error_pieces = ['\nTests have a cyclic dependency!\n']

        itercycle = iter(reversed(cycle))
        try:
            while True:
                item = next(itercycle)
                error_pieces.append(str(item))
                item2 = next(itercycle)
                error_pieces.append('--(')
                error_pieces.append(str(item2))
                error_pieces.append(')->')
        except StopIteration:
            pass

        raise RuntimeError(''.join(error_pieces))

    unmarked = v.signatory_functionality_graph.get_unmarked()
    if len(unmarked):
        print('These tests assume that the following functionality is correct:\n' + ', '.join(unmarked))

    if 'Logsignature' in v.signatory_functionality_graph.vertices():
        raise RuntimeError('Use LogSignature, not Logsignature')


@pytest.fixture(scope='module')
def path_hack(request):
    """Hacks the PYTHONPATH to be able to import other things."""
    original_path = copy.copy(sys.path)
    add_to_path = getattr(request.module, 'add_to_path')
    if isinstance(add_to_path, (tuple, list)):
        sys.path.extend(add_to_path)
    else:
        sys.path.append(add_to_path)
    yield
    sys.path = original_path
