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
"""Generic helpers for testing purposes."""


import iisignature
import pytest
import signatory
import torch

with_grad = object()
without_grad = object()


expand_mode = 'expand'
words_mode = 'words'
brackets_mode = 'brackets'
all_modes = (expand_mode, words_mode, brackets_mode)


def diff(arg1, arg2, atol=1e-8):
    """Considers the difference between the two arguments, and raises an error if they are not close."""
    if not arg1.allclose(arg2, atol=atol):
        diff = arg1 - arg2
        max_diff = torch.max(torch.abs(diff))
        pytest.fail('\ndiff={diff}\nmax_diff={max_diff}\narg1={arg1}\narg2={arg2}'.format(diff=diff, max_diff=max_diff,
                                                                                          arg1=arg1, arg2=arg2))


def get_devices():
    """Gets the currently available devices."""
    if torch.cuda.is_available():
        return 'cuda', 'cpu'
    else:
        return ('cpu',)


def get_path(batch_size, input_stream, input_channels, device, path_grad):
    """Gets a path."""
    return torch.rand(batch_size, input_stream, input_channels, device=device, requires_grad=path_grad, 
                      dtype=torch.double)


def get_basepoint(batch_size, input_channels, device, basepoint):
    """Gets a basepoint."""
    if basepoint == without_grad:
        return torch.rand(batch_size, input_channels, device=device, dtype=torch.double)
    elif basepoint == with_grad:
        return torch.rand(batch_size, input_channels, device=device, requires_grad=True,
                          dtype=torch.double)
    else:
        return basepoint


def get_initial(batch_size, input_channels, device, depth, initial, scalar_term):
    """Gets a value for the 'initial' argument of signatory.signature."""
    if initial in (without_grad, with_grad):
        initial_path = torch.rand(batch_size, 2, input_channels, device=device, dtype=torch.double)
        initial_signature = signatory.signature(initial_path, depth, scalar_term=scalar_term)
        if initial == with_grad:
            initial_signature.requires_grad_()
        return initial_signature
    else:
        return initial


def random_sizes():
    """Generates some random sizes; in addition it includes all small sizes."""
    params = []
    for batch_size in (1, 2):
        for input_stream in (2,):
            for input_channels in (1, 2):
                params.append((batch_size, input_stream, input_channels))
    for _ in range(5):
        batch_size = int(torch.randint(low=1, high=10, size=(1,)))
        input_stream = int(torch.randint(low=2, high=10, size=(1,)))
        input_channels = int(torch.randint(low=1, high=10, size=(1,)))
        params.append((batch_size, input_stream, input_channels))
    return params


def random_sizes_and_basepoint():
    """Generates some random sizes with basepoints; in addition it includes all small sizes and basepoints."""
    params = []
    for batch_size in (1, 2):
        for input_stream in (1, 2):
            for input_channels in (1, 2):
                for basepoint in (True, without_grad, with_grad):
                    params.append((batch_size, input_stream, input_channels, basepoint))
    for batch_size in (1, 2):
        for input_stream in (2,):
            for input_channels in (1, 2):
                for basepoint in (False,):
                    params.append((batch_size, input_stream, input_channels, basepoint))
    for _ in range(5):
        for basepoint in (True, without_grad, with_grad):
            batch_size = int(torch.randint(low=1, high=10, size=(1,)))
            input_stream = int(torch.randint(low=2, high=10, size=(1,)))
            input_channels = int(torch.randint(low=1, high=10, size=(1,)))
            params.append((batch_size, input_stream, input_channels, basepoint))
    return params


class NullContext(object):
    """A null context."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


_iisignature_prepare_cache = {}


def iisignature_prepare(channels, depth, method='d'):
    """Like iisignature.prepare, but caches every result."""
    try:
        return _iisignature_prepare_cache[(channels, depth, method)]
    except KeyError:
        prepared = iisignature.prepare(channels, depth, method)
        _iisignature_prepare_cache[(channels, depth, method)] = prepared
        return prepared
