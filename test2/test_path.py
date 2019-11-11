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
"""Tests the Path class."""


import gc
import pytest
import random
import torch
import weakref

from helpers import helpers as h
from helpers import validation as v


tests = ['Path']
depends = ['signature', 'logsignature']
signatory = v.validate_tests(tests, depends)


def _update_lengths_update_grads():
    num = int(torch.randint(low=0, high=2, size=(1,)))
    update_lengths = []
    update_grads = []
    for _ in range(num):
        update_lengths.append(int(torch.randint(low=1, high=10, size=(1,))))
        update_grads.append(random.choice([True, False]))
    return zip(update_lengths, update_grads)


def test_path():
    """Tests that Path behaves correctly."""
    for device in h.get_devices():
        for path_grad in (False, True):
            for batch_size in (1, 2, 5):
                for input_stream, basepoints in zip((1, 2, 3, 10), ((True, h.without_grad, h.with_grad),
                                                                    (False, True, h.without_grad, h.with_grad),
                                                                    (False, True, h.without_grad, h.with_grad),
                                                                    (False, True, h.without_grad, h.with_grad))):
                    for input_channels in (1, 2, 6):
                        for depth in (1, 2, 4, 6):
                            for basepoint in basepoints:
                                for update_lengths, update_grads in _update_lengths_update_grads():
                                    _test_path(device, path_grad, batch_size, input_stream, input_channels, depth,
                                               basepoint, update_lengths, update_grads)


def _test_path(device, path_grad, batch_size, input_stream, input_channels, depth, basepoint, update_lengths,
               update_grads):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)
    path_obj = signatory.Path(path, depth, basepoint=basepoint)

    if isinstance(basepoint, torch.Tensor):
        full_path = torch.cat([basepoint, path], dim=1)
    elif basepoint is True:
        full_path = torch.cat([torch.zeros(batch_size, 1, input_channels), path], dim=1)
    else:
        full_path = path

    has_grad = path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad)

    # First of all test a Path with no updates
    _test_signature(path_obj, full_path, depth, has_grad)
    _test_logsignature(path_obj, full_path, depth, has_grad)
    assert path_obj.depth == depth

    # Then test Path with variable amounts of updates
    for length, grad in zip(update_lengths, update_grads):
        has_grad = has_grad or grad
        new_path = torch.rand(batch_size, length, input_channels, dtype=torch.double, device=device,
                              requires_grad=grad)
        path_obj.update(new_path)
        full_path = torch.cat([full_path, new_path], dim=1)

    _test_signature(path_obj, full_path, depth, has_grad)
    _test_logsignature(path_obj, full_path, depth, has_grad)
    assert path_obj.depth == depth


def _test_signature(path_obj, full_path, depth, has_grad):
    def candidate(start=None, end=None):
        return path_obj.signature(start, end)

    def true(start, end):
        return signatory.signature(full_path[start:end], depth)

    def extra(true_signature):
        assert (path_obj.signature_size(-3), path_obj.signature_size(-1)) == true_signature.shape
        assert path_obj.signature_channels() == true_signature.size(-1)
        assert path_obj.shape == full_path.shape
        assert path_obj.channels() == full_path.size(-1)

    _test_signature_or_logsignature(has_grad, path_obj, candidate, true, extra,
                                    '_BackwardShortcutBackward')


def _test_logsignature(path_obj, full_path, depth, has_grad):
    for mode in h.all_modes:
        print('mode=' + str(mode))

        def candidate(start=None, end=None):
            return path_obj.logsignature(start, end, mode=mode)

        def true(start, end):
            return signatory.logsignature(full_path[start:end], depth, mode=mode)

        def extra(true_logsignature):
            if mode != h.expand_mode:
                assert (path_obj.logsignature_size(-3),
                        path_obj.logsignature_size(-1)) == true_logsignature.shape
                assert path_obj.logsignature_channels() == true_logsignature.size(-1)

        _test_signature_or_logsignature(has_grad, path_obj, candidate, true, extra,
                                        '_SignatureToLogsignatureFunctionBackward')


def _test_signature_or_logsignature(has_grad, path_obj, candidate, true, extra, backward_name):
    # We perform multiple tests here.
    # Test #1: That the memory usage is consistent
    # Test #2: That the backward 'ctx' is correctly garbage collected
    # Test #3: The forward accuracy of a particular operation
    # Test #4: The backward accuracy of the same operation
    gc.collect()
    torch.cuda.reset_max_memory_allocated()
    candidate()
    memory_used = torch.cuda.memory_allocated()  # Test #1
    for start in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
        for end in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
            print('start=' + str(start))
            print('end=' + str(end))
            gc.collect()
            torch.cuda.reset_max_memory_allocated()
            try:
                tensor = candidate(start, end)
            except ValueError:
                try:
                    true(start, end)
                except ValueError:
                    continue
                else:
                    pytest.fail()
            assert torch.cuda.memory_allocated() <= memory_used  # Test #1
            try:
                true_tensor = true(start, end)
            except ValueError:
                pytest.fail()
            h.diff(tensor, true_tensor)  # Test #3

            extra(true_tensor)  # Any extra tests

            if has_grad:
                grad = torch.rand_like(tensor)
                tensor.backward(grad)
                path_grads = []
                for path in path_obj.path:
                    if path.grad is None:
                        path_grads.append(None)
                    else:
                        path_grads.append(path.grad.clone())
                        path.grad.zero_()
                true_tensor.backward(grad)
                for path, path_grad in zip(path_obj.path, path_grads):
                    if path.grad is None:
                        assert path_grad is None
                    else:
                        h.diff(path.grad, path_grad)  # Test #4
                        path.grad.zero_()
                ctx = tensor.grad_fn
                assert type(ctx).__name__ == backward_name
                ref = weakref.ref(ctx)
                del ctx
                del tensor
                gc.collect()
                assert ref() is None  # Test #2
            else:
                assert tensor.grad_fn is None  # Can't run test 2 or 4 because there is no backward
