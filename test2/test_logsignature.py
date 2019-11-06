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
"""Tests the logsignature function and the LogSignature class."""


import gc
import iisignature
import pytest
import signatory
import torch
from torch import autograd
import weakref

import helpers as h


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
# sometimes we do different calculations depending on whether we expect to take a gradient later, so we need to
# check both of these cases
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('batch_size', (0, 1, 2, 5))
@pytest.mark.parametrize('input_stream', (0, 1, 2, 3, 10))
@pytest.mark.parametrize('input_channels', (0, 1, 2, 6))
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('basepoint', (False, True, h.without_grad, h.with_grad))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_forward(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                 mode, iisignature_prepare):
    """Tests that the forward calculations of the logsignature behave correctly."""

    with h.Information(device=device, path_grad=path_grad, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, basepoint=basepoint,
                       inverse=inverse, mode=mode) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)

        expected_exception = (batch_size < 1) or (input_channels < 1) or (not basepoint and input_stream < 2) or \
                             (basepoint and input_stream < 1)
        try:
            if class_:
                logsignature = signatory.LogSignature(depth, mode=mode, stream=stream,
                                                      inverse=inverse)(path, basepoint=basepoint)
            else:
                logsignature = signatory.logsignature(path, depth, mode=mode, stream=stream, basepoint=basepoint,
                                                      inverse=inverse)
        except ValueError:
            if expected_exception:
                return
            else:
                raise
        else:
            assert not expected_exception

        _test_shape(logsignature, info)
        _test_forward_accuracy(logsignature, path, info, iisignature_prepare)

        # Check that the 'ctx' object is properly garbage collected and we don't have a memory leak
        # (Easy to accidentally happen due to PyTorch bug 25340)
        if path_grad or info.basepoint is h.with_grad:
            ctx = logsignature.grad_fn
            if stream:
                ctx = ctx.next_functions[0][0]
            assert type(ctx).__name__ == '_SignatureToLogSignatureFunctionBackward'
            ref = weakref.ref(ctx)
            del ctx
            del logsignature
            gc.collect()
            assert ref() is None
        else:
            assert logsignature.grad_fn is None


def _test_shape(logsignature, info):
    """Tests the the logsignature is of the expected shape."""
    if info.mode == h.expand_mode:
        correct_channels = signatory.signature_channels(info.input_channels, info.depth)
    else:
        correct_channels = signatory.logsignature_channels(info.input_channels, info.depth)
    if info.stream:
        if info.basepoint is (True, h.without_grad, h.with_grad):
            correct_shape = (info.batch_size,
                             info.input_stream,
                             correct_channels)
        else:
            correct_shape = (info.batch_size,
                             info.input_stream - 1,
                             correct_channels)
    else:
        correct_shape = (info.batch_size,
                         correct_channels)
    assert logsignature.shape == correct_shape


def _test_forward_accuracy(logsignature, path, info, iisignature_prepare):
    """Tests that the logsignature computes the correct values."""
    if info.input_channels > 1:
        def compute_logsignature(max_path_index):
            iisignature_path_pieces = []
            if isinstance(info.basepoint, torch.Tensor) or info.basepoint is True:
                if info.basepoint is True:
                    iisignature_basepoint = torch.zeros(info.batch_size, info.input_channels, dtype=torch.double)
                else:
                    iisignature_basepoint = info.basepoint.cpu()
                iisignature_path_pieces.append(iisignature_basepoint.unsqueeze(1))
            iisignature_path_pieces.append(path.detach().cpu()[:, :max_path_index])
            if info.inverse:
                iisignature_path_pieces_reversed = []
                for tensor in reversed(iisignature_path_pieces):
                    iisignature_path_pieces_reversed.append(tensor.flip(1))
                iisignature_path_pieces = iisignature_path_pieces_reversed
            iisignature_path = torch.cat(iisignature_path_pieces, dim=1)

            if info.mode == h.expand_mode:
                iisignature_mode = 'x'
            else:
                iisignature_mode = 'd'

            return iisignature.logsig(iisignature_path, iisignature_prepare(info.input_channels, info.depth),
                                      iisignature_mode)

        if info.stream:
            iisignature_logsignature_pieces = []
            for max_path_index in range(0 if info.basepoint else 1, info.input_stream + 1):
                iisignature_logsignature_pieces.append(compute_logsignature(max_path_index))
            iisignature_logsignature = torch.cat(iisignature_logsignature_pieces, dim=1)
        else:
            iisignature_logsignature = compute_logsignature(info.input_stream)
    else:
        # iisignature doesn't actually support channels == 1, but the logsignature is just the increment in this
        # case.
        if info.stream:
            iisignature_logsignature = path - path[:, 0].unsqueeze(1)
        else:
            iisignature_logsignature = path[:, -1] - path[:, 0]

    if info.mode == h.words_mode:
        transforms = signatory.unstable.lyndon_words_to_basis_transform(info.input_channels, info.depth)
        logsignature = logsignature.clone()
        for source, target, coefficient in transforms:
            logsignature[:, :, target] -= coefficient * logsignature[:, :, source]

    h.diff(logsignature, torch.tensor(iisignature_logsignature, dtype=torch.double, device=info.device))


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse, mode):
    """Tests the the backwards operation through the logsignature gives the correct values."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, basepoint=basepoint,
                       inverse=inverse, mode=mode, path_grad=True) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)
        if class_:
            def check_fn(path, basepoint):
                return signatory.LogSignature(depth, stream=stream, inverse=inverse, mode=mode)(path,
                                                                                                basepoint=basepoint)
        else:
            def check_fn(path, basepoint):
                return signatory.logsignature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                              mode=mode)
        try:
            autograd.gradcheck(check_fn, (path, basepoint), atol=2e-05, rtol=0.002)
        except RuntimeError:
            pytest.fail()


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                        mode, path_grad):
    """Tests that the logsignature computations don't modify any memory that they're not supposed to."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, basepoint=basepoint,
                       inverse=inverse, mode=mode, path_grad=path_grad) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)

        path_clone = path.clone()
        if isinstance(basepoint, torch.Tensor):
            basepoint_clone = basepoint.clone()

        if class_:
            logsignature = signatory.LogSignature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint)
        else:
            logsignature = signatory.logsignature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse)

        if path_grad or info.basepoint is h.with_grad:
            grad = torch.rand_like(logsignature)

            logsignature_clone = logsignature.clone()
            grad_clone = grad.clone()

            logsignature.backward(grad)
        else:
            assert logsignature.grad_fn is None

        h.diff(path, path_clone)
        if isinstance(basepoint, torch.Tensor):
            h.diff(basepoint, basepoint_clone)
        if path_grad or info.basepoint is h.with_grad:
            h.diff(logsignature, logsignature_clone)
            h.diff(grad, grad_clone)


@pytest.mark.skipif(not torch.cuda.is_available())
@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                                 inverse, mode):
    """Performs two separate tests.

    First, that the computations are deterministic, and always give the same result when run multiple times; in
    particular that using the class signatory.LogSignature multiple times is fine.

    Second, that there are no memory leaks.
    """

    # device='cuda' because it's just easier to keep track of GPU memory
    with h.Information(class_=class_, batch_size=batch_size, input_stream=input_stream, input_channels=input_channels,
                       depth=depth, stream=stream, basepoint=basepoint, inverse=inverse, mode=mode,
                       path_grad=path_grad, device='cuda') as info:

        cpu_path = h.get_path(info).detach().cpu()
        if basepoint in (h.without_grad, h.with_grad):
            cpu_basepoint = h.get_basepoint(info).detach().cpu()
        else:
            cpu_basepoint = basepoint
        if class_:
            logsignature_instance = signatory.LogSignature(depth, stream=stream, inverse=inverse, mode=mode)
            cpu_logsignature = logsignature_instance(cpu_path, basepoint=cpu_basepoint)
        else:
            cpu_logsignature = signatory.logsignature(cpu_path, depth, stream=stream, basepoint=cpu_basepoint,
                                                      inverse=inverse, mode=mode)
        cpu_grad = torch.rand_like(cpu_logsignature)

        def one_iteration():
            gc.collect()
            torch.cuda.reset_max_memory_allocated()
            cuda_path = cpu_path.to('cuda')
            if path_grad:
                cuda_path.requires_grad_()
            if isinstance(cpu_basepoint, torch.Tensor):
                cuda_basepoint = cpu_basepoint.cuda()
                if basepoint is h.with_grad:
                    cuda_basepoint.requires_grad_()
            else:
                cuda_basepoint = basepoint

            if class_:
                cuda_logsignature = logsignature_instance(cuda_path, basepoint=cuda_basepoint)
            else:
                cuda_logsignature = signatory.logsignature(cuda_path, depth, stream=stream, basepoint=cuda_basepoint,
                                                           inverse=inverse)

            h.diff(cuda_logsignature.cpu(), cpu_logsignature)

            if path_grad:
                cuda_grad = cpu_grad.cuda()
                cuda_logsignature.backward(cuda_grad)
            return torch.cuda.max_memory_allocated()

        memory_used = one_iteration()

        for repeat in range(10):
            assert one_iteration() <= memory_used
