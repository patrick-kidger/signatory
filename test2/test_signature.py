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
"""Tests the signature function and the Signature class."""


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
@pytest.mark.parametrize('initial', (None, h.without_grad, h.with_grad))
def test_forward(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                 initial):
    """Tests that the forward calculations of the signature behave correctly."""

    with h.Information(class_=class_, device=device, path_grad=path_grad, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, stream=stream,
                       basepoint=basepoint, inverse=inverse, initial=initial) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)
        initial = h.get_initial(info)

        expected_exception = (batch_size < 1) or (input_channels < 1) or (not basepoint and input_stream < 2) or \
                             (basepoint and input_stream < 1)
        try:
            if class_:
                signature = signatory.Signature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint,
                                                                                       initial=initial)
            else:
                signature = signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                                initial=initial)
        except ValueError:
            if expected_exception:
                return
            else:
                raise
        else:
            assert not expected_exception

        _test_shape(signature, info)
        _test_forward_accuracy(signature, path, info)

        if path_grad or info.basepoint is h.with_grad or info.initial is h.with_grad:
            ctx = signature.grad_fn
            if stream:
                ctx = ctx.next_functions[0][0]
            assert type(ctx).__name__ in ('_SignatureFunctionBackward', '_SignatureCombineFunctionBackward')
            ref = weakref.ref(ctx)
            del ctx
            del signature
            gc.collect()
            assert ref() is None
        else:
            assert signature.grad_fn is None


def _test_shape(signature, info):
    """Tests the the signature is of the expected shape."""
    correct_channels = signatory.signature_channels(info.input_channels, info.depth)
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
    assert signature.shape == correct_shape


def _test_forward_accuracy(signature, path, info):
    """Tests that the signature computes the correct values."""
    iisignature_path_pieces = []
    if isinstance(info.initial, torch.Tensor):
        if info.basepoint is True:
            basepoint_adjustment = 0
        elif info.basepoint is False:
            basepoint_adjustment = path[:, 0, :].unsqueeze(1)
        else:
            basepoint_adjustment = info.basepoint
        adjusted_initial_path = info.initial_path - info.initial_path[:, -1, :].unsqueeze(1) + basepoint_adjustment
        iisignature_path_pieces.append(adjusted_initial_path.cpu())
    if isinstance(info.basepoint, torch.Tensor) or info.basepoint is True:
        if info.basepoint is True:
            iisignature_basepoint = torch.zeros(info.batch_size, info.input_channels, dtype=torch.double)
        else:
            iisignature_basepoint = info.basepoint.cpu()
        iisignature_path_pieces.append(iisignature_basepoint.unsqueeze(1))
    iisignature_path_pieces.append(path.detach().cpu())
    if info.inverse:
        iisignature_path_pieces_reversed = []
        for tensor in reversed(iisignature_path_pieces):
            iisignature_path_pieces_reversed.append(tensor.flip(1))
        iisignature_path_pieces = iisignature_path_pieces_reversed
    iisignature_path = torch.cat(iisignature_path_pieces, dim=1)

    iisignature_signature = iisignature.sig(iisignature_path, info.depth, 2 if info.stream else 0)

    h.diff(signature, torch.tensor(iisignature_signature, dtype=torch.double, device=info.device))


def _device_path_grad():
    if torch.cuda.is_available():
        return ('cuda', False), ('cuda', True), ('cpu', True)
    else:
        return (('cpu', True),)


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device,path_grad', _device_path_grad())
@pytest.mark.parametrize('batch_size', (1, 2, 5))
@pytest.mark.parametrize('input_stream', (6, 10))
@pytest.mark.parametrize('input_channels', (1, 2, 6))
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False,))
@pytest.mark.parametrize('basepoint', (False, True, h.without_grad, h.with_grad))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('initial', (None, h.without_grad, h.with_grad))
def _test_batch_trick(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                      inverse, initial):
    """Tests that the batch trick method for computing signatures, which is sometimes selected for speed, does
    produce the correct values."""

    if device == 'cuda':
        threshold = 512
    else:
        import signatory._impl
        threshold = signatory._impl.hardware_concurrency()
        if threshold < 2:
            return  # can't test the batch trick in this case
    if round(float(threshold) / batch_size) < 2:
        batch_size = int(threshold / 2)

    with h.Information(class_=class_, device=device, path_grad=path_grad, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, stream=stream,
                       basepoint=basepoint, inverse=inverse, initial=initial) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)
        initial = h.get_initial(info)

        if class_:
            signature = signatory.Signature(depth, stream=stream,
                                            inverse=inverse)(path, basepoint=basepoint, initial=initial)
        else:
            signature = signatory.signature(path, depth, stream=stream, basepoint=basepoint,
                                            inverse=inverse, initial=initial)

        batch_trick_signature = signatory.signature.__globals__['_signature_batch_trick'](path,
                                                                                          depth,
                                                                                          stream=stream,
                                                                                          basepoint=basepoint,
                                                                                          inverse=inverse,
                                                                                          initial=initial)

        assert batch_trick_signature is not None  # that the batch trick is viable in this case

        h.diff(signature, batch_trick_signature)

        can_backward = path_grad or info.basepoint is h.with_grad or info.initial is h.with_grad
        try:
            grad = torch.rand_like(signature)
            signature.backward(grad)
        except RuntimeError:
            assert not can_backward
            return
        else:
            assert can_backward

        if path_grad:
            path_grad = path.grad.clone()
            path.grad.zero_()
        if info.basepoint is h.with_grad:
            basepoint_grad = basepoint.grad.clone()
            basepoint.grad.zero_()
        if info.initial is h.with_grad:
            initial_grad = initial.grad.clone()
            initial.grad.zero_()
        batch_trick_signature.backward(grad)
        if path_grad:
            h.diff(path_grad, path.grad)
            path.grad.zero_()
        if info.basepoint is h.with_grad:
            h.diff(basepoint_grad, basepoint.grad)
            basepoint.grad.zero_()
        if info.initial is h.with_grad:
            h.diff(initial_grad, initial.grad)
            initial.grad.zero_()


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('initial', (None, h.without_grad, h.with_grad))
def test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse, initial):
    """Tests the the backwards operation through the signature gives the correct values."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, basepoint=basepoint,
                       inverse=inverse, initial=initial, path_grad=True) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)
        initial = h.get_initial(info)
        if class_:
            def check_fn(path, basepoint, initial):
                return signatory.Signature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint,
                                                                                  initial=initial)
        else:
            def check_fn(path, basepoint, initial):
                return signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                           initial=initial)
        try:
            autograd.gradcheck(check_fn, (path, basepoint, initial), atol=2e-05, rtol=0.002)
        except RuntimeError:
            pytest.fail()


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('initial', (None, h.without_grad, h.with_grad))
def test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint,
                        inverse, initial, path_grad):
    """Tests that the signature computations don't modify any memory that they're not supposed to."""
    with h.Information(class_=class_, device=device, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, stream=stream,
                       basepoint=basepoint, inverse=inverse, initial=initial, path_grad=path_grad) as info:
        path = h.get_path(info)
        basepoint = h.get_basepoint(info)
        initial = h.get_initial(info)

        path_clone = path.clone()
        if isinstance(basepoint, torch.Tensor):
            basepoint_clone = basepoint.clone()
        if isinstance(initial, torch.Tensor):
            initial_clone = initial.clone()

        if class_:
            signature = signatory.Signature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint,
                                                                                   initial=initial)
        else:
            signature = signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                            initial=initial)

        if path_grad or info.basepoint is h.with_grad or info.initial is h.with_grad:
            grad = torch.rand_like(signature)

            signature_clone = signature.clone()
            grad_clone = grad.clone()

            signature.backward(grad)
        else:
            assert signature.grad_fn is None

        h.diff(path, path_clone)
        if isinstance(basepoint, torch.Tensor):
            h.diff(basepoint, basepoint_clone)
        if isinstance(initial, torch.Tensor):
            h.diff(initial, initial_clone)
        if path_grad or info.basepoint is h.with_grad or info.initial is h.with_grad:
            h.diff(signature, signature_clone)
            h.diff(grad, grad_clone)


@pytest.mark.skipif(not torch.cuda.is_available())
@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('inverse', (False, True))
@pytest.mark.parametrize('initial', (None, h.without_grad, h.with_grad))
def test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                                 inverse, initial):
    """Performs two separate tests.

    First, that the computations are deterministic, and always give the same result when run multiple times; in
    particular that using the class signatory.Signature multiple times is fine.

    Second, that there are no memory leaks.
    """

    # device='cuda' because it's just easier to keep track of GPU memory
    with h.Information(class_=class_, batch_size=batch_size, input_stream=input_stream, input_channels=input_channels,
                       depth=depth, stream=stream, basepoint=basepoint, inverse=inverse, initial=initial,
                       path_grad=path_grad, device='cuda') as info:

        cpu_path = h.get_path(info).detach().cpu()
        if basepoint in (h.without_grad, h.with_grad):
            cpu_basepoint = h.get_basepoint(info).detach().cpu()
        else:
            cpu_basepoint = basepoint
        if initial in (h.without_grad, h.with_grad):
            cpu_initial = h.get_initial(info).detach().cpu()
        else:
            cpu_initial = initial
        if class_:
            signature_instance = signatory.Signature(depth, stream=stream, inverse=inverse)
            cpu_signature = signature_instance(cpu_path, basepoint=cpu_basepoint, initial=cpu_initial)
        else:
            cpu_signature = signatory.signature(cpu_path, depth, stream=stream, basepoint=cpu_basepoint, inverse=inverse,
                                                initial=cpu_initial)
        cpu_grad = torch.rand_like(cpu_signature)

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
            if isinstance(cpu_initial, torch.Tensor):
                cuda_initial = cpu_initial.cuda()
                if initial is h.with_grad:
                    cuda_initial.requires_grad_()
            else:
                cuda_initial = initial

            if class_:
                cuda_signature = signature_instance(cuda_path, basepoint=cuda_basepoint, initial=cuda_initial)
            else:
                cuda_signature = signatory.signature(cuda_path, depth, stream=stream, basepoint=cuda_basepoint,
                                                     inverse=inverse, initial=cuda_initial)

            h.diff(cuda_signature.cpu(), cpu_signature)

            if path_grad:
                cuda_grad = cpu_grad.cuda()
                cuda_signature.backward(cuda_grad)
            return torch.cuda.max_memory_allocated()

        memory_used = one_iteration()

        for repeat in range(10):
            assert one_iteration() <= memory_used
