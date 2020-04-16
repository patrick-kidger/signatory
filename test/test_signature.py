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


import iisignature
import gc
import pytest
import torch
from torch import autograd
import warnings
import weakref

from helpers import helpers as h
from helpers import reimplementation as r
from helpers import validation as v

tests = ['signature', 'Signature']
depends = ['signature_channels', 'signature_combine']
signatory = v.validate_tests(tests, depends)


class _IisignatureSignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth):
        ctx.path = path.detach().cpu()
        ctx.depth = depth
        ctx.device = path.device
        ctx.dtype = path.dtype
        return torch.tensor(iisignature.sig(ctx.path, ctx.depth), device=ctx.device, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad):
        return torch.tensor(iisignature.sigbackprop(grad.cpu(), ctx.path, ctx.depth), device=ctx.device,
                            dtype=ctx.dtype), None


def iisignature_signature(path, depth, stream, basepoint, inverse, initial, scalar_term):
    """Duplicates signatory.signature's functionality using iisignature, for testing purposes."""

    def fn(path, depth):
        signature = _IisignatureSignatureFunction.apply(path, depth)
        if scalar_term:
            out = torch.ones(signature.size(0), 1 + signature.size(1), dtype=signature.dtype,
                             device=signature.device)
            out[:, 1:] = signature
            signature = out
        if isinstance(initial, torch.Tensor):
            signature = signatory.signature_combine(initial, signature, path.size(-1), depth, inverse, scalar_term)
        return signature

    return r.iisignature_signature_or_logsignature(fn, path, depth, stream, basepoint, inverse)


def signatory_signature(class_, path, depth, stream, basepoint, inverse, initial, scalar_term):
    """Wraps signatory.Signature and signatory.signature and filters out warnings, for convenience."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="Argument 'initial' has been set but argument 'basepoint' has "
                                                  "not.", category=UserWarning)
        if class_:
            return signatory.Signature(depth, stream=stream, inverse=inverse,
                                       scalar_term=scalar_term)(path,basepoint=basepoint, initial=initial)
        else:
            return signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                       initial=initial, scalar_term=scalar_term)


def test_forward():
    """Tests that the forward calculations of the signature behave correctly."""
    for class_ in (False, True):
        for device in h.get_devices():
            # sometimes we do different calculations depending on whether we expect to take a gradient later, so we need
            # to check both of these cases
            for path_grad in (False, True):
                for batch_size in (0, 1, 2, 5):
                    for input_stream in (0, 1, 2, 3, 10):
                        for input_channels in (0, 1, 2, 6):
                            for depth in (1, 2, 4, 6):
                                for stream in (False, True):
                                    for basepoint in (False, True, h.without_grad, h.with_grad):
                                        for inverse in (False, True):
                                            for initial in (None, h.without_grad, h.with_grad):
                                                for scalar_term in (False, True):
                                                    _test_forward(class_, device, path_grad, batch_size, input_stream,
                                                                  input_channels, depth, stream, basepoint, inverse,
                                                                  initial, scalar_term)


def _test_forward(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                  inverse, initial, scalar_term):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)

    expected_exception = (batch_size < 1) or (input_channels < 1) or (basepoint is False and input_stream < 2) or \
                         (input_stream < 1)
    try:
        initial = h.get_initial(batch_size, input_channels, device, depth, initial, scalar_term)
        signature = signatory_signature(class_, path, depth, stream, basepoint, inverse, initial, scalar_term)
    except ValueError:
        if expected_exception:
            return
        else:
            raise
    else:
        assert not expected_exception

    _test_shape(signature, stream, basepoint, batch_size, input_stream, input_channels, depth, scalar_term)
    h.diff(signature, iisignature_signature(path, depth, stream, basepoint, inverse, initial, scalar_term))

    if path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad) or \
            (isinstance(initial, torch.Tensor) and initial.requires_grad):
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


def _test_shape(signature, stream, basepoint, batch_size, input_stream, input_channels, depth, scalar_term):
    """Tests the the signature is of the expected shape."""
    correct_channels = signatory.signature_channels(input_channels, depth, scalar_term)
    if stream:
        if isinstance(basepoint, torch.Tensor) or basepoint is True:
            correct_shape = (batch_size,
                             input_stream,
                             correct_channels)
        else:
            correct_shape = (batch_size,
                             input_stream - 1,
                             correct_channels)
    else:
        correct_shape = (batch_size,
                         correct_channels)
    assert signature.shape == correct_shape


def test_batch_trick():
    """Tests that the batch trick method for computing signatures, which is sometimes selected for speed, does
    produce the correct values."""
    if torch.cuda.is_available():
        device_path_grad = ('cuda', False), ('cuda', True), ('cpu', True)
    else:
        device_path_grad = (('cpu', True),)

    for class_ in (False, True):
        for device, path_grad in device_path_grad:
            for batch_size in (1, 2, 5):
                for input_stream in (6, 10):
                    for input_channels in (1, 2, 6):
                        for depth in (1, 2, 4, 6):
                            stream = False
                            for basepoint in (False, True, h.without_grad, h.with_grad):
                                for inverse in (False, True):
                                    for initial in (None, h.without_grad, h.with_grad):
                                        for scalar_term in (False, True):
                                            _test_batch_trick(class_, device, path_grad, batch_size, input_stream,
                                                              input_channels, depth, stream, basepoint, inverse,
                                                              initial, scalar_term)


def _no_batch_trick(path, depth, stream, basepoint, inverse, initial, scalar_term):
    return


def _test_batch_trick(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                      inverse, initial, scalar_term):
    if device == 'cuda':
        threshold = 512
    else:
        threshold = torch.get_num_threads()
        if threshold < 2:
            return  # can't test the batch trick in this case
    if round(float(threshold) / batch_size) < 2:
        batch_size = int(threshold / 2)

    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)
    initial = h.get_initial(batch_size, input_channels, device, depth, initial, scalar_term)

    _signature_batch_trick = signatory.signature.__globals__['_signature_batch_trick']
    try:
        # disable batch trick
        signatory.signature.__globals__['_signature_batch_trick'] = _no_batch_trick
        signature = signatory_signature(class_, path, depth, stream, basepoint, inverse, initial, scalar_term)
    finally:
        signatory.signature.__globals__['_signature_batch_trick'] = _signature_batch_trick

    batch_trick_signature = signatory.signature.__globals__['_signature_batch_trick'](path,
                                                                                      depth,
                                                                                      stream=stream,
                                                                                      basepoint=basepoint,
                                                                                      inverse=inverse,
                                                                                      initial=initial,
                                                                                      scalar_term=scalar_term)

    assert batch_trick_signature is not None  # that the batch trick is viable in this case

    h.diff(signature, batch_trick_signature)

    can_backward = path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad) or \
                   (isinstance(initial, torch.Tensor) and initial.requires_grad)
    try:
        grad = torch.rand_like(signature)
        signature.backward(grad)
    except RuntimeError:
        assert not can_backward
        return
    else:
        assert can_backward

    if path_grad:
        path_grad_ = path.grad.clone()
        path.grad.zero_()
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        basepoint_grad = basepoint.grad.clone()
        basepoint.grad.zero_()
    if isinstance(initial, torch.Tensor) and initial.requires_grad:
        initial_grad = initial.grad.clone()
        initial.grad.zero_()
    batch_trick_signature.backward(grad)
    if path_grad:
        h.diff(path_grad_, path.grad)
        path.grad.zero_()
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        h.diff(basepoint_grad, basepoint.grad)
        basepoint.grad.zero_()
    if isinstance(initial, torch.Tensor) and initial.requires_grad:
        h.diff(initial_grad, initial.grad)
        initial.grad.zero_()


def test_backward():
    """Tests that the backwards operation through the signature gives the correct values."""
    for class_ in (False, True):
        for device in h.get_devices():
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 4, 6):
                    for stream in (False, True):
                        for inverse in (False, True):
                            for initial in (None, h.without_grad, h.with_grad):
                                for scalar_term in (False, True):
                                    _test_backward(class_, device, batch_size, input_stream, input_channels, depth,
                                                   stream, basepoint, inverse, initial, scalar_term)


def _test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                   initial, scalar_term):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad=True)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)
    initial = h.get_initial(batch_size, input_channels, device, depth, initial, scalar_term)

    # This is the test we'd like to run. Unfortunately it takes forever, so we do something else instead.
    #
    # if class_:
    #     def check_fn(path, basepoint, initial):
    #         return signatory.Signature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint,
    #                                                                           initial=initial)
    # else:
    #     def check_fn(path, basepoint, initial):
    #         return signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
    #                                    initial=initial)
    # try:
    #     autograd.gradcheck(check_fn, (path, basepoint, initial), atol=2e-05, rtol=0.002)
    # except RuntimeError:
    #     pytest.fail()

    signature = signatory_signature(class_, path, depth, stream, basepoint, inverse, initial, scalar_term)

    grad = torch.rand_like(signature)
    signature.backward(grad)

    path_grad = path.grad.clone()
    path.grad.zero_()
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        basepoint_grad = basepoint.grad.clone()
        basepoint.grad.zero_()
    if isinstance(initial, torch.Tensor) and initial.requires_grad:
        initial_grad = initial.grad.clone()
        initial.grad.zero_()

    iisignature_signature_result = iisignature_signature(path, depth, stream, basepoint, inverse, initial, scalar_term)
    iisignature_signature_result.backward(grad)

    # iisignature uses float32 for this calculation so we need a lower tolerance
    h.diff(path.grad, path_grad, atol=1e-4)
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        h.diff(basepoint.grad, basepoint_grad, atol=1e-4)
    if isinstance(initial, torch.Tensor) and initial.requires_grad:
        h.diff(initial.grad, initial_grad, atol=1e-4)


def test_no_adjustments():
    """Tests that the signature computations don't modify any memory that they're not supposed to."""

    for class_ in (False, True):
        for path_grad in (False, True):
            for device in h.get_devices():
                for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                    for depth in (1, 2, 5):
                        for stream in (False, True):
                            for inverse in (False, True):
                                for initial in (None, h.without_grad, h.with_grad):
                                    for scalar_term in (False, True):
                                        _test_no_adjustments(class_, device, batch_size, input_stream, input_channels,
                                                             depth, stream, basepoint, inverse, initial, path_grad,
                                                             scalar_term)


def _test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint,
                         inverse, initial, path_grad, scalar_term):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)
    initial = h.get_initial(batch_size, input_channels, device, depth, initial, scalar_term)

    path_clone = path.clone()
    if isinstance(basepoint, torch.Tensor):
        basepoint_clone = basepoint.clone()
    if isinstance(initial, torch.Tensor):
        initial_clone = initial.clone()

    signature = signatory_signature(class_, path, depth, stream, basepoint, inverse, initial, scalar_term)

    can_backward = path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad) or \
                   (isinstance(initial, torch.Tensor) and initial.requires_grad)
    if can_backward:
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
    if can_backward:
        h.diff(signature, signature_clone)
        h.diff(grad, grad_clone)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_repeat_and_memory_leaks():
    """Performs two separate tests.

    First, that the computations are deterministic, and always give the same result when run multiple times; in
    particular that using the class signatory.Signature multiple times is fine.

    Second, that there are no memory leaks.
    """
    for class_ in (False, True):
        for path_grad in (False, True):
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 5):
                    for stream in (False, True):
                        for inverse in (False, True):
                            for initial in (None, h.without_grad, h.with_grad):
                                for scalar_term in (False, True):
                                    _test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream,
                                                                  input_channels, depth, stream, basepoint, inverse,
                                                                  initial, scalar_term)


def _test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                                  inverse, initial, scalar_term):
    cpu_path = h.get_path(batch_size, input_stream, input_channels, device='cpu', path_grad=False)
    cpu_basepoint = h.get_basepoint(batch_size, input_channels, device='cpu', basepoint=basepoint)
    cpu_initial = h.get_initial(batch_size, input_channels, device='cpu', depth=depth, initial=initial,
                                scalar_term=scalar_term)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="Argument 'initial' has been set but argument 'basepoint' has "
                                                  "not.", category=UserWarning)
        if class_:
            signature_instance = signatory.Signature(depth, stream=stream, inverse=inverse, scalar_term=scalar_term)
            cpu_signature = signature_instance(cpu_path, basepoint=cpu_basepoint, initial=cpu_initial)
        else:
            cpu_signature = signatory.signature(cpu_path, depth, stream=stream, basepoint=cpu_basepoint,
                                                inverse=inverse, initial=cpu_initial, scalar_term=scalar_term)
    cpu_grad = torch.rand_like(cpu_signature)

    def one_iteration():
        gc.collect()
        torch.cuda.synchronize()
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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="Argument 'initial' has been set but argument 'basepoint' has "
                                                      "not.", category=UserWarning)
            if class_:
                cuda_signature = signature_instance(cuda_path, basepoint=cuda_basepoint, initial=cuda_initial)
            else:
                cuda_signature = signatory.signature(cuda_path, depth, stream=stream, basepoint=cuda_basepoint,
                                                     inverse=inverse, initial=cuda_initial, scalar_term=scalar_term)

        h.diff(cuda_signature.cpu(), cpu_signature)

        if path_grad:
            cuda_grad = cpu_grad.cuda()
            cuda_signature.backward(cuda_grad)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    memory_used = one_iteration()

    for repeat in range(10):
        assert one_iteration() <= memory_used
