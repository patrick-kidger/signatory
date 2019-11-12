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
import random
import torch
from torch import autograd
import warnings
import weakref

from helpers import helpers as h
from helpers import reimplementation as r
from helpers import validation as v


tests = ['logsignature', 'LogSignature', 'unstable.lyndon_words_to_basis_transform']
depends = ['lyndon_words', 'signature_channels', 'logsignature_channels']
signatory = v.validate_tests(tests, depends)


_lyndon_indices_cache = {}


def _lyndon_indices(channels, depth):
    try:
        return _lyndon_indices_cache[(channels, depth)]
    except KeyError:
        words = signatory.lyndon_words(channels, depth)
        indices = []
        for word in words:
            index = signatory.signature_channels(channels, len(word) - 1) if len(word) > 1 else 0
            stride = 1
            for letter in reversed(word):
                index += stride * letter
                stride *= channels
            indices.append(index)
        indices = torch.tensor(indices)
        _lyndon_indices_cache[(channels, depth)] = indices
        return indices


_lyndon_words_to_basis_transform_cache = {}


def _lyndon_words_to_basis_transform(channels, depth):
    try:
        return _lyndon_words_to_basis_transform_cache[(channels, depth)]
    except KeyError:
        prepared = signatory.unstable.lyndon_words_to_basis_transform(channels, depth)
        _lyndon_words_to_basis_transform_cache[(channels, depth)] = prepared
        return prepared


class _iisignatureLogsignatureFunction(autograd.Function):
    # Note that 'x' is the only method that I know to have the correct backprop values for iisignature.logsigbackprop
    # (See iisignature bug 8)
    @staticmethod
    def forward(ctx, path, depth):
        ctx.path = path.detach().cpu()
        ctx.prepare = h.iisignature_prepare(path.size(-1), depth, 'x')
        ctx.device = path.device
        ctx.dtype = path.dtype
        return torch.tensor(iisignature.logsig(ctx.path, ctx.prepare, 'x'), device=ctx.device, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad):
        return torch.tensor(iisignature.logsigbackprop(grad.cpu(), ctx.path, ctx.prepare, 'x'), device=ctx.device,
                            dtype=ctx.dtype), None


def _iisignature_logsignature(path, depth, mode):
    result = _iisignatureLogsignatureFunction.apply(path, depth)
    if mode != h.expand_mode:
        result = torch.index_select(result, -1, _lyndon_indices(path.size(-1), depth).to(path.device))
        if mode != h.words_mode:
            for transform_class in _lyndon_words_to_basis_transform(path.size(-1), depth):
                for source, target, coefficient in transform_class:
                    result[..., target] -= coefficient * result[..., source]

            # We can't use this directly because we can't backprop through it. (iisignature has a bug in logsigbackprop
            # for this one)
            # But we do test that the forward result is correct.
            direct_iisignature_logsignature = iisignature.logsig(path.detach().cpu(),
                                                                 h.iisignature_prepare(path.size(-1), depth))
            assert result.allclose(torch.tensor(direct_iisignature_logsignature, dtype=path.dtype, device=path.device))
    return result


def iisignature_logsignature(path, depth, stream, basepoint, inverse, mode):
    """Duplicates signatory.logsignature's functionality using iisignature, for testing purposes."""

    batch_size, input_stream, input_channels = path.shape
    device = path.device
    dtype = path.dtype

    if input_channels > 1:
        fn = lambda path, depth: _iisignature_logsignature(path, depth, mode)
        result = r.iisignature_signature_or_logsignature(fn, path, depth, stream, basepoint, inverse)
    else:
        # iisignature doesn't actually support channels == 1, but the logsignature is just the increment in this
        # case.

        # Add on the basepoint if necessary
        if basepoint is True:
            iisignature_path = torch.cat([torch.zeros(batch_size, 1, input_channels,
                                                      dtype=dtype, device=device), path], dim=1)
        elif isinstance(basepoint, torch.Tensor):
            iisignature_path = torch.cat([basepoint.unsqueeze(1), path], dim=1)
        else:
            iisignature_path = path

        # Calculate increments
        result = iisignature_path[:, 1:] - iisignature_path[:, 0].unsqueeze(1)

        # Extract final piece if necessary
        if not stream:
            result = result[:, -1]

        # Invert if necessary
        # (The simple code below works in this special case! Inverting is usually a bit trickier.)
        if inverse:
            result = -result

        if mode == h.expand_mode:
            result_ = result
            if stream:
                if basepoint is False:
                    stream_length = input_stream - 1
                else:
                    stream_length = input_stream
                result = torch.zeros(batch_size, stream_length, depth, dtype=dtype, device=device)
            else:
                result = torch.zeros(batch_size, depth, dtype=dtype, device=device)
            result[..., 0].unsqueeze(-1).copy_(result_)

    return result


def signatory_logsignature(class_, path, depth, stream, basepoint, inverse, mode):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on the "
                                                  "GPU.", category=UserWarning)
        if class_:
            return signatory.LogSignature(depth, mode=mode, stream=stream, inverse=inverse)(path, basepoint=basepoint)
        else:
            return signatory.logsignature(path, depth, mode=mode, stream=stream, basepoint=basepoint, inverse=inverse)


def test_forward():
    """Tests that the forward calculations of the logsignature behave correctly."""
    for class_ in (False, True):
        for device in h.get_devices():
            for batch_size in (0, 1, 2, 3):
                for input_stream in (0, 1, 2, 3, 10):
                    for input_channels in (0, 1, 2, 6):
                        for depth in (1, 2, 4, 6):
                            for mode in h.all_modes:
                                # Cuts down on the amount of iterations dramatically!
                                # We're randomising over the arguments which we happen to know aren't really important
                                # to the operation of logsignatures (as compared to signatures), so we're not missing
                                # much by doing this.
                                stream = random.choice([False, True])
                                path_grad = random.choice([False, True])
                                basepoint = random.choice([False, True, h.without_grad, h.with_grad])
                                inverse = random.choice([False, True])
                                _test_forward(class_, device, path_grad, batch_size, input_stream, input_channels,
                                              depth, stream, basepoint, inverse, mode)


def _test_forward(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                  inverse, mode):

    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)

    expected_exception = (batch_size < 1) or (input_channels < 1) or (basepoint is False and input_stream < 2) or \
                         (input_stream < 1)
    try:
        logsignature = signatory_logsignature(class_, path, depth, stream, basepoint, inverse, mode)
    except ValueError:
        if expected_exception:
            return
        else:
            raise
    else:
        assert not expected_exception

    _test_shape(logsignature, mode, batch_size, input_stream, input_channels, depth, stream, basepoint)
    h.diff(logsignature, iisignature_logsignature(path, depth, stream, basepoint, inverse, mode))

    # Check that the 'ctx' object is properly garbage collected and we don't have a memory leak
    # (Easy to accidentally happen due to PyTorch bug 25340)
    if path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad):
        ctx = logsignature.grad_fn
        if stream:
            ctx = ctx.next_functions[0][0]
        assert type(ctx).__name__ == '_SignatureToLogsignatureFunctionBackward'
        ref = weakref.ref(ctx)
        del ctx
        del logsignature
        gc.collect()
        assert ref() is None
    else:
        assert logsignature.grad_fn is None


def _test_shape(logsignature, mode, batch_size, input_stream, input_channels, depth, stream, basepoint):
    """Tests the the logsignature is of the expected shape."""
    if mode == h.expand_mode:
        correct_channels = signatory.signature_channels(input_channels, depth)
    else:
        correct_channels = signatory.logsignature_channels(input_channels, depth)
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
    assert logsignature.shape == correct_shape


def test_backward_expand_words():
    """Tests that the backwards operation through the logsignature gives the correct values."""
    for class_ in (False, True):
        for device in h.get_devices():
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 4, 6):
                    for mode in (h.expand_mode, h.words_mode):
                        stream = random.choice([False, True])
                        inverse = random.choice([False, True])
                        _test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream,
                                       basepoint, inverse, mode)


@pytest.mark.slow
def test_backward_brackets():
    """Tests that the backwards operation through the logsignature gives the correct values."""
    for class_ in (False, True):
        for device in h.get_devices():
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 4, 6):
                    for mode in (h.brackets_mode,):
                        stream = random.choice([False, True])
                        inverse = random.choice([False, True])
                        _test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream,
                                       basepoint, inverse, mode)


def _test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse, mode):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad=True)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)

    # This is the test we'd like to run, but it takes forever
    #
    # if class_:
    #     def check_fn(path, basepoint):
    #         return signatory.LogSignature(depth, stream=stream, inverse=inverse, mode=mode)(path,
    #                                                                                         basepoint=basepoint)
    # else:
    #     def check_fn(path, basepoint):
    #         return signatory.logsignature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
    #                                       mode=mode)
    # try:
    #     autograd.gradcheck(check_fn, (path, basepoint), atol=2e-05, rtol=0.002)
    # except RuntimeError:
    #     pytest.fail()

    logsignature = signatory_logsignature(class_, path, depth, stream, basepoint, inverse, mode)

    grad = torch.rand_like(logsignature)
    logsignature.backward(grad)

    path_grad = path.grad.clone()
    path.grad.zero_()
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        basepoint_grad = basepoint.grad.clone()
        basepoint.grad.zero_()

    iisignature_logsignature_result = iisignature_logsignature(path, depth, stream, basepoint, inverse, mode)
    iisignature_logsignature_result.backward(grad)

    # iisignature uses float32 for this calculation so we need a lower tolerance
    h.diff(path.grad, path_grad, atol=1e-6)
    if isinstance(basepoint, torch.Tensor) and basepoint.requires_grad:
        h.diff(basepoint.grad, basepoint_grad, atol=1e-6)


def test_no_adjustments():
    """Tests that the logsignature computations don't modify any memory that they're not supposed to."""
    for class_ in (False, True):
        for device in h.get_devices():
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 5):
                    for mode in h.all_modes:
                        path_grad = random.choice([False, True])
                        stream = random.choice([False, True])
                        inverse = random.choice([False, True])
                        _test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream,
                                             basepoint, inverse, mode, path_grad)


def _test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                         mode, path_grad):

    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)

    path_clone = path.clone()
    if isinstance(basepoint, torch.Tensor):
        basepoint_clone = basepoint.clone()

    logsignature = signatory_logsignature(class_, path, depth, stream, basepoint, inverse, mode)

    can_backward = path_grad or (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad)
    if can_backward:
        grad = torch.rand_like(logsignature)

        logsignature_clone = logsignature.clone()
        grad_clone = grad.clone()

        logsignature.backward(grad)
    else:
        assert logsignature.grad_fn is None

    h.diff(path, path_clone)
    if isinstance(basepoint, torch.Tensor):
        h.diff(basepoint, basepoint_clone)
    if can_backward:
        h.diff(logsignature, logsignature_clone)
        h.diff(grad, grad_clone)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_repeat_and_memory_leaks():
    """Performs two separate tests.

    First, that the computations are deterministic, and always give the same result when run multiple times; in
    particular that using the class signatory.LogSignature multiple times is fine.

    Second, that there are no memory leaks.
    """
    for class_ in (False, True):
        for path_grad in (False, True):
            for batch_size, input_stream, input_channels, basepoint in h.random_sizes_and_basepoint():
                for depth in (1, 2, 5):
                    for mode in h.all_modes:
                        stream = random.choice([False, True])
                        inverse = random.choice([False, True])
                        _test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels,
                                                      depth, stream, basepoint, inverse, mode)


def _test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                                  inverse, mode):
    cpu_path = h.get_path(batch_size, input_stream, input_channels, device='cpu', path_grad=False)
    cpu_basepoint = h.get_basepoint(batch_size, input_channels, device='cpu', basepoint=basepoint)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on the "
                                                  "GPU.", category=UserWarning)
        if class_:
            logsignature_instance = signatory.LogSignature(depth, stream=stream, inverse=inverse, mode=mode)
            cpu_logsignature = logsignature_instance(cpu_path, basepoint=cpu_basepoint)
        else:
            cpu_logsignature = signatory.logsignature(cpu_path, depth, stream=stream, basepoint=cpu_basepoint,
                                                      inverse=inverse, mode=mode)
    cpu_grad = torch.rand_like(cpu_logsignature)

    def one_iteration():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated()
        # device='cuda' because it's just easier to keep track of GPU memory
        cuda_path = cpu_path.to('cuda')
        if path_grad:
            cuda_path.requires_grad_()
        if isinstance(cpu_basepoint, torch.Tensor):
            cuda_basepoint = cpu_basepoint.cuda()
            if basepoint is h.with_grad:
                cuda_basepoint.requires_grad_()
        else:
            cuda_basepoint = basepoint

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on the "
                                                      "GPU.", category=UserWarning)
            if class_:
                cuda_logsignature = logsignature_instance(cuda_path, basepoint=cuda_basepoint)
            else:
                cuda_logsignature = signatory.logsignature(cuda_path, depth, stream=stream, basepoint=cuda_basepoint,
                                                           inverse=inverse, mode=mode)

        h.diff(cuda_logsignature.cpu(), cpu_logsignature)

        if path_grad:
            cuda_grad = cpu_grad.cuda()
            cuda_logsignature.backward(cuda_grad)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    memory_used = one_iteration()

    # The calculations are essentially parallel and therefore not quite deterministic in the stream==True case. This
    # means that they sometimes use a bit of extra peak memory.
    if stream:
        memory_used *= 2

    for repeat in range(10):
        try:
            assert one_iteration() <= memory_used
        except AssertionError:
            print(repeat)
            raise
