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
"""Tests the functions for combining signatures."""


import gc
import iisignature
import pytest
import random
import torch
from torch import autograd
import weakref

from helpers import helpers as h
from helpers import validation as v
from helpers import reimplementation as r


tests = ['signature_combine', 'multi_signature_combine']
depends = []
signatory = v.validate_tests(tests, depends)


# We have to use the iisignature implementation here, rather than our own, as else we end up with a dependency cycle
# in the tests, between signatory.signature and signatory.signature_combine.
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


def iisignature_signature(path, depth, stream=False, basepoint=False, inverse=False, scalar_term=False):
    """Duplicates signatory.signature's functionality using iisignature, for testing purposes."""

    def fn(path, depth):
        signature = _IisignatureSignatureFunction.apply(path, depth)
        if scalar_term:
            out = torch.ones(signature.size(0), 1 + signature.size(1), dtype=signature.dtype,
                             device=signature.device)
            out[:, 1:] = signature
            signature = out
        return signature

    return r.iisignature_signature_or_logsignature(fn, path, depth, stream, basepoint, inverse)


def test_forward():
    """Tests that the forward calculation for combing signatures produces the correct values."""
    for signature_combine, amount in ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)):
        for signature_grad in (False, True):
            for device in h.get_devices():
                for batch_size in (1, 2, 5):
                    input_stream = 2
                    for input_channels in (1, 2, 6):
                        for depth in (1, 2, 4, 6):
                            for inverse in (False, True):
                                for scalar_term in (False, True):
                                    _test_forward(signature_combine, signature_grad, amount, device, batch_size,
                                                  input_stream, input_channels, depth, inverse, scalar_term)


def _test_forward(signature_combine, signature_grad, amount, device, batch_size, input_stream, input_channels, depth,
                  inverse, scalar_term):
    paths = []
    for _ in range(amount):
        paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))
    signatures = []
    basepoint = False
    for path in paths:
        signature = iisignature_signature(path, depth, basepoint=basepoint, inverse=inverse, scalar_term=scalar_term)
        if signature_grad:
            signature.requires_grad_()
        signatures.append(signature)
        basepoint = path[:, -1]
    if signature_combine:
        combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                          inverse=inverse, scalar_term=scalar_term)
    else:
        combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                inverse=inverse, scalar_term=scalar_term)
    combined_paths = torch.cat(paths, dim=1)
    true_combined_signatures = iisignature_signature(combined_paths, depth, inverse=inverse, scalar_term=scalar_term)
    h.diff(combined_signatures, true_combined_signatures)

    if signature_grad:
        ctx = combined_signatures.grad_fn
        assert type(ctx).__name__ == '_SignatureCombineFunctionBackward'
        ref = weakref.ref(ctx)
        del ctx
        del combined_signatures
        gc.collect()
        assert ref() is None
    else:
        assert combined_signatures.grad_fn is None


def test_backward():
    """Tests that the backwards calculation for combining signatures produces the correct values."""
    for signature_combine, amount in ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)):
        for device in h.get_devices():
            for batch_size, input_stream, input_channels in h.random_sizes():
                for depth in (1, 2, 4, 6):
                    for scalar_term in (False, True):
                        inverse = random.choice([False, True])
                        _test_backward(signature_combine, amount, device, batch_size, input_stream, input_channels,
                                       depth, inverse, scalar_term)


def _test_backward(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse,
                   scalar_term):
    paths = []
    for _ in range(amount):
        paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double,
                                requires_grad=True))
    signatures = []
    basepoint = False
    for path in paths:
        signature = iisignature_signature(path, depth, basepoint=basepoint, inverse=inverse, scalar_term=scalar_term)
        signatures.append(signature)
        basepoint = path[:, -1]

    # This is the test we'd like to run here, but it takes too long.
    # Furthermore we'd also prefer to only go backwards through the signature combine, not through the signature, but
    # we can't really do that with our faster alternative.
    #
    # if signature_combine:
    #     def check_fn(*signatures):
    #         return signatory.signature_combine(signatures[0], signatures[1], input_channels, depth, inverse=inverse)
    # else:
    #     def check_fn(*signatures):
    #         return signatory.multi_signature_combine(signatures, input_channels, depth, inverse=inverse)
    # try:
    #     autograd.gradcheck(check_fn, tuple(signatures), atol=2e-05, rtol=0.002)
    # except RuntimeError:
    #     pytest.fail()

    if signature_combine:
        combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                          inverse=inverse, scalar_term=scalar_term)
    else:
        combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth, inverse=inverse,
                                                                scalar_term=scalar_term)
    grad = torch.rand_like(combined_signatures)
    combined_signatures.backward(grad)
    path_grads = [path.grad.clone() for path in paths]
    for path in paths:
        path.grad.zero_()

    true_signature = iisignature_signature(torch.cat(paths, dim=1), depth, inverse=inverse, scalar_term=scalar_term)
    true_signature.backward(grad)
    for path_grad, path in zip(path_grads, paths):
        h.diff(path_grad, path.grad)


def test_no_adjustments():
    """Tests that the calculations for combining signatures don't modify memory they're not supposed to."""
    for signature_combine, amount in ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)):
        for signature_grad in (False, True):
            for device in h.get_devices():
                for batch_size, input_stream, input_channels in h.random_sizes():
                    for depth in (1, 2, 5):
                        for inverse in (False, True):
                            for scalar_term in (False, True):
                                _test_no_adjustments(signature_combine, amount, device, batch_size, input_stream,
                                                     input_channels, depth, inverse, signature_grad, scalar_term)


def _test_no_adjustments(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse,
                         signature_grad, scalar_term):
    paths = []
    for _ in range(amount):
        paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))

    signatures = []
    signatures_clone = []
    basepoint = False
    for path in paths:
        signature = iisignature_signature(path, depth, basepoint=basepoint, inverse=inverse, scalar_term=scalar_term)
        signatures_clone.append(signature.clone())
        if signature_grad:
            signature.requires_grad_()
        signatures.append(signature)
        basepoint = path[:, -1]
    if signature_combine:
        combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                          inverse=inverse, scalar_term=scalar_term)
    else:
        combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                inverse=inverse, scalar_term=scalar_term)

    if signature_grad:
        grad = torch.rand_like(combined_signatures)
        grad_clone = grad.clone()
        combined_signatures_clone = combined_signatures.clone()
        combined_signatures.backward(grad)

    for signature, signature_clone in zip(signatures, signatures_clone):
        h.diff(signature, signature_clone)
    if signature_grad:
        h.diff(grad, grad_clone)
        h.diff(combined_signatures, combined_signatures_clone)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_memory_leaks():
    """Checks that there are no memory leaks."""
    for signature_combine, amount in ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)):
        for signature_grad in (False, True):
            for batch_size, input_stream, input_channels in h.random_sizes():
                for depth in (1, 2, 5):
                    for inverse in (False, True):
                        for scalar_term in (False, True):
                            _test_memory_leaks(signature_combine, amount, batch_size, input_stream, input_channels,
                                               depth, inverse, signature_grad, scalar_term)


def _test_memory_leaks(signature_combine, amount, batch_size, input_stream, input_channels, depth, inverse,
                       signature_grad, scalar_term):

    def one_iteration():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated()
        paths = []
        for _ in range(amount):
            paths.append(torch.rand(batch_size, input_stream, input_channels, device='cuda', dtype=torch.double))
        signatures = []
        basepoint = False
        for path in paths:
            signature = iisignature_signature(path, depth, basepoint=basepoint, inverse=inverse,
                                              scalar_term=scalar_term)
            if signature_grad:
                signature.requires_grad_()
            signatures.append(signature)
        if signature_combine:
            combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                              inverse=inverse, scalar_term=scalar_term)
        else:
            combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                    inverse=inverse, scalar_term=scalar_term)
        if signature_grad:
            grad = torch.rand_like(combined_signatures)
            combined_signatures.backward(grad)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    memory_used = one_iteration()

    for repeat in range(10):
        assert one_iteration() <= memory_used
