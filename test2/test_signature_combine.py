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
import pytest
import signatory
import torch
from torch import autograd
import weakref

import helpers as h


pytestmark = pytest.mark.usefixtures('no_parallelism')


@pytest.mark.parametrize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parametrize('signature_grad', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size', (1, 2, 5))
@pytest.mark.parametrize('input_stream', (2,))
@pytest.mark.parametrize('input_channels', (1, 2, 6))
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('inverse', (False, True))
def test_forward(signature_combine, signature_grad, amount, device, batch_size, input_stream, input_channels, depth,
                 inverse):
    """Tests that the forward calculation for combing signatures produces the correct values."""
    with h.Information(signature_combine=signature_combine, amount=amount, device=device, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, inverse=inverse,
                       signature_grad=signature_grad, path_grad=False):
        paths = []
        for _ in range(amount):
            paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))
        signatures = []
        basepoint = False
        for path in paths:
            signature = signatory.signature(path, depth, basepoint=basepoint, inverse=inverse)
            if signature_grad:
                signature.requires_grad_()
            signatures.append(signature)
            basepoint = path[:, -1]
        if signature_combine:
            combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                              inverse=inverse)
        else:
            combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                    inverse=inverse)
        combined_paths = torch.cat(paths, dim=1)
        true_combined_signatures = signatory.signature(combined_paths, depth, inverse=inverse)
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


@pytest.mark.parametrize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('inverse', (False, True))
def test_backward(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse):
    """Tests that the backwards calculation for combing signatures produces the correct values."""
    with h.Information(signature_combine=signature_combine, amount=amount, device=device, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, inverse=inverse):
        paths = []
        for _ in range(amount):
            paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))
        signatures = []
        basepoint = False
        for path in paths:
            signature = signatory.signature(path, depth, basepoint=basepoint, inverse=inverse)
            signature.requires_grad_()
            signatures.append(signature)
            basepoint = path[:, -1]
        if signature_combine:
            def check_fn(*signatures):
                return signatory.signature_combine(signatures[0], signatures[1], input_channels, depth, inverse=inverse)
        else:
            def check_fn(*signatures):
                return signatory.multi_signature_combine(signatures, input_channels, depth, inverse=inverse)
        try:
            autograd.gradcheck(check_fn, tuple(signatures), atol=2e-05, rtol=0.002)
        except RuntimeError:
            pytest.fail()


@pytest.mark.parametrize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parametrize('signature_grad', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('inverse', (False, True))
def test_no_adjustments(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse,
                        signature_grad):
    """Tests that the calculations for combining signatures don't modify memory they're not supposed to."""
    with h.Information(signature_combine=signature_combine, amount=amount, device=device, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, inverse=inverse,
                       signature_grad=signature_grad, path_grad=False) as info:
        paths = []
        for _ in range(amount):
            paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))

        signatures = []
        signatures_clone = []
        basepoint = False
        for path in paths:
            signature = signatory.signature(path, depth, basepoint=basepoint, inverse=inverse)
            signatures_clone.append(signature.clone())
            if signature_grad:
                signature.requires_grad_()
            signatures.append(signature)
            basepoint = path[:, -1]
        if signature_combine:
            combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                              inverse=inverse)
        else:
            combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                    inverse=inverse)

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


@pytest.mark.skipif(not torch.cuda.is_available())
@pytest.mark.parametrize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parametrize('signature_grad', (False, True))
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('inverse', (False, True))
def test_memory_leaks(signature_combine, amount, batch_size, input_stream, input_channels, depth, inverse,
                      signature_grad):
    """Checks that there are no memory leaks."""

    with h.Information(signature_combine=signature_combine, amount=amount, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, inverse=inverse,
                       signature_grad=signature_grad):

        def one_iteration():
            gc.collect()
            torch.cuda.reset_max_memory_allocated()
            paths = []
            for _ in range(amount):
                paths.append(torch.rand(batch_size, input_stream, input_channels, device='cuda', dtype=torch.double))
            signatures = []
            basepoint = False
            for path in paths:
                signature = signatory.signature(path, depth, basepoint=basepoint, inverse=inverse)
                if signature_grad:
                    signature.requires_grad_()
                signatures.append(signature)
            if signature_combine:
                combined_signatures = signatory.signature_combine(signatures[0], signatures[1], input_channels, depth,
                                                                  inverse=inverse)
            else:
                combined_signatures = signatory.multi_signature_combine(signatures, input_channels, depth,
                                                                        inverse=inverse)
            if signature_grad:
                grad = torch.rand_like(combined_signatures)
                combined_signatures.backward(grad)
            return torch.cuda.max_memory_allocated()

        memory_used = one_iteration()

        for repeat in range(10):
            assert one_iteration() <= memory_used
