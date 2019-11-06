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
"""Tests that the computations for converting signatues to logsignatures behave correctly."""


import gc
import pytest
import signatory
import torch
from torch import autograd
import weakref

import helpers as h


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('signature_grad', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_forward(class_, device, batch_size, input_stream, input_channels, depth, stream, mode, signature_grad):
    """Tests that the forward calculations produce the correct values."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, mode=mode,
                       signature_grad=signature_grad, path_grad=False) as info:
        path = h.get_path(info)
        signature = signatory.signature(path, depth, stream=stream)
        if signature_grad:
            signature.requires_grad_()
        if class_:
            logsignature = signatory.SignatureToLogsignature(input_channels, depth, stream=stream, mode=mode)(signature)
        else:
            logsignature = signatory.signature_to_logsignature(signature, input_channels, depth, stream=stream,
                                                               mode=mode)
        true_logsignature = signatory.logsignature(path, depth, stream=stream, mode=mode)
        h.diff(logsignature, true_logsignature)

        if signature_grad:
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


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
def test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, mode):
    """Tests that the backward calculations produce the correct values."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, mode=mode, path_grad=False) as info:
        path = h.get_path(info)
        signature = signatory.signature(path, depth, stream=stream)
        signature.requires_grad_()
        if class_:
            def check_fn(signature):
                return signatory.SignatureToLogSignature(input_channels, depth, stream=stream, mode=mode)(signature)
        else:
            def check_fn(signature):
                return signatory.signature_to_logsignature(signature, input_channels, depth, stream=stream, mode=mode)
        try:
            autograd.gradcheck(check_fn, (signature,), atol=2e-05, rtol=0.002)
        except RuntimeError:
            pytest.fail()


@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('device', h.get_devices())
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
@pytest.mark.parametrize('signature_grad', (False, True))
def test_no_adjustments(class_, device, batch_size, input_stream, input_channels, depth, stream, mode, signature_grad):
    """Tests that no memory is modified that shouldn't be modified."""
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, mode=mode, path_grad=False,
                       signature_grad=signature_grad) as info:
        path = h.get_path(info)
        signature = signatory.signature(path, depth, stream=stream)
        signature_clone = signature.clone()
        if signature_grad:
            signature.requires_grad_()
        if class_:
            logsignature = signatory.SignatureToLogsignature(input_channels, depth, stream=stream, mode=mode)(signature)
        else:
            logsignature = signatory.signature_to_logsignature(signature, input_channels, depth, stream=stream,
                                                               mode=mode)

        if signature_grad:
            grad = torch.rand_like(logsignature)
            logsignature_clone = logsignature.clone()
            grad_clone = grad.clone()
            logsignature.backward(grad)
        else:
            assert logsignature.grad_fn is None

        h.diff(signature, signature_clone)
        if signature_grad:
            h.diff(logsignature, logsignature_clone)
            h.diff(grad, grad_clone)


@pytest.mark.skipif(not torch.cuda.is_available())
@pytest.mark.parametrize('class_', (False, True))
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parametrize('depth', (1, 2, 5))
@pytest.mark.parametrize('stream', (False, True))
@pytest.mark.parametrize('mode', h.all_modes)
@pytest.mark.parametrize('signature_grad', (False, True))
def test_repeat_and_memory_leaks(class_, path_grad, batch_size, input_stream, input_channels, depth, stream, mode,
                                 signature_grad):
    """Performs two separate tests.

    First, that the computations are deterministic, and always give the same result when run multiple times; in
    particular that using the class signatory.SignatureToLogSignature multiple times is fine.

    Second, that there are no memory leaks.
    """

    # device='cuda' because it's just easier to keep track of GPU memory
    with h.Information(class_=class_, batch_size=batch_size, input_stream=input_stream, input_channels=input_channels,
                       depth=depth, stream=stream, mode=mode, signature_grad=signature_grad, path_grad=False,
                       device='cuda') as info:


        cpu_path = h.get_path(info).cpu()
        cpu_signature = signatory.signature(cpu_path, depth, stream=stream)
        if class_:
            signature_to_logsignature_instance = signatory.SignatureToLogsignature(input_channels, depth, stream=stream,
                                                                                   mode=mode)
            cpu_logsignature = signature_to_logsignature_instance(cpu_signature)
        else:
            cpu_logsignature = signatory.signature_to_logsignature(cpu_signature, input_channels, depth, stream=stream,
                                                                   mode=mode)
        cpu_grad = torch.rand_like(cpu_logsignature)

        def one_iteration():
            gc.collect()
            torch.cuda.reset_max_memory_allocated()
            cuda_signature = cpu_signature.to('cuda')
            if signature_grad:
                cuda_signature.requires_grad_()
            if class_:
                cuda_logsignature = signature_to_logsignature_instance(cuda_signature)
            else:
                cuda_logsignature = signatory.signature_to_logsignature(cuda_signature, input_channels, depth,
                                                                        stream=stream, mode=mode)

            h.diff(cuda_logsignature.cpu(), cpu_logsignature)

            if signature_grad:
                cuda_grad = cpu_grad.cuda()
                cuda_logsignature.backward(cuda_grad)
            return torch.cuda.max_memory_allocated()

        memory_used = one_iteration()

        for repeat in range(10):
            assert one_iteration() <= memory_used
