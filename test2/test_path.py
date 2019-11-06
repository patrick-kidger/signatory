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
import signatory
import torch
import weakref


import helpers as h


def _update_lengths_update_grads():
    num = int(torch.randint(low=0, high=2, size=(1,)))
    update_lengths = []
    update_grads = []
    for _ in range(num):
        update_lengths.append(int(torch.randint(low=1, high=10, size=(1,))))
        update_grads.append(random.choice([True, False]))
    return zip(update_lengths, update_grads)


@pytest.mark.parametrize('device', h.get_devices())
# sometimes we do different calculations depending on whether we expect to take a gradient later, so we need to
# check both of these cases
@pytest.mark.parametrize('path_grad', (False, True))
@pytest.mark.parametrize('batch_size', (0, 1, 2, 5))
@pytest.mark.parametrize('input_stream', (0, 1, 2, 3, 10))
@pytest.mark.parametrize('input_channels', (0, 1, 2, 6))
@pytest.mark.parametrize('depth', (1, 2, 4, 6))
@pytest.mark.parametrize('basepoint', (False, True, h.without_grad, h.with_grad))
@pytest.mark.parametrize('update_lengths,update_grads', _update_lengths_update_grads())
def test_path(device, path_grad, batch_size, input_stream, input_channels, depth, basepoint, update_lengths,
              update_grads):
    """Tests that Path behaves correctly."""

    with h.Information(device=device, path_grad=path_grad, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, basepoint=basepoint, update_lengths=update_lengths,
                       update_grads=update_grads) as info:
        path = h.get_path(info)
        basepoint = h.get_path(info)
        path_obj = signatory.Path(path, depth, basepoint=basepoint)

        if isinstance(basepoint, torch.Tensor):
            full_path = torch.cat([basepoint, path], dim=1)
        elif basepoint is True:
            full_path = torch.cat([torch.zeros(batch_size, 1, input_channels), path], dim=1)
        else:
            full_path = path

        has_grad = path_grad or info.basepoint is with_grad

        _test_signature(path_obj, full_path, info, has_grad)
        _test_logsignature(path_obj, full_path, info, has_grad)
        assert path_obj.depth == depth

        for length, grad in zip(update_lengths, update_grads):
            has_grad = has_grad or grad
            new_path = torch.rand(batch_size, length, input_channels, dtype=torch.double, device=device,
                                  requires_grad=grad)
            path_obj.update(new_path)
            full_path = torch.cat([full_path, new_path], dim=1)

            _test_signature(path_obj, full_path, info, has_grad)
            _test_logsignature(path_obj, full_path, info, has_grad)
            assert path_obj.depth == depth


def _test_signature(path_obj, full_path, info, has_grad):
    gc.collect()
    torch.cuda.reset_max_memory_allocated()
    path_obj.signature()
    memory_used = torch.cuda.memory_allocated()
    for start in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
        for end in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
            with h.Information(start=start, end=end):
                gc.collect()
                torch.cuda.reset_max_memory_allocated()
                try:
                    signature = path_obj.signature(start, end)
                except ValueError:
                    try:
                        signatory.signature(full_path, info.depth)
                    except ValueError:
                        continue
                    else:
                        pytest.fail()
                assert torch.cuda.memory_allocated() <= memory_used
                try:
                    true_signature = signatory.signature(full_path, info.depth)
                except ValueError:
                    pytest.fail()

                h.diff(signature, true_signature)
                assert (path_obj.signature_size(-3), path_obj.signature_size(-1)) == true_signature.shape
                assert path_obj.signature_channels() == true_signature.size(-1)
                assert path_obj.shape == full_path.shape
                assert path_obj.channels() == full_path.size(-1)

                if has_grad:
                    ctx = signature.grad_fn
                    assert type(ctx).__name__ == '_BackwardShortcutBackward'
                    ref = weakref.ref(ctx)
                    del ctx
                    del signature
                    gc.collect()
                    assert ref() is None
                else:
                    assert signature.grad_fn is None


def _test_logsignature(path_obj, full_path, info, has_grad):
    for mode in h.all_modes:
        gc.collect()
        torch.cuda.reset_max_memory_allocated()
        path_obj.logsignature(mode=mode)
        memory_used = torch.cuda.memory_allocated()
        for start in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
            for end in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
                with h.Information(start=start, end=end, mode=mode):
                    gc.collect()
                    torch.cuda.reset_max_memory_allocated()
                    try:
                        logsignature = path_obj.logsignature(start, end, mode=mode)
                    except ValueError:
                        try:
                            signatory.logsignature(full_path, info.depth, mode=mode)
                        except ValueError:
                            continue
                        else:
                            pytest.fail()
                    assert torch.cuda.memory_allocated() <= memory_used
                    try:
                        true_logsignature = signatory.logsignature(full_path, info.depth, mode=mode)
                    except ValueError:
                        pytest.fail()

                    h.diff(logsignature, true_logsignature)
                    if mode != h.expand_mode:
                        assert (path_obj.logsignature_size(-3),
                                path_obj.logsignature_size(-1)) == true_logsignature.shape
                        assert path_obj.logsignature_channels() == true_logsignature.size(-1)

                    if has_grad:
                        ctx = logsignature.grad_fn
                        assert type(ctx).__name__ == '_SignatureToLogSignatureFunctionBackward'
                        ref = weakref.ref(ctx)
                        del ctx
                        del logsignature
                        gc.collect()
                        assert ref() is None
                    else:
                        assert logsignature.grad_fn is None


# TODO: backward tests
