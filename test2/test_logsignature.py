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


import iisignature
import pytest
import signatory
import torch
from torch import autograd

import helpers as h


@pytest.mark.parameterize('class_', (False, True))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
# sometimes we do different calculations depending on whether we expect to take a gradient later, so we need to
# check both of these cases
@pytest.mark.parameterize('path_grad', (False, True))
@pytest.mark.parameterize('batch_size', (0, 1, 2, 5))
@pytest.mark.parameterize('input_stream', (0, 1, 2, 3, 10))
@pytest.mark.parameterize('input_channels', (0, 1, 2, 6))
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('stream', (False, True))
@pytest.mark.parameterize('basepoint', (False, True, h.without_grad, h.with_grad))
@pytest.mark.parameterize('inverse', (False, True))
@pytest.mark.parameterize('mode', h.all_modes)
def test_forward(class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse,
                 mode, iisignature_prepare):

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


def _test_shape(logsignature, info):
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


@pytest.mark.parameterize('class_', (False, True))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
@pytest.mark.parameterize('batch_size,input_stream,input_channels,basepoint', h.random_sizes_and_basepoint())
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('stream', (False, True))
@pytest.mark.parameterize('inverse', (False, True))
@pytest.mark.parameterize('mode', h.all_modes)
def test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, basepoint, inverse, mode):
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
