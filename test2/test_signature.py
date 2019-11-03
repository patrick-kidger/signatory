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
import signatory
import torch

import helpers as h


# Replace the weak dictionary with a regular dictionary for speed
if not hasattr(signatory.SignatureToLogSignature, '_lyndon_info_capsule_cache'):
    raise RuntimeError('Expected signatory.SignatureToLogSignature to have a cache for lyndon info capsules')
signatory.SignatureToLogSignature._lyndon_info_capsule_cache = {}


class TestSignature(object):
    def __init__(self):
        self.tested_batch_trick = False
        super(TestSignature, self).__init__()

    def test_forward(self):
        for class_ in (True, False):
            for device in ('cuda', 'cpu'):
                # sometimes we do different calculations depending on whether we expect to take a gradient later, so we
                # need to check both of these cases
                for path_grad in (False, True):
                    for batch_size in (0, 1, 2, 5):
                        for input_stream in (0, 1, 2, 3, 10):
                            for input_channels in (0, 1, 2, 6):
                                for depth in (1, 2, 4, 6):
                                    for stream in (False, True):
                                        for basepoint in (False, True, h.without_grad, h.with_grad):
                                            for inverse in (False, True):
                                                for initial in (None, h.without_grad, h.with_grad):
                                                    self._test_forward(class_, device, path_grad, batch_size,
                                                                       input_stream, input_channels, depth, stream,
                                                                       basepoint, inverse, initial)

    def _test_forward(self, class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream, basepoint,
                      inverse, initial):

        with h.Information(class_=class_, device=device, path_grad=path_grad, batch_size=batch_size,
                           input_stream=input_stream, input_channels=input_channels, depth=depth, stream=stream,
                           basepoint=basepoint, inverse=inverse, initial=initial) as info:
            path = torch.rand(batch_size, input_stream, input_channels, device=device, requires_grad=path_grad,
                              dtype=torch.double)
            if basepoint is h.without_grad:
                basepoint = torch.rand(batch_size, input_channels, device=device, dtype=torch.double)
            elif basepoint is h.with_grad:
                basepoint = torch.rand(batch_size, input_channels, device=device, requires_grad=True, dtype=torch.double)
            if initial in (h.without_grad, h.with_grad):
                initial_path = torch.rand(batch_size, 2, input_channels, device=device, dtype=torch.double)
                initial_ = signatory.signature(initial_path, depth)
                if initial is h.with_grad:
                    initial_.requires_grad_()
                initial = initial_

            try:
                if class_:
                    signature = signatory.Signature(depth, stream=stream, inverse=inverse)(path, basepoint=basepoint,
                                                                                           initial=initial)
                else:
                    signature = signatory.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse,
                                                    initial=initial)
            except ValueError:
                if (batch_size < 1) or (input_stream < 2) or (input_channels < 1):
                    # expected exception
                    return
                else:
                    raise
            else:
                assert not ((batch_size < 1) or (input_stream < 2) or (input_channels < 1))

            self._test_shape(signature)
            self._test_forward_accuracy(signature, path, info)
            self._test_batch_trick(signature, path, info)
            self._test_signature_to_logsignature(signature, info)

    def _test_forward_accuracy(self, signature, path, info):
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

    def _test_batch_trick(self, signature, path, info):
        batch_trick_signature = signatory.signature.__globals__['_signature_batch_trick'](path,
                                                                                          info.depth,
                                                                                          stream=info.stream,
                                                                                          basepoint=info.basepoint,
                                                                                          inverse=info.inverse,
                                                                                          initial=info.initial)
        if batch_trick_signature is None:
            return
        self.tested_batch_trick = True

        h.diff(signature, batch_trick_signature)

        path_has_grad = path.requires_grad
        basepoint_has_grad = isinstance(info.basepoint, torch.Tensor) and info.basepoint.requires_grad
        initial_has_grad = isinstance(info.initial, torch.Tensor) and info.initial.requires_grad
        if not (path_has_grad or basepoint_has_grad or initial_has_grad):
            return

        grad = torch.rand_like(signature)
        signature.backward(grad)
        if path_has_grad:
            path_grad = path.grad.clone()
            path.grad.zero_()
        if basepoint_has_grad:
            basepoint_grad = info.basepoint.grad.clone()
            info.basepoint.grad.zero_()
        if initial_has_grad:
            initial_grad = info.initial.grad.clone()
            info.initial.grad.zero_()
        batch_trick_signature.backward(grad)
        if path_has_grad:
            h.diff(path_grad, path.grad)
            path.grad.zero_()
        if basepoint_has_grad:
            h.diff(basepoint_grad, info.basepoint.grad)
            info.basepoint.grad.zero_()
        if initial_has_grad:
            h.diff(initial_grad, info.initial.grad)
            info.initial.grad.zero_()


class TestLogSignature(object):
    def test_forward(self):
        for class_ in (True, False):
            for device in ('cuda', 'cpu'):
                # sometimes we do different calculations depending on whether we expect to take a gradient later, so we
                # need to check both of these cases
                for path_grad in (False, True):
                    for batch_size in (0, 1, 2, 5):
                        for input_stream in (0, 1, 2, 3, 10):
                            for input_channels in (0, 1, 2, 6):
                                for depth in (1, 2, 4, 6):
                                    for stream in (False, True):
                                        for basepoint in (False, True, h.without_grad, h.with_grad):
                                            for inverse in (False, True):
                                                for mode in h.all_modes:
                                                    self._test_forward(class_, device, path_grad, batch_size,
                                                                       input_stream, input_channels, depth, stream,
                                                                       basepoint, inverse, mode)

    def _test_forward(self, class_, device, path_grad, batch_size, input_stream, input_channels, depth, stream,
                      basepoint, inverse, mode):

        with h.Information(device=device, path_grad=path_grad, batch_size=batch_size, input_stream=input_stream,
                           input_channels=input_channels, depth=depth, stream=stream, basepoint=basepoint,
                           inverse=inverse, mode=mode) as info:
            path = torch.rand(batch_size, input_stream, input_channels, device=device, requires_grad=path_grad,
                              dtype=torch.double)
            if basepoint is h.without_grad:
                basepoint = torch.rand(batch_size, input_channels, device=device, dtype=torch.double)
            elif basepoint is h.with_grad:
                basepoint = torch.rand(batch_size, input_channels, device=device, requires_grad=True, dtype=torch.double)

            try:
                if class_:
                    logsignature = signatory.LogSignature(depth, mode=mode, stream=stream,
                                                          inverse=inverse)(path, basepoint=basepoint)
                else:
                    logsignature = signatory.logsignature(path, depth, mode=mode, stream=stream, basepoint=basepoint,
                                                          inverse=inverse)
            except ValueError:
                if (batch_size < 1) or (input_stream < 2) or (input_channels < 1):
                    # expected exception
                    return
                else:
                    raise
            else:
                assert not ((batch_size < 1) or (input_stream < 2) or (input_channels < 1))

            self._test_shape(logsignature)
            self._test_forward_accuracy(logsignature, path, info)

    def _test_forward_accuracy(self, logsignature, path, info):
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

                if info.mode is h.expand_mode:
                    iisignature_mode = 'x'
                else:
                    iisignature_mode = 'd'

                return iisignature.logsig(iisignature_path, h.iisignature_prepare(info.input_channels, info.depth),
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

        if info.mode is h.words_mode:
            transforms = signatory.unstable.lyndon_words_to_basis_transform(info.input_channels, info.depth)
            logsignature = logsignature.clone()
            for source, target, coefficient in transforms:
                logsignature[:, :, target] -= coefficient * logsignature[:, :, source]

        h.diff(logsignature, torch.tensor(iisignature_logsignature, dtype=torch.double, device=info.device))


class TestSignatureCombine:
    pass
    # TODO
