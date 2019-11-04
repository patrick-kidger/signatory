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


import pytest
import signatory
import torch
from torch import autograd

import helpers as h


pytestmark = pytest.mark.usefixtures('no_parallelism')


@pytest.mark.parameterize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
@pytest.mark.parameterize('batch_size', (1, 2, 5))
@pytest.mark.parameterize('input_stream', (2,))
@pytest.mark.parameterize('input_channels', (1, 2, 6))
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('inverse', (False, True))
def test_forward(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse):
    with h.Information(signature_combine=signature_combine, amount=amount, device=device, batch_size=batch_size,
                       input_stream=input_stream, input_channels=input_channels, depth=depth, inverse=inverse):
        paths = []
        for _ in range(amount):
            paths.append(torch.rand(batch_size, input_stream, input_channels, device=device, dtype=torch.double))
        signatures = []
        basepoint = False
        for path in paths:
            signatures.append(signatory.signature(path, depth, basepoint=basepoint, inverse=inverse))
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


@pytest.mark.parameterize('signature_combine,amount', ((True, 2), (False, 1), (False, 2), (False, 3), (False, 10)))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
@pytest.mark.parameterize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('inverse', (False, True))
def test_backward(signature_combine, amount, device, batch_size, input_stream, input_channels, depth, inverse):
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
