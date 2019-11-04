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
from torch import autograd

import helpers as h


@pytest.mark.parameterize('class_', (False, True))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
@pytest.mark.parameterize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('stream', (False, True))
@pytest.mark.parameterize('mode', h.all_modes)
def test_forward(class_, device, batch_size, input_stream, input_channels, depth, stream, mode):
    with h.Information(class_=class_, device=device, batch_size=batch_size, input_stream=input_stream,
                       input_channels=input_channels, depth=depth, stream=stream, mode=mode, path_grad=True) as info:
        path = h.get_path(info)
        signature = signatory.signature(path, depth, stream=stream)
        if class_:
            logsignature = signatory.SignatureToLogsignature(input_channels, depth, stream=stream, mode=mode)(signature)
        else:
            logsignature = signatory.signature_to_logsignature(signature, input_channels, depth, stream=stream,
                                                               mode=mode)
        true_logsignature = signatory.logsignature(path, depth, stream=stream, mode=mode)
        h.diff(logsignature, true_logsignature)


@pytest.mark.parameterize('class_', (False, True))
@pytest.mark.parameterize('device', ('cuda', 'cpu'))
@pytest.mark.parameterize('batch_size,input_stream,input_channels', h.random_sizes())
@pytest.mark.parameterize('depth', (1, 2, 4, 6))
@pytest.mark.parameterize('stream', (False, True))
@pytest.mark.parameterize('mode', h.all_modes)
def test_backward(class_, device, batch_size, input_stream, input_channels, depth, stream, mode):
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
