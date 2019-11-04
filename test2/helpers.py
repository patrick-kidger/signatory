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


import signatory
import torch


with_grad = object()
without_grad = object()


expand_mode = 'expand'
words_mode = 'words'
brackets_mode = 'brackets'
all_modes = (expand_mode, words_mode, brackets_mode)


class Information(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)
        super(Information, self).__init__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            pieces = []
            for key, val in self.kwargs:
                pieces.append("{}: {}".format(key, val))
            raise exc_type('\n'.join(pieces))


def diff(arg1, arg2):
    if not arg1.allclose(arg2):
        diff = arg1 - arg2
        max_diff = torch.max(torch.abs(diff))
        assert False, 'diff={diff}\nmax_diff={max_diff}\narg1={arg1}arg2={arg2}'.format(diff=diff, max_diff=max_diff,
                                                                                        arg1=arg1, arg2=arg2)


def get_path(info):
    return torch.rand(info.batch_size, info.input_stream, info.input_channels, device=info.device,
                      requires_grad=info.path_grad, dtype=torch.double)


def get_basepoint(info):
    if info.basepoint == without_grad:
        return torch.rand(info.batch_size, info.input_channels, device=info.device, dtype=torch.double)
    elif info.basepoint == with_grad:
        return torch.rand(info.batch_size, info.input_channels, device=info.device, requires_grad=True,
                          dtype=torch.double)
    else:
        return info.basepoint


def get_initial(info):
    if info.initial in (without_grad, with_grad):
        initial_path = torch.rand(info.batch_size, 2, info.input_channels, device=info.device, dtype=torch.double)
        initial = signatory.signature(initial_path, info.depth)
        if info.initial == with_grad:
            initial.requires_grad_()
        return initial
    else:
        return info.initial


def random_sizes():
    params = []
    for batch_size in (1, 2):
        for input_stream in (1, 2):
            for input_channels in (1, 2):
                params.append((batch_size, input_stream, input_channels))
    for _ in range(5):
        batch_size = int(torch.randint(low=1, high=10, size=(1,)))
        input_stream = int(torch.randint(low=2, high=10, size=(1,)))
        input_channels = int(torch.randint(low=1, high=10, size=(1,)))
        params.append((batch_size, input_stream, input_channels))
    return params


def random_sizes_and_basepoint():
    params = []
    for batch_size in (1, 2):
        for input_stream in (1, 2):
            for input_channels in (1, 2):
                for basepoint in (True, without_grad, with_grad):
                    params.append((batch_size, input_stream, input_channels, basepoint))
    for batch_size in (1, 2):
        for input_stream in (2,):
            for input_channels in (1, 2):
                for basepoint in (False,):
                    params.append((batch_size, input_stream, input_channels, basepoint))
    for _ in range(5):
        for basepoint in (True, h.without_grad, h.with_grad):
            batch_size = int(torch.randint(low=1, high=10, size=(1,)))
            input_stream = int(torch.randint(low=2, high=10, size=(1,)))
            input_channels = int(torch.randint(low=1, high=10, size=(1,)))
            params.append((batch_size, input_stream, input_channels, basepoint))
    return params
