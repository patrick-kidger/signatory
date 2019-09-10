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
"""Provides the Path class, a high-level object capable of giving signatures over intervals."""

import torch
from torch import autograd

from . import signature_module as smodule
from . import _impl

# noinspection PyUnreachableCode
if False:
    from typing import Union


class _TensorAlgebraMult(autograd.Function):
    @staticmethod
    def forward(ctx, arg1, arg2, input_channels, depth):
        ctx.save_for_backward(arg1, arg2)
        ctx.input_channels = input_channels
        ctx.depth = depth
        return _impl.tensor_algebra_mult_forward(arg1, arg2, input_channels, depth)

    @staticmethod
    def backward(ctx, grad):
        arg1, arg2 = ctx.saved_tensors
        grad_arg1, grad_arg2 = _impl.tensor_algebra_mult_backward(grad, arg1, arg2, ctx.input_channels, ctx.depth)
        return grad_arg1, grad_arg2, None, None


class Path(object):
    """Calculates signatures on intervals of an input path.

    By doing some precomputation, it can rapidly calculate the signature of the input path over any interval.

    Arguments:
        path (torch.Tensor): As :func:`signatory.signature`.

        depth (int): As :func:`signatory.signature`.

        basepoint (bool or torch.Tensor, optional): As :func:`signatory.signature`.
    """
    def __init__(self, path, depth, basepoint=False):
        # type: (torch.Tensor, int, Union[bool, torch.Tensor]) -> None
        self._depth = depth
        self._batch = path.size(0)
        self._length = path.size(1)
        self._channels = path.size(2)
        self._end = path[:, -1, :]

        self._signature = smodule.signature(path, depth, stream=True, basepoint=basepoint)
        self._reverse_signature = smodule.signature(path, depth, stream=True, basepoint=basepoint, inverse=True)
        self._signature_length = self._signature.size(1)
        self._signature_channels = self._signature.size(2)

    def signature(self, start, end):
        # type: (int, int) -> torch.Tensor
        """Returns the signature on a particular interval.

        Arguments:
            start (int): The start point of the interval to calculate the signature on.

            end (int): The end point of the interval to calcluate the signature on.

        Returns:
            The signature on the interval :attr:`[start, end]`. That is, let :attr:`p` be the input :attr:`path`
            with basepoint prepended. Then this function is equivalent to
            :attr:`signatory.signature(p[start:end], depth)`.
        """

        old_start = start
        old_end = end

        # We're duplicating slicing behaviour, which means to accept values even beyond the normal indexing range
        length = self.signature_length + 1
        if start < -length:
            start = -length
        elif start > length:
            start = length
        if end < -length:
            end = -length
        elif end > length:
            end = length
        # Accept negative indices
        if start < 0:
            start += length
        if end < 0:
            end += length

        if end - start < 2:
            raise ValueError("start={}, end={} is interpreted as start={}, end={} for path of length {}, which "
                             "does not describe a valid interval.".format(old_start, old_end, start, end, self.length))

        start -= 1
        end -= 2

        if start == -1:
            return self._signature[:, end, :]
        rev = self._reverse_signature[:, start, :]
        sig = self._signature[:, end, :]

        return _TensorAlgebraMult.apply(rev, sig, self.channels, self.depth)

    # TODO: update
    # TODO: logsignature

    @property
    def depth(self):
        return self._depth

    @property
    def batch(self):
        return self._batch

    @property
    def length(self):
        return self._length

    @property
    def channels(self):
        return self._channels

    @property
    def signature_length(self):
        return self._signature_length

    @property
    def signature_channels(self):
        return self._signature_channels
