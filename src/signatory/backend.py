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
"""Provides certain low-level helpers."""


import torch
from torch import autograd

from . import compatibility as compat
from . import _impl


def interpret_basepoint(basepoint, path):
    if basepoint is True:
        basepoint_value = torch.zeros((path.shape[0], path.shape[2]), dtype=path.dtype, device=path.device)
    elif isinstance(basepoint, torch.Tensor):
        basepoint_value = basepoint
        basepoint = True
    else:
        basepoint_value = torch.Tensor()
    return basepoint, basepoint_value


class TensorAlgebraMult(autograd.Function):
    @staticmethod
    def forward(ctx, arg1, arg2, input_channels, depth):
        ctx.save_for_backward(arg1, arg2)
        ctx.input_channels = input_channels
        ctx.depth = depth
        with compat.mac_exception_catcher:
            return _impl.tensor_algebra_mult_forward(arg1, arg2, input_channels, depth)

    @staticmethod
    def backward(ctx, grad):
        arg1, arg2 = ctx.saved_tensors
        with compat.mac_exception_catcher:
            grad_arg1, grad_arg2 = _impl.tensor_algebra_mult_backward(grad, arg1, arg2, ctx.input_channels, ctx.depth)
        return grad_arg1, grad_arg2, None, None
