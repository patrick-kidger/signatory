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

# noinspection PyUnresolvedReferences
from . import _impl


def forward(ctx, path, depth, stream, basepoint, fn_forward, extra_args=()):
    ctx.basepoint = basepoint

    if basepoint is True:
        basepoint_value = torch.zeros((path.shape[0], path.shape[2]), dtype=path.dtype, device=path.device)
    elif isinstance(basepoint, torch.Tensor):
        basepoint_value = basepoint
        basepoint = True
    else:
        basepoint_value = torch.Tensor()

    result, backwards_info = fn_forward(path, depth, stream, basepoint, basepoint_value, *extra_args)
    if ctx.requires_grad:
        ctx.backwards_info = backwards_info
        ctx.save_for_backward(result)

    return result


def backward(ctx, grad_result, fn_backward):
    # Just to check that the result of the forward pass hasn't been modified in-place. (Which would make the result
    # of the backwards calculation be incorrect!) The reason we don't use the tensor itself is because another
    # handle to the same information is already saved in ctx.backwards_info.
    _ = ctx.saved_tensors

    grad_path, grad_basepoint_value = fn_backward(grad_result, ctx.backwards_info)
    if not isinstance(ctx.basepoint, torch.Tensor):
        grad_basepoint_value = None

    return grad_path, None, None, grad_basepoint_value


def mode_convert(mode):
    if mode == "expand":
        return _impl.LogSignatureMode.Expand
    elif mode == "brackets":
        return _impl.LogSignatureMode.Brackets
    elif mode == "words":
        return _impl.LogSignatureMode.Words
    else:
        raise ValueError("Invalid values for argument 'mode'. Valid values are 'expand', 'brackets', or 'words'.")
