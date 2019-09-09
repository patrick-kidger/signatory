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


# Requirement: must be such that
# interpret_basepoint(basepoint, path)[1] == interpret_basepoint(interpret_basepoint(basepoint, path)[1], path)
# i.e. in some sense it is a projection.
# This is because sometimes we may we wish to pass an interpreted basepoint in as the 'basepoint' argument.
def interpret_basepoint(basepoint, path):
    if basepoint is True:
        basepoint_value = torch.zeros((path.shape[0], path.shape[2]), dtype=path.dtype, device=path.device)
    elif isinstance(basepoint, torch.Tensor):
        basepoint_value = basepoint
        basepoint = True
    else:
        basepoint_value = torch.Tensor()
    return basepoint, basepoint_value


def forward(ctx, path, depth, stream, basepoint, inverse, fn_forward, extra_args=()):
    ctx.basepoint = basepoint

    basepoint, basepoint_value = interpret_basepoint(basepoint, path)

    path = path.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)
    result, backwards_info = fn_forward(path, depth, stream, basepoint, basepoint_value, inverse, *extra_args)
    if ctx.requires_grad:
        ctx.backwards_info = backwards_info
        ctx.save_for_backward(result)

    # would like to transpose here but we can't because of PyTorch bug 24413, so instead we have to transpose at every
    # call site instead.
    return result


def backward(ctx, grad_result, fn_backward):
    # Because in the forward pass we transpose at every call site, our grad_result comes to us here already-transposed.
    # so we don't need to do it here.

    # Just to check that the result of the forward pass hasn't been modified in-place. (Which would make the result
    # of the backwards calculation be incorrect!) The reason we don't actually use the tensor is because another
    # handle to it is already saved in ctx.backwards_info, which we do use.
    _ = ctx.saved_tensors

    grad_path, grad_basepoint_value = fn_backward(grad_result, ctx.backwards_info)
    if not isinstance(ctx.basepoint, torch.Tensor):
        grad_basepoint_value = None
    grad_path = grad_path.transpose(0, 1)  # (stream, batch, channel) to (batch, stream, channel)

    return grad_path, None, None, grad_basepoint_value, None


def mode_convert(mode):
    if mode == "expand":
        return _impl.LogSignatureMode.Expand
    elif mode == "brackets":
        return _impl.LogSignatureMode.Brackets
    elif mode == "words":
        return _impl.LogSignatureMode.Words
    else:
        raise ValueError("Invalid values for argument 'mode'. Valid values are 'expand', 'brackets', or 'words'.")
