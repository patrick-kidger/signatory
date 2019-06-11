import torch

from signatory._impl import (signature_channels,
                             signature as _signature)


def signature(path: torch.Tensor, depth: int, basepoint: bool = False, stream: bool = False, flatten: bool = True):
    result = _signature(path, depth, basepoint, stream, flatten)
    if flatten:
        result = result[0]
    return result


signature.__doc__ = _signature.__doc__





# TODO: remove
import torch.autograd as autograd
from signatory._impl import signature_backward
autograd.gradgradcheck

class sig(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, basepoint=False, stream=False, flatten=True):
        out, original_out = _signature(path, depth, basepoint, stream, flatten)
        ctx.save_for_backward(path, *original_out)
        ctx.depth = depth
        ctx.basepoint = basepoint
        ctx.stream = stream
        ctx.flatten = flatten
        if flatten:
            # out is a list of a single tensor, so unpack it
            out = out[0]
        else:
            # out is a list of several tensors, but apparently only tuples are allowed return values
            out = tuple(out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        path, *out = ctx.saved_tensors
        out = signature_backward(list(grads), out, path, ctx.depth, ctx.basepoint, ctx.stream, ctx.flatten)
        return out, None, None, None, None
