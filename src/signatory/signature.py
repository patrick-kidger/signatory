import torch
import torch.autograd as autograd
from typing import Any, Union, List

from signatory._impl import (signature_channels,
                             _signature_forward,
                             _signature_backward)


# It would be lovely to do all of this at the C++ level. (In particular sigspec is really a struct that has no
# business being passed around at the Python level.) But unfortunately the documentation for how to create autograd
# Functions in C++ is nonexistent. Presumably that means it's all still subject to change, so we're just going to stick
# to the Python way of doings things for now.
class SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, path: torch.Tensor, depth: int, basepoint: bool = False, stream: bool = False,
                flatten: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        result, result_as_vector, path_increments, sigspec = _signature_forward(path, depth, basepoint, stream, flatten)
        ctx._call_info = (result_as_vector, path_increments, sigspec, depth, basepoint, stream, flatten)
        if flatten:
            result = result[0]
        else:
            result = tuple(result)  # okay to return tuples, not okay to return lists. For some reason.
        return result

    @staticmethod
    def backward(ctx: Any, *grad_outputs: List[torch.Tensor]) -> torch.Tensor:
        return _signature_backward(grad_outputs, *ctx._call_info), None, None, None, None


def signature(path: torch.Tensor, depth: int, basepoint: bool = False, stream: bool = False,
              flatten: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    return SignatureFunction.apply(path, depth, basepoint, stream, flatten)
