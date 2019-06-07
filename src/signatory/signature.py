import torch

from signatory._impl import signature as _signature


def signature(path: torch.Tensor, depth: int, basepoint: bool = False, stream: bool = False, flatten: bool = True):
    result = _signature(path, depth, basepoint, stream, flatten)
    if flatten:
        result = result[0]
    return result
