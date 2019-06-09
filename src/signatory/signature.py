import torch

from signatory._impl import (signature_channels,
                             signature as _signature)


def signature(path: torch.Tensor, depth: int, basepoint: bool = False, stream: bool = False, flatten: bool = True,
              batch_first: bool = False):
    result = _signature(path, depth, basepoint, stream, flatten, batch_first)
    if flatten:
        result = result[0]
    return result


signature.__doc__ = _signature.__doc__
