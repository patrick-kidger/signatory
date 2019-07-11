import torch
from torch import autograd

# noinspection PyUnreachableCode
if False:
    from typing import Any, Tuple, Union

from ._impl import (_signature_channels,
                    _signature_forward,
                    _signature_backward,
                    _logsignature_forward,
                    _logsignature_backward)


# It would be lovely to do all of this at the C++ level. (In particular sigspec is really a struct that has no
# business being passed around at the Python level.) But unfortunately the documentation for how to create autograd
# Functions in C++ is nonexistent. Presumably that means it's all still subject to change, so we're just going to stick
# to the Python way of doings things for now.
class _SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, basepoint_value):
        # type: (Any, torch.Tensor, int, bool, torch.Tensor, bool) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        result, backwards_info = _signature_forward(path, depth, stream, basepoint, basepoint_value)
        ctx._backwards_info = backwards_info
        ctx._basepoint = basepoint
        return result

    @staticmethod
    @autograd.function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        # type: (Any, Tuple[torch.Tensor]) -> Tuple[torch.Tensor, None, None, None, Union[None, torch.Tensor]]
        grad_path, grad_basepoint_value = _signature_backward(grad_result, ctx._backwards_info)
        return grad_path, None, None, None, grad_basepoint_value


class _LogSignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, basepoint_value):
        # type: (Any, torch.Tensor, int, bool, torch.Tensor, bool) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        result, backwards_info = _logsignature_forward(path, depth, stream, basepoint, basepoint_value)
        ctx._backwards_info = backwards_info
        ctx._basepoint = basepoint
        return result

    @staticmethod
    @autograd.function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        # type: (Any, Tuple[torch.Tensor]) -> Tuple[torch.Tensor, None, None, None, Union[None, torch.Tensor]]
        grad_path, grad_basepoint_value = _logsignature_backward(grad_result, ctx._backwards_info)
        return grad_path, None, None, None, grad_basepoint_value


def _parse_basepoint(basepoint, path):
    if basepoint is True:
        basepoint_value = torch.zeros((path.shape[0], path.shape[2]), dtype=path.dtype, device=path.device)
    elif isinstance(basepoint, torch.Tensor):
        basepoint_value = basepoint
        basepoint = True
    else:
        basepoint_value = torch.Tensor()
    return basepoint, basepoint_value


def signature(path, depth, stream=False, basepoint=False):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    r"""Applies the signature transform to a stream of data.

    The input :attr:`path` is expected to be a three-dimensional tensor, with dimensions :math:`(N, L, C)`, where
    :math:`N` is the batch size, :math:`L` is the length of the input sequence, and :math:`C` denotes the number of
    channels. Thus each batch element is interpreted as a stream of data :math:`(x_1, \ldots, x_L)`, where each
    :math:`x_i \in \mathbb{R}^C`. (This is the same as :class:`torch.nn.Conv1d`, for example.)

    If :attr:`basepoint` is True then an additional point :math:`x_0 = 0 \in \mathbb{R}^C` is prepended to the path.
    (Alternatively it can be a :class:`torch.Tensor1` of shape :math:`(N, C)` specifying the point to prepend.)

    Each path is then lifted to a piecewise linear path :math:`X \colon [0, 1] \to \mathbb{R}^C`, and the signature
    transform of this path is then computed. This (by the definition of the signature transform) gives a sequence of
    tensors of shape

    .. math::
        (N, C), (N, C, C), \ldots (N, C, \ldots, C),

    where the final tensor has :attr:`depth` many dimensions of size :math:`C`. These are then flattened down and
    concatenated to give a single tensor of shape

    .. math::
        (N, C + C^2 + \cdots + C^\text{depth}).

    (This value may be computed via the :func:`signatory.signature_channels` function.)

    If :attr:`stream` is True then  the signatures of all paths :math:`(x_1, \ldots, x_j)`, for :math:`j=2, \ldots, L`,
    are computed. (Or :math:`(x_0, \ldots, x_j)`, for :math:`j=1, \ldots, L` if :attr:`basepoint` is provided. In
    neither case is the signature of the path of a single element computed, as that isn't defined.)

    Examples:
        If :attr:`stream` is False then the returned tensor will have shape

        .. math::
            (N, C + C^2 + \cdots + C^d).

        If :attr:`basepoint` is True and :attr:`stream` is True then the returned tensor will
        have shape

        .. math::
            (N, L, C + C^2 + \cdots + C^d),

        as the stream dimension is now preserved. See also the 'Returns' section below.

    Arguments:
        path (:class:`torch.Tensor`): The batch of input paths to apply the signature transform to.

        depth (int): The depth to truncate the signature at.

        stream (bool, optional): Defaults to False. If False then the signature transform of the whole path is computed.
            If True then the signature of all intermediate paths are also computed.

        basepoint (bool or :class:`torch.Tensor`, optional): Defaults to False. If True, then the input paths will have
            an additional point at the origin prepended to the start of the sequence. (If this is False then the
            signature transform is invariant to translations of the path, which may or may not be desirable.)
            Alternatively it may be a :class:`torch.Tensor` specifying the point to prepend, in which case it should
            have shape :math:`(N, C)`

    Returns:
        A :class:`torch.Tensor`. Given an input :class:`torch.Tensor` of shape :math:`(N, L, C)`, and input arguments
        :attr:`depth`, :attr:`basepoint`, :attr:`stream`, then the return value is, in pseudocode:

        .. code-block:: python

            if stream:
                if basepoint is True or isinstance(basepoint, torch.Tensor):
                    return torch.Tensor of shape (N, L, C + C^2 + ... + C^depth)
                else:
                    return torch.Tensor of shape (N, L - 1, C + C^2 + ... + C^depth)
            else:
                return torch.Tensor of shape (N, C + C^2 + ... + C^depth)

    """

    basepoint, basepoint_value = _parse_basepoint(basepoint, path)
    return _SignatureFunction.apply(path, depth, stream, basepoint, basepoint_value)


def logsignature(path, depth, stream=False, basepoint=False):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    """TODO"""
    basepoint, basepoint_value = _parse_basepoint(basepoint, path)
    return _LogSignatureFunction.apply(path, depth, stream, basepoint, basepoint_value)


# A wrapper for the sake of consistent documentation on signatures
def signature_channels(in_channels, depth):
    # type: (int, int) -> int
    """Computes the number of output channels from a signature call.

    Arguments:
        in_channels (int): The number of channels in the input; that is, the dimension of the space that the input path
            resides in.

        depth (int): The depth of the signature that is being computed.

    Returns:
        An int specifying the number of channels in the signature of the path.
    """

    return _signature_channels(in_channels, depth)


def extract_term(signature, in_channels, depth):
    # type: (torch.Tensor, int, int) -> torch.Tensor
    r"""Extracts a particular term from a signature.

    The signature to depth :math:`d` of a batch of paths in :math:`\mathbb{R}^\text{C}` is a tensor with
    :math:`C + C^2 + \cdots + C^d` channels. (See :func:`signatory.signature`.) This function extracts the :attr:`depth`
    term of that, returning a tensor with just :math:`C^\text{depth}` channels.

    This is really just here as a convenience function: if you want to extract multiple terms and care about speed then
    it will be probably quicker to write your own function.

    Arguments:
        signature (:class:`torch.Tensor`): The signature to extract the term from. Should be the result of the
            :func:`signatory.signature` function.

        in_channels (int): The number of input channels :math:`C`.

        depth (int): The depth of the term to be extracted from the signature.

    Returns:
        The :class:`torch.Tensor` corresponding to the :attr:`depth` term of the signature.
    """

    if depth == 1:
        start = 0
    else:
        start = signature_channels(in_channels, depth - 1)
    return signature.narrow(dim=-1, start=start, length=in_channels ** depth)
