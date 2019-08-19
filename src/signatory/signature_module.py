import torch
from torch import nn
from torch import autograd
from torch.autograd import function as autograd_function

from . import backend

# noinspection PyUnresolvedReferences
from . import _impl

# noinspection PyUnreachableCode
if False:
    from typing import Any, Union


class _SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint):
        return backend.forward(ctx, path, depth, stream, basepoint, _impl.signature_forward)

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        return backend.backward(ctx, grad_result, _impl.signature_backward)


def signature(path, depth, stream=False, basepoint=False):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor]) -> torch.Tensor
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

        Note that the number of output channels may be calculated via the convenience function
        :func:`signatory.signature_channels`.

    """
    # noinspection PyUnresolvedReferences
    return _SignatureFunction.apply(path, depth, stream, basepoint)


class Signature(nn.Module):
    """Module wrapper around the :func:`signatory.signature` function.

    Arguments:
        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

    Called with a single argument :attr:`path` of type :class:`torch.Tensor`.
    """

    def __init__(self, depth, stream=False, basepoint=False, **kwargs):
        # type: (int, bool, Union[bool, torch.Tensor], **Any) -> None
        super(Signature, self).__init__(**kwargs)
        self.depth = depth
        self.stream = stream
        self.basepoint = basepoint

    def forward(self, path):
        # type: (torch.Tensor) -> torch.Tensor
        return signature(path, self.depth, self.stream, self.basepoint)

    def extra_repr(self):
        return 'depth={depth}, stream={stream}, basepoint={basepoint}'.format(depth=self.depth, stream=self.stream,
                                                                              basepoint=str(self.basepoint)[:6])


# A wrapper for the sake of consistent documentation
def signature_channels(channels, depth):
    # type: (int, int) -> int
    r"""Computes the number of output channels from a signature call. Specifically, it computes

    .. math::
        \text{channels} + \text{channels}^2 + \cdots + \text{channels}^\text{depth}.

    Arguments:
        channels (int): The number of channels in the input; that is, the dimension of the space that the input path
            resides in.

        depth (int): The depth of the signature that is being computed.

    Returns:
        An int specifying the number of channels in the signature of the path.
    """

    return _impl.signature_channels(channels, depth)


def extract_signature_term(sig_tensor, channels, depth):
    # type: (torch.Tensor, int, int) -> torch.Tensor
    r"""Extracts a particular term from a signature.

    The signature to depth :math:`d` of a batch of paths in :math:`\mathbb{R}^\text{C}` is a tensor with
    :math:`C + C^2 + \cdots + C^d` channels. (See :func:`signatory.signature`.) This function extracts the :attr:`depth`
    term of that, returning a tensor with just :math:`C^\text{depth}` channels.

    Arguments:
        sig_tensor (:class:`torch.Tensor`): The signature to extract the term from. Should be the result of the
            :func:`signatory.signature` function.

        channels (int): The number of input channels :math:`C`. (In principle this is determined by the size of
            :attr:`sig_tensor`, but it is hard to compute from this.)

        depth (int): The depth of the term to be extracted from the signature.

    Returns:
        The :class:`torch.Tensor` corresponding to the :attr:`depth` term of the signature.
    """

    if channels < 1:
        raise ValueError("in_channels must be at least 1")

    if depth == 1:
        start = 0
    else:
        start = signature_channels(channels, depth - 1)
    return sig_tensor.narrow(dim=-1, start=start, length=channels ** depth)
