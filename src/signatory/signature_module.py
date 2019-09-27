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
"""Provides operations relating to the signature transform."""


import torch
from torch import nn
from torch import autograd
from torch.autograd import function as autograd_function

from . import backend
from . import compatibility as compat
# noinspection PyUnresolvedReferences
from . import _impl

# noinspection PyUnreachableCode
if False:
    from typing import Any, Union


class _SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, inverse, initial):
        ctx.basepoint = basepoint

        basepoint, basepoint_value = backend.interpret_basepoint(basepoint, path)
        if isinstance(initial, torch.Tensor):
            initial_value = initial
            initial = True
        else:
            initial_value = torch.Tensor()
            initial = False
        ctx.initial = initial

        path = path.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)
        with compat.mac_exception_catcher:
            result, backwards_info = _impl.signature_forward(path, depth, stream, basepoint, basepoint_value, inverse,
                                                             initial, initial_value)
        if ctx.requires_grad:
            ctx.backwards_info = backwards_info
            ctx.save_for_backward(result)

        # would like to transpose here but we can't because of PyTorch bug 24413, so instead we have to transpose at
        # every call site instead.
        return result

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        # Because in the forward pass we transpose at every call site, our grad_result comes to us here
        # already-transposed. so we don't need to do it here.

        # Just to check that the result of the forward pass hasn't been modified in-place. (Which would make the result
        # of the backwards calculation be incorrect!) The reason we don't actually use the tensor is because another
        # handle to it is already saved in ctx.backwards_info, which we do use.
        _ = ctx.saved_tensors

        with compat.mac_exception_catcher:
            grad_path, grad_basepoint, grad_initial = _impl.signature_backward(grad_result, ctx.backwards_info)
        grad_path = grad_path.transpose(0, 1)  # (stream, batch, channel) to (batch, stream, channel)
        if not isinstance(ctx.basepoint, torch.Tensor):
            grad_basepoint = None
        if not ctx.initial:
            grad_initial = None

        return grad_path, None, None, grad_basepoint, None, grad_initial


def signature(path, depth, stream=False, basepoint=False, inverse=False, initial=None):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor], bool, Union[None, torch.Tensor]) -> torch.Tensor
    r"""Applies the signature transform to a stream of data.

    The input :attr:`path` is expected to be a three-dimensional tensor, with dimensions :math:`(N, L, C)`, where
    :math:`N` is the batch size, :math:`L` is the length of the input sequence, and :math:`C` denotes the number of
    channels. Thus each batch element is interpreted as a stream of data :math:`(x_1, \ldots, x_L)`, where each
    :math:`x_i \in \mathbb{R}^C`.

    Each path is then lifted to a piecewise linear path :math:`X \colon [0, 1] \to \mathbb{R}^C` and the signature
    transform of :attr:`path` to depth :attr:`depth`, is computed, defined by

    .. math::
        \exp(x_2 - x_1) \otimes \exp(x_3 - x_2) \otimes \cdots \otimes \exp(x_L - x_{L - 1}),

    which gives a tensor of shape

    .. math::
        (N, C + C^2 + \cdots + C^\text{depth}).

    If :attr:`basepoint` is True then an additional point :math:`x_0 = 0 \in \mathbb{R}^C` is prepended to the path
    before the signature transform is applied. Alternatively it can be a :class:`torch.Tensor` of shape :math:`(N, C)`
    specifying the point to prepend.

    If :attr:`stream` is True then  the signatures of all paths :math:`(x_1, \ldots, x_j)`, for :math:`j=2, \ldots, L`,
    are computed. (Or :math:`(x_0, \ldots, x_j)`, for :math:`j=1, \ldots, L` if :attr:`basepoint` is provided. In
    neither case is the signature of the path of a single element computed, as that isn't defined.)

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

        inverse (bool, optional): Defaults to False. If True then it is in fact the inverse signature that is computed.
            That is,

            .. math::
                \exp(x_{L - 1} - x_L) \otimes \cdots \otimes \exp(x_2 - x_3) \otimes \exp(x_1 - x_2).

            From a machine learning perspective it does not particularly matter whether the signature or the inverse
            signature is computed - both represent essentially the same information as each other.

        initial (None or :class:`torch.Tensor`, optional): Defaults to None. If it is a :class:`torch.Tensor` then it
            must be of size :math:`(N, C + C^2 + ... + C^\text{depth})`, and it will be premultiplied to the signature,
            so that in fact

            .. math::
                \text{initial} \otimes \exp(x_2 - x_1) \otimes \exp(x_3 - x_2) \otimes \cdots \otimes \exp(x_L - x_{L - 1})

            is computed. (Or the appropriate modification of this if :attr:`inverse=True` or if :attr:`basepoint` is
            passed.) If this argument is None then this extra multiplication is not done, and the signature is
            calculated as previously described.

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
    result = _SignatureFunction.apply(path, depth, stream, basepoint, inverse, initial)

    # We have to do the transpose outside of autograd.Function.apply to avoid a PyTorch bug.
    # https://github.com/pytorch/pytorch/issues/24413
    if stream:
        result = result.transpose(0, 1)  # NOT .transpose_ - the underlying TensorImpl (in C++) is used elsewhere and we
                                         # don't want to change it.
    return result


class Signature(nn.Module):
    """Module wrapper around the :func:`signatory.signature` function.

    Arguments:
        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        inverse (bool, optional): as :func:`signatory.signature`.
    """

    def __init__(self, depth, stream=False, inverse=False, **kwargs):
        # type: (int, bool, bool, **Any) -> None
        super(Signature, self).__init__(**kwargs)
        self.depth = depth
        self.stream = stream
        self.inverse = inverse

    def forward(self, path, basepoint=False, initial=None):
        # type: (torch.Tensor, Union[bool, torch.Tensor], Union[None, torch.Tensor]) -> torch.Tensor
        """The forward operation.

        Arguments:
            path (torch.Tensor): As :func:`signatory.signature`.

            basepoint (bool or torch.Tensor, optional): As :func:`signatory.signature`.

            initial (None or torch.Tensor, optional): As :func:`signatory.signature`.

        Returns:
            As :func:`signatory.signature`.
        """
        return signature(path, self.depth, stream=self.stream, basepoint=basepoint, inverse=self.inverse,
                         initial=initial)

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

    with compat.mac_exception_catcher:
        return _impl.signature_channels(channels, depth)


def extract_signature_term(sigtensor, channels, depth):
    # type: (torch.Tensor, int, int) -> torch.Tensor
    r"""Extracts a particular term from a signature.

    The signature to depth :math:`d` of a batch of paths in :math:`\mathbb{R}^\text{C}` is a tensor with
    :math:`C + C^2 + \cdots + C^d` channels. (See :func:`signatory.signature`.) This function extracts the :attr:`depth`
    term of that, returning a tensor with just :math:`C^\text{depth}` channels.

    Arguments:
        sigtensor (:class:`torch.Tensor`): The signature to extract the term from. Should be a result from the
            :func:`signatory.signature` function.

        channels (int): The number of input channels :math:`C`.

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
    return sigtensor.narrow(dim=-1, start=start, length=channels ** depth)


def signature_combine(sigtensor1, sigtensor2, input_channels, depth, inverse=False):
    # type: (torch.Tensor, torch.Tensor, int, int, bool) -> torch.Tensor
    r"""Combines two signatures into a single signature.

    This is done by computing a single tensor product:

    .. math::

        \text{sigtensor1} \otimes \text{sigtensor2}

    Usage is most clear by example. See :ref:`examples-combine`.

    Arguments:
        sigtensor1 (torch.Tensor): The signature of a path, of dimensions :attr:`(batch, signature_channels)`.

        sigtensor2 (torch.Tensor): The signature of a second path, of dimensions :attr:`(batch, signature_channels)`.
            When the signature of the second path was created, it must have been called with :attr:`basepoint` set to
            the final value of the path that created :attr:`sigtensor1`. (See :ref:`examples-combine`.)

        input_channels (int): The number of channels in the two paths that were used to compute :attr:`sigtensor1` and
            :attr:`sigtensor2`. This must be the same for both :attr:`sigtensor1` and :attr:`sigtensor2`.

        depth (int): The depth that :attr:`sigtensor1` and :attr:`sigtensor2` have been calculated to. This must be
            the same for both :attr:`sigtensor1` and :attr:`sigtensor2`.

        inverse (bool, optional): Defaults to False. Whether :attr:`sigtensor1` and :attr:`sigtensor2` were created
            with :attr:`inverse=True`. This must be the same for both :attr:`sigtensor1` and :attr:`sigtensor2`.

    Returns:
        Let :attr:`path1` be the path whose signature is :attr:`sigtensor1`. Let :attr:`path2` be the path whose
        signature is :attr:`sigtensor2`. Then this function returns the signature of :attr:`path1` and :attr:`path2`
        concatenated with each other. (The interpretation is usually that :attr:`path2` represents an extension of
        :attr:`path1`.)

    .. danger::

        There are two subtle bugs which can occur when using this function incautiously. First of all, make sure
        that :attr:`sigtensor2` is created with an appropriate :attr:`basepoint`. Secondly, ensure that :attr:`inverse`
        is set to whatever value of :attr:`inverse` was used to create :attr:`sigtensor1` and :attr:`sigtensor2`.

        If this is not done then the return value of this function will be essentially meaningless numbers.

        Neither of these errors can be caught programmatically by this function; it is up to the software developer to
        get it right!
    """
    if inverse:
        sigtensor1, sigtensor2 = sigtensor2, sigtensor1
    return backend.TensorAlgebraMult.apply(sigtensor1, sigtensor2, input_channels, depth)
