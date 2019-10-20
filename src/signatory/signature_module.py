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
# noinspection PyUnresolvedReferences
from . import _impl

# noinspection PyUnreachableCode
if False:
    from typing import Any, List, Union


# So coming up we've got the functionality for creating signtures. The system is a little convoluted and unfortunately
# has a few warts, so we're detailing them here, and why.
#
# The thing that actually matters is _impl.signature_forward and _impl.signature_backward, which do exactly what you
# think they do.
#
# Unfortunately it's not at all clear how to bring these together into an autograd.Function in the C++, so we export
# both to Python and to that here instead, to create _SignatureFunction.
#
# (This uses the helpers interpet_forward_args and interpret_backward_grad as the Python signature function and the C++
# signature_forward don't have exactly the same interface - for some sensible reasons, and some not-so-sensible but now
# unchangable-because-of-backwards-compatability reasons - and these convert between them.)
#
# In particular one annoying wart that I would definitely change if I could is the order of dimensions. For speed
# reasons this is (stream, batch, channel) in C++. For convention reasons it is (batch, stream, channel) in the visible
# API. Somewhere a transpose has to be performed. Unfortunately, some transposes cannot be performed in anything within
# the autograd machinery, as else certain correctness checks don't trigger. (PyTorch bug 24413). So for simplicity we
# put all transposes outside the autograd framework.
# If I could I would just make (stream, batch, channel) the convention throughout.
#
# Moving on, the next wart is that there's some somewhat convoluted logic in the main signature function to test for
# more efficient parallelisation cases. This has to be done outside of _SignatureFunction (and thus in Python), so that
# the backward logic is hooked up correctly.
#
# And in turn we then have to perform argument checking before that set of logic, which is the purpose of
# _signature_checkargs.
#
# If only we could do autograd in C++ pretty much all of this unpleasantness could be tided up, but the documentation
# for how to do that is literally non-existent.


def interpet_forward_args(ctx, path, basepoint, initial):
    ctx.basepoint_is_tensor = isinstance(basepoint, torch.Tensor)
    ctx.initial_is_tensor = isinstance(initial, torch.Tensor)
    basepoint, basepoint_value = backend.interpret_basepoint(basepoint, path.size(-2), path.size(-1), path.dtype,
                                                             path.device)
    initial, initial_value = backend.interpret_initial(initial)
    return basepoint, basepoint_value, initial, initial_value


def interpret_backward_grad(ctx, grad_basepoint, grad_initial):
    if not ctx.basepoint_is_tensor:
        grad_basepoint = None
    if not ctx.initial_is_tensor:
        grad_initial = None
    return grad_basepoint, grad_initial


class _SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, inverse, initial, open_mp_parallelise):

        basepoint, basepoint_value, initial, initial_value = interpet_forward_args(ctx, path, basepoint, initial)

        signature_, path_increments = _impl.signature_forward(path, depth, stream, basepoint, basepoint_value, inverse,
                                                              initial, initial_value, open_mp_parallelise)
        ctx.save_for_backward(signature_, path_increments)
        ctx.depth = depth
        ctx.stream = stream
        ctx.basepoint = basepoint
        ctx.inverse = inverse
        ctx.initial = initial

        return signature_

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        signature_, path_increments = ctx.saved_tensors

        grad_path, grad_basepoint, grad_initial = _impl.signature_backward(grad_result, signature_, path_increments,
                                                                           ctx.depth, ctx.stream, ctx.basepoint,
                                                                           ctx.inverse, ctx.initial)

        grad_basepoint, grad_initial = interpret_backward_grad(ctx, grad_basepoint, grad_initial)

        return grad_path, None, None, grad_basepoint, None, grad_initial, None


def _signature_checkargs(path, depth, basepoint, initial):
    path = path.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)
    basepoint, basepoint_value = backend.interpret_basepoint(basepoint, path.size(-2), path.size(-1), path.dtype,
                                                             path.device)
    initial, initial_value = backend.interpret_initial(initial)
    _impl.signature_checkargs(path, depth, basepoint, basepoint_value, initial, initial_value)


def _signature_openmp(path, depth, stream, basepoint, inverse, initial):
    if stream:
        return

    grad = False
    grad = grad or path.requires_grad
    if isinstance(basepoint, torch.Tensor):
        grad = grad or basepoint.requires_grad
    if isinstance(initial, torch.Tensor):
        grad = grad or initial.requires_grad
    open_mp_parallelise = _impl.built_with_open_mp() and not grad and not path.is_cuda

    if not open_mp_parallelise:
        return

    # noinspection PyUnresolvedReferences
    return _SignatureFunction.apply(path.transpose(0, 1), depth, stream, basepoint, inverse, initial, True)


def _signature_batch_trick(path, depth, stream, basepoint, inverse, initial):
    if stream:
        return

    _threshold = _impl.hardware_concurrency()
    threshold = _threshold if _threshold != 0 else 32

    batch_size, stream_size, channel_size = path.shape

    # If we don't have available parallelisation capacity
    if batch_size >= threshold:
        return

    # Number of chunks to split the stream in to
    mult = int(round(float(threshold) / batch_size))
    mult = min(mult, int(stream_size / 2))

    # If the problem isn't large enough to be worth parallelising
    if mult < 2:
        return

    remainder = stream_size % mult                 # How much of the stream is left over as a remainder
    reduced_bulk_length = int(stream_size / mult)  # Size of each chunk of the stream
    bulk_length = stream_size - remainder          # Size of all of the chunks of the stream put together,
    # excluding the remainder
    path_bulk = path[:, 0:bulk_length]
    path_remainder = path[:, bulk_length:]

    # If the data isn't laid out such that some of the stream dimension can be pushed into the batch dimension
    try:
        path_bulk.view(batch_size * mult, reduced_bulk_length, channel_size)
    except RuntimeError:
        return

    # Need to set basepoints to the end of each previous chunk
    path_bulk = path_bulk.view(batch_size, mult, reduced_bulk_length, channel_size)
    ends = path_bulk[:, :, -1].roll(shifts=1, dims=-2)
    if remainder != 0:
        basepoint_remainder = ends[:, 0].clone()
    if isinstance(basepoint, torch.Tensor):
        # noinspection PyUnresolvedReferences
        ends[:, 0].copy_(basepoint)
    elif basepoint is True:
        ends[:, 0].zero_()
    else:
        # noinspection PyUnresolvedReferences
        ends[:, 0].copy_(path_bulk[:, 0, 0])
    path_bulk = path_bulk.view(batch_size * mult, reduced_bulk_length, channel_size)
    basepoint = ends.view(batch_size * mult, channel_size)

    # noinspection PyUnresolvedReferences
    result_bulk = _SignatureFunction.apply(path_bulk.transpose(0, 1), depth, stream, basepoint, inverse, False, False)
    result_bulk = result_bulk.view(batch_size, mult, result_bulk.size(-1))
    chunks = []
    if isinstance(initial, torch.Tensor):
        chunks.append(initial)
    chunks.extend(result_bulk.unbind(dim=-2))
    if remainder != 0:
        # noinspection PyUnresolvedReferences
        result_remainder = _SignatureFunction.apply(path_remainder.transpose(0, 1), depth, stream, basepoint_remainder,
                                                    inverse, False, False)
        chunks.append(result_remainder)

    return multi_signature_combine(chunks, channel_size, depth, inverse)


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

    _signature_checkargs(path, depth, basepoint, initial)

    # Coming up is a somewhat involved set of optimisations via parallelisation.
    #
    # Parallelisation can be accomplished via OpenMP (in the C++ code) in the special case of:
    #     stream==False,
    #     the path is on the CPU,
    #     gradient isn't required,
    #     and Signatory was compiled with OpenMP.
    # Rationale: stream==False is an inherently serial problem, so we can't parallelise at all then.
    #            If the problem is on the GPU then OpenMP slows things down. I don't know why.
    #            If a gradient is required then whilst OpenMP would be quicker on the forward pass, it would be slower
    #               on the backward pass than our second parallelisation approach, below. (And a hybrid approach is not
    #               possible as the OpenMP approach doesn't generate certain information needed for the more-efficient
    #               backward pass.)
    #
    # So we start off by checking if we can parallelise via OpenMP.
    #
    # If not then we try a trick: we split apart the stream dimension into chunks and push them into the batch
    # dimension. Then afterwards we pull it out and perform the remaining few tensor multiplications. This requires:
    #     stream==False
    #     That the batch size is currently small enough that this will speed things up, not slow things down
    #     That the stream size is currently large enough, for the same reason.
    #     That the data is laid out in a 'quasicontiguous' manner, so that this moving of dimensions can be done without
    #         copying the data.
    # (As a final remark, this optimisation is done in the Python rather than the C++ so that the backward pass is
    # hooked up.)

    for fn in (_signature_openmp,        # Try to parallelise via OpenMP
               _signature_batch_trick):  # Try to parallelise via the push-into-batch-dimension trick
        result = fn(path, depth, stream, basepoint, inverse, initial)
        if result is not None:
            break
    else:
        # Default case: no parallelisation
        # transpose to go from Python convention of (batch, stream, channel) to autograd/C++ convention of
        # (stream, batch, channel)
        # noinspection PyUnresolvedReferences
        result = _SignatureFunction.apply(path.transpose(0, 1), depth, stream, basepoint, inverse, initial, False)

    # We have to do the transpose outside of autograd.Function.apply to avoid PyTorch bug 24413
    if stream:
        # NOT .transpose_ - the underlying TensorImpl (in C++) is used elsewhere and we don't want to change it.
        result = result.transpose(0, 1)
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


class _SignatureCombineFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_channels, depth, *sigtensors):
        ctx.input_channels = input_channels
        ctx.depth = depth
        ctx.save_for_backward(*sigtensors)
        return _impl.signature_combine_forward(list(sigtensors), input_channels, depth)

    @staticmethod
    def backward(ctx, grad):
        sigtensors = ctx.saved_tensors
        grad = _impl.signature_combine_backward(grad, list(sigtensors), ctx.input_channels, ctx.depth)
        return (None, None) + tuple(grad)


def signature_combine(sigtensor1, sigtensor2, input_channels, depth, inverse=False):
    # type: (torch.Tensor, torch.Tensor, int, int, bool) -> torch.Tensor
    r"""Combines two signatures into a single signature.

    This is done by computing a single tensor product:

    .. math::

        \text{sigtensor1} \otimes \text{sigtensor2}

    Usage is most clear by example. See :ref:`examples-combine`.

    See also :func:`signatory.multi_signature_combine` for a more general version. (There are no advantages to using
    :func:`signatory.signature_combine` over :func:`signatory.multi_signature_combine`.)

    Arguments:
        sigtensor1 (:class:`torch.Tensor`): The signature of a path, of dimensions :attr:`(batch, signature_channels)`.

        sigtensor2 (:class:`torch.Tensor`): The signature of a second path, of dimensions
            :attr:`(batch, signature_channels)`. When the signature of the second path was created, it must have been
            called with :attr:`basepoint` set to the final value of the path that created :attr:`sigtensor1`. (See
            :ref:`examples-combine`.)

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
    """
    return multi_signature_combine([sigtensor1, sigtensor2], input_channels, depth, inverse)


def multi_signature_combine(sigtensors, input_channels, depth, inverse=False):
    # type: (List[torch.Tensor], int, int, bool) -> torch.Tensor
    r"""Combines multiple signatures into a single signature.

    This is done by computing multiple tensor products:

    .. math::

        \text{sigtensor}_0 \otimes \text{sigtensor}_1 \otimes \cdots \otimes \text{sigtensor}_k

    where each :attr:`sigtensors` is a list of :math:`\text{sigtensor}_i` for :math:`i = 0, 1, \ldots, k`.

    See also :func:`signatory.signature_combine`.

    Arguments:
        sigtensors (list of :class:`torch.Tensor`): Signature of multiple paths, each with dimensions
            :attr:`(batch, signature_channels)`.

        input_channels (int): As :func:`signatory.signature_combine`.

        depth (int): As :func:`signatory.signature_combine`.

        inverse (bool, optional): As :func:`signatory.signature_combine`.

    Returns:
        Let :math:`\text{path}_i` be the path whose signature is :math:`\text{sigtensor}_i`. Then this function returns
        the signature of every :math:`\text{path}_i` concatenated together.

    .. danger::

        Make sure that each element of :attr:`sigtensors` is created with an appropriate :attr:`basepoint`, as with
        :func:`signatory.signature_combine`.
    """
    if inverse:
        sigtensors = reversed(sigtensors)
    return _SignatureCombineFunction.apply(input_channels, depth, *sigtensors)
