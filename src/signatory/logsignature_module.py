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
"""Provides operations relating to the logsignature transform."""


import math
import torch
from torch import nn
from torch import autograd
from torch.autograd import function as autograd_function
import warnings
import weakref

from . import signature_module as smodule
from . import impl

from typing import Union


def _interpret_mode(mode):
    if mode == "expand":
        return impl.LogSignatureMode.Expand
    elif mode == "brackets":
        return impl.LogSignatureMode.Brackets
    elif mode == "words":
        return impl.LogSignatureMode.Words
    else:
        raise ValueError("Invalid values for argument 'mode'. Valid values are 'expand', 'brackets', or 'words'.")


class _SignatureToLogsignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, signature, channels, depth, stream, mode, lyndon_info, scalar_term):
        mode = _interpret_mode(mode)

        logsignature_, lyndon_info_capsule = impl.signature_to_logsignature_forward(signature, channels, depth, stream,
                                                                                     mode, lyndon_info, scalar_term)
        ctx.save_for_backward(signature.detach())
        ctx.channels = channels
        ctx.depth = depth
        ctx.stream = stream
        ctx.mode = mode
        ctx.lyndon_info_capsule = lyndon_info_capsule
        ctx.scalar_term = scalar_term

        return logsignature_

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_logsignature):
        signature, = ctx.saved_tensors

        grad_signature = impl.signature_to_logsignature_backward(grad_logsignature, signature, ctx.channels, ctx.depth,
                                                                 ctx.stream, ctx.mode, ctx.lyndon_info_capsule,
                                                                 ctx.scalar_term)

        return grad_signature, None, None, None, None, None, None


def _signature_to_logsignature(signature, channels, depth, stream, mode, lyndon_info, scalar_term):
    if stream:
        signature = signature.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)
    logsignature_ = _SignatureToLogsignatureFunction.apply(signature, channels, depth, stream, mode, lyndon_info,
                                                           scalar_term)
    if stream:
        logsignature_ = logsignature_.transpose(0, 1)  # (stream, batch, channel) to (batch, stream, channel)
    return logsignature_


def signature_to_logsignature(signature: torch.Tensor, channels: int, depth: int, stream: bool = False,
                              mode: str = "words", scalar_term: bool = False) -> torch.Tensor:
    """Calculates the logsignature corresponding to a signature.

    Arguments:
        signature (:class:`torch.Tensor`): The result of a call to :func:`signatory.signature`.

        channels (int): The number of input channels of the :attr:`path` that :func:`signatory.signature` was called
            with.

        depth (int): The value of :attr:`depth` that :func:`signatory.signature` was called with.

        stream (bool, optional): Defaults to False. The value of :attr:`stream` that :func:`signatory.signature` was
            called with.

        mode (str, optional): Defaults to :code:`"words"`. As :func:`signatory.logsignature`.

        scalar_term (bool, optional): Defaults to False. The value of :attr:`scalar_term` that
            :func:`signatory.signature` was called with.

    Example:
        .. code-block:: python

            import signatory
            import torch
            batch, stream, channels = 8, 8, 8
            depth = 3
            path = torch.rand(batch, stream, channels)
            signature = signatory.signature(path, depth)
            logsignature = signatory.signature_to_logsignature(signature, channels, depth)

    Returns:
        A :class:`torch.Tensor` representing the logsignature corresponding to the given signature. See
        :func:`signatory.logsignature`.
    """
    # Go via the class so that it uses a cached lyndon info capsule, if we have one already for some reason.
    return SignatureToLogSignature(channels, depth, stream, mode, scalar_term)(signature)


class SignatureToLogSignature(nn.Module):
    """:class:`torch.nn.Module` wrapper around the :func:`signatory.signature_to_logsignature` function.

    Calling this :class:`torch.nn.Module` on an input :code:`signature` with the same number of channels as the last
    :code:`signature` it was called with will be faster than multiple calls to the
    :func:`signatory.signature_to_logsignature` function, in the same way that :class:`signatory.LogSignature` will be
    faster than :func:`signatory.logsignature`.

    Arguments:
        channels (int): as :func:`signatory.signature_to_logsignature`.

        depth (int): as :func:`signatory.signature_to_logsignature`.

        stream (bool, optional): as :func:`signatory.signature_to_logsignature`.

        mode (str, optional): as :func:`signatory.signature_to_logsignature`.

        scalar_term (bool, optional): as :func:`signatory.signature_to_logsignature`.
    """

    _lyndon_info_capsule_cache = weakref.WeakValueDictionary()

    # Many objects - in particular PyCapsules - aren't weakref-able, so we wrap them in this.
    class _RefHolder(object):
        def __init__(self, item):
            self.item = item

        def __copy__(self):
            return self

        def __deepcopy__(self, memodict):
            return self

    def __init__(self, channels: int, depth: int, stream: bool = False, mode: str = "words", scalar_term: bool = False,
                 **kwargs):
        super(SignatureToLogSignature, self).__init__(**kwargs)

        self._channels = channels
        self._depth = depth
        self._stream = stream
        self._mode = mode
        self._scalar_term = scalar_term

        self._lyndon_info_capsule = self._get_lyndon_info(channels, depth, mode)

    @classmethod
    def _get_lyndon_info(cls, in_channels, depth, mode):
        try:
            # This computation can be pretty slow! We definitely want to reuse it between instances
            return cls._lyndon_info_capsule_cache[(in_channels, depth, mode)]
        except KeyError:
            mode = _interpret_mode(mode)
            lyndon_info_capsule = cls._RefHolder(impl.make_lyndon_info(in_channels, depth, mode))
            cls._lyndon_info_capsule_cache[(in_channels, depth, mode)] = lyndon_info_capsule
            return lyndon_info_capsule

    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        """The forward operation.

        Arguments:
            signature (:class:`torch.Tensor`): As :func:`signatory.signature_to_logsignature`.

        Returns:
            As :func:`signatory.signature_to_logsignature`.
        """
        if signature.is_cuda and self._mode == 'brackets':
            warnings.warn("The logsignature with mode='brackets' has been requested on the GPU. This mode is quite "
                          "slow to calculate, and the GPU offers no speedup. Consider mode='words' instead.")

        return _signature_to_logsignature(signature, self._channels, self._depth, self._stream, self._mode,
                                          self._lyndon_info_capsule.item, self._scalar_term)

    def extra_repr(self):
        return ('channels={channels}, depth={depth}, stream={stream}, mode={mode}'
                .format(channels=self._channels, depth=self._depth, stream=self._stream, mode=repr(self._mode)))


# Alias
SignatureToLogsignature = SignatureToLogSignature


def logsignature(path: torch.Tensor, depth: int, stream: Union[bool, torch.Tensor] = False, basepoint: bool = False,
                 inverse=False, mode: str = "words") -> torch.Tensor:
    """Applies the logsignature transform to a stream of data.

    The :attr:`modes` argument determines how the logsignature is represented.

    Note that if performing many logsignature calculations for the same depth and size of input, then you will
    see a performance boost (at the cost of using a little extra memory) by using :class:`signatory.LogSignature`
    instead of :func:`signatory.logsignature`.

    Arguments:
        path (:class:`torch.Tensor`): as :func:`signatory.signature`.

        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

        inverse (bool, optional): as :func:`signatory.signature`.

        mode (str, optional): Defaults to :code:`"words"`. How the output should be presented. Valid values are
            :code:`"words"`, :code:`"brackets"`, or :code:`"expand"`. Precisely what each of these options mean is
            described in the
            "Returns" section below. For machine learning applications, :code:`"words"` is the appropriate choice. The
            other two options are mostly only interesting for mathematicians.

    Returns:
        A :class:`torch.Tensor`, of almost the same shape as the tensor returned from :func:`signatory.signature` called
        with the same arguments.

        If :code:`mode == "expand"` then it will be exactly the same shape as the returned tensor from
        :func:`signatory.signature`.

        If :code:`mode in ("brackets", "words")` then the channel dimension will instead be of size
        :code:`signatory.logsignature_channels(path.size(-1), depth)`. (Where :code:`path.size(-1)` is the number of
        input channels.)

        The different modes correspond to different mathematical representations of the logsignature.

        .. tip::

            If you haven't studied tensor algebras and free Lie algebras, and none of the following explanation makes
            sense to you, then you probably want to leave :attr:`mode` on its default value of :code:`"words"` and it
            will all be fine!

        If :code:`mode == "expand"` then the logsignature is presented as a member of the tensor algebra; the numbers
        returned correspond to the coefficients of all words in the tensor algebra.

        If :code:`mode == "brackets"` then the logsignature is presented in terms of the coefficients of the Lyndon
        basis of the free Lie algebra.

        If :code:`mode == "words"` then the logsignature is presented in terms of the coefficients of a particular
        computationally efficient basis of the free Lie algebra (that is not a Hall basis). Every basis element is given
        as a sum of Lyndon brackets. When each bracket is expanded out and the sum computed, the sum will contain
        precisely one Lyndon word (and some collection of non-Lyndon words). Moreover
        every Lyndon word is represented uniquely in this way. We identify these basis elements with each corresponding
        Lyndon word. This is natural as the coefficients in this basis are found just by extracting the coefficients of
        all Lyndon words from the tensor algebra representation of the logsignature.

        In all cases, the ordering corresponds to the ordering on words given by first ordering the words by length,
        and then ordering each length class lexicographically.
    """
    return LogSignature(depth, stream=stream, inverse=inverse, mode=mode)(path, basepoint=basepoint)


class LogSignature(nn.Module):
    """:class:`torch.nn.Module` wrapper around the :func:`signatory.logsignature` function.

    This :class:`torch.nn.Module` performs certain optimisations to allow it to calculate multiple logsignatures faster
    than multiple calls to :func:`signatory.logsignature`.

    Specifically, these optimisations will apply if this :class:`torch.nn.Module` is called with an input :code:`path`
    with the same number of channels as the last input :code:`path` it was called with, as is likely to be very common
    in machine learning set-ups. For larger depths or numbers of channels, this speedup will be substantial.

    Arguments:
        depth (int): as :func:`signatory.logsignature`.

        stream (bool, optional): as :func:`signatory.logsignature`.

        inverse (bool, optional): as :func:`signatory.logsignature`.

        mode (str, optional): as :func:`signatory.logsignature`.
    """

    def __init__(self, depth: int, stream: bool = False, inverse: bool = False, mode: str = "words", **kwargs):
        super(LogSignature, self).__init__(**kwargs)
        self._depth = depth
        self._stream = stream
        self._inverse = inverse
        self._mode = mode

        self._signature_to_logsignature_instance = None
        self._last_channels = None

    def _get_signature_to_logsignature_instance(self, channels):
        if self._signature_to_logsignature_instance is None or self._last_channels != channels:
            self._last_channels = channels
            self._signature_to_logsignature_instance = SignatureToLogSignature(channels, self._depth, self._stream,
                                                                               self._mode)
        return self._signature_to_logsignature_instance

    def prepare(self, in_channels: int) -> None:
        """Prepares for computing logsignatures for paths of the specified number of channels. This will be done
        anyway automatically whenever this :class:`torch.nn.Module` is called, if it hasn't been called already; this
        method simply allows to have it done earlier, for example when benchmarking.

        Arguments:
            in_channels (int): The number of input channels of the path that this instance will subsequently be called
                with. (corresponding to :code:`path.size(-1)`.)
        """

        # In particular does not return anything
        self._get_signature_to_logsignature_instance(in_channels)

    # Deliberately no 'initial' argument. To support that for logsignatures we'd need to be able to expand a
    # (potentially compressed) logsignature into a signature first. (Which is possible in principle.)
    def forward(self, path: torch.Tensor, basepoint: Union[bool, torch.Tensor] = False) -> torch.Tensor:
        """The forward operation.

        Arguments:
            path (:class:`torch.Tensor`): As :func:`signatory.logsignature`.

            basepoint (bool or torch.Tensor, optional): As :func:`signatory.logsignature`.

        Returns:
            As :func:`signatory.logsignature`.
        """

        signature = smodule.signature(path, self._depth, stream=self._stream, basepoint=basepoint,
                                      inverse=self._inverse, initial=None)
        return self._get_signature_to_logsignature_instance(path.size(-1))(signature)

    def extra_repr(self):
        return ('depth={depth}, stream={stream}, inverse={inverse}, mode={mode}'
                .format(depth=self._depth, stream=self._stream, inverse=self._inverse, mode=repr(self._mode)))


# Alias
Logsignature = LogSignature


# Computes the list of prime factors of x
def _get_prime_factors(x):
    if x == 1:
        return []
    prime_factors = []
    largest_i_so_far = 2
    while True:
        for i in range(largest_i_so_far, int(round(math.sqrt(x))) + 1):  # int needed for Py2 compatability
            if x % i == 0:
                largest_i_so_far = i
                break
        else:
            prime_factors.append(x)  # x is prime
            break
        x = x // i
        prime_factors.append(i)
    return prime_factors


# Evaluate the Mobius function of x
def _mobius_function(x):
    prime_factors = _get_prime_factors(x)
    prev_elem = None
    for elem in prime_factors:
        if elem == prev_elem:
            return 0
        prev_elem = elem
    num_unique_factors = len(set(prime_factors))
    if num_unique_factors % 2 == 0:
        return 1
    else:
        return -1


def logsignature_channels(in_channels: int, depth: int) -> int:
    """Computes the number of output channels from a logsignature call with :attr:`mode in ("words", "brackets")`.

    Arguments:
        in_channels (int): The number of channels in the input; that is, the dimension of the space that the input path
            resides in. If calling :func:`signatory.logsignature` with argument :attr:`path` then :attr:`in_channels`
            should be equal to :attr:`path.size(-1)`.

        depth (int): The depth of the signature that is being computed.

    Returns:
        An :attr:`int` specifying the number of channels in the logsignature of the path.
    """

    if in_channels < 1:
        raise ValueError("in_channels must be at least 1")

    if depth < 1:
        raise ValueError("depth must be at least 1")

    total = 0
    for d in range(1, depth + 1):
        subtotal = 0
        for d_divisor in range(1, d + 1):
            if d % d_divisor == 0:
                subtotal += _mobius_function(d // d_divisor) * in_channels ** d_divisor
        total += subtotal // d
    return total
