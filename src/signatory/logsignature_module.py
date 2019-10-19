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
import weakref

from . import signature_module as smodule
# noinspection PyUnresolvedReferences
from . import _impl

# noinspection PyUnreachableCode
if False:
    from typing import Any, Union


def _mode_convert(mode):
    if mode == "expand":
        return _impl.LogSignatureMode.Expand
    elif mode == "brackets":
        return _impl.LogSignatureMode.Brackets
    elif mode == "words":
        return _impl.LogSignatureMode.Words
    else:
        raise ValueError("Invalid values for argument 'mode'. Valid values are 'expand', 'brackets', or 'words'.")


class _SignatureToLogsignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, signature, channels, depth, stream, mode, lyndon_info):
        mode = _mode_convert(mode)
        if stream:
            signature = signature.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)

        logsignature_, backwards_info = _impl.signature_to_logsignature_forward(signature, channels, depth, stream,
                                                                                mode, lyndon_info)
        ctx.save_for_backward(signature.detach())
        ctx.backwards_info = backwards_info
        ctx.stream = stream

        # Call to detach works around PyTorch bug 25340, which is a won't-fix. Basically, it makes sure that a reference
        # cycle doesn't occur. No call to detach gives:
        #   result -> ctx -> backwards_info -> result
        # Whilst detach gives:
        #   result.detach()-> ctx -> backwards_info -> result
        # No cycle!
        # If a reference cycle is present then a memory leak occurs. (As the result -> ctx reference is in C++ so it's
        # not handled by Python.)
        return logsignature_.detach()

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_logsignature):
        # Because in the forward pass we transpose in the wrapper, our grad_result comes to us here
        # already-transposed. so we don't need to do it here.

        # Just to check that the input of the forward pass hasn't been modified in-place. (Which would make the result
        # of the backwards calculation be incorrect!)
        _ = ctx.saved_tensors

        grad_signature = _impl.signature_to_logsignature_backward(grad_logsignature, ctx.backwards_info)

        if ctx.stream:
            grad_signature = grad_signature.transpose(0, 1)  # (stream, batch, channel) to (batch, stream, channel)

        return grad_signature, None, None, None, None, None


def _signature_to_logsignature(signature, channels, depth, stream, mode, lyndon_info):
    logsignature_ = _SignatureToLogsignatureFunction.apply(signature, channels, depth, stream, mode, lyndon_info)
    if stream:
        logsignature_ = logsignature_.transpose(0, 1)
    return logsignature_


def signature_to_logsignature(signature, channels, depth, stream=False, mode="words"):
    # type: (torch.Tensor, int, int, bool, str) -> torch.Tensor
    """Converts a signature to a logsignature.

    Arguments:
        signature (:class:`torch.Tensor`): The result of a call to :func:`signatory.signature`.

        channels (int): The number of input channels of the path that :func:`signatory.signature` was called with.

        depth (int): The :attr:`depth` that :func:`signatory.signature` was called with.

        stream (bool, optional): Defaults to False. The :attr:`stream` that :func:`signatory.signature` was called with.

        mode (str, optional): Defaults to "words". As :func:`signatory.logsignature`.

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
        A :class:`torch.Tensor` representing the logsignature corresponding to the given signature.
    """
    return _signature_to_logsignature(signature, channels, depth, stream, mode, None)


class SignatureToLogSignature(nn.Module):
    """Module wrapper around the :func:`signatory.signature_to_logsignature` function.

    Calling this Module on an input :code:`signature` with the same depth and number of channels as the last input
    :code:`path` it was called with will be faster than the corresponding :func:`signatory.signature_to_logsignature`
    function, as this Module caches the result of certain computations which depend only on this value. (For larger
    depths or numbers of channels, this speedup will be substantial.)

    Arguments:
        channels (int): as :func:`signatory.signature_to_logsignature`.

        depth (int): as :func:`signatory.signature_to_logsignature`.

        stream (bool, optional): as :func:`signatory.signature_to_logsignature`.

        mode (str, optional): as :func:`signatory.signature_to_logsignature`.
    """

    _lyndon_info_cache = weakref.WeakValueDictionary()

    # Many objects - in particular PyCapsules - aren't weakref-able, so we wrap them in this.
    class _RefHolder(object):
        def __init__(self, item):
            self.item = item

    def __init__(self, channels, depth, stream=False, mode="words", **kwargs):
        # type: (int, int, bool, str, **Any) -> None
        super(SignatureToLogSignature, self).__init__(**kwargs)

        self._channels = channels
        self._depth = depth
        self._stream = stream
        self._mode = mode

        self._lyndon_info = self._get_lyndon_info(channels, depth, mode)

    @classmethod
    def _get_lyndon_info(cls, in_channels, depth, mode):
        try:
            # This computation can be pretty slow! We definitely want to reuse it between instances
            return cls._lyndon_info_cache[(in_channels, depth, mode)]
        except KeyError:
            mode = _mode_convert(mode)
            lyndon_info = cls._RefHolder(_impl.make_lyndon_info(in_channels, depth, mode))
            cls._lyndon_info_cache[(in_channels, depth, mode)] = lyndon_info
            return lyndon_info

    def forward(self, signature):
        # type: (torch.Tensor) -> torch.Tensor
        """The forward operation.

        Arguments:
            signature (torch.Tensor): As :func:`signatory.signature_to_logsignature`.

        Returns:
            As :func:`signatory.signature_to_logsignature`.
        """
        return _signature_to_logsignature(signature, self._channels, self._depth, self._stream, self._mode,
                                          self._lyndon_info.item)

    def extra_repr(self):
        return ('channels={channels}, depth={depth}, stream={stream}, mode{mode}'
                .format(channels=self._channels, depth=self._depth, stream=self._stream, mode=self._mode))


# Alias
SignatureToLogsignature = SignatureToLogSignature


def logsignature(path, depth, stream=False, basepoint=False, inverse=False, mode="words"):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor], bool, str) -> torch.Tensor
    """Applies the logsignature transform to a stream of data.

    The :attr:`modes` argument determines how the logsignature is represented.

    Note that if performing many logsignature calculations for the same depth and size of input, then you will
    see a performance boost by using :class:`signatory.LogSignature` over :func:`signatory.logsignature`.

    Arguments:
        path (:class:`torch.Tensor`): as :func:`signatory.signature`.

        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

        inverse (bool, optional): as :func:`signatory.signature`.

        mode (str, optional): Defaults to :attr:`"words"`. How the output should be presented. Valid values are
            :attr:`"expand"`, :attr:`"brackets"`, or :attr:`"words"`. Precisely what each of these options mean is
            described in the
            "Returns" section below. As a rule of thumb: use :attr:`"words"` for new projects (as it is the fastest),
            and use :attr:`"brackets"` for compatibility with other projects which do not provide equivalent
            functionality to :attr:`"words"`. (Such as `iisignature <https://github.com/bottler/iisignature>`__). The
            mode :attr:`"expand"` is mostly only interesting for mathematicians.

    Returns:
        A :class:`torch.Tensor`. If :attr:`mode == "expand"` then it will be of the same shape as the returned tensor
        from :func:`signatory.signature`. If :attr:`mode in ("brackets", "words")` then it will again be of the
        same shape, except that the channel dimension will instead be of size
        :attr:`signatory.logsignature_channels(C, depth)`, where :attr:`C` is the number of input channels, i.e.
        :attr:`path.size(-1)`.
        (Thus the logsignature is much smaller than the signature, which is the whole point of using the logsignature
        over the signature in the first place.)

        We now go on to explain what the different values for :attr:`mode` mean. This discussion is in the "Returns"
        section because the value of :attr:`mode` essentially just determines how the output is represented; the
        mathematical meaning is the same in all cases.

        If :attr:`mode == "expand"` then the logsignature is presented as a member of the tensor algebra; the numbers
        returned correspond to the coefficients of all words in the tensor algebra.

        If :attr:`mode == "brackets"` then the logsignature is presented in terms of the coefficients of the Lyndon
        basis of the free Lie algebra.

        If :attr:`mode == "words"` then the logsignature is presented in terms of the coefficients of a particular
        computationally efficient basis of the free Lie algebra that is not a Hall basis. Every basis element is given
        as a sum of Lyndon brackets. When each bracket is expanded out and the sum computed, the sum will contain
        precisely one Lyndon word (and some collection of non-Lyndon words). Moreover
        every Lyndon word is represented uniquely in this way. We identify these basis elements with each corresponding
        Lyndon word. This is natural as the coefficients in this basis are found just by extracting the coefficients of
        all Lyndon words from the tensor algebra representation of the logsignature.

        In all cases, the ordering corresponds to the ordering on words given by first ordering the words by length,
        and then ordering each length class lexicographically.
    """

    # Deliberately no 'initial' argument. To support that for logsignatures we'd need to be able to expand a
    # (potentially compressed) logsignature into a signature first. (Which is certainly possible in principle)
    signature = smodule.signature(path, depth, stream=stream, basepoint=basepoint, inverse=inverse, initial=None)
    # lyndon_info=None because that's supported through the LogSignature class.
    return _signature_to_logsignature(signature, path.size(-1), depth, stream=stream, mode=mode,
                                      lyndon_info=None)


class LogSignature(nn.Module):
    """Module wrapper around the :func:`signatory.logsignature` function.

    Calling this Module on an input :code:`path` with the same number of channels as the last input :code:`path` it was
    called with will be faster than the corresponding :func:`signatory.logsignature` function, as this Module caches the
    result of certain computations which depend only on this value. (For larger depths or numbers of channels, this
    speedup will be substantial.)

    Arguments:
        depth (int): as :func:`signatory.logsignature`.

        stream (bool, optional): as :func:`signatory.logsignature`.

        inverse (bool, optional): as :func:`signatory.logsignature`.

        mode (str, optional): as :func:`signatory.logsignature`.
    """

    def __init__(self, depth, stream=False, inverse=False, mode="words", **kwargs):
        # type: (int, bool, bool, str, **Any) -> None
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

    def prepare(self, in_channels):
        # type: (int) -> None
        """Prepares for computing logsignatures of a certain size.

        There is some nontrivial computation which must be done for every logsignature computation of a certain size,
        and which is the same for all logsignature computations of that size. (Where 'size' refers to a specific
        combination of input channels, depth of logsignature, and mode.)

        :class:`signatory.LogSignature` caches this information, to help speed up later logsignature computations.
        Normally this information will simply be computed and cached the first time it is needed.

        This method allows for computing and cache this information up front, before performing any logsignature
        computations at all (for example, for benchmarking reasons).

        Arguments:
            in_channels (int): The number of input channels of the path that this instance will subsequently be called
                with. (corresponding to :code:`path.size(-1)`.)
        """

        # In particular does not return anything
        self._get_signature_to_logsignature_instance(in_channels)

    def forward(self, path, basepoint=False):
        # type: (torch.Tensor, Union[bool, torch.Tensor]) -> torch.Tensor
        """The forward operation.

        Arguments:
            path (torch.Tensor): As :func:`signatory.logsignature`.

            basepoint (bool or torch.Tensor, optional): As :func:`signatory.logsignature`.

        Returns:
            As :func:`signatory.logsignature`.
        """

        # Deliberately no 'initial' argument. To support that for logsignatures we'd need to be able to expand a
        # (potentially compressed) logsignature into a signature first. (Which is certainly possible in principle)
        signature = smodule.signature(path, self._depth, stream=self._stream, basepoint=basepoint,
                                      inverse=self._inverse, initial=None)
        return self._get_signature_to_logsignature_instance(path.size(-1))(signature)

    def extra_repr(self):
        return ('depth={depth}, stream={stream}, inverse={inverse}, mode{mode}'
                .format(depth=self._depth, stream=self._stream, inverse=self._inverse, mode=self._mode))


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


def logsignature_channels(in_channels, depth):
    # type: (int, int) -> int
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
