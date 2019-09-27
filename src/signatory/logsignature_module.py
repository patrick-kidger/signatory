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

from . import backend
from . import compatibility as compat
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


# noinspection PyProtectedMember
class _LogSignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, inverse, mode, lyndon_info):
        # lyndon_info isn't a documented parameter because it's only used internally in the package.
        # It must be either None (as in logsignature) or the result of a call to _impl._make_lyndon_info (as in
        # LogSignature).

        mode = _mode_convert(mode)
        ctx.basepoint = basepoint

        basepoint, basepoint_value = backend.interpret_basepoint(basepoint, path)

        path = path.transpose(0, 1)  # (batch, stream, channel) to (stream, batch, channel)
        with compat.mac_exception_catcher:
            result, backwards_info = _impl.logsignature_forward(path, depth, stream, basepoint, basepoint_value,
                                                                inverse, mode, lyndon_info)
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
            grad_path, grad_basepoint = _impl.logsignature_backward(grad_result, ctx.backwards_info)
        grad_path = grad_path.transpose(0, 1)  # (stream, batch, channel) to (batch, stream, channel)
        if not isinstance(ctx.basepoint, torch.Tensor):
            grad_basepoint = None

        return grad_path, None, None, grad_basepoint, None, None, None


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
    # noinspection PyUnresolvedReferences
    result = _LogSignatureFunction.apply(path, depth, stream, basepoint, inverse, mode, None)

    # We have to do the transpose outside of autograd.Function.apply to avoid a PyTorch bug.
    # https://github.com/pytorch/pytorch/issues/24413
    if stream:
        result = result.transpose(0, 1)  # NOT .transpose_ - the underlying TensorImpl (in C++) is used elsewhere and we
                                         # don't want to change it.
    return result


class LogSignature(nn.Module):
    """Module wrapper around the :func:`signatory.logsignature` function.

    Calling this module on an input :attr:`path` with the same number of channels as the last input :attr:`path` it was
    called with will be faster than the corresponding :func:`signatory.logsignature` function, as this module caches the
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
        self.depth = depth
        self.stream = stream
        self.inverse = inverse
        self.mode = mode

    @staticmethod
    # This computation can be pretty slow! We definitely want to reuse it between instances
    @compat.lru_cache(maxsize=None)
    def lyndon_info_cache(channels, depth, mode):
        mode = _mode_convert(mode)
        with compat.mac_exception_catcher:
            return _impl.make_lyndon_info(channels, depth, mode)

    def forward(self, path, basepoint=False):
        # type: (torch.Tensor, Union[bool, torch.Tensor]) -> torch.Tensor
        """The forward operation.

        Arguments:
            path (torch.Tensor): As :func:`signatory.logsignature`.

            basepoint (bool or torch.Tensor, optional): As :func:`signatory.logsignature`.

        Returns:
            As :func:`signatory.logsignature`.
        """

        lyndon_info = self.lyndon_info_cache(path.size(-1), self.depth, self.mode)
        # don't call logsignature itself because that (deliberately) doesn't expose a lyndon_info argument.
        # noinspection PyProtectedMember, PyUnresolvedReferences
        result = _LogSignatureFunction.apply(path, self.depth, self.stream, basepoint, self.inverse, self.mode,
                                             lyndon_info)

        # We have to do the transpose outside of autograd.Function.apply to avoid a PyTorch bug.
        # https://github.com/pytorch/pytorch/issues/24413
        if self.stream:
            result = result.transpose(0, 1)  # NOT .transpose_ - the underlying TensorImpl (in C++) is used elsewhere
                                             # and we don't want to change it.
        return result

    def extra_repr(self):
        return ('depth={depth}, stream={stream}, basepoint={basepoint}, mode{mode}'
                .format(depth=self.depth, stream=self.stream, basepoint=str(self.basepoint)[:6], mode=self.mode))


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
            resides in.

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
