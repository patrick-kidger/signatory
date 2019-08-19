import functools as ft
import math
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


# noinspection PyProtectedMember
class _LogSignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, mode, lyndon_info):
        # lyndon_info isn't a documented parameter because it's only used internally in the package.
        # It must be either None or the result of a call to _impl._make_lyndon_info

        mode = backend.mode_convert(mode)
        return backend.forward(ctx, path, depth, stream, basepoint, _impl.logsignature_forward, (mode, lyndon_info))

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        return (*backend.backward(ctx, grad_result, _impl.logsignature_backward), None, None)


def logsignature(path, depth, stream=False, basepoint=False, mode="brackets"):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor], str) -> torch.Tensor
    """Applies the logsignature transform to a stream of data.

    Applies the signatures to the stream of data, then takes the logarithm (in the truncated tensor algebra) of the
    result. (It is *not* taking the pointwise logarithm: the high-dimensional Euclidean space in which the signature
    lives has an interpretation as a "truncated tensor algebra", on which a particular notion of "logarithm" is defined.
    See the "Returns" section below for a little more detail.)

    In particular the :attr`modes` argument determines how the logsignature is represented.

    Note that if performing many logsignature calculations for the same depth and size of input, then you will likely
    see a performance boost by using :class:`signatory.LogSignature` over :class:`signatory.logsignature`.

    Arguments:
        path (:class:`torch.Tensor`): as :func:`signatory.signature`.

        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

        mode (str, optional): How the output should be presented. Valid values are "words", "brackets", or
            "expand". Precisely what each of these options mean is described in the "Returns" section below. As a rule
            of thumb: use "words" for new projects (as it is the fastest); use "brackets" if you want compatibility with
            other projects (such as `iisignature <https://github.com/bottler/iisignature>`__, which defaults to this).
            The mode "expand" is mostly only interesting for mathematicians.

    Returns:
        A :class:`torch.Tensor`. If :code:`mode == "expand"` then it will be of the same shape as the returned tensor
        from :func:`signatory.signature`. If :code:`mode in ("words", "brackets")` then it will again be of the
        same shape, except that the channel dimension will instead be of size
        :code:`logsignature_channels(in_channels, depth)`. (i.e. much smaller, which is the whole point of using the
        logsignature over the signature in the first place.)

        We now go on to explain what the different values for :attr:`mode` mean. This discussion is the "Returns"
        section a the value of :attr:`mode` essentially just determines how the output is represented; the
        mathematical meaning is the same in all cases. We start with an explanation for the reader who is not familiar
        with notions such as free Lie algebras and Lyndon words. Which is most people! For the more mathematically
        inclined reader, we have an more in-depth explanation later.

        First the low-math explanation.

        The signatures computed by the :func:`signatory.signature` function are a large collection of numbers; that is,
        they live in a high-dimensional Euclidean space (formally called the "truncated tensor algebra"). This space has
        a a particular notion of "logarithm", and when called with :code:`mode == "expand"` then the logsignature will
        also be displayed as a member of the truncated tensor algebra.

        However it turns out that this is space-inefficient: the logsignature may be represented by a smaller collection
        of numbers, that is, it's really a member of a smaller-dimensional Euclidean space (formally it is a "free Lie
        algebra"). Its representation in the high-dimensional truncated tensor algebra is simply a projection from this
        small space into the larger space. As such this probably isn't a useful value for :attr:`mode` when doing
        machine learning; you instead want a representation in the smaller space..

        A representation in this smaller space is what will be returned if :code:`mode in ("words", "brackets")`.
        (The keen-eyed reader will notice that since the logarithm is bijective, then this also means that the signature
        itself is also using more space than necessary to represent itself -- as it can just be represented by its
        corresponding logsignature, which lives in a smaller dimensional space. This is completely correct. However in
        some sense this larger representation is the whole point of the signature: the dependency between its terms is
        nonlinear, which give lots of interesting features to learn machine learning models on. The same larger
        representation for the logsignature only has linear dependencies amongst its terms, so the same statement isn't
        true there.)

        The smaller space has multiple different possible bases. That is, there are multiple ways to represent elements
        of this smaller space. We offer three different bases; all of them are just linear transformations of each
        other.

        A popular basis is called the "Lyndon basis", and this is the basis that is used if :code:`mode == "brackets"`.
        This is thus the default choice, as it is the result that is typically expected.

        However it turns out that there is a more computationally efficient basis which may be used: this is what is
        used if :code:`mode == "words"`. Thus this is the recommended choice for new projects.

        Now the high-math explanation.

        The explanation will now assume that you are familiar with Lyndon words, the Lyndon basis, and Hall bases.

        As already explained, :code:`mode == "expand"` represents the logsignature in the truncated tensor algebra, so
        that the numbers you see are the coefficients of all possible words. These words are first ordered by length,
        and within each length class they are ordered lexicographically.

        Next, :code:`mode == "brackets"` represents the logsignature as coefficients of the Lyndon basis. The order of
        the coefficients corresponds to an ordering of the foliage of the elements of the Lyndon basis (i.e. the Lyndon
        words usually associated with each basis element.) These words are first ordered by length, and within each
        length class they are ordered lexicographically.

        Now it turns out that the coefficients in the Lyndon basis may be computed by calculated by (a) computing the
        logsignature in the truncated tensor algebra, (b) extracting the coefficents of all Lyndon words, and (c)
        applying a unitriangular (i.e. triangular with 1 on the diagonal) linear transformation to them. In particular,
        this transformation is clearly invertible (as you'd expect, given that we're computing a basis), so we see that
        the coefficients of all the Lyndon words also contain the same information; the Lyndon words may be interpreted
        as forming a basis for the free Lie algebra. (And this basis in terms of the Lyndon words is *not* the same as
        the Lyndon basis!) This gives a non-Hall basis for the free Lie algebra; each basis element is a sum of Lyndon
        brackets such that when expanded out, each sum contains precisely one Lyndon word, and every Lyndon word is part
        of such a sum. The representation in this basis is what is computed when :code:`mode == "words"`. As usual the
        coefficients are given by an ordering of the words, in which the words are first ordered by length, and within
        each length class are ordered lexicographically.
    """
    # noinspection PyUnresolvedReferences
    return _LogSignatureFunction.apply(path, depth, stream, basepoint, mode, None)


class LogSignature(nn.Module):
    """Module wrapper around the :func:`signatory.logsignature` function.

    Calling this module on an input `path` with the same number of channels as the last input `path` it was called with
    will be faster than the corresponding :func:`signatory.logsignature` function, as this module caches the result of
    certain computations which depend only on this value. (For larger numbers of channels, this speedup will be
    substantial.)

    Arguments:
        depth (int): as :func:`signatory.logsignature`.

        stream (bool, optional): as :func:`signatory.logsignature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.logsignature`.

        mode (str, optional): as :func:`signatory.logsignature`.

    Called with a single argument :attr:`path` of type :class:`torch.Tensor`.
    """

    def __init__(self, depth, stream=False, basepoint=False, mode="brackets", **kwargs):
        # type: (int, bool, Union[bool, torch.Tensor], str, **Any) -> None
        super(LogSignature, self).__init__(**kwargs)
        self.depth = depth
        self.stream = stream
        self.basepoint = basepoint
        self.mode = mode

    @staticmethod
    @ft.lru_cache(maxsize=None)  # This computation can be pretty slow! We definitely want to reuse it between instances
    def lyndon_info_cache(channels, depth, mode):
        mode = backend.mode_convert(mode)
        return _impl.make_lyndon_info(channels, depth, mode)

    def forward(self, path):
        # type: (torch.Tensor) -> torch.Tensor

        lyndon_info = self.lyndon_info_cache(path.size(-1), self.depth, self.mode)
        # don't call logsignature itself because that (deliberately) doesn't expose a lyndon_info argument.
        # noinspection PyProtectedMember, PyUnresolvedReferences
        return _LogSignatureFunction.apply(path, self.depth, self.stream, self.basepoint, self.mode, lyndon_info)

    def extra_repr(self):
        return ('depth={depth}, stream={stream}, basepoint={basepoint}, mode{mode}'
                .format(depth=self.depth, stream=self.stream, basepoint=str(self.basepoint)[:6], mode=self.mode))


def _get_prime_factors(x):
    if x == 1:
        return []
    prime_factors = []
    largest_i_so_far = 2
    while True:
        for i in range(largest_i_so_far, round(math.sqrt(x)) + 1):
            if x % i == 0:
                largest_i_so_far = i
                break
        else:
            prime_factors.append(x)  # x is prime
            break
        x = x // i
        prime_factors.append(i)
    return prime_factors


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
    """Computes the number of output channels from a logsignature call with :code:`mode in ("words", "brackets")`.

    Arguments:
        in_channels (int): The number of channels in the input; that is, the dimension of the space that the input path
            resides in.

        depth (int): The depth of the signature that is being computed.

    Returns:
        An int specifying the number of channels in the logsignature of the path.
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
