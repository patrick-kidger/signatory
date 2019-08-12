import math
import torch

# noinspection PyProtectedMember, PyUnresolvedReferences
from ._impl import _signature_channels


# A wrapper for the sake of consistent documentation
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


def extract_term(sig_tensor, in_channels, depth):
    # type: (torch.Tensor, int, int) -> torch.Tensor
    r"""Extracts a particular term from a signature.

    The signature to depth :math:`d` of a batch of paths in :math:`\mathbb{R}^\text{C}` is a tensor with
    :math:`C + C^2 + \cdots + C^d` channels. (See :func:`signatory.signature`.) This function extracts the :attr:`depth`
    term of that, returning a tensor with just :math:`C^\text{depth}` channels.

    Arguments:
        sig_tensor (:class:`torch.Tensor`): The signature to extract the term from. Should be the result of the
            :func:`signatory.signature` function.

        in_channels (int): The number of input channels :math:`C`.

        depth (int): The depth of the term to be extracted from the signature.

    Returns:
        The :class:`torch.Tensor` corresponding to the :attr:`depth` term of the signature.
    """

    if in_channels < 1:
        raise ValueError("in_channels must be at least 1")

    if depth == 1:
        start = 0
    else:
        start = signature_channels(in_channels, depth - 1)
    return sig_tensor.narrow(dim=-1, start=start, length=in_channels ** depth)
