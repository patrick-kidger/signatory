# noinspection PyUnresolvedReferences
from . import _impl


# noinspection PyUnreachableCode
if False:
    from typing import List, Tuple, Union
    LyndonBracket = Union[int, Tuple['LyndonBracket', 'LyndonBracket']]


def lyndon_words(channels, depth):
    # type: (int, int) -> List[List[int]]
    r"""Computes the collection of all Lyndon words up to length :attr:`depth` in an alphabet of size
    :attr:`in_channels`. Each letter is represented by an integer :math:`i` in the range
    :math:`0 \leq i < \text{in_channels}`.

    Arguments:
        channels (int): The size of the alphabet.
        depth (int): The maximum word length.

    Returns:
        A list of lists of integers. Each sub-list corresponds to one Lyndon word.
    """

    return _impl.lyndon_words(channels, depth)


def lyndon_brackets(channels, depth):
    # type: (int, int) -> List[LyndonBracket]
    r"""Computes the collection of all Lyndon words, in their standard bracketing, up to length :attr:`depth` in an
    alphabet of size :attr:`in_channels`. Each letter is represented by an integer :math:`i` in the range
    :math:`0 \leq i < \text{in_channels}`.

    Arguments:
        channels (int): The size of the alphabet.
        depth (int): The maximum word length.

    Returns:
        A list. Each element corresponds to a single Lyndon word with its standard bracketing."""

    return _impl.lyndon_brackets(channels, depth)


def lyndon_words_to_basis_transform(channels, depth):
    # type: (int, int) -> List[Tuple[int, int, int]]
    """Computes the collection of transforms needed to go from a basis of the free Lie algebra in terms of Lyndon words
    to a basis of the free Lie algebra in terms of the Lyndon basis."""
    # TODO: put example code here to explain

    return _impl.lyndon_words_to_basis_transform(channels, depth)
