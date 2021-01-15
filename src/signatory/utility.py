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
"""Provides operations relating to Lyndon words and Lyndon brackets."""


import itertools as it

from . import impl


from typing import List, Union
# what we actually want, but can't make sense of in the auto-generated documentation
# LyndonBracket = Union[int, List['LyndonBracket']]
LyndonBracket = Union[int, List]


def lyndon_words(channels: int, depth: int) -> List[List[int]]:
    r"""Computes the collection of all Lyndon words up to length :attr:`depth` in an alphabet of size
    :attr:`channels`. Each letter is represented by an integer :math:`i` in the range
    :math:`0 \leq i < \text{channels}`.

    Logsignatures may be thought of as a sum of coefficients of Lyndon words. This gives the words in the order that
    they correspond to the values returned by :func:`signatory.logsignature` with :code:`mode="words"`.

    Arguments:
        channels (int): The size of the alphabet.
        depth (int): The maximum word length.

    Returns:
        A list of lists of integers. Each sub-list corresponds to one Lyndon word. The words are ordered by length, and
        then ordered lexicographically within each length class.
    """

    return impl.lyndon_words(channels, depth)


def lyndon_brackets(channels: int, depth: int) -> List[LyndonBracket]:
    r"""Computes the collection of all Lyndon words, in their standard bracketing, up to length :attr:`depth` in an
    alphabet of size :attr:`channels`. Each letter is represented by an integer :math:`i` in the range
    :math:`0 \leq i < \text{channels}`.

    Logsignatures may be thought of as a sum of coefficients of Lyndon brackets. This gives the brackets in the order
    that they correspond to the values returned by :func:`signatory.logsignature` with :code:`mode="brackets"`.

    Arguments:
        channels (int): The size of the alphabet.
        depth (int): The maximum word length.

    Returns:
        A list. Each element corresponds to a single Lyndon word with its standard bracketing. The words are ordered by
        length, and then ordered lexicographically within each length class."""

    return impl.lyndon_brackets(channels, depth)


def all_words(channels: int, depth: int) -> List[List[int]]:
    r"""Computes the collection of all words up to length :attr:`depth` in an alphabet of size
    :attr:`channels`. Each letter is represented by an integer :math:`i` in the range
    :math:`0 \leq i < \text{channels}`.

    Signatures may be thought of as a sum of coefficients of words. This gives the words in the order that they
    correspond to the values returned by :func:`signatory.signature`.

    Logsignatures may be thought of as a sum of coefficients of words. This gives the words in the order that they
    correspond to the values returned by :func:`signatory.logsignature` with :code:`mode="expand"`.

    Arguments:
        channels (int): The size of the alphabet.
        depth (int): The maximum word length.

    Returns:
        A list of lists of integers. Each sub-list corresponds to one word. The words are ordered by length, and
        then ordered lexicographically within each length class.
    """

    def generator():
        r = range(channels)
        for depth_index in range(1, depth + 1):
            ranges = (r,) * depth_index
            for elem in it.product(*ranges):
                yield elem
    # Just returning the generator would be much nicer, programmatically speaking, but then this is inconsistent with
    # the lyndon_words function. This isn't expected to use a lot of memory so this is acceptable.
    return list(generator())
