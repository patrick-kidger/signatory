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
"""Provides benchmarking helpers."""


import collections as co
import itertools as it
import numpy as np


class namedarray(object):
    """Wraps a numpy array with name-based lookup along axes."""

    def __init__(self, *size):
        self.array = np.empty(size, dtype=object)
        self.numdims = len(size)
        self.dim_lookups = [co.OrderedDict() for _ in range(self.numdims)]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            raise ValueError
        if len(key) != self.numdims:
            raise ValueError
        indices = []
        for elem, lookup in zip(key, self.dim_lookups):
            if isinstance(elem, slice):
                raise ValueError
            try:
                index = lookup[elem]
            except KeyError:
                index = lookup[elem] = len(lookup)
            indices.append(index)
        indices = tuple(indices)
        self.array[indices] = value

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise ValueError
        if len(key) != self.numdims:
            raise ValueError
        indices = []
        for elem, lookup in zip(key, self.dim_lookups):
            try:
                index = lookup[elem]
            except KeyError:
                index = elem
            indices.append(index)
        indices = tuple(indices)
        return self.array[indices]

    def __iter__(self):
        lookups = tuple(lookup.keys() for lookup in self.dim_lookups)
        for index in it.product(*lookups):
            yield index, self[index]


# Allows for using classes a bit like dictionaries, specifying key-value pairs. They can inherit from one another.
# e.g.
#
# class A(Container):
#   x = 4
# class B(A):
#   y = 3
# 4 in A  # True
# 3 in A  # False
# 4 in B  # True
# 3 in B  # True
class MetaContainer(type):
    def __contains__(self, item):
        return item in self.__dict__.values() or any(item in base for base in self.__bases__
                                                     if isinstance(base, MetaContainer))


Container = MetaContainer('Container', (object,), {})  # Python 2/3 compatible metaclasses
