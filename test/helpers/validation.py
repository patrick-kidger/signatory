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
"""Helpers for validating the tests themselves."""


import argparse
import collections as co
import inspect
import itertools as it
import os
import signatory


# We could easily use something like networkx here but I'd prefer not to introduce extra dependencies.
class _Graph(object):
    def __init__(self):
        super(_Graph, self).__init__()
        # Maps vertices to the other vertices they are adjacent to.
        self._graph = co.defaultdict(list)
        self._marked = set()

    def add_directed_edge(self, start, end, info):
        self._graph[start].append((end, info))
        self._graph[end]  # create if it doesn't already exist

    def vertices(self):
        return set(self._graph.keys())

    def _is_cyclic(self, vertex, visited, current):
        visited[vertex] = True
        current[vertex] = True

        for neighbour, info in self._graph[vertex]:
            if current[neighbour]:
                return [neighbour, info], False
            if not visited[neighbour]:
                cycle_info = self._is_cyclic(neighbour, visited, current)
                if cycle_info is not None:
                    cycle, cycle_is_complete = cycle_info
                    if not cycle_is_complete:
                        cycle.append(neighbour)
                        if neighbour == cycle[0]:
                            cycle_is_complete = True
                        else:
                            cycle.append(info)
                    return cycle, cycle_is_complete

        current[vertex] = False
        return

    def get_cycle(self):
        master_vertex = object()
        self._graph[master_vertex] = [(key, None) for key in self._graph.keys()]
        visited = {key: False for key in self._graph.keys()}
        current = {key: False for key in self._graph.keys()}
        cycle_info = self._is_cyclic(master_vertex, visited, current)
        del self._graph[master_vertex]
        if cycle_info is not None:
            cycle, _ = cycle_info
            return cycle

    def mark(self, vertex):
        self._marked.add(vertex)

    def get_unmarked(self):
        return {key for key in self._graph.keys() if key not in self._marked}


signatory_functionality_graph = _Graph()


def validate_tests(tests, depends):
    """Some of the tests depend upon other parts of the Signatory library, which are assumed to be correct. So we have
    to be careful not to accidentally create a cyclic dependency, where X assumes that Y is correct and tests against
    it, whilst (possibly indirectly) Y assumes that X is correct, and tests against it... which could potentially result
    in both being wrong and neither test detecting it.

    This function is the solution. Signatory should never be directly imported in any of the tests, but 'imported' via
    this function instead, where the module specifies what functionality it tests, and what functionality it depends
    upon.

    That and only that functionality is then exposed through the object returned by this function.

    Furthermore, this function keeps track of every time it is called, and makes sure that no cyclic dependencies arise.
    """

    if signatory in inspect.currentframe().f_back.f_globals.values():
        raise RuntimeError('Signatory has already been imported separately. Remove the import statement and use the '
                           'mock of Signatory returned by this function.')

    for test in tests:
        signatory_functionality_graph.mark(test)
        for depend in depends:
            signatory_functionality_graph.add_directed_edge(test, depend, os.path.basename(inspect.stack()[1][1]))

    signatory_mock = argparse.Namespace()
    for string in it.chain(tests, depends):
        obj = signatory
        obj_mock = signatory_mock
        split_string = string.split('.')
        last = len(split_string) - 1
        for i, string_elem in enumerate(split_string):
            obj = getattr(obj, string_elem)
            obj_mock_new = argparse.Namespace()
            setattr(obj_mock, string_elem, obj if i == last else obj_mock_new)
            obj_mock = obj_mock_new
    return signatory_mock
