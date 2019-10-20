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
"""Provides Python 2/3 compatibility for tests."""


import functools as ft
import sys
import unittest


try:
    # Python 2
    stringtype = basestring
except NameError:
    # Python 3
    stringtype = str

        
try:
    lru_cache = ft.lru_cache
except AttributeError:
    # Python 2

    class LruCacheSizeNone:
        """# A poor man's lru_cache. No maxsize argument for simplicity (hardcoded to None)"""

        def __init__(self, fn):
            self.memodict = {}
            self.fn = fn

        def __call__(self, *args):
            try:
                return self.memodict[args]
            except KeyError:
                out = self.fn(*args)
                self.memodict[args] = out
                return out
    
    class LruCacheSizeOne:
        """# A poor man's lru_cache. No maxsize argument for simplicity (hardcoded to one)"""
        def __init__(self, fn):
            self.memoargs = object()
            self.memoout = None
            self.fn = fn

        def __call__(self, *args):
            if args == self.memoargs:
                return self.memoout
            else:
                out = self.fn(*args)
                self.memoargs = args
                self.memoout = out
                return out
    
    def lru_cache(maxsize=128, typed=False):
        if typed:
            raise ValueError("Only typed=False supported.")

        if maxsize is None:
            return LruCacheSizeNone
        elif maxsize == 1:
            return LruCacheSizeOne
        else:
            raise ValueError("Only maxsize=1 or maxsize=None are supported.")


def skip(fn):
    """Python 2/3-compatible skip function."""
    
    if sys.version_info.major == 2:
        # unittest.skip seems to have a bug in Python 2
        def skipped_fn(self, *args, **kwargs):
            pass
        return skipped_fn
    else:
        return unittest.skip(fn)
