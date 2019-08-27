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
"""Provides Python 2/3 compatibility for tests.

Builds on the compatibility file for the main Signatory project. (But is not included in it so as not to pollute the
main project with things only needed for the tests).
"""


import functools as ft
import signatory.compatibility as compat
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
        try:
            return compat.lru_cache(maxsize, typed)
        except ValueError:
            if maxsize != 1:
                raise ValueError("Only maxsize=1 supported.")
            if typed:
                raise ValueError("Only type=False supported.")
            # If we ever need to support either of those then we can modify this
            return LruCacheSizeOne

       
def skip(fn):
    """Python 2/3-compatible skip function."""
    
    if sys.version_info.major == 2:
        # unittest.skip seems to have a bug in Python 2
        def skipped_fn(self, *args, **kwargs):
            pass
        return skipped_fn
    else:
        return unittest.skip(fn)
