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
"""Provides Python 2/3 compatibility and different OS compatability"""


import functools as ft
import sys


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

    def lru_cache(maxsize=128, typed=False):
        if maxsize != None:
            raise ValueError("Only maxsize=None supported.")
        if typed:
            raise ValueError("Only type=False supported.")
        # If we ever need to support either of those then we can modify this
        return LruCacheSizeNone


# It seems that either the use of clang or running on Mac means that exceptions don't get properly translated by
# pybind11. In particular it raises RuntimeError("Caught an unknown exception!") rather than anything else.
# We only use ValueErrors so we just translate to that; this is pretty much the best we can do.
class _MacExceptionCatcherClass(object):
    """Translates exceptions caught on Macs, which do things slightly differently."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is RuntimeError and 'unknown exception' in str(exc_val) and 'darwin' in sys.platform:
            raise ValueError("Exception raised. Unfortunately C++-to-Python translation of exceptions doesn't work "
                             "properly on a Mac so that's all we know.")


mac_exception_catcher = _MacExceptionCatcherClass()
