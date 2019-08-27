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
"""Provides Python 2/3 compatibility."""


import functools as ft


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
