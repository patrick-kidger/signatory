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


import functools as ft
import sys
import torch  # must be imported before anything from signatory
import types

try:
    from . import _impl
except ImportError as e:
    if 'specified procedure could not be found' in str(e):
        raise ImportError('Caught ImportError: {}. This can probably be fixed by updating your version of Python, '
                          'e.g. from 3.6.6 to 3.6.9'.format(str(e)))
    else:
        raise


if 'darwin' in sys.platform:
    # It seems that either the use of clang or running on Mac means that exceptions don't get properly translated by
    # pybind11. In particular it raises RuntimeError("Caught an unknown exception!") rather than anything else.
    # We only use ValueErrors so we just translate to that; this is pretty much the best we can do.
    class compat_module(_impl.__class__):
        wraps = {}  # should really be an instance attribute but we only have one instance of this class, so it doesn't
                    # matter

        def _get_wrapped(self, obj):
            if isinstance(obj, types.BuiltinFunctionType):
                try:
                    return self.__class__.wraps[obj]
                except KeyError:
                    @ft.wraps(obj)
                    def obj_wrapper(*args, **kwargs):
                        try:
                            return obj(*args, **kwargs)
                        except RuntimeError as e:
                            if 'unknown exception' in str(e):
                                raise ValueError("Exception raised. Unfortunately C++-to-Python translation of "
                                                 "exceptions doesn't work properly on a Mac so that's all we know.")
                            else:
                                raise
                    self.__class__.wraps[obj] = obj_wrapper
                    return obj_wrapper
            else:
                return obj

        def __getattribute__(self, item):
            obj = super(compat_module, self).__getattribute(item)
            return self._get_wrapped(obj)

    _impl.__class__ = compat_module


from .augment import Augment
from .logsignature_module import (logsignature,
                                  LogSignature,
                                  Logsignature,  # alias for LogSignature
                                  logsignature_channels)
from .path import Path
from .signature_module import (signature,
                               Signature,
                               signature_channels,
                               extract_signature_term,
                               signature_combine)
from .utility import (lyndon_words,
                      lyndon_brackets,
                      all_words)


__version__ = "1.1.4"

del torch
