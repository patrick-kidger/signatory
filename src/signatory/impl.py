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
"""Provides an interface to _impl."""


import functools as ft
import sys
import types

# noinspection PyUnresolvedReferences
from . import _impl


def wrap(fn):
    if not isinstance(fn, types.BuiltinFunctionType):
        raise ValueError("Can't wrap a non-function.")
    if sys.platform.startswith('darwin'):
        @ft.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except RuntimeError as e:
                if 'unknown exception' in str(e):
                    raise ValueError("Exception raised. Unfortunately C++-to-Python translation of exceptions doesn't "
                                     "work properly on a Mac so that's all we know. See the FAQ.")
                else:
                    raise
        return wrapped
    else:
        return fn


LogSignatureMode = _impl.LogSignatureMode
signature_to_logsignature_forward = wrap(_impl.signature_to_logsignature_forward)
signature_to_logsignature_backward = wrap(_impl.signature_to_logsignature_backward)
make_lyndon_info = wrap(_impl.make_lyndon_info)
signature_forward = wrap(_impl.signature_forward)
signature_backward = wrap(_impl.signature_backward)
signature_checkargs = wrap(_impl.signature_checkargs)
hardware_concurrency = wrap(_impl.hardware_concurrency)
signature_channels = wrap(_impl.signature_channels)
signature_combine_forward = wrap(_impl.signature_combine_forward)
signature_combine_backward = wrap(_impl.signature_combine_backward)
lyndon_words_to_basis_transform = wrap(_impl.lyndon_words_to_basis_transform)
lyndon_words = wrap(_impl.lyndon_words)
lyndon_brackets = wrap(_impl.lyndon_brackets)
set_max_parallelism = wrap(_impl.set_max_parallelism)
get_max_parallelism = wrap(_impl.get_max_parallelism)
