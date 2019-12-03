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


# noinspection PyUnresolvedReferences
from . import _impl


# For some reason some exceptions on a Mac are converted to RuntimeErrors rather than ValueErrors.
# So we have to make a conversion.
# This isn't perfect; any genuine RuntimeErrors will now always be ValueErrors.
# So for consistency across platforms we _always_ convert RuntimeErrors to ValueErrors.
def _wrap(fn):
    # We'd like to perform a check that fn is actually a function here
    # But that throws an error with the mocking used in the documentation
    # Easiest to not check and rely on the tests to, y'know, test.

    # We'd also like to @functools.wrap(fn) this
    # but again it fails with autodoc.
    # Not super important, as nothing in this module should be public anyway.
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            raise ValueError(str(e))
    return wrapped


LogSignatureMode = _impl.LogSignatureMode  # not wrapped because it's not a function
signature_to_logsignature_forward = _wrap(_impl.signature_to_logsignature_forward)
signature_to_logsignature_backward = _wrap(_impl.signature_to_logsignature_backward)
make_lyndon_info = _wrap(_impl.make_lyndon_info)
signature_forward = _wrap(_impl.signature_forward)
signature_backward = _wrap(_impl.signature_backward)
signature_checkargs = _wrap(_impl.signature_checkargs)
signature_channels = _wrap(_impl.signature_channels)
signature_combine_forward = _wrap(_impl.signature_combine_forward)
signature_combine_backward = _wrap(_impl.signature_combine_backward)
lyndon_words_to_basis_transform = _wrap(_impl.lyndon_words_to_basis_transform)
lyndon_words = _wrap(_impl.lyndon_words)
lyndon_brackets = _wrap(_impl.lyndon_brackets)
