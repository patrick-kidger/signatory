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
"""This module exposes functionality that isn't documented as part of the public API. In particular that means that
everything here is experimental and may change without notice.

The functionality here is either exposed for developer convenience, or because the functionality is niche enough that it
hasn't been worth documenting and exposing.
"""


from .signature_module import (set_signature_calculation_methods,
                               reset_signature_calculation_methods)

from ._impl import (lyndon_words_to_basis_transform,
                    built_with_open_mp,)
