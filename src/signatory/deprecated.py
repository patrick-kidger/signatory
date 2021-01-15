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
"""Keeps deprecated interfaces around."""


import warnings

from typing import Optional


max_parallel_value = [8]


def max_parallelism(value: Optional[int] = None) -> int:
    """Deprecated and now does nothing. Use torch.set_num_threads and torch.get_num_threads instead.

    Old docstring:

    Gets or sets the maximum amount of parallelism used in Signatory's computations. Higher values will typically
    result in quicker computations but will use more memory.

    Calling without arguments will return the current value.
    Passing a value of 1 will disable parallelism.
    Passing :code:`-1`, :code:`math.inf`, :code:`np.inf` or :code:`float('inf')` will enable unlimited parallelism.
    """
    warnings.warn("max_parallelism is deprecated and now does nothing. Use torch.set_num_threads and "
                  "torch.get_num_threads to control the amount of parallelism.", DeprecationWarning)
    if value == -1:
        value = float('inf')
    if value is not None:
        max_parallel_value[0] = value
    return max_parallel_value[0]
