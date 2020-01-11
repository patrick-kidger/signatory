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
import esig.tosig
import torch


def setup(obj):
    obj.path = torch.rand(obj.size, dtype=torch.float).numpy()


def run(obj):
    first_result = esig.tosig.stream2logsig(obj.path[0], obj.depth)
    if not len(first_result):
        raise Exception

    result = [first_result]
    for batch_elem in obj.path[1:]:
        result.append(esig.tosig.stream2logsig(batch_elem, obj.depth))
    return result
