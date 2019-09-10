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
"""Tests the Path class."""


import signatory
import torch
from torch import autograd

import utils_testing as utils


class TestPath(utils.EnhancedTestCase):
    def test_path(self):
        for c in utils.ConfigIter(inverse=False, stream=False, N=(1, 2), depth=(1, 2, 4)):
            path_obj = signatory.Path(c.path, c.depth, basepoint=c.basepoint)
            for start in range(-2 * c.path.size(1), 2 * c.path.size(1)):
                for end in range(-2 * c.path.size(1), 2 * c.path.size(1)):
                    if c.basepoint is True:
                        basepointed_path = torch.cat([torch.zeros(c.N, 1, c.C, dtype=torch.double, device=c.device),
                                                      c.path], dim=1)
                    elif c.basepoint is False:
                        basepointed_path = c.path
                    else:  # isinstance(self.basepoint, torch.Tensor) == True
                        basepointed_path = torch.cat([c.basepoint.unsqueeze(1), c.path], dim=1)

                    try:
                        a_path = path_obj.signature(start, end)
                    except ValueError:
                        try:
                            c.signature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                        except ValueError:
                            continue
                        else:
                            self.fail(c.fail(start=start, end=end))
                    true_path = c.signature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                    if not true_path.allclose(a_path):
                        self.fail(c.fail(start=start, end=end))

    def test_gradient(self):
        def gradchecked(path, depth, basepoint, start, end):
            return signatory.Path(path, depth, basepoint=basepoint).signature(start, end)

        for c in utils.ConfigIter(inverse=False,
                                  stream=False,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            length = c.path.size(1)
            if isinstance(c.basepoint, torch.Tensor) or c.basepoint:
                length += 1
            for start in range(0, length + 1):
                for end in range(start + 2, length + 1):
                    autograd.gradcheck(gradchecked, (c.path, c.depth, c.basepoint, start, end))
