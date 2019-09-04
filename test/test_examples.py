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
"""Tests the examples given in the documentation to make sure they don't raise errors."""


import torch

import example1
import example2
import example3
import utils_testing as utils


class TestExamples(utils.EnhancedTestCase):
    def example_tester(self, example_fn):
        batch_size = 4
        in_channels = 4
        out_dimension = 4
        sig_depth = 4

        x = torch.rand(batch_size, 4, in_channels)
        signet = example_fn(in_channels, out_dimension, sig_depth)
        y = signet(x)
        self.assertEqual(y.shape, (batch_size, out_dimension))

    def test_example1(self):
        self.example_tester(example1.SigNet)

    def test_example2(self):
        self.example_tester(example2.SigNet2)

    def test_example3(self):
        self.example_tester(example3.SigNet3)
