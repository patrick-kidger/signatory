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


import os
import pytest
import torch


_here = os.path.realpath(os.path.dirname(__file__))
add_to_path = os.path.join(_here, '..', 'examples')
pytestmark = pytest.mark.usefixtures('path_hack')


def test_example1():
    import example1
    _example_tester(example1.SigNet)


def test_example2():
    import example2
    _example_tester(example2.SigNet2)


def test_example3():
    import example3
    _example_tester(example3.SigNet3)


def _example_tester(example_fn):
    batch_size = 4
    in_channels = 4
    out_dimension = 4
    sig_depth = 4

    x = torch.rand(batch_size, 10, in_channels)
    signet = example_fn(in_channels, out_dimension, sig_depth)
    y = signet(x)

    assert y.shape == (batch_size, out_dimension)
