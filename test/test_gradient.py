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
"""Tests that the results of the backward computations are accurate by using the gradcheck function provided by
PyTorch."""


import signatory
import torch.autograd as autograd

import compatibility as compat
import utils_testing as utils


class TestSignatureGrad(utils.EnhancedTestCase):
    @staticmethod
    def gradcheck(path, depth, stream, basepoint, inverse, **kwargs):
        return autograd.gradcheck(signatory.signature, (path, depth, stream, basepoint, inverse), **kwargs)

    def test_gradcheck_edge(self):
        for c in utils.ConfigIter(depth=(1, 2, 3),
                                  requires_grad=True,
                                  size=((1, 2, 1), (1, 4, 4), (4, 2, 4), (4, 4, 1))):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse)
            except RuntimeError:
                self.fail(c.fail())

    @compat.skip  # takes forever
    def test_gradcheck_grid(self):
        for c in utils.ConfigIter(requires_grad=True):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse)
            except RuntimeError:
                self.fail(c.fail())

    def test_gradcheck_random(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.random_size()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse)
            except RuntimeError:
                self.fail(c.fail())

    @compat.skip  # takes forever
    def test_gradcheck_large(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.large_size(),
                                  depth=utils.large_depth()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse)
            except RuntimeError:
                self.fail(c.fail())

    # We don't do gradgradcheck because our backwards function uses a whole bunch of in-place operations for memory
    # efficiency, so it's not automatically differentiable. (And I'm not writing a custom double backward function...)


class TestLogSignatureGrad(utils.EnhancedTestCase):
    @staticmethod
    def gradcheck(path, depth, stream, basepoint, inverse, mode, **kwargs):
        return autograd.gradcheck(signatory.logsignature, (path, depth, stream, basepoint, inverse, mode), **kwargs)

    def test_gradcheck_edge(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  depth=(1, 2),
                                  requires_grad=True,
                                  size=((1, 2, 1), (1, 3, 3), (3, 2, 3), (3, 3, 1))):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())

    @compat.skip  # takes forever
    def test_gradcheck_grid(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())

    def test_gradcheck_random(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=utils.random_size()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())

    @compat.skip  # takes forever
    def test_gradcheck_large(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=utils.large_size(),
                                  depth=utils.large_depth()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.inverse, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())
