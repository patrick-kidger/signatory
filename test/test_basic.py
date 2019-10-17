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
"""Performs various 'basic' tests: that no errors are thrown in normal use (and conversely that errors are thrown when
they're supposed to), and that the outputs are the correct shapes."""


import signatory
import torch

import utils_testing as utils


class TestSignatureArguments(utils.EnhancedTestCase):
    def test_minimal_axes(self):
        for c in utils.ConfigIter(basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 2, 4), (4, 4, 1), (1, 2, 1)),
                                  requires_grad=True):
            try:
                c.signature()
                c.signature_backward()
            except Exception:
                self.fail(c.fail())

        for c in utils.ConfigIter(basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)),
                                  requires_grad=True):
            try:
                c.signature()
                c.signature_backward()
            except Exception:
                self.fail(c.fail())

    def test_too_small_axes(self):
        for c in utils.ConfigIter(basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 1, 4), (4, 0, 4), (4, 4, 0), (1, 1, 1)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.signature()

        for c in utils.ConfigIter(basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 0, 4), (4, 4, 0)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.signature()

    @staticmethod
    def correct_shape(size, depth, stream, basepoint):
        N, L, C = size
        if stream:
            if isinstance(basepoint, torch.Tensor) or basepoint:
                return N, L, signatory.signature_channels(C, depth)
            else:
                return N, L - 1, signatory.signature_channels(C, depth)
        else:
            return N, signatory.signature_channels(C, depth)

    # Hybrid test for speed
    def test_arguments_and_shape(self):
        for c in utils.ConfigIter(N=(1, 2),   # reduced space to test because this test is too slow otherwise
                                  L=(1, 2, 3),
                                  depth=(1, 2, 3)):
            if not c.has_basepoint() and c.L == 1:
                with self.assertRaises(ValueError):
                    c.signature()
            else:
                try:
                    signatory_out = c.signature()
                except Exception:
                    self.fail(c.fail())
                correct_shape = self.correct_shape(c.size, c.depth, c.stream, c.basepoint)
                self.assertEqual(signatory_out.shape, correct_shape, c.fail())


class TestLogSignatureArguments(utils.EnhancedTestCase):
    def test_minimal_axes(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 2, 4), (4, 4, 1), (1, 2, 1)),
                                  requires_grad=True):
            try:
                c.logsignature()
                c.logsignature_backward()
            except Exception:
                self.fail(c.fail())

        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)),
                                  requires_grad=True):
            try:
                c.logsignature()
                c.logsignature_backward()
            except Exception:
                self.fail(c.fail())

    def test_too_small_axes(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 1, 4), (4, 0, 4), (4, 4, 0), (1, 1, 1)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.logsignature()

        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 0, 4), (4, 4, 0)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.logsignature()

    @staticmethod
    def correct_shape(size, depth, stream, basepoint, mode):
        N, L, C = size
        if mode == "expand":
            channel_fn = signatory.signature_channels
        else:
            channel_fn = signatory.logsignature_channels
        if stream:
            if isinstance(basepoint, torch.Tensor) or basepoint:
                return N, L, channel_fn(C, depth)
            else:
                return N, L - 1, channel_fn(C, depth)
        else:
            return N, channel_fn(C, depth)

    # Hybrid test for speed
    def test_arguments_and_shape(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  N=(1, 2),   # reduced space to test because this test is too slow otherwise
                                  L=(1, 2, 3),
                                  depth=(1, 2, 3)):
            if not c.has_basepoint() and c.L == 1:
                with self.assertRaises(ValueError):
                    c.logsignature()
            else:
                try:
                    signatory_out = c.logsignature()
                except Exception:
                    self.fail(c.fail())
                correct_shape = self.correct_shape(c.size, c.depth, c.stream, c.basepoint, c.signatory_mode)
                self.assertEqual(signatory_out.shape, correct_shape, c.fail())


class TestSignatureCombine(utils.EnhancedTestCase):
    def test_signature_combine(self):
        for c in utils.ConfigIter(requires_grad=True, stream=False):
            path2 = torch.rand_like(c.path, requires_grad=True)
            sig = c.signature()
            sig2 = c.signature(store=False, path=path2, basepoint=c.path[:, -1, :])
            sig_combined = signatory.signature_combine(sig, sig2, c.C, c.depth, inverse=c.inverse)

            path_combined = torch.cat([c.path, path2], dim=1)
            sig_combined2 = c.signature(store=False, path=path_combined)
            if not sig_combined.allclose(sig_combined2):
                self.fail(c.diff_fail(sig_combined=sig_combined, sig_combined2=sig_combined2))

            grad = torch.rand_like(sig_combined)
            sig_combined.backward(grad)
            path_grad = c.path.grad.clone()
            path_grad2 = path2.grad.clone()
            c.path.grad.zero_()
            path2.grad.zero_()
            sig_combined2.backward(grad)
            if not path_grad.allclose(c.path.grad):
                self.fail(c.diff_fail(path_grad=path_grad, true_path_grad=c.path.grad))
            if not path_grad2.allclose(path2.grad):
                self.fail(c.diff_fail(path_grad2=path_grad, true_path2_grad=path2.grad))

    def test_multi_signature_combine(self):
        for c in utils.ConfigIter(requires_grad=True, stream=False):
            for amount in (1, 2, 3, 10):
                prev_path = c.path
                paths = [prev_path]
                sigs = [c.signature()]
                for _ in range(amount):
                    next_path = torch.rand_like(c.path, requires_grad=True)
                    paths.append(next_path)
                    sigs.append(c.signature(store=False, path=next_path, basepoint=prev_path[:, -1, :]))
                    prev_path = next_path

                sig_combined = signatory.signature_combine(sigs, c.C, c.depth, inverse=c.inverse)

                path_combined = torch.cat(paths, dim=1)
                true_sig_combined = c.signature(store=False, path=path_combined)
                if not sig_combined.allclose(true_sig_combined):
                    self.fail(c.diff_fail(sig_combined=sig_combined, true_sig_combined=true_sig_combined))

                grad = torch.rand_like(sig_combined)
                sig_combined.backward(grad)
                path_grads = []
                for path in paths:
                    path_grads.append(path.grad.clone())
                    path.grad.zero_()
                true_sig_combined.backward(grad)
                for i, (path, path_grad) in enumerate(zip(paths, path_grads)):
                    if not path.grad.allclose(path_grad):
                        self.fail(c.diff_fail(path_grad=path_grad, true_path_grad=path.grad, extras={'i': i}))
