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
"""The implementation of our signature and logsignature functions uses as little memory as possible, often by
performing operations inplace. It is thus plausible that we accidentally modify something we shouldn't modify;
checking that that doesn't occur is the purpose of these tests."""


import gc
import torch
import weakref

import utils_testing as utils


class TestSignatureMemory(utils.EnhancedTestCase):
    def test_no_adjustments(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.random_size(5)):
            path_copy = c.path.clone()
            signatory_out = c.signature()
            self.assertTrue(c.path.allclose(path_copy))
            signatory_out_copy = signatory_out.clone()
            grad = torch.rand_like(signatory_out)
            grad_copy = grad.clone()
            c.signature_backward(grad)
            self.assertTrue(c.path.allclose(path_copy))
            self.assertTrue(c.signatory_out.allclose(signatory_out_copy))
            self.assertTrue(c.grad.allclose(grad_copy))

    def test_inplace_caught(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.signature()
            signatory_out += 1
            with self.assertRaises(RuntimeError, msg=c.fail()):
                c.signature_backward()

    def test_ctx_dies(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.signature(store=False)
            ctx = signatory_out.grad_fn
            if c.stream:
                ctx = ctx.next_functions[0][0]
            ref = weakref.ref(ctx)
            del ctx
            del signatory_out
            gc.collect()
            self.assertIsNone(ref(), c.fail())

    def test_no_leaks(self):
        # device='cuda' because it's just easier to keep track of GPU memory
        for c in utils.ConfigIter(requires_grad=True, size=utils.random_size(5), device='cuda', repeats=10):
            if c.rep == 0:
                torch.cuda.reset_max_memory_allocated()
            signatory_out = c.signature()
            back = c.signature_backward()
            memory_used = torch.cuda.max_memory_allocated()
            if c.rep == 0:
                memory_used_first = memory_used
            else:
                if memory_used > memory_used_first:
                    self.fail(c.fail(memory_used=memory_used, memory_used_first=memory_used_first))
            del c
            del signatory_out
            del back
            gc.collect()


class TestLogSignatureMemory(utils.EnhancedTestCase):
    def test_no_adjustments(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  size=utils.random_size(5),
                                  requires_grad=True):
            path_copy = c.path.clone()
            signatory_out = c.logsignature()
            self.assertTrue(c.path.allclose(path_copy))
            signatory_out_copy = signatory_out.clone()
            grad = torch.rand_like(signatory_out)
            grad_copy = grad.clone()
            c.logsignature_backward(grad)
            self.assertTrue(c.path.allclose(path_copy))
            self.assertTrue(c.signatory_out.allclose(signatory_out_copy))
            self.assertTrue(c.grad.allclose(grad_copy))

    def test_inplace_caught(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.logsignature()
            signatory_out += 1
            with self.assertRaises(RuntimeError, msg=c.fail()):
                c.logsignature_backward()

    def test_ctx_dies(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.logsignature(store=False)
            ctx = signatory_out.grad_fn
            if c.stream:
                ctx = ctx.next_functions[0][0]
            ref = weakref.ref(ctx)
            del ctx
            del signatory_out
            gc.collect()
            self.assertIsNone(ref(), c.fail())

    def test_no_leaks(self):
        # device='cuda' because it's just easier to keep track of GPU memory
        for c in utils.ConfigIter(requires_grad=True, size=utils.random_size(5), device='cuda', repeats=10,
                                  mode=utils.all_modes):
            if c.rep == 0:
                torch.cuda.reset_max_memory_allocated()
            signatory_out = c.logsignature()
            back = c.logsignature_backward()
            memory_used = torch.cuda.max_memory_allocated()
            if c.rep == 0:
                memory_used_first = memory_used
            else:
                if memory_used > memory_used_first:
                    self.fail(c.fail(memory_used=memory_used, memory_used_first=memory_used_first))
            del c
            del signatory_out
            del back
            gc.collect()
