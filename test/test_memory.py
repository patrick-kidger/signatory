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


class TestSignatureMemory(utils.TimedTestCase):
    def test_no_adjustments(self):
        for c in utils.ConfigIter(requires_grad=True):
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
            with self.assertRaises(RuntimeError):
                c.signature_backward()

    def test_ctx_dies(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.signature(store=False)
            ctx = signatory_out.grad_fn
            ref = weakref.ref(ctx)
            del signatory_out
            gc.collect()
            if ref() is not None:
                self.fail(c.fail())

    def test_no_leaks(self):
        torch.cuda.reset_max_memory_allocated()
        # It's just easier to keep track of GPU memory)
        iterator = utils.ConfigIter(requires_grad=True, size=utils.random_size(5), device='cuda')
        c = next(iterator)
        signatory_out = c.signature()
        back = c.signature_backward()
        memory_used = torch.cuda.max_memory_allocated()
        del c
        del signatory_out
        del back
        gc.collect()
        for c in iterator:
            signatory_out = c.signature()
            back = c.signature_backward()
            memory_allocated = torch.cuda.max_memory_allocated()
            if memory_allocated > memory_used:
                self.fail(c.fail(memory_used=memory_used, memory_allocated=memory_allocated))
            # Can't wait for them to be 'overwritten' on the next loop if we want to call gc.collect()
            del c
            del signatory_out
            del back
            gc.collect()


class TestLogSignatureMemory(utils.TimedTestCase):
    def test_no_adjustments(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
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
            with self.assertRaises(RuntimeError):
                c.logsignature_backward()

    def test_ctx_dies(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            signatory_out = c.logsignature(store=False)
            ctx = signatory_out.grad_fn
            ref = weakref.ref(ctx)
            del signatory_out
            gc.collect()
            if ref() is not None:
                self.fail(c.fail())

    def test_no_leaks(self):
        for mode in utils.all_modes:
            torch.cuda.reset_max_memory_allocated()
            # It's just easier to keep track of GPU memory)
            iterator = utils.ConfigIter(requires_grad=True, size=utils.random_size(5), device='cuda', mode=mode)
            c = next(iterator)
            signatory_out = c.logsignature()
            back = c.logsignature_backward()
            memory_used = torch.cuda.max_memory_allocated()
            del c
            del signatory_out
            del back
            gc.collect()
            for c in iterator:
                signatory_out = c.signature()
                back = c.signature_backward()
                memory_allocated = torch.cuda.max_memory_allocated()
                if memory_allocated > memory_used:
                    self.fail(c.fail(memory_used=memory_used, memory_allocated=memory_allocated))
                # Can't wait for them to be 'overwritten' on the next loop if we want to call gc.collect()
                del c
                del signatory_out
                del back
                gc.collect()
