import torch
import unittest

import utils


class TestSignatureMemory(unittest.TestCase):
    def test_memory(self):
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


class TestLogSignatureMemory(unittest.TestCase):
    def test_memory(self):
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
