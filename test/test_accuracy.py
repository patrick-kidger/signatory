import iisignature
import signatory
import torch
import unittest


class TestAccuracy(unittest.TestCase):
    def test_forward(self):
        for stream in (True, False):
            for N in (1, 2, 3, 4):
                for L in (2, 3, 4, 5):
                    for C in (1, 2, 3, 4):
                        size = (N, L, C)
                        path = torch.rand(size, requires_grad=True)
                        depth = int(torch.randint(low=1, high=4, size=(1,)))
                        out = signatory.signature(path, depth, stream)
                        iiout = iisignature.sig(path.detach(), depth, 2 if stream else 0)
                        self.assertTrue(out.allclose(torch.tensor(iiout)))

    def test_backward(self):
        for N in (1, 2, 3, 4):
            for L in (2, 3, 4, 5):
                for C in (1, 2, 3, 4):
                    size = (N, L, C)
                    path = torch.rand(size, requires_grad=True)
                    depth = int(torch.randint(low=1, high=4, size=(1,)))
                    out = signatory.signature(path, depth, stream=False)
                    grad = torch.rand_like(out)
                    out.backward(grad)
                    iisig_backward = iisignature.sigbackprop(grad, path.detach(), depth)
                    self.assertTrue(path.grad.allclose(torch.tensor(iisig_backward)))
