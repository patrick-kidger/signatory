import iisignature
import signatory
import torch
import unittest


class TestAccuracy(unittest.TestCase):
    # def test_single_axis(self):
    #     for stream in (True, False):
    #         for basepoint in (True, False):
    #             for size in ((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)):
    #                 out = signatory.signature(torch.rand(size, requires_grad=True), 2, stream, basepoint)

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
