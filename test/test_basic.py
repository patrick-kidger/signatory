import signatory
import torch
import unittest


class TestArguments(unittest.TestCase):
    def test_single_axis(self):
        for stream in (True, False):
            for basepoint in (True, False):
                for size in ((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)):
                    out = signatory.signature(torch.rand(size, requires_grad=True), 2, stream, basepoint)
                    out.backward(torch.rand_like(out))

    def test_arguments(self):
        for stream in (True, False):
            for basepoint in (True, False):
                for N in (1, 2, 3, 4):
                    for C in (1, 2, 3, 4):
                        for L in (1, 2, 3, 4):
                            size = (N, C, L)
                            path = torch.rand(size, requires_grad=True)
                            depth = int(torch.randint(low=1, high=4, size=(1,)))
                            try:
                                out = signatory.signature(path, depth, stream, basepoint)
                                out.backward(torch.rand_like(out))
                            except Exception:
                                self.fail("stream={stream}, basepoint={basepoint}, size={size}, depth={depth}"
                                          .format(stream=stream, basepoint=basepoint, size=size, depth=depth))


class TestShapes(unittest.TestCase):
    @staticmethod
    def correct_shape(size, depth, stream, basepoint):
        N, C, L = size
        if stream:
            if basepoint:
                return N, signatory.signature_channels(C, depth), L
            else:
                return N, signatory.signature_channels(C, depth), L - 1
        else:
            return N, signatory.signature_channels(C, depth)

    def test_shapes(self):
        for stream in (True, False):
            for basepoint in (True, False):
                for N in (1, 2, 3, 4):
                    for C in (1, 2, 3, 4):
                        for L in (1, 2, 3, 4):
                            size = (N, C, L)
                            path = torch.rand(size)
                            depth = int(torch.randint(low=1, high=4, size=(1,)))
                            out = signatory.signature(path, depth, stream, basepoint)
                            correct_shape = self.correct_shape(size, depth, stream, basepoint)
                            self.assertEqual(out.shape, correct_shape, "stream={stream}, basepoint={basepoint}, "
                                                                       "N={N}, C={C}, L={L}, depth={depth}"
                                                                       .format(stream=stream, basepoint=basepoint,
                                                                               N=N, C=C, L=L, depth=depth))
