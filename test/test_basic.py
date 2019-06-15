import signatory
import torch
import unittest


class TestArguments(unittest.TestCase):
    def test_single_axis(self):
        for basepoint in (True, False):
            for stream in (True, False):
                for flatten in (True, False):
                    for size in ((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)):
                        out = signatory.signature(torch.rand(size, requires_grad=True), 2, basepoint, stream, flatten)
                        if flatten:
                            out = (out,)
                        for o in out:
                            o.backward(torch.rand_like(o))

    def test_arguments(self):
        for basepoint in (True, False):
            for stream in (True, False):
                for flatten in (True, False):
                    for N in (1, 2, 3, 4):
                        for C in (1, 2, 3, 4):
                            for L in (1, 2, 3, 4):
                                size = (N, C, L)
                                path = torch.rand(size, requires_grad=True)
                                depth = int(torch.randint(low=1, high=4, size=(1,)))
                                try:
                                    out = signatory.signature(path, depth, basepoint, stream, flatten)
                                    if flatten:
                                        out = (out,)
                                    for o in out:
                                        o.backward(torch.rand_like(o))
                                except Exception:
                                    self.fail("basepoint={basepoint}, stream={stream}, flatten={flatten}, size={size}, "
                                              "depth={depth}".format(basepoint=basepoint, stream=stream,
                                                                     flatten=flatten, size=size, depth=depth))


class TestShapes(unittest.TestCase):
    def correct_shape(self, size, depth, basepoint, stream, flatten):
        N, C, L = size
        if flatten:
            if stream:
                if basepoint:
                    return N, signatory.signature_channels(C, depth), L
                else:
                    return N, signatory.signature_channels(C, depth), L - 1
            else:
                return N, signatory.signature_channels(C, depth)
        else:
            if stream:
                if basepoint:
                    return tuple((N, C ** i, L) for i in range(1, depth + 1))
                else:
                    return tuple((N, C ** i, L - 1) for i in range(1, depth + 1))
            else:
                return tuple((N, C ** i) for i in range(1, depth + 1))

    def test_shapes(self):
        for basepoint in (True, False):
            for stream in (True, False):
                for flatten in (True, False):
                    for N in (1, 2, 3, 4):
                        for C in (1, 2, 3, 4):
                            for L in (1, 2, 3, 4):
                                size = (N, C, L)
                                path = torch.rand(size)
                                depth = int(torch.randint(low=1, high=4, size=(1,)))
                                out = signatory.signature(path, depth, basepoint, stream, flatten)
                                if flatten:
                                    got_shape = out.shape
                                else:
                                    got_shape = tuple(o.shape for o in out)
                                correct_shape = self.correct_shape(size, depth, basepoint, stream, flatten)
                                self.assertEqual(got_shape, correct_shape, "basepoint={basepoint}, stream={stream}, "
                                                                           "flatten={flatten}, N={N}, C={C}, L={L}, "
                                                                           "depth={depth}"
                                                                           .format(basepoint=basepoint, stream=stream,
                                                                                   flatten=flatten, N=N, C=C, L=L,
                                                                                   depth=depth))
