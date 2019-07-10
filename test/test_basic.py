import signatory
import torch
import unittest


class TestArguments(unittest.TestCase):
    def test_minimal_axes(self):
        depth = 2
        for stream in (True, False):
            for size in ((1, 4, 4), (4, 2, 4), (4, 4, 1), (1, 2, 1)):
                basepoint_size = (size[0], size[2])
                for basepoint in (True,
                                  False,
                                  torch.rand(basepoint_size, requires_grad=True),
                                  torch.rand(basepoint_size, requires_grad=False)):
                    try:
                        out = signatory.signature(torch.rand(size, requires_grad=True), depth, stream, basepoint)
                        out.backward(torch.rand_like(out))
                    except Exception:
                        self.fail("stream={stream}, basepoint={basepoint}, size={size}, depth={depth}, "
                                  "basepoint_size={basepoint_size}"
                                  .format(stream=stream, basepoint=basepoint, size=size, depth=depth,
                                          basepoint_size=basepoint_size))

    def test_too_small_axes(self):
        depth = 2
        for stream in (True, False):
            for size in ((0, 4, 4), (4, 1, 4), (4, 0, 4), (4, 4, 0), (1, 1, 1)):
                basepoint_size = (size[0], size[2])
                for basepoint in (True,
                                  False,
                                  torch.rand(basepoint_size, requires_grad=True),
                                  torch.rand(basepoint_size, requires_grad=False)):
                    with self.assertRaises(ValueError):
                        signatory.signature(torch.rand(size, requires_grad=True), depth, stream, basepoint)

    def test_arguments(self):
        for stream in (True, False):
            for N in (1, 2, 3, 4):
                for L in (1, 2, 3, 4):
                    for C in (1, 2, 3, 4):
                        size = (N, L, C)
                        basepoint_size = (N, C)
                        for basepoint in (True,
                                          False,
                                          torch.rand(basepoint_size, requires_grad=True),
                                          torch.rand(basepoint_size, requires_grad=False)):
                            path = torch.rand(size, requires_grad=True)
                            depth = int(torch.randint(low=1, high=4, size=(1,)))
                            try:
                                if L == 1:
                                    with self.assertRaises(ValueError):
                                        signatory.signature(path, depth, stream, basepoint)
                                else:
                                    out = signatory.signature(path, depth, stream, basepoint)
                                    out.backward(torch.rand_like(out))
                            except Exception:
                                self.fail("stream={stream}, basepoint={basepoint}, size={size}, depth={depth}, "
                                          "basepoint_size={basepoint_size}"
                                          .format(stream=stream, basepoint=basepoint, size=size, depth=depth,
                                                  basepoint_size=basepoint_size))


class TestShapes(unittest.TestCase):
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

    def test_shapes(self):
        for stream in (True, False):
            for N in (1, 2, 3, 4):
                for L in (2, 3, 4, 5):
                    for C in (1, 2, 3, 4):
                        basepoint_size = (N, C)
                        for basepoint in (True,
                                          False,
                                          torch.rand(basepoint_size, requires_grad=True),
                                          torch.rand(basepoint_size, requires_grad=False)):
                            size = (N, L, C)
                            path = torch.rand(size)
                            depth = int(torch.randint(low=1, high=4, size=(1,)))
                            out = signatory.signature(path, depth, stream, basepoint)
                            correct_shape = self.correct_shape(size, depth, stream, basepoint)
                            self.assertEqual(out.shape, correct_shape, "stream={stream}, basepoint={basepoint}, "
                                                                       "N={N}, L={L}, C={C}, depth={depth},"
                                                                       "basepoint_size={basepoint_size}"
                                                                       .format(stream=stream, basepoint=basepoint,
                                                                               N=N, L=L, C=C, depth=depth,
                                                                               basepoint_size=basepoint_size))
