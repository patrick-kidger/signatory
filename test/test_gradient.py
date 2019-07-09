import random
import signatory
import torch
import torch.autograd as autograd
import unittest


class TestGrad(unittest.TestCase):
    @staticmethod
    def gradcheck(size=(1, 4, 3), depth=2, stream=False, basepoint=False):
        path = torch.rand(*size, requires_grad=True, dtype=torch.double)
        return autograd.gradcheck(signatory.signature, (path, depth, stream, basepoint))

    def test_gradcheck_edge(self):
        sentinel = object()
        for stream in (True, False):
            for basepoint in (True, False, None, sentinel):
                for depth in (1, 2):
                    for size in ((1, 2, 1), (1, 4, 4), (4, 2, 4), (4, 4, 1)):
                        if basepoint is None:
                            basepoint = torch.rand((size[0], size[2]), requires_grad=True)
                        if basepoint is sentinel:
                            basepoint = torch.rand((size[0], size[2]), requires_grad=False)
                        try:
                            self.gradcheck(size, depth, stream, basepoint)
                        except RuntimeError:
                            self.fail("Failed with stream={stream}, basepoint={basepoint}, size={size}, depth={depth}"
                                      .format(stream=stream, basepoint=basepoint, size=size, depth=depth))

    def test_gradcheck_random(self):
        for stream in (True, False):
            for basepoint in (True, False, None):
                for _ in range(5):
                    size = torch.randint(low=1, high=10, size=(3,))
                    if basepoint is None:
                        basepoint_grad = random.choice([True, False])
                        basepoint = torch.rand((size[0], size[2]), requires_grad=basepoint_grad)
                    else:
                        basepoint_grad = None
                    depth = int(torch.randint(low=1, high=4, size=(1,)))
                    try:
                        self.gradcheck(size, depth, stream, basepoint)
                    except RuntimeError:
                        self.fail("Failed with stream={stream}, basepoint={basepoint}, size={size}, depth={depth}, "
                                  "basepoint_grad={basepoint_grad}"
                                  .format(stream=stream, basepoint=basepoint, size=size, depth=depth,
                                          basepoint_grad=basepoint_grad))

    # We don't do gradgradcheck because our backwards function uses a whole bunch of in-place operations for memory
    # efficiency, so it's not automatically differentiable. (And I'm not writing a custom double backward function...)
