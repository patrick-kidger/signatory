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
        for stream in (True, False):
            for size in ((1, 2, 1), (1, 4, 4), (4, 2, 4), (4, 4, 1)):
                basepoint_size = (size[0], size[2])
                for basepoint in (True,
                                  False,
                                  torch.rand(basepoint_size, requires_grad=True, dtype=torch.double),
                                  torch.rand(basepoint_size, requires_grad=False, dtype=torch.double)):
                    for depth in (1, 2):
                        try:
                            self.gradcheck(size, depth, stream, basepoint)
                        except RuntimeError:
                            self.fail("Failed with stream={stream}, basepoint={basepoint}, size={size}, depth={depth}, "
                                      "basepoint_size={basepoint_size}"
                                      .format(stream=stream, basepoint=basepoint, size=size, depth=depth,
                                              basepoint_size=basepoint_size))

    def test_gradcheck_random(self):
        for stream in (True, False):
            for _ in range(5):
                for basepoint_ in (True, False, None):
                    size = torch.randint(low=1, high=10, size=(3,))
                    size[1] += 1  # stream dimension must be at least size 2
                    basepoint_size = (size[0], size[2])
                    if basepoint_ is None:
                        basepoint_grad = random.choice([True, False])
                        basepoint = torch.rand(basepoint_size, requires_grad=basepoint_grad, dtype=torch.double)
                    else:
                        basepoint_grad = None
                        basepoint = basepoint_
                    depth = int(torch.randint(low=1, high=4, size=(1,)))
                    try:
                        self.gradcheck(size, depth, stream, basepoint)
                    except Exception:
                        self.fail("Failed with stream={stream}, basepoint={basepoint}, size={size}, depth={depth}, "
                                  "basepoint_size={basepoint_size}, basepoint_grad={basepoint_grad}"
                                  .format(stream=stream, basepoint=basepoint, size=size, depth=depth,
                                          basepoint_size=basepoint_size, basepoint_grad=basepoint_grad))

    # We don't do gradgradcheck because our backwards function uses a whole bunch of in-place operations for memory
    # efficiency, so it's not automatically differentiable. (And I'm not writing a custom double backward function...)
