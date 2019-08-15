import signatory
import torch
import torch.autograd as autograd
import unittest

import utils


def random_size():
    for _ in range(5):
        size0 = int(torch.randint(low=1, high=10, size=(1,)))
        size1 = int(torch.randint(low=2, high=10, size=(1,)))
        size2 = int(torch.randint(low=1, high=10, size=(1,)))
        yield size0, size1, size2


class TestSignatureGrad(unittest.TestCase):
    @staticmethod
    def gradcheck(path, depth, stream, basepoint, **kwargs):
        return autograd.gradcheck(signatory.signature, (path, depth, stream, basepoint), **kwargs)

    def test_gradcheck_edge(self):
        for c in utils.ConfigIter(depth=(1, 2, 3),
                                  requires_grad=True,
                                  size=((1, 2, 1), (1, 4, 4), (4, 2, 4), (4, 4, 1))):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint)
            except RuntimeError:
                self.fail(c.fail())

    @unittest.skip  # takes forever
    def test_gradcheck_grid(self):
        for c in utils.ConfigIter(requires_grad=True):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint)
            except RuntimeError:
                self.fail(c.fail())

    def test_gradcheck_random(self):
        for c in utils.ConfigIter(requires_grad=True,
                                  size=random_size()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint)
            except RuntimeError:
                self.fail(c.fail())

    # We don't do gradgradcheck because our backwards function uses a whole bunch of in-place operations for memory
    # efficiency, so it's not automatically differentiable. (And I'm not writing a custom double backward function...)


class TestLogSignatureGrad(unittest.TestCase):
    @staticmethod
    def gradcheck(path, depth, stream, basepoint, mode, **kwargs):
        return autograd.gradcheck(signatory.logsignature, (path, depth, stream, basepoint, mode), **kwargs)

    def test_gradcheck_edge(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  depth=(1, 2, 3),
                                  requires_grad=True,
                                  size=((1, 2, 1), (1, 4, 4), (4, 2, 4), (4, 4, 1))):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())

    @unittest.skip  # takes forever
    def test_gradcheck_grid(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())

    def test_gradcheck_random(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True,
                                  size=random_size()):
            try:
                self.gradcheck(c.path, c.depth, c.stream, c.basepoint, c.signatory_mode)
            except RuntimeError:
                self.fail(c.fail())
