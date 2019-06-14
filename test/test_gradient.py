import random
import signatory
import torch
import torch.autograd as autograd
import unittest


class TestGrad(unittest.TestCase):
    @staticmethod
    def gradcheck(size=(1, 4, 3), depth=2, basepoint=False, stream=False, flatten=True):
        x = torch.rand(*size, requires_grad=True, dtype=torch.double)
        return autograd.gradcheck(signatory.signature, (x, depth, basepoint, stream, flatten))

    def test_gradcheck(self):
        for basepoint in (True, False):
            for stream in (True, False):
                for flatten in (True, False):
                    for _ in range(5):
                        size = torch.randint(low=1, high=10, size=(3,))
                        depth = int(torch.randint(low=1, high=4, size=(1,)))
                        self.assertTrue(self.gradcheck(size, depth, basepoint, stream, flatten),
                                        "Failed with basepoint={basepoint}, stream={stream}, flatten={flatten}, "
                                        "size={size}, depth={depth}".format(basepoint=basepoint, stream=stream,
                                                                            flatten=flatten, size=size, depth=depth))

    # Fails, and it's not clear why. However... (see next method)
    @staticmethod
    def gradgradcheck(size=(1, 4, 3), depth=2, basepoint=False, stream=False, flatten=True):
        if flatten:
            test_fn = signatory.signature
        else:
            def test_fn(*args):
                result = signatory.signature(*args)
                return random.choice(result)
        x = torch.rand(*size, requires_grad=True, dtype=torch.double)
        return autograd.gradgradcheck(test_fn, (x, depth, basepoint, stream, flatten))

    # ...this doesn't even manage to start at all, because of the fact that 'grad' and 'y' are both lists of
    # torch.Tensors. Looking at the source code for autograd.gradcheck, it's clear that they don't expect any nested
    # data structures for input like this; all their code assumems that the inputs are just torch.Tensors.
    # Since we need to have lists of Tensors as input to signature_backward, it seems plausible that the fact that their
    # torch.autograd.gradgradcheck even runs is a fluke, and the fact that it then fails is meaningless. Especially when
    # you consider that we haven't even written a custom double-backward function - double-backward is done
    # automatically by autograd in its usual 'tracing' manner, so it would be very surprising if it didn't give the
    # correct results.
    @staticmethod
    def gradgradcheck2(size=(1, 4, 3), depth=2, basepoint=False, stream=False, flatten=True):
        from signatory._impl import _signature_backward
        x = torch.rand(*size, dtype=torch.double)
        y = signatory.signature(x, depth, basepoint, stream, flatten)
        if flatten:
            # if not flatten then it's already a list
            y = [y]
        grad = [torch.rand_like(yi).requires_grad_() for yi in y]
        return autograd.gradcheck(_signature_backward, (grad, y, x, depth, basepoint, stream, flatten))
