import signatory
import torch
import torch.autograd as autograd


def gradcheck(depth=2, size=(1, 4, 3), basepoint=False, stream=False, flatten=True):
    x = torch.rand(*size, requires_grad=True, dtype=torch.double)
    # TODO: remove sig.sig
    return autograd.gradcheck(signatory.sig.apply, (x, depth, basepoint, stream, flatten))


# Fails, and it's not clear why. However...
def gradgradcheck(depth=2, size=(1, 4, 3), basepoint=False, stream=False, flatten=True):
    x = torch.rand(*size, requires_grad=True, dtype=torch.double)
    # TODO: remove sig.sig
    return autograd.gradgradcheck(signatory.sig.apply, (x, depth, basepoint, stream, flatten))


# ...this doesn't even manage to start at all, because of the fact that 'grad' and 'y' are both lists of torch.Tensors.
# Looking at the source code for autograd.gradcheck, it's clear that they don't expect any nested data structures for
# input like this; all their code assumems that the inputs are just torch.Tensors.
# Since we need to have lists of Tensors as input to signature_backward, it seems plausible that the fact that their
# torch.autograd.gradgradcheck even runs is a fluke, and the fact that it then fails is meaningless. Especially when
# you consider that we haven't even written a custom double-backward function - double-backward is done automatically by
# autograd in its usual 'tracing' manner, so it would be very surprising if it didn't give the correct results.
def gradgradcheck2(depth=2, size=(1, 4, 3), basepoint=False, stream=False, flatten=True):
    from signatory._impl import signature_backward
    x = torch.rand(*size, dtype=torch.double)
    y = signatory.sig.apply(x, depth, basepoint, stream, flatten)
    if flatten:
        # if not flatten then it's already a list
        y = [y]
    grad = [torch.rand_like(yi).requires_grad_() for yi in y]
    return autograd.gradcheck(signature_backward, (grad, y, x, depth, basepoint, stream, flatten))
