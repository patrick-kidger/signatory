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
                        for basepoint in (True, False, torch.rand(N, C, dtype=torch.float64)):
                            size = (N, L, C)
                            path = torch.rand(size, requires_grad=True, dtype=torch.float64)
                            depth = int(torch.randint(low=1, high=4, size=(1,)))
                            out = signatory.signature(path, depth, stream=stream, basepoint=basepoint)
                            if basepoint is True:
                                ii_in = torch.cat([torch.zeros(N, 1, C, dtype=torch.float64), path.detach()], dim=1)
                            elif basepoint is False:
                                ii_in = path.detach()
                            else:
                                ii_in = torch.cat([basepoint.unsqueeze(1), path.detach()], dim=1)
                            iiout = iisignature.sig(ii_in, depth, 2 if stream else 0)
                            close = out.allclose(torch.tensor(iiout, dtype=torch.float64))
                            if not close:
                                self.fail("out={out}, iiout={iiout}".format(out=out, iiout=iiout))

    def test_backward(self):
        for N in (1, 2, 3, 4):
            for L in (2, 3, 4, 5):
                for C in (1, 2, 3, 4):
                    for basepoint in (True,
                                      False,
                                      torch.rand(N, C, requires_grad=True, dtype=torch.float64),
                                      torch.rand(N, C, requires_grad=False, dtype=torch.float64)):
                        size = (N, L, C)
                        path = torch.rand(size, requires_grad=True, dtype=torch.float64)
                        depth = int(torch.randint(low=1, high=4, size=(1,)))
                        out = signatory.signature(path, depth, stream=False, basepoint=basepoint)
                        grad = torch.rand_like(out)
                        out.backward(grad)
                        gradresult = path.grad
                        if basepoint is True:
                            ii_in = torch.cat([torch.zeros(N, 1, C, dtype=torch.float64), path], dim=1)
                        elif basepoint is False:
                            ii_in = path
                        else:  # implies isinstance(basepoint, torch.Tensor) == True
                            ii_in = torch.cat([basepoint.unsqueeze(1), path], dim=1)
                            if basepoint.requires_grad:
                                # get the gradient on the basepoint as well
                                gradresult = torch.cat([basepoint.grad.unsqueeze(1), gradresult], dim=1)
                        iisig_backward = iisignature.sigbackprop(grad, ii_in.detach(), depth)
                        if basepoint is True or (isinstance(basepoint, torch.Tensor) and not basepoint.requires_grad):
                            # if we don't have a gradient through the basepoint then discard the corresponding basepoint
                            # part of the iisig_backward result
                            iisig_backward = iisig_backward[:, 1:, :]
                        # strangely iisignature returns float32 in the backward calculation, even if the input was
                        # float64, so we have to reduce the tolerance slightly
                        close = gradresult.allclose(torch.tensor(iisig_backward, dtype=torch.float64), atol=1e-6)
                        if not close:
                            self.fail("gradresult={gradresult}, iisig_backward={iisig_backward}, basepoint={basepoint} "
                                      .format(gradresult=gradresult, iisig_backward=iisig_backward,
                                              basepoint=basepoint))
