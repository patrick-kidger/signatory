import signatory
import iisignature
import timeit
import torch
import unittest
import warnings


class TestSpeed(unittest.TestCase):
    @staticmethod
    def speed(batch_size=1000, number=10, depth=3, stream=True, backward=False):
        if stream and backward:
            raise ValueError("iisignature does not support backpropagation for a full stream of data, so we can't "
                             "compare against them.")
        signatory_x = torch.rand(batch_size, 10, 100, requires_grad=backward)
        iisignature_x = torch.rand(batch_size, 100, 10).numpy()

        def signatory_fn():
            y = signatory.signature(signatory_x, depth, stream=stream)
            if backward:
                y.backward(torch.rand_like(y))

        def iisignature_fn():
            for iisignature_ix in iisignature_x:
                y = iisignature.sig(iisignature_ix, depth, 2 if stream else 0)
                if backward:
                    iisignature.sigbackprop(torch.rand_like(torch.tensor(y)), iisignature_ix, depth)

        signatory_time = timeit.timeit(signatory_fn, number=number)
        iisignature_time = timeit.timeit(iisignature_fn, number=number)
        return signatory_time, iisignature_time, signatory_time / iisignature_time

    def wrapped_speed_test(self, **kwargs):
        signatory_time, iisignature_time, ratio = self.speed(**kwargs)
        self.assertLess(ratio, 1.0)
        if ratio > 0.5:
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio}"
                          .format(stime=signatory_time, itime=iisignature_time, ratio=ratio))

    def test_speed_forward_nostream(self):
        self.wrapped_speed_test(number=20, stream=False, backward=False)

    def test_speed_forward_stream(self):
        self.wrapped_speed_test(number=20, stream=True, backward=False)

    def test_speed_backward(self):
        self.wrapped_speed_test(number=20, stream=False, backward=True)
