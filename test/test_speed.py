import signatory
import iisignature
import timeit
import torch
import unittest
import warnings


class TestSpeed(unittest.TestCase):
    @staticmethod
    def speed(batch_size=1000, number=20, depth=3, stream=True, backward=False):
        # We speed test by testing against another library
        # Which is admittedly a bit fragile - what if they update their library to operate faster?
        # But just testing raw speeds is going to be very hardware-dependent, so this is good enough

        if stream and backward:
            raise ValueError("iisignature does not support backpropagation for a full stream of data, so we can't "
                             "compare against them.")
        signatory_x = torch.rand(batch_size, 100, 10, requires_grad=backward)
        iisignature_x = signatory_x.detach().numpy()

        def signatory_fn():
            y = signatory.signature(signatory_x, depth, stream=stream)
            if backward:
                y.backward(torch.rand_like(y))

        def iisignature_fn():
            y = iisignature.sig(iisignature_x, depth, 2 if stream else 0)
            if backward:
                iisignature.sigbackprop(torch.rand_like(torch.tensor(y)), iisignature_x, depth)

        signatory_time = timeit.timeit(signatory_fn, number=number)
        iisignature_time = timeit.timeit(iisignature_fn, number=number)
        return signatory_time, iisignature_time, signatory_time / iisignature_time

    def wrapped_speed_test(self, stream, backward):
        signatory_time, iisignature_time, ratio = self.speed(stream=stream, backward=backward)
        self.assertLess(ratio, 1.0)

        # normal speeds are about:
        # if backward:
        #     if stream:
        #         (iisignature doesn't support backward+stream)
        #     else:
        #         0.7
        # else:
        #     if stream:
        #         0.2
        #     else:
        #         0.3
        #
        # but we give ourselves some margin
        if ratio > (0.8 if backward else 0.4):
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "stream: {stream}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                          ratio=ratio, stream=stream,
                                                                          backward=backward))

    def test_speed_forward_nostream(self):
        self.wrapped_speed_test(stream=False, backward=False)

    def test_speed_forward_stream(self):
        self.wrapped_speed_test(stream=True, backward=False)

    def test_speed_backward(self):
        self.wrapped_speed_test(stream=False, backward=True)
