import signatory
import iisignature
import timeit
import torch
import unittest
import warnings


@unittest.skip
class TestSignatureSpeed(unittest.TestCase):
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


@unittest.skip
class TestLogSignatureSpeed(unittest.TestCase):
    @classmethod
    def speed(cls, batch_size=1000, number=20, depth=3, backward=False, signatory_mode="brackets"):
        # We speed test by testing against another library
        # Which is admittedly a bit fragile - what if they update their library to operate faster?
        # But just testing raw speeds is going to be very hardware-dependent, so this is good enough

        signatory_x = torch.rand(batch_size, 100, 10, requires_grad=backward)
        iisignature_x = signatory_x.detach().numpy()

        iisignature_mode = {"expand": "x", "brackets": "d"}[signatory_mode]

        # TODO: allow signatory.logsignature time to factor out its preparation as well

        prep = iisignature.prepare(10, depth)

        def signatory_fn():
            y = signatory.logsignature(signatory_x, depth, mode=signatory_mode)
            if backward:
                y.backward(torch.rand_like(y))

        def iisignature_fn():
            y = iisignature.logsig(iisignature_x, prep, iisignature_mode)
            if backward:
                iisignature.logsigbackprop(torch.rand_like(torch.tensor(y)), iisignature_x, prep, iisignature_mode)

        signatory_time = timeit.timeit(signatory_fn, number=number)
        iisignature_time = timeit.timeit(iisignature_fn, number=number)
        return signatory_time, iisignature_time, signatory_time / iisignature_time

    def wrapped_speed_test(self, backward, mode):
        signatory_time, iisignature_time, ratio = self.speed(backward=backward, signatory_mode=mode)
        self.assertLess(ratio, 1.0)

        # normal speeds are about:
        # if backward:
        #     if mode == "expand":
        #         ?
        #     elif mode == "brackets":
        #         ?
        # else:
        #     if mode == "expand":
        #         ?
        #     elif mode == "brackets":
        #         ?
        #
        # but we give ourselves some margin
        # TODO: enable these warnings once we know what typical values are
        if ratio > (1 if backward else 1):
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "mode: {mode}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                      ratio=ratio, mode=mode, backward=backward))

    def test_speed(self):
        for backward in (True, False):
            for mode in ("brackets", "expand"):
                self.wrapped_speed_test(backward, mode)
