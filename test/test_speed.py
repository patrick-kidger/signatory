# Copyright 2019 Patrick Kidger. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""Tests that we do indeed perform faster than iisignature. We only test on the CPU (not the GPU) as we'll obviously be
faster on the GPU!

In principle these are quite a fragile set of tests - what if iisignature is updated to operate faster? But just
testing raw speeds is very hardware dependent, so this will do. In particular we give ourselves quite a large margin
before we actually fail the tests, although warnings will be emitted if we're unusually slow.
"""


import signatory
import iisignature
import timeit
import torch
import warnings

import utils_testing as utils


class TestSignatureSpeed(utils.TimedTestCase):
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
        grad_seed = torch.rand_like(signatory.signature(signatory_x, depth, stream=stream))

        def signatory_fn():
            y = signatory.signature(signatory_x, depth, stream=stream)
            if backward:
                y.backward(grad_seed)

        def iisignature_fn():
            iisignature.sig(iisignature_x, depth, 2 if stream else 0)
            if backward:
                iisignature.sigbackprop(grad_seed, iisignature_x, depth)

        signatory_time = 100000000  # very scientific
        iisignature_time = 100000000
        for _ in range(number):
            signatory_time = min(signatory_time, timeit.timeit(signatory_fn, number=1))
            iisignature_time = min(iisignature_time, timeit.timeit(iisignature_fn, number=1))
        return signatory_time, iisignature_time, signatory_time / iisignature_time

    def wrapped_speed_test(self, stream, backward):
        signatory_time, iisignature_time, ratio = self.speed(stream=stream, backward=backward)

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
        # but we give ourselves a large margin: there's quite a lot of variation.
        # The point of Signatory is to run on the GPU, not the CPU, anyway

        if ratio > 2:
            self.fail("Speed was too slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                      "stream: {stream}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                      ratio=ratio, stream=stream, backward=backward))
        if ratio > (1.1 if backward else 0.7):
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "stream: {stream}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                          ratio=ratio, stream=stream,
                                                                          backward=backward))

    def test_speed(self):
        self.speed(batch_size=100, number=10, stream=False, backward=True)  # warm up
        for stream in (True, False):
            for backward in (True, False):
                if stream and backward:
                    continue
                self.wrapped_speed_test(stream=stream, backward=backward)


class TestLogSignatureSpeed(utils.TimedTestCase):
    @classmethod
    def speed(cls, batch_size=1000, number=20, depth=3, backward=False, signatory_mode="brackets"):
        # We speed test by testing against another library
        # Which is admittedly a bit fragile - what if they update their library to operate faster?
        # But just testing raw speeds is going to be very hardware-dependent, so this is good enough

        signatory_x = torch.rand(batch_size, 100, 10, requires_grad=backward)
        iisignature_x = signatory_x.detach().numpy()

        iisignature_mode = {"expand": "x", "brackets": "d"}[signatory_mode]

        signatory_logsignature = signatory.LogSignature(depth, mode=signatory_mode)
        u = signatory_logsignature(signatory_x)  # to do the equivalent of prepare for iisignature
        grad_seed = torch.rand_like(u)
        prep = iisignature.prepare(10, depth)

        def signatory_fn():
            y = signatory_logsignature(signatory_x)
            if backward:
                y.backward(grad_seed)

        def iisignature_fn():
            iisignature.logsig(iisignature_x, prep, iisignature_mode)
            if backward:
                iisignature.logsigbackprop(grad_seed, iisignature_x, prep, iisignature_mode)

        signatory_time = 100000000  # very scientific
        iisignature_time = 100000000
        for _ in range(number):
            signatory_time = min(signatory_time, timeit.timeit(signatory_fn, number=1))
            iisignature_time = min(iisignature_time, timeit.timeit(iisignature_fn, number=1))
        return signatory_time, iisignature_time, signatory_time / iisignature_time

    def wrapped_speed_test(self, backward, mode):
        signatory_time, iisignature_time, ratio = self.speed(backward=backward, signatory_mode=mode)

        # normal speeds are about:
        # if backward:
        #     if mode == "expand":
        #         0.84
        #     elif mode == "brackets":
        #         0.83
        # else:
        #     if mode == "expand":
        #         0.59
        #     elif mode == "brackets":
        #         0.64
        #
        # but there can be quite a lot of variation, so
        # we give ourselves a large margin
        # The point of Signatory is to run on the GPU, not the CPU, anyway
        if backward:
            if mode == "expand":
                expected_ratio = 1.1
            elif mode == "brackets":
                expected_ratio = 1.1
        else:
            if mode == "expand":
                expected_ratio = 0.9
            elif mode == "brackets":
                expected_ratio = 0.9

        if ratio > 2:
            self.fail("Speed was too slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                      "mode: {mode}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                      ratio=ratio, mode=mode, backward=backward))

        if ratio > expected_ratio:
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "mode: {mode}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                      ratio=ratio, mode=mode, backward=backward))

    def test_speed(self):
        self.speed(batch_size=100, number=10, backward=True, signatory_mode="brackets")  # warm up
        for backward in (True, False):
            for mode in ("brackets", "expand"):
                self.wrapped_speed_test(backward, mode)
