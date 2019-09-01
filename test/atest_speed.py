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


import torch
import warnings

import speed_comparison as speedc
import utils_testing as utils


class TestSignatureSpeed(utils.EnhancedTestCase):
    def inner_test_speed(self, stream, backward):
        if backward and stream:
            raise ValueError("iisignature does not support backward and stream")

        size = (64, 16, 8)
        depth = 6
        kwargs = {}
        if stream:  # NOT just passing as stream=stream; we can't pass extra kwargs when backward=True.
            kwargs['stream'] = stream
        results = speedc.run_test(speedc.signature_backward_fns if backward else speedc.signature_forward_fns,
                                  size,
                                  depth,
                                  repeat=20,
                                  number=5,
                                  print_name=False,
                                  skip=lambda library_name: library_name in ('esig', 'signature_gpu'),
                                  **kwargs)
        signatory_time = results['signatory'].min
        iisignature_time = results['iisignature'].min
        ratio = signatory_time / iisignature_time

        # normal speeds are about:
        # if backward:
        #     if stream:
        #         (iisignature doesn't support backward+stream)
        #     else:
        #         TODO
        # else:
        #     if stream:
        #         TODO
        #     else:
        #         TODO
        #
        # but we give ourselves a large margin: there's quite a lot of variation.
        # The point of Signatory is to run on the GPU, not the CPU, anyway
        if backward:
            if stream:
                pass
            else:
                expected_ratio = 0.7
        else:
            if stream:
                expected_ratio = 0.4
            else:
                expected_ratio = 0.4

        if ratio > 2:
            self.fail("Speed was too slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                      "stream: {stream}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                      ratio=ratio, stream=stream, backward=backward))
        if ratio > expected_ratio:
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "stream: {stream}, backward: {backward}".format(stime=signatory_time, itime=iisignature_time,
                                                                          ratio=ratio, stream=stream,
                                                                          backward=backward))

    def test_speed(self):
        for stream in (True, False):
            for backward in (True, False):
                if stream and backward:
                    continue
                self.wrapped_speed_test(stream=stream, backward=backward)


class TestLogSignatureSpeed(utils.EnhancedTestCase):
    def inner_test_speed(self, backward):
        size = (64, 16, 8)
        depth = 6
        results = speedc.run_test(speedc.logsignature_backward_fns if backward else speedc.logsignature_forward_fns,
                                  size,
                                  depth,
                                  repeat=20,
                                  number=5,
                                  print_name=False,
                                  skip=lambda library_name: library_name in ('esig', 'signature_gpu'))
        signatory_time = results['signatory'].min
        iisignature_time = results['iisignature'].min
        ratio = signatory_time / iisignature_time

        # normal speeds are about:
        # if backward:
        #     TODO
        # else:
        #     TODO
        #
        # but there can be quite a lot of variation, so
        # we give ourselves a large margin
        # The point of Signatory is to run on the GPU, not the CPU, anyway
        if backward:
            expected_ratio = 1.1
        else:
            expected_ratio = 0.9

        if ratio > 2:
            self.fail("Speed was too slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                      "backward: {backward}".format(stime=signatory_time, itime=iisignature_time, ratio=ratio,
                                                    backward=backward))

        if ratio > expected_ratio:
            warnings.warn("Speed was unusually slow: signatory time: {stime} iisignature time: {itime}, ratio: {ratio} "
                          "backward: {backward}".format(stime=signatory_time, itime=iisignature_time, ratio=ratio,
                                                        backward=backward))

    def test_speed(self):
        for backward in (True, False):
            self.inner_test_speed(backward)
