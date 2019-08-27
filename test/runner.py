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
"""Central location from which to run all tests."""


import os
import sys
import unittest


def main(failfast=False, record_test_times=True):
    if record_test_times:
        # What an ugly hack. I don't see nice ways to record extra diagnostic information in tests, though.
        unittest.test_times = []
    
    loc = os.path.dirname(__file__)
    loader = unittest.defaultTestLoader
    suite = loader.discover(loc)

    runner = unittest.TextTestRunner(failfast=failfast)
    try:
        result = runner.run(suite)
    finally:  # in case of KeyboardInterrupt on a long test
        if record_test_times:
            print('Time taken for each test:')
            for r in unittest.test_times:
                print(r)
            del unittest.test_times
