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
"""Performs memory benchmarking on a particular operation.

There's quite a few intricacies to this. In particular the only accurate way to measure memory usage seems to
necessitate having it in a separate process (to avoid noise from everything else going on), and to use memory_profiler
(rather than something like valgrind), to be able to accurately measure the baseline memory usage (which varies across
runs with valgrind. Probably valgrind has an option to make it work anyway, but this is simpler.)
"""


import argparse
import gc
import importlib
import memory_profiler
import numpy as np
import time
import torch
import sys


def main():
    obj = argparse.Namespace(size=tuple(int(i) for i in size.split(',')), depth=int(depth))
    library_module = importlib.import_module('.functions.' + library_module_name, __package__)
    library_module.setup(obj)

    # The operation that we're going to benchmark (must be defined up here, so that the memory consumed by this function
    # definition is part of the baseline)
    def run_wrapper():
        # store result to make sure it's in memory
        result = library_module.run(obj)
        if result is None:
            raise RuntimeError("'run' did not return anything, so the thing to measure might not be held in memory.")
        # wait to make sure we measure it
        time.sleep(0.5)

    # Now measure the actual memory used!
    try:
        # Now measure the baseline memory usage
        gc.collect()
        baseline = min(memory_profiler.memory_usage(proc=-1, interval=.2, timeout=1))

        run_wrapper()  # warm up. Not totally clear if that really matters here or not, but it can't hurt.

        gc.collect()
        used = max(memory_profiler.memory_usage((run_wrapper, (), {})))
        amount = used - baseline
    except Exception:
        amount = np.inf

    # Report results
    print(amount)


# Perform setup
library_module_name, size, depth, device = sys.argv[1:]
device = int(device)
if device == -1:
    main()
else:
    with torch.cuda.device(device):
        main()
