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
"""Performs time benchmarking on a particular operation.

This is pulled out in a separate process to work around an apparent PyTorch bug, that only allows for setting the number
of threads during the 'preamble', and not during the runtime of the program.
"""


import argparse
import importlib
import math
import sys
import timeit
import torch


def main():
    obj = argparse.Namespace(size=tuple(int(i) for i in size.split(',')), depth=int(depth))
    library_module = importlib.import_module('.functions.' + library_module_name, __package__)
    library_module.setup(obj)

    try:
        result = min(timeit.Timer(setup=lambda: library_module.run(obj),  # warm up
                                  stmt=lambda: library_module.run(obj)).repeat(repeat=50, number=1))
    except Exception:
        result = math.inf

    # Report results
    print(result)


# Perform setup
library_module_name, size, depth, device = sys.argv[1:]
device = int(device)
if device == -1:
    main()
else:
    with torch.cuda.device(device):
        main()
