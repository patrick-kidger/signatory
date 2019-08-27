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
"""Not providing tests that we test against; this module provides speed benchmarks against esig and iisignature.
(It just happens to fit most naturally in with the tests.)
"""


import collections as co
import esig.tosig
import iisignature
import signatory
import time
import torch

import compatibility as compat


@compat.lru_cache(maxsize=None)
def prepare(channels, depth):
    return iisignature.prepare(channels, depth)


def esig_signature_forward(path, depth):
    path = path.detach().numpy()

    start = time.time()
    for batch_elem in path:
        result = esig.tosig.stream2sig(batch_elem, depth)
    if len(result):
        return time.time() - start
    else:
        # esig doesn't support larger depths and just results an empty array
        #
        # and also spams stdout complaining
        return 99999999


def esig_logsignature_forward(path, depth):
    path = path.detach().numpy()
    start = time.time()
    for batch_elem in path:
        esig.tosig.stream2logsig(batch_elem, depth)
    return time.time() - start


def esig_signature_backward(path, depth):
    # esig doesn't provide this operation.
    return 99999999


def esig_logsignature_backward(path, depth):
    # esig doesn't provide this operation.
    return 99999999


esig_fns = {'signature_forward': esig_signature_forward,
            'logsignature_forward': esig_logsignature_forward,
            'signature_backward': esig_signature_backward,
            'logsignature_backward': esig_logsignature_backward}


def iisignature_signature_forward(path, depth):
    path = path.detach()
    start = time.time()
    iisignature.sig(path, depth)
    return time.time() - start


def iisignature_logsignature_forward(path, depth):
    path = path.detach()
    prep = prepare(path.size(-1), depth)
    start = time.time()
    iisignature.logsig(path, prep)
    return time.time() - start


def iisignature_signature_backward(path, depth):
    path = path.detach()
    signature = iisignature.sig(path, depth)
    grad = torch.rand_like(torch.tensor(signature))
    start = time.time()
    iisignature.sigbackprop(grad, path, depth)
    return time.time() - start


def iisignature_logsignature_backward(path, depth):
    path = path.detach()
    prep = prepare(path.size(-1), depth)
    logsignature = iisignature.logsig(path, prep)
    grad = torch.rand_like(torch.tensor(logsignature))
    start = time.time()
    iisignature.logsigbackprop(grad, path, prep)
    return time.time() - start


iisignature_fns = {'signature_forward': iisignature_signature_forward,
                   'logsignature_forward': iisignature_logsignature_forward,
                   'signature_backward': iisignature_signature_backward,
                   'logsignature_backward': iisignature_logsignature_backward}


def signatory_signature_forward(path, depth):
    start = time.time()
    signatory.signature(path, depth)
    return time.time() - start


def signatory_logsignature_forward(path, depth):
    # ensure that we're doing a fair test by caching if we can
    # (equivalent to the call to 'prepare' in iisignature)
    signatory.LogSignature(depth)(path)
    start = time.time()
    signatory.LogSignature(depth)(path)
    return time.time() - start


def signatory_signature_backward(path, depth):
    signature = signatory.signature(path, depth)
    grad = torch.rand_like(signature)
    start = time.time()
    signature.backward(grad)
    return time.time() - start


def signatory_logsignature_backward(path, depth):
    logsignature = signatory.LogSignature(depth)(path)
    grad = torch.rand_like(logsignature)
    start = time.time()
    logsignature.backward(grad)
    return time.time() - start


def signatory_signature_forward_gpu(path, depth):
    return signatory_signature_forward(path.to('cuda'), depth)


def signatory_logsignature_forward_gpu(path, depth):
    return signatory_logsignature_forward(path.to('cuda'), depth)


def signatory_signature_backward_gpu(path, depth):
    return signatory_signature_backward(path.to('cuda'), depth)


def signatory_logsignature_backward_gpu(path, depth):
    return signatory_logsignature_backward(path.to('cuda'), depth)


signatory_fns = {'signature_forward': signatory_signature_forward,
                 'logsignature_forward': signatory_logsignature_forward,
                 'signature_backward': signatory_signature_backward,
                 'logsignature_backward': signatory_logsignature_backward,
                 'signature_forward_gpu': signatory_signature_forward_gpu,
                 'logsignature_forward_gpu': signatory_logsignature_forward_gpu,
                 'signature_backward_gpu': signatory_signature_backward_gpu,
                 'logsignature_backward_gpu': signatory_logsignature_backward_gpu}


all_library_fns = {'esig': esig_fns, 'iisignature': iisignature_fns, 'signatory': signatory_fns}


def run_tests():
    size = (64, 16, 8)
    depths = (3, 6)
    repeats = 20

    results = co.defaultdict(lambda: co.defaultdict(dict))
    path = torch.rand(size, dtype=torch.double, requires_grad=True)
    for library_name, library_fns in all_library_fns.items():
        for library_fn_name, library_fn in library_fns.items():
            for depth in depths:
                print(library_name, library_fn_name, depth)
                results[library_name][library_fn_name][depth] = [library_fn(path, depth) for _ in range(repeats)]

    return results


def process_results(results):
    min_results = co.defaultdict(lambda: co.defaultdict(dict))
    for library_name, results_by_library in results.items():
        for library_fn_name, results_by_library_fn in results_by_library.items():
            for depth, results_by_depth in results_by_library_fn.items():
                min_results[library_name][library_fn_name][depth] = min(results_by_depth)

    ratios = co.defaultdict(dict)

    def remove_gpu_str(fn_str):
        return fn_str.split('_gpu')[0]

    depths = tuple(next(next(results.keys()).keys()).keys())
    for signatory_fn_name in signatory_fns.keys():
        for depth in depths:
            signatory_time = min_results['signatory'][signatory_fn_name][depth]
            iisignature_time = min_results['iisignature'][remove_gpu_str(signatory_fn_name)][depth]
            esig_time = min_results['esig'][remove_gpu_str(signatory_fn_name)][depth]

            ratios[signatory_fn_name][depth] = signatory_time / min(iisignature_time, esig_time)

    return min_results, ratios
