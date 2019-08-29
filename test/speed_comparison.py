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
"""This module provides speed benchmarks against esig and iisignature. Can be called separately, as well as being part
of some tests."""


import esig.tosig
import iisignature
import signatory
import time
import torch

import compatibility as compat


@compat.lru_cache(maxsize=None)
def prepare(channels, depth):
    return iisignature.prepare(channels, depth)


@compat.lru_cache(maxsize=1)
def path_cuda(path):
    return path.to('cuda')


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


def iisignature_signature_forward(path, depth, stream=False):
    path = path.detach()
    start = time.time()
    iisignature.sig(path, depth, 2 if stream else 0)
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


def signatory_signature_forward(path, depth, stream=False):
    start = time.time()
    signatory.signature(path, depth, stream=stream)
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
    return signatory_signature_forward(path_cuda(path), depth)


def signatory_logsignature_forward_gpu(path, depth):
    return signatory_logsignature_forward(path_cuda(path), depth)


def signatory_signature_backward_gpu(path, depth):
    return signatory_signature_backward(path_cuda(path), depth)


def signatory_logsignature_backward_gpu(path, depth):
    return signatory_logsignature_backward(path_cuda(path), depth)


signature_forward_fns = {'esig': esig_signature_forward,
                         'iisignature': iisignature_signature_forward,
                         'signatory': signatory_signature_forward,
                         'signatory_gpu': signatory_signature_forward_gpu}

signature_backward_fns = {'esig': esig_signature_backward,
                          'iisignature': iisignature_signature_backward,
                          'signatory': signatory_signature_backward,
                          'signatory_gpu': signatory_signature_backward_gpu}

logsignature_forward_fns = {'esig': esig_logsignature_forward,
                            'iisignature': iisignature_logsignature_forward,
                            'signatory': signatory_logsignature_forward,
                            'signatory_gpu': signatory_logsignature_forward_gpu}

logsignature_backward_fns = {'esig': esig_logsignature_backward,
                             'iisignature': iisignature_logsignature_backward,
                             'signatory': signatory_logsignature_backward,
                             'signatory_gpu': signatory_logsignature_backward_gpu}

signature_fns = {'forward': signature_forward_fns, 'backward': signature_backward_fns}

logsignature_fns = {'forward': logsignature_forward_fns, 'backward': logsignature_backward_fns}

all_fns = {'signature': signature_fns, 'logsignature': logsignature_fns}


class Result:
    def __init__(self, results):
        self.results = results
        self.min = min(results)

    def __repr__(self):
        return "{:.3}".format(self.min)


def run_test(fn_dict, path, depth, repeats, skip=lambda library_name: False, **kwargs):
    library_results = {}
    for library_name, library_fn in fn_dict.values():
        if skip(library_name):
            continue
        library_results[library_name] = Result([library_fn(path, depth, **kwargs) for _ in range(repeats)])
    return library_results


def run_tests(size=(64, 16, 8), depths=(3, 6), repeats=20):
    results = {}
    path = torch.rand(size, dtype=torch.double, requires_grad=True)
    for fn_name, fns in all_fns.items():
        fn_results = results[fn_name] = {}
        for direction_name, directions in fns:
            direction_results = fn_results[direction_name] = {}
            for depth in depths:
                direction_results[depth] = run_test(directions, path, depth, repeats)
    return results


def get_ratios(results):
    ratios = {}
    for fn_name, fn_results in results.items():
        fn_ratios = ratios[fn_name] = {}
        for direction_name, direction_results in fn_results:
            direction_ratios = fn_ratios[direction_name] = {}
            for depth, depth_results in direction_results.items():
                depth_ratios = direction_ratios[depth] = {}

                esig_results = depth_results['esig']
                iisignature_results = depth_results['iisignature']
                signatory_results = depth_results['signatory']
                signatory_gpu_results = depth_results['signatory_gpu']

                depth_ratios['cpu'] = min(esig_results.min, iisignature_results.min) / signatory_results.min
                depth_ratios['gpu'] = min(esig_results.min, iisignature_results.min) / signatory_gpu_results.min
    return ratios
