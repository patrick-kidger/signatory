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
import math
import signatory
import timeit
import torch

try:
    # Being run directly
    from . import compatibility as compat
except ImportError:
    # Being run via tests
    import compatibility as compat


@compat.lru_cache(maxsize=None)
def prepare(channels, depth):
    return iisignature.prepare(channels, depth)


def esig_signature_forward(size, depth, repeat, number):
    path = torch.rand(size, dtype=torch.float).numpy()
    if not len(esig.tosig.stream2sig(path[0], depth)):
        # esig doesn't support larger depths and just returns an empty array
        #
        # and also spams stdout complaining
        return [math.inf]

    def stmt():
        for batch_elem in path:
            esig.tosig.stream2sig(batch_elem, depth)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def esig_logsignature_forward(size, depth, repeat, number):
    path = torch.rand(size, dtype=torch.float).numpy()
    if not len(esig.tosig.stream2logsig(path[0], depth)):
        # esig doesn't support larger depths and just returns an empty array
        #
        # and also spams stdout complaining
        return [math.inf]

    def stmt():
        for batch_elem in path:
            esig.tosig.stream2logsig(batch_elem, depth)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def esig_signature_backward(size, depth, repeat, number):
    # esig doesn't provide this operation.
    return [math.inf]


def esig_logsignature_backward(size, depth, repeat, number):
    # esig doesn't provide this operation.
    return [math.inf]


def iisignature_signature_forward(size, depth, repeat, number, stream=False):
    path = torch.rand(size, dtype=torch.float).numpy()

    if stream:
        def stmt():
            iisignature.sig(path, depth, 2)
    else:
        def stmt():
            iisignature.sig(path, depth, 0)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def iisignature_logsignature_forward(size, depth, repeat, number):
    path = torch.rand(size, dtype=torch.float).numpy()
    prep = prepare(path.shape[-1], depth)

    def stmt():
        iisignature.logsig(path, prep)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def iisignature_signature_backward(size, depth, repeat, number):
    path = torch.rand(size, dtype=torch.float).numpy()
    signature = iisignature.sig(path, depth)
    grad = torch.rand_like(torch.tensor(signature, dtype=torch.float)).numpy()

    def stmt():
        iisignature.sigbackprop(grad, path, depth)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def iisignature_logsignature_backward(size, depth, repeat, number):
    path = torch.rand(size, dtype=torch.float).numpy()
    prep = prepare(path.shape[-1], depth)
    logsignature = iisignature.logsig(path, prep)
    grad = torch.rand_like(torch.tensor(logsignature, dtype=torch.float)).numpy()

    def stmt():
        iisignature.logsigbackprop(grad, path, prep)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def signatory_signature_forward(size, depth, repeat, number, stream=False, gpu=False):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda')
    else:
        path = torch.rand(size, dtype=torch.float)

    if gpu:
        def stmt():
            signatory.signature(path, depth, stream=stream)
            torch.cuda.synchronize()
    else:
        def stmt():
            signatory.signature(path, depth, stream=stream)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def signatory_logsignature_forward(size, depth, repeat, number, gpu=False, mode='words'):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda')
    else:
        path = torch.rand(size, dtype=torch.float)
    # ensure that we're doing a fair test by caching if we can
    # (equivalent to the call to 'prepare' in iisignature)
    signatory.LogSignature(depth, mode=mode)(path)

    if gpu:
        def stmt():
            signatory.LogSignature(depth, mode=mode)(path)
            torch.cuda.synchronize()
    else:
        def stmt():
            signatory.LogSignature(depth, mode=mode)(path)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def signatory_signature_backward(size, depth, repeat, number, gpu=False):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda', requires_grad=True)
    else:
        path = torch.rand(size, dtype=torch.float, requires_grad=True)
    signature = signatory.signature(path, depth)
    grad = torch.rand_like(signature)

    if gpu:
        def stmt():
            signature.backward(grad, retain_graph=True)
            torch.cuda.synchronize()
    else:
        def stmt():
            signature.backward(grad, retain_graph=True)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def signatory_logsignature_backward(size, depth, repeat, number, gpu=False, mode='words'):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda', requires_grad=True)
    else:
        path = torch.rand(size, dtype=torch.float, requires_grad=True)
    logsignature = signatory.LogSignature(depth, mode=mode)(path)
    grad = torch.rand_like(logsignature)

    if gpu:
        def stmt():
            logsignature.backward(grad, retain_graph=True)
            torch.cuda.synchronize()
    else:
        def stmt():
            logsignature.backward(grad, retain_graph=True)

    stmt()  # warm up

    return timeit.Timer(stmt=stmt).repeat(repeat=repeat, number=number)


def signatory_signature_forward_gpu(size, depth, repeat, number):
    return signatory_signature_forward(size, depth, repeat, number, gpu=True)


def signatory_logsignature_forward_gpu(size, depth, repeat, number):
    return signatory_logsignature_forward(size, depth, repeat, number, gpu=True)


def signatory_signature_backward_gpu(size, depth, repeat, number):
    return signatory_signature_backward(size, depth, repeat, number, gpu=True)


def signatory_logsignature_backward_gpu(size, depth, repeat, number):
    return signatory_logsignature_backward(size, depth, repeat, number, gpu=True)


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
        if self.min is math.inf:
            return '-'
        else:
            return "{:.3}".format(self.min)


def run_test(fn_dict, size, depth, repeat, number, print_name, skip=lambda library_name: False, **kwargs):
    library_results = {}
    for library_name, library_fn in fn_dict.items():
        if skip(library_name):
            continue
        if print_name:
            print(*print_name, library_name)
        library_results[library_name] = Result(library_fn(size, depth, repeat, number, **kwargs))
    return library_results


def run_tests(size=(32, 16, 8), depths=(4, 6), repeat=20, number=5):
    results = {}
    for fn_name, fns in all_fns.items():
        fn_results = results[fn_name] = {}
        for direction_name, directions in fns.items():
            direction_results = fn_results[direction_name] = {}
            for depth in depths:
                direction_results[depth] = run_test(directions, size, depth, repeat, number,
                                                    print_name=(fn_name, direction_name, depth))
    return results


def get_ratios(results):
    ratios = {}
    for fn_name, fn_results in results.items():
        fn_ratios = ratios[fn_name] = {}
        for direction_name, direction_results in fn_results.items():
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
