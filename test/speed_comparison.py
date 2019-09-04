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


import collections as co
import esig.tosig
import iisignature
import math
import signatory
import timeit
import torch

try:
    # Being run via commands.py
    from . import compatibility as compat
except ImportError:
    # Being run via tests
    import compatibility as compat


@compat.lru_cache(maxsize=None)
def prepare(channels, depth):
    """A cache wrapper around iisignature.prepare, since it takes a long time."""
    return iisignature.prepare(channels, depth)


# All of the following functions wrap the analogous functions from each package, to provide a consistent interface for
# testing against each other.


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


def signatory_logsignature_forward(size, depth, repeat, number, gpu=False, mode='words', stream=False):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda')
    else:
        path = torch.rand(size, dtype=torch.float)
    # ensure that we're doing a fair test by caching if we can
    # (equivalent to the call to 'prepare' in iisignature)
    signatory.LogSignature(depth, mode=mode, stream=stream)(path)

    if gpu:
        def stmt():
            signatory.LogSignature(depth, mode=mode, stream=stream)(path)
            torch.cuda.synchronize()
    else:
        def stmt():
            signatory.LogSignature(depth, mode=mode, stream=stream)(path)

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


def signatory_logsignature_backward(size, depth, repeat, number, gpu=False, mode='words', stream=False):
    if gpu:
        path = torch.rand(size, dtype=torch.float, device='cuda', requires_grad=True)
    else:
        path = torch.rand(size, dtype=torch.float, requires_grad=True)
    logsignature = signatory.LogSignature(depth, mode=mode, stream=stream)(path)
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


signature_forward_fns = co.OrderedDict([('signatory_cpu', signatory_signature_forward),
                                        ('signatory_gpu', signatory_signature_forward_gpu),
                                        ('iisignature', iisignature_signature_forward),
                                        ('esig', esig_signature_forward)])


signature_backward_fns = co.OrderedDict([('signatory_cpu', signatory_signature_backward),
                                         ('signatory_gpu', signatory_signature_backward_gpu),
                                         ('iisignature', iisignature_signature_backward),
                                         ('esig', esig_signature_backward),])

logsignature_forward_fns = co.OrderedDict([('signatory_cpu', signatory_logsignature_forward),
                                           ('signatory_gpu', signatory_logsignature_forward_gpu),
                                           ('iisignature', iisignature_logsignature_forward),
                                           ('esig', esig_logsignature_forward),])

logsignature_backward_fns = co.OrderedDict([('signatory_cpu', signatory_logsignature_backward),
                                            ('signatory_gpu', signatory_logsignature_backward_gpu),
                                            ('iisignature', iisignature_logsignature_backward),
                                            ('esig', esig_logsignature_backward)])

signature_fns = co.OrderedDict([('forward', signature_forward_fns), ('backward', signature_backward_fns)])

logsignature_fns = co.OrderedDict([('forward', logsignature_forward_fns), ('backward', logsignature_backward_fns)])

all_fns = co.OrderedDict([('Signature', signature_fns), ('Logsignature', logsignature_fns)])


class Result:
    """Represents the speed of a particular function, across multiple runs."""
    def __init__(self, results):
        self.results = results
        self.min = min(results)

    def __repr__(self):
        if self.min is math.inf:
            return '-'
        else:
            return "{:.3}".format(self.min)


def run_test(fn_dict, size, depth, repeat, number, print_name, skip=lambda library_name: False, **kwargs):
    """Runs a particular function across multiple different libraries and records their times."""
    library_results = {}
    for library_name, library_fn in fn_dict.items():
        if skip(library_name):
            continue
        if print_name:
            print(print_name, library_name)
        library_results[library_name] = Result(library_fn(size, depth, repeat, number, **kwargs))
    return library_results


def run_tests(size=(16, 32, 8), depths=(4, 6), repeat=100, number=1):
    """Runs all functions across all libraries and records their times."""
    results = {}
    for fn_name, fns in all_fns.items():
        for direction_name, directions in fns.items():
            for depth in depths:
                name = "{}, {}, depth {}".format(fn_name, direction_name, depth)
                result = run_test(directions, size, depth, repeat, number, name)
                esig_results = result['esig']
                iisignature_results = result['iisignature']
                signatory_results = result['signatory']
                signatory_gpu_results = result['signatory_gpu']
                result['speedup_cpu'] = min(esig_results.min, iisignature_results.min) / signatory_results.min
                result['speedup_gpu'] = min(esig_results.min, iisignature_results.min) / signatory_gpu_results.min
                results[name] = result
    return results


def display_results(results):
    """Formats the result of run_tests into a table."""
    # Now we just make a pretty table out of the results.
    # Coding this up was quite therapeutic.
    operation_str = 'Operation'
    padding = 1
    max_row_heading_len = len(operation_str) + 2 * padding
    for row_heading in results:
        max_row_heading_len = max(max_row_heading_len, len(row_heading))
    column_width_lower_bound = 4
    column_widths = [max_row_heading_len]
    column_headings = list(results[row_heading])
    for column_heading in column_headings:
        column_widths.append(max(column_width_lower_bound, len(column_heading)) + 2 * padding)
    heading_str = "|".join("{{:^{}}}".format(column_width) for column_width in column_widths)
    border_str = '+'.join('-' * column_width for column_width in column_widths)
    print(heading_str.format(operation_str, *column_headings))
    print(border_str)
    for row_heading, row_values in results.items():
        print("{{:<{}}}".format(max_row_heading_len).format(row_heading), end='')
        for column_width, (column_heading, column_value), true_column_heading in zip(column_widths, row_values.items(),
                                                                                     column_headings):
            assert column_heading == true_column_heading
            print("|{{:>{}}}".format(column_width).format(column_value), end='')
        print('\n')
