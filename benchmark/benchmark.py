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
"""This module provides speed benchmarks against esig and iisignature."""


import collections as co
import itertools as it
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import timeit

import esig.tosig
import iisignature
import signatory
import torch


# Increasing this tends to result in quicker computations that use more memory
# For the sake of a fair benchmark in both parameters we just disable this.
signatory.max_parallelisation(1)


class BenchmarkBase(object):
    """Abstract base class. Subclasses should correspond to a particular function to be benchmarked.

    Each subclass should specify the multiline strings 'not_include', 'mem_include' and 'run'. These are then run one
    after the other, _in that order_.

    not_include will be initial setup that should not be benchmarked.
    mem_include is setup that should be included in total memory used but not total time taken.
    run should define a single function run(self) specifying the operation to be benchmarked.

    A single object 'self' is available to assign any variables to during setup in not_include and mem_include; this
    will then be passed to run.

    The esig.tosig, iisignature, signatory, and torch libraries may be assumed to be imported.

    The function defined in run will be ran once for memory benchmarking, and several times for speed benchmarking.


    As for why we implement it this way: memory benchmarking uses valgrind, meaning that practically speaking it has to
    occur in a separate process, so we have to specify what we want to happen via strings that will be eval'd.
    """

    # Not counted for either timing or memory
    not_include = ""

    # Defines a 'run' function which is counted for memory and timing
    run = """
def run(self):
    raise NotImplementedError
"""

    # Counted for memory but not timing
    mem_include = ""

    def __init__(self, size, depth, repeat, number, measure):
        self.size = size
        self.depth = depth
        self.repeat = repeat
        self.number = number
        self.measure = measure

        # https://stackoverflow.com/questions/1463306/how-does-exec-work-with-locals
        local_dict = locals()
        exec(self.not_include, globals(), local_dict)
        exec(self.run, globals(), local_dict)
        exec(self.mem_include, globals(), local_dict)
        # Construct the statement to use for speed benchmarking
        self.time_statement = local_dict['run'].__get__(self, type(self))

        # Construct the program to run for memory benchmarking
        self.mem_statement = '\n'.join(["import argparse",
                                        "import gc",
                                        "import memory_profiler",
                                        "import numpy as np",
                                        "import time",
                                        "import esig.tosig",
                                        "import iisignature",
                                        "import signatory",
                                        "import torch",
                                        "signatory.max_parallelisation(1)",
                                        "self = argparse.Namespace()",
                                        'self.size = {size}'.format(size=repr(size)),
                                        'self.depth = {depth}'.format(depth=repr(depth)),
                                        'self.repeat = {repeat}'.format(repeat=repr(repeat)),
                                        'self.number = {number}'.format(number=repr(number)),
                                        'self.measure = {measure}'.format(measure=repr(measure)),
                                        self.not_include,
                                        self.run,
                                        '',
                                        'def run_wrapper():',
                                        # store result to make sure it's in memory
                                        '    result = run(self)',
                                        '    if result is None:',
                                        '        raise RuntimeError("run did not return anything, so the thing to "',
                                        '                           "measure might not be held in memory.")',
                                        # wait to make sure we measure it
                                        '    time.sleep(0.5)',
                                        '',
                                        'gc.collect()',
                                        'baseline = min(memory_profiler.memory_usage(proc=-1, interval=.2, timeout=1))',
                                        self.mem_include,
                                        'try:',
                                        '    used = max(memory_profiler.memory_usage((run_wrapper, (), {})))',
                                        'except Exception:',
                                        '    used = np.inf',
                                        'print(used - baseline)'])

    def action(self):
        if self.measure == 'time':
            return self.time()
        elif self.measure == 'memory':
            return self.memory()
        else:
            raise ValueError("I don't know how to measure '{}'".format(self.measure))

    def time(self):
        try:
            try:
                self.time_statement()  # warm up
            except Exception:
                return math.inf
            return min(timeit.Timer(stmt=self.time_statement).repeat(repeat=self.repeat, number=self.number))
        except KeyboardInterrupt:
            return math.inf

    def memory(self):
        files = os.listdir('.')
        memory_tmp = 'memory.tmp'

        if memory_tmp in files:
            raise RuntimeError('Could not write due to existing memory files.')
        with open(memory_tmp, 'w') as f:
            f.write(self.mem_statement)

        try:
            p = subprocess.run('python {}'.format(memory_tmp), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
            stderr = p.stderr.decode()
            if stderr != '':
                print('Error:')
                print('------')
                print(stderr)
                print('File:')
                print('-----')
                print(self.mem_statement)
                print('')
                print('=========================')
                print('Raising an error to stop.')
                print('=========================')
                raise RuntimeError
            else:
                stdout = p.stdout.decode().strip()
                for line in stdout.split('\n'):
                    if 'Legitimate' not in line:
                        stdout = line
                        break
                return float(stdout)
        finally:
            try:
                os.remove(memory_tmp)
            except FileNotFoundError:
                pass


class esig_signature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
"""

    run = """
def run(self):
    if not len(esig.tosig.stream2logsig(self.path[0], self.depth)):
        raise Exception
    
    result = []
    for batch_elem in self.path:
        result.append(esig.tosig.stream2sig(batch_elem, self.depth))
    return result
"""


class esig_logsignature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
"""

    run = """
def run(self):
    if not len(esig.tosig.stream2logsig(self.path[0], self.depth)):
        raise Exception

    result = []
    for batch_elem in self.path:
        result.append(esig.tosig.stream2logsig(batch_elem, self.depth))
    return result
"""


class esig_signature_backward(BenchmarkBase):
    run = """
def run(self):
    # esig doesn't provide this operation.
    raise Exception
"""


class esig_logsignature_backward(BenchmarkBase):
    run = """
def run(self):
    # esig doesn't provide this operation.
    raise Exception
"""


class iisignature_signature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
"""

    run = """
def run(self):
    return iisignature.sig(self.path, self.depth)
"""


class iisignature_logsignature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
"""

    run = """
def run(self):
    return iisignature.logsig(self.path, self.prep)
"""

    mem_include = """
self.prep = iisignature.prepare(self.path.shape[-1], self.depth)
"""


class iisignature_signature_backward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
shape = self.size[-3], iisignature.siglength(self.size[-1], self.depth)
self.grad = torch.rand(shape).numpy()
"""

    run = """
def run(self):
    return iisignature.sigbackprop(self.grad, self.path, self.depth)
"""


class iisignature_logsignature_backward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float).numpy()
shape = self.size[-3], iisignature.logsiglength(self.size[-1], self.depth)
self.grad = torch.rand(shape).numpy()
"""

    run = """
def run(self):
    return iisignature.logsigbackprop(self.grad, self.path, self.prep)
"""

    mem_include = """
self.prep = iisignature.prepare(self.path.shape[-1], self.depth)
"""


class signatory_signature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float)
"""

    run = """
def run(self):
    return signatory.signature(self.path, self.depth)
"""


class signatory_signature_forward_gpu(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, device='cuda')
"""

    run = """
def run(self):
    result = signatory.signature(self.path, self.depth)
    torch.cuda.synchronize()
    return result
"""


class signatory_logsignature_forward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float)
"""

    run = """
def run(self):
    return self.logsignature_instance(self.path)
"""

    mem_include = """
self.logsignature_instance = signatory.LogSignature(self.depth)
self.logsignature_instance.prepare(self.size[-1])
"""


class signatory_logsignature_forward_gpu(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, device='cuda')
"""

    run = """
def run(self):
    result = self.logsignature_instance(self.path)
    torch.cuda.synchronize()
    return result
"""

    mem_include = """
self.logsignature_instance = signatory.LogSignature(self.depth)
self.logsignature_instance.prepare(self.size[-1])
"""


class signatory_signature_backward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, requires_grad=True)
shape = self.size[-3], signatory.signature_channels(self.size[-1], self.depth)
self.grad = torch.rand(shape)
"""

    run = """
def run(self):
    self.signature.backward(self.grad, retain_graph=self.measure == 'time')
    return self.path.grad
"""

    mem_include = """
self.signature = signatory.signature(self.path, self.depth)
"""


class signatory_signature_backward_gpu(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, requires_grad=True, device='cuda')
shape = self.size[-3], signatory.signature_channels(self.size[-1], self.depth)
self.grad = torch.rand(shape, device='cuda')
"""

    run = """
def run(self):
    self.signature.backward(self.grad, retain_graph=self.measure == 'time')
    torch.cuda.synchronize()
    return self.path.grad
"""

    mem_include = """
self.signature = signatory.signature(self.path, self.depth)
"""


class signatory_logsignature_backward(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, requires_grad=True)
shape = self.size[-3], signatory.logsignature_channels(self.size[-1], self.depth)
self.grad = torch.rand(shape)
"""

    run = """
def run(self):
    self.logsignature.backward(self.grad, retain_graph=self.measure == 'time')
    return self.path.grad
"""

    mem_include = """
self.logsignature = signatory.LogSignature(self.depth)(self.path)
"""


class signatory_logsignature_backward_gpu(BenchmarkBase):
    not_include = """
self.path = torch.rand(self.size, dtype=torch.float, requires_grad=True, device='cuda')
shape = self.size[-3], signatory.logsignature_channels(self.size[-1], self.depth)
self.grad = torch.rand(shape, device='cuda')
"""

    run = """
def run(self):
    self.logsignature.backward(self.grad, retain_graph=self.measure == 'time')
    torch.cuda.synchronize()
    return self.path.grad
"""

    mem_include = """
self.logsignature = signatory.LogSignature(self.depth)(self.path)
"""


signatory_cpu_str = 'Signatory CPU'
signatory_gpu_str = 'Signatory GPU'
iisignature_str = 'iisignature'
esig_str = 'esig'
speedup_cpu_str = 'Ratio CPU'
speedup_gpu_str = 'Ratio GPU'
speedup_str = 'Ratio'


signature_forward_fns = co.OrderedDict([(signatory_cpu_str, signatory_signature_forward),
                                        (signatory_gpu_str, signatory_signature_forward_gpu),
                                        (iisignature_str, iisignature_signature_forward),
                                        (esig_str, esig_signature_forward)])


signature_backward_fns = co.OrderedDict([(signatory_cpu_str, signatory_signature_backward),
                                         (signatory_gpu_str, signatory_signature_backward_gpu),
                                         (iisignature_str, iisignature_signature_backward),
                                         (esig_str, esig_signature_backward)])

logsignature_forward_fns = co.OrderedDict([(signatory_cpu_str, signatory_logsignature_forward),
                                           (signatory_gpu_str, signatory_logsignature_forward_gpu),
                                           (iisignature_str, iisignature_logsignature_forward),
                                           (esig_str, esig_logsignature_forward)])

logsignature_backward_fns = co.OrderedDict([(signatory_cpu_str, signatory_logsignature_backward),
                                            (signatory_gpu_str, signatory_logsignature_backward_gpu),
                                            (iisignature_str, iisignature_logsignature_backward),
                                            (esig_str, esig_logsignature_backward)])

signature_forward_fns_wrapper = {'Signature forward': signature_forward_fns}
signature_backward_fns_wrapper = {'Signature backward': signature_backward_fns}
logsignature_forward_fns_wrapper = {'Logsignature forward': logsignature_forward_fns}
logsignature_backward_fns_wrapper = {'Logsignature backward': logsignature_backward_fns}

all_fns = co.OrderedDict()
all_fns.update(signature_forward_fns_wrapper)
all_fns.update(signature_backward_fns_wrapper)
all_fns.update(logsignature_forward_fns_wrapper)
all_fns.update(logsignature_backward_fns_wrapper)


class namedarray(object):
    """Wraps a numpy array with name-based lookup along axes.

    Just a minimal helper for our needs elsewhere in this file. There are definitely fancier solutions available.
    """

    def __init__(self, *size):
        self.array = np.empty(size, dtype=object)
        self.numdims = len(size)
        self.dim_lookups = [co.OrderedDict() for _ in range(self.numdims)]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            raise ValueError
        if len(key) != self.numdims:
            raise ValueError
        indices = []
        for elem, lookup in zip(key, self.dim_lookups):
            if isinstance(elem, slice):
                raise ValueError
            try:
                index = lookup[elem]
            except KeyError:
                index = lookup[elem] = len(lookup)
            indices.append(index)
        indices = tuple(indices)
        self.array[indices] = value

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise ValueError
        if len(key) != self.numdims:
            raise ValueError
        indices = []
        for elem, lookup in zip(key, self.dim_lookups):
            try:
                index = lookup[elem]
            except KeyError:
                index = elem
            indices.append(index)
        indices = tuple(indices)
        return self.array[indices]

    def __iter__(self):
        lookups = tuple(lookup.keys() for lookup in self.dim_lookups)
        for index in it.product(*lookups):
            yield index, self[index]


class BenchmarkRunner(object):
    """Runs all functions across all libraries and records their times or memory usage for multiple sizes and depths."""

    def __init__(self, sizes, depths, ratio, test_esig, test_signatory_gpu, measure, fns):
        if measure == 'memory' and test_signatory_gpu:
            raise RuntimeError('Memory comparisons for Signatory GPU are not meaningful, as everything else operates '
                               'on the CPU.')

        self.sizes = sizes
        self.depths = depths
        self.repeat = 50
        self.number = 1
        self.ratio = ratio
        self.test_esig = test_esig
        self.test_signatory_gpu = test_signatory_gpu
        self.measure = measure
        if fns == 'all':
            self.fns = all_fns
        elif fns == 'sigf':
            self.fns = signature_forward_fns_wrapper
        elif fns == 'sigb':
            self.fns = signature_backward_fns_wrapper
        elif fns == 'logsigf':
            self.fns = logsignature_forward_fns_wrapper
        elif fns == 'logsigb':
            self.fns = logsignature_backward_fns_wrapper
        else:
            raise RuntimeError

        self._results = None

    @property
    def results(self):
        return self._results

    def check_graph(self):
        """Checks whether or not this benchmark is suitable for being plotted as a graph."""

        if len(self.sizes) > 1 and len(self.depths) > 1:
            raise RuntimeError("Cannot output as graph with multiple sizes and multiple depths.")
        if len(list(self.fns.keys())) > 1:
            raise RuntimeError("Cannot output as graph with multiple functions.")
        batch_size, stream_size, _ = next(iter(self.sizes))
        for size in self.sizes:
            if size[0] != batch_size or size[1] != stream_size:
                raise RuntimeError("Cannot output as graph with multiple batch or stream sizes.")

    def run(self):
        """Runs the benchmarks."""

        results = namedarray(len(self.fns), len(self.sizes), len(self.depths))
        for fn_name, fn_dict in self.fns.items():
            for size in self.sizes:
                for depth in self.depths:
                    result = self._run_test(fn_name, fn_dict, size, depth)
                    results[fn_name, size, depth] = result
        self._results = results

    def _run_test(self, fn_name, fn_dict, size, depth):
        """Runs a particular function across multiple different libraries and records their times."""

        library_results = co.OrderedDict()
        for library_name, library_fn in fn_dict.items():
            if (not self.test_esig) and (library_name == esig_str):
                continue
            if (not self.test_signatory_gpu) and (library_name == signatory_gpu_str):
                continue
            print(self._table_format_index(fn_name, size, depth), library_name)
            test = library_fn(size=size, depth=depth, repeat=self.repeat, number=self.number, measure=self.measure)
            library_results[library_name] = test.action()

        if self.ratio:
            other_best = library_results[iisignature_str]
            if self.test_esig:
                other_best = min(library_results[esig_str], other_best)
            try:
                library_results[speedup_cpu_str] = other_best / library_results[signatory_cpu_str]
            except ZeroDivisionError:
                library_results[speedup_cpu_str] = math.inf
            if self.test_signatory_gpu:
                try:
                    library_results[speedup_gpu_str] = other_best / library_results[signatory_gpu_str]
                except ZeroDivisionError:
                    library_results[speedup_gpu_str] = math.inf
        return library_results

    @staticmethod
    def _table_format_index(fn_name, size, depth):
        return "{}, size {}, depth {}".format(fn_name, size, depth)

    def _graph_format_index(self, fn_name, size, depth):
        if len(self.sizes) > 1:
            return size[-1]
        elif len(self.depths) > 1:
            return depth

    @classmethod
    def typical(cls, **kwargs):
        """Tests two typical use cases."""
        new_kwargs = dict(sizes=((32, 128, 8),),
                          depths=(5, 7))
        new_kwargs.update(kwargs)
        return cls(**new_kwargs)

    @classmethod
    def channels(cls, **kwargs):
        """Tests a number of channels for a fixed depth."""
        new_kwargs = dict(sizes=((32, 128, 2), (32, 128, 3), (32, 128, 4), (32, 128, 5), (32, 128, 6), (32, 128, 7)),
                          depths=(7,))
        new_kwargs.update(kwargs)
        return cls(**new_kwargs)

    @classmethod
    def depths(cls, **kwargs):
        """Tests depths for a fixed number of channels."""
        new_kwargs = dict(sizes=((32, 128, 4),),
                          depths=(2, 3, 4, 5, 6, 7, 8, 9))
        new_kwargs.update(kwargs)
        return cls(**new_kwargs)

    @classmethod
    def small(cls, **kwargs):
        """Tests on very small data. This doesn't given meaningful results - the millisecond overhead of
        PyTorch/NumPy/etc. ends up giving a greater noise than there is signal - but it serves to test the speed-testing
        framework.
        """
        new_kwargs = dict(sizes=((1, 2, 2),),
                          depths=(2, 3, 4, 5))
        new_kwargs.update(kwargs)
        return cls(**new_kwargs)

    def graph(self, log=True, save=False):
        """Plots the result as a graph."""

        self.check_graph()

        fig = plt.figure()
        ax = fig.gca()

        _, example_row_value = next(iter(self.results))
        x_axes = [[] for _ in range(len(example_row_value))]
        y_axes = [[] for _ in range(len(example_row_value))]
        labels = []
        for column_heading in example_row_value.keys():
            if speedup_str in column_heading:
                continue
            labels.append(column_heading)
        for row_heading, row_value in self.results:
            for x_axis, y_axis, (column_heading, column_value) in zip(x_axes, y_axes, row_value.items()):
                if speedup_str in column_heading:
                    continue
                x_axis.append(self._graph_format_index(*row_heading))
                y_axis.append(column_value)

        for x_axis, y_axis, label in zip(x_axes, y_axes, labels):
            ax.plot(x_axis, y_axis, label=label)
        ax.legend(mode='expand', ncol=len(example_row_value), bbox_to_anchor=(0, 1.1, 1, 0), borderaxespad=0.)
        title_string = list(self.fns.keys())[0] + ': ' + self.measure
        ax.set_title(title_string, y=1.1)
        if self.measure == 'time':
            ax.set_ylabel("Time in seconds")
        elif self.measure == 'memory':
            ax.set_ylabel("Memory usage in MB")
        if len(self.sizes) > 1:
            ax.set_xlabel("Number of channels")
            tag = '_channels'
        elif len(self.depths) > 1:
            ax.set_xlabel("Depth")
            tag = '_depths'
        else:
            raise RuntimeError
        if log:
            ax.set_yscale('log')
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(range(int(math.ceil(start)), int(math.floor(end)) + 1))
        plt.tight_layout()
        if save:
            dirname = title_string.lower().replace(' ', '_').replace(':', '') + tag
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            plt.savefig(os.path.join(dirname, str(time.time())))
        else:
            plt.show()

    def table(self):
        """Formats the results into a table."""

        def val_to_str(val):
            if val == math.inf:
                return '-'
            if isinstance(val, float):
                return '{:.3}'.format(val)
            return str(val)

        operation_str = 'Operation'
        padding = 1
        max_row_heading_len = len(operation_str) + 2 * padding
        for row_heading, _ in self.results:
            max_row_heading_len = max(max_row_heading_len, len(self._table_format_index(*row_heading)))

        column_headings = []
        column_width_lower_bound = 4
        column_widths = [max_row_heading_len]
        _, example_row_value = next(iter(self.results))
        for column_heading, _ in example_row_value.items():
            column_headings.append(column_heading)
            column_widths.append(max(column_width_lower_bound, len(column_heading)) + 2 * padding)
        for _, row_value in self.results:
            for column_width_index, (_, column_value) in enumerate(row_value.items()):
                column_widths[column_width_index] = max(column_widths[column_width_index],
                                                        len(val_to_str(column_value)) + 2 * padding)

        heading_str = "|".join("{{:^{}}}".format(column_width) for column_width in column_widths)
        border_str = '+'.join('-' * column_width for column_width in column_widths)
        print(heading_str.format(operation_str, *column_headings))
        print(border_str)
        for row_heading, row_value in self.results:
            print("{{:<{}}}".format(max_row_heading_len).format(self._table_format_index(*row_heading)), end='')
            for column_width, (column_heading, column_value), true_column_heading in zip(column_widths[1:],
                                                                                         row_value.items(),
                                                                                         column_headings):
                assert column_heading == true_column_heading
                print("|{{:>{}}}".format(column_width).format(val_to_str(column_value)), end='')
            print('')
