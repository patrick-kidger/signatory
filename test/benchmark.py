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
import gc
import iisignature
import itertools as it
import functools as ft
import math
import matplotlib.pyplot as plt
import memory_profiler as mem
import numpy as np
import signatory
import timeit
import torch

try:
    # Being run via command.py
    from . import compatibility as compat
except ImportError:
    # Being run via tests
    import compatibility as compat


@compat.lru_cache(maxsize=None)
def prepare(channels, depth):
    """A cache wrapper around iisignature.prepare, since it takes a long time."""
    return iisignature.prepare(channels, depth)


class EsigError(Exception):
    pass


# All of the following functions wrap the analogous functions from each package, to provide a consistent interface for
# testing against each other.


class BenchmarkBase(object):
    def __init__(self, depth, repeat, number):
        self.depth = depth
        self.repeat = repeat
        self.number = number

    def stmt(self):
        raise NotImplementedError

    def action(self, measure):
        if measure == 'time':
            return self.time()
        elif measure == 'memory':
            return self.memory()
        else:
            raise ValueError("I don't know how to measure '{}'".format(measure))

    def time(self):
        try:
            self.stmt()  # warm up
        except EsigError:
            return [math.inf]
        return timeit.Timer(stmt=self.stmt).repeat(repeat=self.repeat, number=self.number)

    def memory(self):
        try:
            # load things that need loading; this may use extra memory
            self.stmt()
        except EsigError:
            return [math.inf]

        gc.collect()
        if hasattr(self, 'gpu') and self.gpu:
            torch.cuda.reset_max_memory_allocated()
            self.stmt()
            return [torch.cuda.max_memory_allocated()]
        else:
            background_usage = mem.memory_usage((lambda: None, (), {}), interval=0.000001)
            # max because memory_used is a list of the memory used over 0.1s increments
            # megabytes -> bytes
            background_usage = max(background_usage) * 10 ** 6
            # we don't do any repeats because we expect memory usage to be pretty independent of system state, unlike
            # speed
            memory_used = mem.memory_usage((self.stmt, (), {}), interval=0.000001)
            memory_used = max(memory_used) * 10 ** 6
            return [memory_used - background_usage]


class esig_signature_forward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        super(esig_signature_forward, self).__init__(**kwargs)

    def stmt(self):
        if not len(esig.tosig.stream2logsig(self.path[0], self.depth)):
            raise EsigError

        for batch_elem in self.path:
            esig.tosig.stream2sig(batch_elem, self.depth)


class esig_logsignature_forward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        super(esig_logsignature_forward, self).__init__(**kwargs)

    def stmt(self):
        if not len(esig.tosig.stream2logsig(self.path[0], self.depth)):
            raise EsigError

        for batch_elem in self.path:
            esig.tosig.stream2logsig(batch_elem, self.depth)


class esig_signature_backward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        super(esig_signature_backward, self).__init__(**kwargs)

    def stmt(self):
        # esig doesn't provide this operation.
        raise EsigError


class esig_logsignature_backward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        super(esig_logsignature_backward, self).__init__(**kwargs)

    def stmt(self):
        # esig doesn't provide this operation.
        raise EsigError


class iisignature_signature_forward(BenchmarkBase):
    def __init__(self, size, stream=False, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        self.stream = stream
        super(iisignature_signature_forward, self).__init__(**kwargs)

    def stmt(self):
        if self.stream:
            iisignature.sig(self.path, self.depth, 2)
        else:
            iisignature.sig(self.path, self.depth, 0)


class iisignature_logsignature_forward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        super(iisignature_logsignature_forward, self).__init__(**kwargs)
        self.prep = prepare(self.path.shape[-1], self.depth)

    def stmt(self):
        iisignature.logsig(self.path, self.prep)


class iisignature_signature_backward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        super(iisignature_signature_backward, self).__init__(**kwargs)
        signature = iisignature.sig(self.path, self.depth)
        self.grad = torch.rand_like(torch.tensor(signature, dtype=torch.float)).numpy()

    def stmt(self):
        iisignature.sigbackprop(self.grad, self.path, self.depth)


class iisignature_logsignature_backward(BenchmarkBase):
    def __init__(self, size, **kwargs):
        self.path = torch.rand(size, dtype=torch.float).numpy()
        super(iisignature_logsignature_backward, self).__init__(**kwargs)
        self.prep = prepare(self.path.shape[-1], self.depth)
        logsignature = iisignature.logsig(self.path, self.prep)
        self.grad = torch.rand_like(torch.tensor(logsignature, dtype=torch.float)).numpy()

    def stmt(self):
        iisignature.logsigbackprop(self.grad, self.path, self.prep)


class signatory_signature_forward(BenchmarkBase):
    def __init__(self, size, stream=False, gpu=False, **kwargs):
        if gpu:
            self.path = torch.rand(size, dtype=torch.float, device='cuda')
        else:
            self.path = torch.rand(size, dtype=torch.float)
        super(signatory_signature_forward, self).__init__(**kwargs)
        self.stream = stream
        self.gpu = gpu

    def stmt(self):
        signatory.signature(self.path, self.depth, stream=self.stream)
        if self.gpu:
            torch.cuda.synchronize()


class signatory_logsignature_forward(BenchmarkBase):
    def __init__(self, size, stream=False, gpu=False, mode='words', **kwargs):
        if gpu:
            self.path = torch.rand(size, dtype=torch.float, device='cuda')
        else:
            self.path = torch.rand(size, dtype=torch.float)
        super(signatory_logsignature_forward, self).__init__(**kwargs)
        # ensure that we're doing a fair test by caching if we can
        # (equivalent to the call to 'prepare' in iisignature)
        signatory.LogSignature(self.depth, mode=mode, stream=stream)(self.path)
        self.stream = stream
        self.gpu = gpu
        self.mode = mode

    def stmt(self):
        signatory.LogSignature(self.depth, mode=self.mode, stream=self.stream)(self.path)
        if self.gpu:
            torch.cuda.synchronize()


class signatory_signature_backward(BenchmarkBase):
    def __init__(self, size, gpu=False, **kwargs):
        if gpu:
            self.path = torch.rand(size, dtype=torch.float, device='cuda', requires_grad=True)
        else:
            self.path = torch.rand(size, dtype=torch.float, requires_grad=True)
        super(signatory_signature_backward, self).__init__(**kwargs)
        self.gpu = gpu
        self.signature = signatory.signature(self.path, self.depth)
        self.grad = torch.rand_like(self.signature)

    def stmt(self):
        self.signature.backward(self.grad, retain_graph=True)
        if self.gpu:
            torch.cuda.synchronize()


class signatory_logsignature_backward(BenchmarkBase):
    def __init__(self, size, stream=False, gpu=False, mode='words', **kwargs):
        if gpu:
            self.path = torch.rand(size, dtype=torch.float, device='cuda', requires_grad=True)
        else:
            self.path = torch.rand(size, dtype=torch.float, requires_grad=True)
        super(signatory_logsignature_backward, self).__init__(**kwargs)
        self.logsignature = signatory.LogSignature(self.depth, mode=mode, stream=stream)(self.path)
        self.grad = torch.rand_like(self.logsignature)
        self.stream = stream
        self.gpu = gpu
        self.mode = mode

    def stmt(self):
        self.logsignature.backward(self.grad, retain_graph=True)
        if self.gpu:
            torch.cuda.synchronize()


signatory_cpu_str = 'Signatory CPU'
signatory_gpu_str = 'Signatory GPU'
iisignature_str = 'iisignature'
esig_str = 'esig'
speedup_cpu_str = 'Ratio CPU'
speedup_gpu_str = 'Ratio GPU'
speedup_str = 'Ratio'


signature_forward_fns = co.OrderedDict([(signatory_cpu_str, signatory_signature_forward),
                                        (signatory_gpu_str, ft.partial(signatory_signature_forward, gpu=True)),
                                        (iisignature_str, iisignature_signature_forward),
                                        (esig_str, esig_signature_forward)])


signature_backward_fns = co.OrderedDict([(signatory_cpu_str, signatory_signature_backward),
                                         (signatory_gpu_str, ft.partial(signatory_signature_backward, gpu=True)),
                                         (iisignature_str, iisignature_signature_backward),
                                         (esig_str, esig_signature_backward)])

logsignature_forward_fns = co.OrderedDict([(signatory_cpu_str, signatory_logsignature_forward),
                                           (signatory_gpu_str, ft.partial(signatory_logsignature_forward, gpu=True)),
                                           (iisignature_str, iisignature_logsignature_forward),
                                           (esig_str, esig_logsignature_forward)])

logsignature_backward_fns = co.OrderedDict([(signatory_cpu_str, signatory_logsignature_backward),
                                            (signatory_gpu_str, ft.partial(signatory_logsignature_backward, gpu=True)),
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
    """Just a minimal helper for our needs elsewhere in this file. There are definitely fancier solutions available."""
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
        if self.fns is all_fns:
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
            test = library_fn(size=size, depth=depth, repeat=self.repeat, number=self.number)
            library_results[library_name] = min(test.action(self.measure))

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
                          depths=(5, 6, 7, 8, 9))
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

    def graph(self, log=True):
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
        ax.legend(loc='upper left')
        ax.set_title(list(self.fns.keys())[0])
        ax.set_ylabel("Time in seconds")
        if len(self.sizes) > 1:
            ax.set_xlabel("Number of channels")
        elif len(self.depths) > 1:
            ax.set_xlabel("Depth")
            if log:
                ax.set_yscale('log')
        else:
            raise RuntimeError
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(range(int(math.ceil(start)), int(math.floor(end)) + 1))
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
