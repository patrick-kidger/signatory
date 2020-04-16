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
"""Provides speed and memory benchmarks against esig and iisignature."""

import collections as co
import datetime
import io
import itertools as it
import math
import matplotlib.pyplot as plt
import os
import subprocess
import torch

from . import helpers


class InvalidBenchmark(Exception):
    """Raised to indicate a set of options that do not define a valid benchmark."""
    def __init__(self, msg, *args, **kwargs):
        msg += ' Run `python command.py benchmark --help` for details on how to do this.'
        super(InvalidBenchmark, self).__init__(msg, *args, **kwargs)


# This section specifies various constants and options that can be selected for benchmarking

# Here we essentially specify the different libraries we can test.
# We call them 'columns' because we can also include ratios between the libraries as well.
class RatioColumns(helpers.Container):
    speedup_cpu_str = 'Ratio CPU (parallel)'
    speedup_cpu_no_parallel_str = 'Ratio CPU (no parallel)'
    speedup_gpu_str = 'Ratio GPU'


class Columns(RatioColumns):
    signatory_cpu_str = 'Signatory CPU (parallel)'
    signatory_cpu_no_parallel_str = 'Signatory CPU (no parallel)'
    signatory_gpu_str = 'Signatory GPU'
    iisignature_str = 'iisignature'
    esig_str = 'esig'


colours = {Columns.signatory_cpu_str: 'b',
           Columns.signatory_cpu_no_parallel_str: 'c',
           Columns.signatory_gpu_str: 'g',
           Columns.iisignature_str: 'r',
           Columns.esig_str: 'm'}


# Now we specify all the different functions we could benchmark
_signature_forward_fns = co.OrderedDict([(Columns.esig_str, 'esig_signature_forward'),
                                         (Columns.iisignature_str, 'iisignature_signature_forward'),
                                         (Columns.signatory_cpu_no_parallel_str, 'signatory_signature_forward_no_parallel'),
                                         (Columns.signatory_cpu_str, 'signatory_signature_forward'),
                                         (Columns.signatory_gpu_str, 'signatory_signature_forward_gpu')])

_signature_backward_fns = co.OrderedDict([(Columns.esig_str, 'esig_signature_backward'),
                                          (Columns.iisignature_str, 'iisignature_signature_backward'),
                                          (Columns.signatory_cpu_no_parallel_str, 'signatory_signature_backward_no_parallel'),
                                          (Columns.signatory_cpu_str, 'signatory_signature_backward'),
                                          (Columns.signatory_gpu_str, 'signatory_signature_backward_gpu')])

_logsignature_forward_fns = co.OrderedDict([(Columns.esig_str, 'esig_logsignature_forward'),
                                            (Columns.iisignature_str, 'iisignature_logsignature_forward'),
                                            (Columns.signatory_cpu_no_parallel_str, 'signatory_logsignature_forward_no_parallel'),
                                            (Columns.signatory_cpu_str, 'signatory_logsignature_forward'),
                                            (Columns.signatory_gpu_str, 'signatory_logsignature_forward_gpu')])

_logsignature_backward_fns = co.OrderedDict([(Columns.esig_str, 'esig_logsignature_backward'),
                                             (Columns.iisignature_str, 'iisignature_logsignature_backward'),
                                             (Columns.signatory_cpu_no_parallel_str, 'signatory_logsignature_backward_no_parallel'),
                                             (Columns.signatory_cpu_str, 'signatory_logsignature_backward'),
                                             (Columns.signatory_gpu_str, 'signatory_logsignature_backward_gpu')])


class BackwardFunctions(helpers.Container):
    signature_backward_fns = {'Signature backward': _signature_backward_fns}
    logsignature_backward_fns = {'Logsignature backward': _logsignature_backward_fns}


class Functions(BackwardFunctions):
    signature_forward_fns = {'Signature forward': _signature_forward_fns}
    logsignature_forward_fns = {'Logsignature forward': _logsignature_forward_fns}
    all_fns = co.OrderedDict()
    all_fns.update(signature_forward_fns)
    all_fns.update(BackwardFunctions.signature_backward_fns)
    all_fns.update(logsignature_forward_fns)
    all_fns.update(BackwardFunctions.logsignature_backward_fns)


# These are the things we can measure
class Measurables(helpers.Container):
    time = 'time'
    memory = 'memory'


# These are the different predefined size/depth combinations that we can produce benchmarks for.
class Types(helpers.Container):
    class typical(object):
        """Tests two typical use cases."""
        sizes = ((32, 128, 8),)
        depths = (5, 7)

    class channels(object):
        """Tests a number of channels for a fixed depth."""
        sizes = ((32, 128, 2), (32, 128, 3), (32, 128, 4), (32, 128, 5), (32, 128, 6), (32, 128, 7))
        depths = (7,)

    class depths(object):
        """Tests depths for a fixed number of channels."""
        sizes = ((32, 128, 4),)
        depths = (2, 3, 4, 5, 6, 7, 8, 9)

    class small(object):
        """Tests on very small data. This doesn't given meaningful results - the overhead of PyTorch/NumPy/etc. ends up
        giving a greater noise than there is signal - but it serves to test the benchmark framework itself.
        """
        sizes = ((1, 2, 2),)
        depths = (2, 3, 4, 5)

# Done with specifying constants and options


class BenchmarkRunner(object):
    """Runs all functions across all libraries and records their times or memory usage for multiple sizes and depths."""

    def __init__(self, type_, test_esig, test_iisignature, test_signatory_gpu, measure, fns, **kwargs):
        assert type_ in Types
        assert measure in Measurables
        assert fns in Functions

        if measure is Measurables.memory and test_signatory_gpu:
            raise InvalidBenchmark('Memory comparisons for Signatory GPU are not meaningful, as everything else '
                                   'operates on the CPU. Please disable GPU testing.')

        if fns in BackwardFunctions and test_esig:
            raise InvalidBenchmark('esig does not support backward computations. Please disable esig testing.')

        self.sizes = type_.sizes
        self.depths = type_.depths
        self.test_esig = test_esig
        self.test_iisignature = test_iisignature
        self.test_signatory_gpu = test_signatory_gpu
        self.measure = measure
        self.fns = fns

        self.title_string = list(fns.keys())[0] + ': ' + measure
        self.dirname = self.title_string.lower().replace(' ', '_').replace(':', '') + '_' + type_.__name__

        self._results = None

        super(BenchmarkRunner, self).__init__(**kwargs)

    @property
    def results(self):
        return self._results

    def check_graph(self):
        """Checks whether or not this benchmark is suitable for being plotted as a graph."""

        if len(self.sizes) > 1 and len(self.depths) > 1:
            raise InvalidBenchmark("Cannot output as graph with multiple sizes and multiple depths.")
        if len(list(self.fns.keys())) > 1:
            raise InvalidBenchmark("Cannot output as graph with multiple functions.")
        batch_size, stream_size, _ = next(iter(self.sizes))
        for size in self.sizes:
            if size[0] != batch_size or size[1] != stream_size:
                raise InvalidBenchmark("Cannot output as graph with multiple batch or stream sizes.")

    def run(self):
        """Runs the benchmarks."""

        running = True
        results = helpers.namedarray(len(self.fns), len(self.sizes), len(self.depths))
        for fn_name, fn_dict in self.fns.items():
            for size in self.sizes:
                for depth in self.depths:
                    running, results[fn_name, size, depth] = self._run_test(fn_name, fn_dict, size, depth, running)
        self._results = results

    def _run_test(self, fn_name, fn_dict, size, depth, running):
        """Runs a particular function across multiple different libraries and records their speed or memory usage."""

        column_results = co.OrderedDict()

        for library_name, library_module_name in fn_dict.items():
            if (not self.test_esig) and (library_name is Columns.esig_str):
                continue
            if (not self.test_signatory_gpu) and (library_name is Columns.signatory_gpu_str):
                continue
            if (not self.test_iisignature) and (library_name is Columns.iisignature_str):
                continue

            result = math.inf
            if running:
                print(self._table_format_index(fn_name, size, depth), library_name)
                try:
                    if self.measure is Measurables.time:
                        result = self._time(library_module_name, size, depth)
                    elif self.measure is Measurables.memory:
                        result = self._memory(library_module_name, size, depth)
                    else:
                        raise RuntimeError
                except KeyboardInterrupt:
                    running = False
            column_results[library_name] = result

        other_best = math.inf
        if self.test_iisignature:
            other_best = min(column_results[Columns.iisignature_str], other_best)
        if self.test_esig:
            other_best = min(column_results[Columns.esig_str], other_best)
        try:
            column_results[Columns.speedup_cpu_str] = other_best / column_results[Columns.signatory_cpu_str]
        except ZeroDivisionError:
            column_results[Columns.speedup_cpu_str] = math.inf
        try:
            column_results[Columns.speedup_cpu_no_parallel_str] = other_best / column_results[Columns.signatory_cpu_no_parallel_str]
        except ZeroDivisionError:
            column_results[Columns.speedup_cpu_no_parallel_str] = math.inf
        if self.test_signatory_gpu:
            try:
                column_results[Columns.speedup_gpu_str] = other_best / column_results[Columns.signatory_gpu_str]
            except ZeroDivisionError:
                column_results[Columns.speedup_gpu_str] = math.inf

        return running, column_results

    @classmethod
    def _time(cls, library_module_name, size, depth):
        return cls._run_file(library_module_name, 'time_', size, depth)

    @classmethod
    def _memory(cls, library_module_name, size, depth):
        result = 0
        for _ in range(5):
            stdout = cls._run_file(library_module_name, 'memory', size, depth)
            if stdout == 0:
                # Sometimes things bug out and give a zero memory reading.
                # I'm not sure why things seem to be flaky
                continue
            # Take the maximum, as we sample based on some frequency, and can easily miss a peak when doing this over
            # just one run.
            # (Yeah this isn't ideal.)
            result = max(result, stdout)
        if result == 0:
            result = math.inf
        return result

    @staticmethod
    def _run_file(library_module_name, filename, size, depth):
        if torch.cuda.is_available():
            device = int(torch.cuda.current_device())
        else:
            device = -1
        p = subprocess.run('python -m {}.{} {} {} {} {}'
                           ''.format(__package__,
                                     filename,
                                     library_module_name,
                                     str(size).replace(' ', '').replace('(', '').replace(')', ''),
                                     depth,
                                     device),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        stderr = p.stderr.decode()
        if stderr != '':
            print('Error:')
            print('------')
            print(stderr)
            print('')
            raise RuntimeError("Error in " + library_module_name)

        stdout = p.stdout.decode().strip()
        for line in stdout.split('\n'):
            if 'Legitimate' not in line:
                stdout = line
                break
        return float(stdout)

    @staticmethod
    def _table_format_index(fn_name, size, depth):
        return "{}, size {}, depth {}".format(fn_name, size, depth)

    def _graph_format_index(self, size, depth):
        if len(self.sizes) > 1:
            return size[-1]
        elif len(self.depths) > 1:
            return depth

    def graph(self, save=False, log=True):
        """Plots the result as a graph."""

        self.check_graph()

        fig = plt.figure()
        ax = fig.gca()

        _, example_row_value = next(iter(self.results))
        x_axes = [[] for _ in range(len(example_row_value))]
        y_axes = [[] for _ in range(len(example_row_value))]
        labels = []
        for column_heading in example_row_value.keys():
            labels.append(column_heading)
        for (fn_name, size, depth), row_value in self.results:
            for x_axis, y_axis, (column_heading, column_value) in zip(x_axes, y_axes, row_value.items()):
                x_axis.append(self._graph_format_index(size, depth))
                y_axis.append(column_value)

        for x_axis, y_axis, label in zip(x_axes, y_axes, labels):
            if label in RatioColumns:
                continue
            ax.plot(x_axis, y_axis, label=label, color=colours[label])

        ncol = 2

        # From https://stackoverflow.com/a/10101532/12254339
        def flip(items):
            return it.chain(*[items[i::ncol] for i in range(ncol)])

        handles, labels = ax.get_legend_handles_labels()

        legend = ax.legend(flip(handles), flip(labels),
                           mode='expand', ncol=ncol, bbox_to_anchor=(0, 1.02, 1, 0), loc='lower left',
                           borderaxespad=0.)
        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
        ax.set_title(self.title_string, y=0.03 + legend_bbox.inverse_transformed(ax.transAxes).ymax)
        if self.measure is Measurables.time:
            ax.set_ylabel("Time in seconds")
        elif self.measure is Measurables.memory:
            ax.set_ylabel("Memory usage in MB")
        else:
            raise RuntimeError
        if len(self.sizes) > 1:
            ax.set_xlabel("Number of channels")
        elif len(self.depths) > 1:
            ax.set_xlabel("Depth")
        else:
            raise RuntimeError
        if log:
            ax.set_yscale('log')
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(range(int(math.ceil(start)), int(math.floor(end)) + 1))
        plt.tight_layout()
        if save:
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            plt.savefig(os.path.join(self.dirname, str(datetime.datetime.utcnow())) + '.png')
        else:
            plt.show()

    def table(self, save=False):
        """Formats the results into a table."""

        out_str = ''

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
        out_str += heading_str.format(operation_str, *column_headings) + '\n'
        out_str += border_str + '\n'
        for row_heading, row_value in self.results:
            out_str += "{{:<{}}}".format(max_row_heading_len).format(self._table_format_index(*row_heading))
            for column_width, (column_heading, column_value), true_column_heading in zip(column_widths[1:],
                                                                                         row_value.items(),
                                                                                         column_headings):
                assert column_heading == true_column_heading
                out_str += "|{{:>{}}}".format(column_width).format(val_to_str(column_value))
            out_str += '\n'

        if save:
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            with io.open(os.path.join(self.dirname, str(datetime.datetime.utcnow())) + '.txt', 'w', encoding='utf-8') as f:
                f.write(out_str)
        else:
            print(out_str)
