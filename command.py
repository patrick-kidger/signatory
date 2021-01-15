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
"""Provides a set of commands for running tests, building documentation etc.

Find out more by running python command.py --help
"""


import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import webbrowser
#### DO NOT IMPORT NON-(STANDARD LIBRARY) MODULES HERE
# Instead, lazily import them inside the command.
# This allows all the commands that don't e.g. require a built version of Signatory to operate without it
# Exception: metadata, as we guarantee that that will not import anything that isn't standard library.
import metadata


def main():
    deviceparser = argparse.ArgumentParser(add_help=False)
    deviceparser.add_argument('-d', '--device', type=int, default=-1,
                              help="Which CUDA device to use, from a range of 0 upwards. May be set to -1 to not "
                                   "try to change the default device e.g. if no CUDA device is available. Defaults to "
                                   "-1.")

    parser = argparse.ArgumentParser(description="Runs various commands for building and testing Signatory.")
    subparsers = parser.add_subparsers(dest='command', help='Which command to run')

    version_parser = subparsers.add_parser('version', description="Prints the version")
    test_parser = subparsers.add_parser('test', parents=[deviceparser], description="Run tests")
    benchmark_parser = subparsers.add_parser('benchmark', parents=[deviceparser], description="Run speed benchmarks")
    docs_parser = subparsers.add_parser('docs', description="Build documentation")
    readme_parser = subparsers.add_parser('readme', description="Generate the README from the documentation.")
    workflows_parser = subparsers.add_parser('workflows', description="Generate the GitHub workflows from templates.")
    should_not_import_parser = subparsers.add_parser('should_not_import', description="Tests that Signatory _cannot_ "
                                                                                      "be imported.")

    version_parser.set_defaults(cmd=version)
    test_parser.set_defaults(cmd=test)
    benchmark_parser.set_defaults(cmd=benchmark)
    docs_parser.set_defaults(cmd=docs)
    readme_parser.set_defaults(cmd=readme)
    workflows_parser.set_defaults(cmd=workflows)
    should_not_import_parser.set_defaults(cmd=should_not_import)

    test_parser.add_argument('-t', '--test', default='', help="What to test. e.g. `test_signature.py::test_forward`. "
                                                              "Defaults to all tests.")
    test_parser.add_argument('-a', '--args', nargs=argparse.REMAINDER,
                             help="All other arguments are forwarded on to pytest.")

    benchmark_parser.add_argument('-e', '--noesig', action='store_false', dest='test_esig',
                                  help="Skip esig tests.")
    benchmark_parser.add_argument('-i', '--noiisignature', action='store_false', dest='test_iisignature',
                                  help="Skip iisignature tests.")
    benchmark_parser.add_argument('-g', '--nogpu', action='store_false', dest='test_signatory_gpu',
                                  help="Skip Signatory GPU tests.")
    benchmark_parser.add_argument('-m', '--measure', choices=('time', 'memory'), default='time',
                                  help="Whether to measure speed or memory usage. Defaults to time.")
    benchmark_parser.add_argument('-f', '--fns', choices=('all', 'sigf', 'sigb', 'logsigf', 'logsigb'), default='all',
                                  help="Which functions to run: signature forwards, signature backwards, logsignature "
                                       "forwards, logsignature backwards, or all of them. Defaults to all.")
    benchmark_parser.add_argument('-t', '--type', choices=('typical', 'depths', 'channels', 'small'), default='typical',
                                  help="What kind of benchmark to run. 'typical' tests on two typical size/depth "
                                       "combinations and prints the results as a table to stdout. 'depth' and "
                                       "'channels' are more thorough benchmarks (and will taking correspondingly "
                                       "longer to run!) testing multiple depths or multiple channels respectively. "
                                       "Defaults to typical.")
    benchmark_parser.add_argument('-o', '--output', choices=('table', 'graph', 'graphtable', 'none'), default='table',
                                  help="How to format the output. 'table' formats as a table, 'graph' formats as a "
                                       "graph. 'graphtable' does both. 'none' prints no output at all (perhaps if "
                                       "you're retrieving the results programmatically by importing command.py "
                                       "instead). Defaults to table.")
    benchmark_parser.add_argument('-s', '--save', action='store_true', help='Save the results to disk rather than '
                                                                            'displaying them. (By default tables are '
                                                                            'printed to stdout and graphs are opened '
                                                                            'in a new window.)')
                                  
    docs_parser.add_argument('-o', '--open', action='store_true',
                             help="Open the documentation in a web browser as soon as it is built.")

    args = parser.parse_args()

    # Have to do it this way for Python 2/3 compatability
    if hasattr(args, 'cmd'):
        return args.cmd(args)
    else:
        # No command was specified
        print("Please enter a command. Use -h to see available commands.")


_here = os.path.realpath(os.path.dirname(__file__))


def _get_device():
    import torch
    try:
        return 'CUDA device ' + str(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        return 'no CUDA device'


class _NullContext(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def version(args):
    print(metadata.version)


def test(args):
    try:
        import iisignature  # fail fast here if necessary
    except ImportError:
        raise ImportError("The iisignature package is required for running tests. It can be installed via 'pip "
                          "install iisignature'")
    import pytest
    import torch
    with torch.cuda.device(args.device) if args.device != -1 else _NullContext():
        print('Using ' + _get_device())
        pytest_args = [os.path.join(_here, 'test', args.test)]
        pytest_args.extend(['--tb=long', '-ra', '--durations=0'])
        if args.args is not None:
            pytest_args.extend(args.args)
        return pytest.main(pytest_args) == 0


def benchmark(args):
    """Run speed benchmarks."""
    try:
        import iisignature  # fail fast here if necessary
    except ImportError:
        raise ImportError("The iisignature package is required for running tests. It can be installed via 'pip "
                          "install iisignature'")
    try:
        import esig  # fail fast here if necessary
    except ImportError:
        raise ImportError("The esig package is required for running tests. It can be installed via 'pip "
                          "install esig'")

    import benchmark.benchmark as bench
    import torch

    if args.measure == 'time':
        measure = bench.Measurables.time
    elif args.measure == 'memory':
        measure = bench.Measurables.memory
        print("Note: memory benchmarking seems to be pretty flakey. Evidence of this is that the values we measure for "
              "iisignature are much lower than that reported in the iisignature paper itself (arxiv:1802.08252).")
    else:
        raise RuntimeError

    if args.fns == 'all':
        fns = bench.Functions.all_fns
    elif args.fns == 'sigf':
        fns = bench.Functions.signature_forward_fns
    elif args.fns == 'sigb':
        fns = bench.Functions.signature_backward_fns
    elif args.fns == 'logsigf':
        fns = bench.Functions.logsignature_forward_fns
    elif args.fns == 'logsigb':
        fns = bench.Functions.logsignature_backward_fns
    else:
        raise RuntimeError

    if args.type == 'typical':
        type_ = bench.Types.typical
    elif args.type == 'depths':
        type_ = bench.Types.depths
    elif args.type == 'channels':
        type_ = bench.Types.channels
    elif args.type == 'small':
        type_ = bench.Types.small
    else:
        raise RuntimeError

    try:
        runner = bench.BenchmarkRunner(type_=type_,
                                       test_esig=args.test_esig,
                                       test_iisignature=args.test_iisignature,
                                       test_signatory_gpu=args.test_signatory_gpu,
                                       measure=measure,
                                       fns=fns)
        if args.output in ('graph', 'graphsave'):
            runner.check_graph()
    except bench.InvalidBenchmark as e:
        print(str(e))
    else:
        with torch.cuda.device(args.device) if args.device != -1 else _NullContext():
            print('Using ' + _get_device())
            runner.run()

        if args.output in ('table', 'graphtable'):
            runner.table(save=args.save)
        if args.output in ('graph', 'graphtable'):
            runner.graph(save=args.save)

        return runner

    
def docs(args=()):
    """Build the documentation. After it has been built then it can be found in ./docs/_build/html/index.html/
    The package 'py2annotate' will need to be installed. It can be installed via `pip install py2annotate`
    Note that the documentation is already available online at https://signatory.readthedocs.io
    """
    try:
        import py2annotate  # fail fast here if necessary
    except ImportError:
        raise ImportError("The py2annotate package is required for running tests. It can be installed via 'pip "
                          "install py2annotate'")
    try:
        shutil.rmtree(os.path.join(_here, "docs", "_build"))
    except FileNotFoundError:
        pass
    subprocess.Popen("sphinx-build -M html {} {}".format(os.path.join(_here, "docs"), os.path.join(_here, "docs", "_build"))).wait()
    if args.open:
        webbrowser.open_new_tab('file:///{}'.format(os.path.join(_here, 'docs', '_build', 'html', 'index.html')))


def workflows(args=()):
    """The GitHub workflows are generated from templates."""
    sys.path.insert(0, os.path.join(_here, '.github', 'workflows_templates'))
    import from_template
    from_template.main()
    sys.path = sys.path[1:]

    
def readme(args=()):
    """The readme is generated automatically from the documentation."""
    
    outs = []
    includestr = '.. include::'
    on = '.. command.readme on'
    off = '.. command.readme off'
    insert = '.. command.readme insert '  # space at end is important
    reference = re.compile(r'^\.\. [\w-]+:$')

    def parse_file(filename):
        out_data = []
        with io.open(filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
            skipping = False
            for line in data:
                stripline = line.strip()
                if stripline.startswith(on):
                    skipping = False
                elif stripline.startswith(off):
                    skipping = True
                elif skipping:
                    pass
                elif reference.match(stripline):
                    pass
                else:
                    if stripline.startswith(insert):
                        indent = line.find(insert)
                        out_line = line[:indent] + line[indent + len(insert):]
                    elif stripline.startswith(includestr):
                        # [1:] to remove the leading / at the start; otherwise ends up being parsed as root
                        subfilename = stripline[len(includestr):].strip()[1:]
                        out_line = parse_file(os.path.join(_here, 'docs', subfilename))
                    else:
                        out_line = line
                    if ':ref:' in out_line:
                        raise RuntimeError('refs not supported')
                    out_line = out_line.replace('|version|', metadata.version)
                    out_data.append(out_line)
        return ''.join(out_data)

    def read_from_files(filenames):
        for filename in filenames:
            filename = os.path.join(_here, filename)
            outs.append(parse_file(filename))

    read_from_files([os.path.join(_here, 'docs', 'index.rst'),
                     os.path.join(_here, 'docs', 'pages', 'understanding', 'whataresignatures.rst'),
                     os.path.join('docs', 'pages', 'usage', 'installation.rst')])

    outs.append("Documentation\n"
                "#############\n"
                "The documentation is available `here <https://signatory.readthedocs.io>`__.")

    outs.append("Example\n"
                "#######\n"
                "Usage is straightforward. As a simple example,\n"
                "\n"
                ".. code-block:: python\n"
                "\n"
                "    import signatory\n"
                "    import torch\n"
                "    batch, stream, channels = 1, 10, 2\n"
                "    depth = 4\n"
                "    path = torch.rand(batch, stream, channels)\n"
                "    signature = signatory.signature(path, depth)\n"
                "    # signature is a PyTorch tensor\n"
                "\n"
                "For further examples, see the `documentation <https://signatory.readthedocs.io/en/latest/pages/examples/examples.html>`__.")

    read_from_files([os.path.join(_here, 'docs', 'pages', 'miscellaneous', 'citation.rst')])

    with io.open(os.path.join(_here, 'README.rst'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(outs))


def should_not_import(args=()):
    """Tests that we _can't_ import Signatory. Doing this before we install it ensures that we're definitely testing
    the version we install and not some other version we can accidentally see.
    """

    try:
        import signatory
    except ImportError as e:
        # Python 3 vs Python 2 difference
        return str(e) in ('No module named signatory', "No module named 'signatory'")
    else:
        return False
        
            
if __name__ == '__main__':
    result = main()
    if isinstance(result, bool):
        sys.exit(not result)
