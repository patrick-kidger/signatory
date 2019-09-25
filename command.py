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
try:
    # Python 3
    from shlex import quote as shlex_quote
except ImportError:
    # Python 2
    from pipes import quote as shlex_quote
import shutil
try:
    # Python 2 on POSIX
    import subprocess32 as subprocess
except ImportError:
    # Python 3 on anything
    import subprocess
# (no support for Python 2 on Windows but that's fine because PyTorch doesn't support that anyway)

import sys
import webbrowser
#### DO NOT IMPORT NON-(STANDARD LIBRARY) MODULES HERE
# Instead, lazily import them inside the command.
# This allows all the commands that don't e.g. require a built version of Signatory to operate without it
# Exception: metadata, as we guarantee that that will not import anything that isn't standard library.
import metadata


def main():
    deviceparser = argparse.ArgumentParser(add_help=False)
    deviceparser.add_argument('-d', '--device', type=int, default=0, help="Which CUDA device to use. Defaults to 0")

    parser = argparse.ArgumentParser(description="Runs various commands for building and testing Signatory.")
    parser.add_argument('-v', '--version', action='version', version=metadata.version)
    subparsers = parser.add_subparsers(dest='command', help='Which command to run')
    
    test_parser = subparsers.add_parser('test', parents=[deviceparser], description="Run tests")
    benchmark_parser = subparsers.add_parser('benchmark', parents=[deviceparser], description="Run speed benchmarks")
    docs_parser = subparsers.add_parser('docs', description="Build documentation")
    genreadme_parser = subparsers.add_parser('genreadme', description="Generate the README from the documentation.")

    test_parser.set_defaults(cmd=test)
    benchmark_parser.set_defaults(cmd=benchmark)
    docs_parser.set_defaults(cmd=docs)
    genreadme_parser.set_defaults(cmd=genreadme)

    test_parser.add_argument('-f', '--failfast', action='store_true', help='Stop tests on first failure.')
    test_parser.add_argument('-n', '--nonames', action='store_false', dest='names',
                             help="Don't print names and start time of the tests being run.")
    test_parser.add_argument('-t', '--notimes', action='store_false', dest='times',
                             help="Don't print the overall times of the tests that have been run.")

    benchmark_parser.add_argument('-e', '--noesig', action='store_false', dest='esig',
                                  help="Skip esig tests as esig is typically very slow.")
    benchmark_parser.add_argument('-t', '--type', choices=('typical', 'depths', 'channels', 'small'), default='typical',
                                  help="What kind of benchmark to run. 'typical' tests on two typical size/depth "
                                       "combinations and prints the results as a table to stdout. 'depth' and "
                                       "'channels' are more thorough benchmarks (and will taking correspondingly "
                                       "longer to run!) testing multiple depths or multiple channels respectively.")
    benchmark_parser.add_argument('-o', '--output', choices=('table', 'graph', 'none'), default='table',
                                  help="How to format the output. 'table' formats as a table, 'graph' formats as a "
                                       "graph. 'none' prints no output at all (perhaps if you're retrieving the results"
                                       "programmatically by importing command.py instead).")
    benchmark_parser.add_argument('-f', '--fns', choices=('sigf', 'sigb', 'logsigf', 'logsigb', 'all'), default='all',
                                  help="Which functions to run: signature forwards, signature backwards, logsignature "
                                       "forwards, logsignature backwards, or all of them.")
    benchmark_parser.add_argument('-m', '--measure', choices=('time', 'memory'), default='time',
                                  help="Whether to measure speed or memory usage.")
    benchmark_parser.add_argument('-r', '--transpose', action='store_true',
                                  help="Whether to pass in transposed tensors (corresponding to batch-last input).")
                                  
    docs_parser.add_argument('-o', '--open', action='store_true',
                             help="Whether to open the documentation in a web browser as soon as it is built.")

    args = parser.parse_args()

    # Have to do it this way for Python 2/3 compatability
    if hasattr(args, 'cmd'):
        args.cmd(args)
    else:
        # No command was specified
        print("Please enter a command. Use -h to see available commands.")


here = os.path.realpath(os.path.dirname(__file__))


def _run_commands(*commands, **kwargs):
    """Runs a collection of commands in a shell. Should be platform-agnostic."""

    # For Python 2 compatability.
    stdout = kwargs.pop('stdout', True)
    if kwargs:
        raise ValueError("kwargs {} not understood".format(kwargs))

    print_commands = ['echo {}'.format(shlex_quote("(running) " + command)) for command in commands]
    all_commands = []
    for i in range(len(commands)):
        all_commands.append(print_commands[i])
        if stdout:
            all_commands.append(commands[i])
        else:
            if 'win' in sys.platform:
                null = ' >nul'
            else:
                null = ' > /dev/null'
            all_commands.append(commands[i] + null)
    completed_process = subprocess.run(' && '.join(all_commands), shell=True)
    return completed_process.returncode
    
    
def test(args):
    """Run all tests.
    The package 'iisignature' will need to be installed, to test against.
    It can be installed via `pip install iisignature`
    """
    try:
        import iisignature  # fail fast here if necessary
    except ImportError:
        raise ImportError("The iisignature package is required for running tests. It can be installed via 'pip "
                          "install iisignature'")
    import test.runner
    import torch
    with torch.cuda.device(args.device):
        print('Using device {}'.format(args.device))
        test.runner.main(failfast=args.failfast, times=args.times, names=args.names)


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
                          
    import test.benchmark as bench
    import torch
    with torch.cuda.device(args.device):
        print('Using device {}'.format(args.device))
        if args.type == 'typical':
            runner = bench.BenchmarkRunner.typical(transpose=args.transpose, test_esig=args.esig, fns=args.fns)
        elif args.type == 'depths':
            runner = bench.BenchmarkRunner.depths(transpose=args.transpose, test_esig=args.esig, fns=args.fns)
        elif args.type == 'channels':
            runner = bench.BenchmarkRunner.channels(transpose=args.transpose, test_esig=args.esig, fns=args.fns)
        elif args.type == 'small':
            runner = bench.BenchmarkRunner.small(transpose=args.transpose, test_esig=args.esig, fns=args.fns)
        else:
            raise RuntimeError
        if args.output == 'graph':
            runner.check_graph()
        runner.run(args.measure)
        if args.output == 'graph':
            runner.graph()
        elif args.output == 'table':
            runner.table()
        elif args.output == 'none':
            pass
        else:
            raise RuntimeError
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
        shutil.rmtree(os.path.join(here, "docs", "_build"))
    except FileNotFoundError:
        pass
    _run_commands("sphinx-build -M html {} {}".format(os.path.join(here, "docs"), os.path.join(here, "docs", "_build")))
    if args.open:
        webbrowser.open_new_tab('file:///{}'.format(os.path.join(here, 'docs', '_build', 'html', 'index.html')))

    
def genreadme(args=()):
    """The readme is generated automatically from the documentation."""
    
    outs = []
    includestr = '.. include::'
    on = '.. genreadme on'
    off = '.. genreadme off'
    insert = '.. genreadme insert '  # space at end is important
    reference = re.compile(r'^\.\. [\w-]+:$')
    
    inserts = {'install_from_source': "Installation from source is also possible; please consult the `documentation "
                                      "<https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#usage-install-from-source>`__."}

    def parse_file(filename):
        out_data = []
        with io.open(filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
            skipping = False
            for line in data:
                stripline = line.strip()
                if stripline == on:
                    skipping = False
                elif stripline == off:
                    skipping = True
                elif skipping:
                    pass
                elif reference.match(stripline):
                    pass
                else:
                    if stripline.startswith(insert):
                        out_line = inserts[stripline[len(insert):]]
                    elif stripline.startswith(includestr):
                        # [1:] to remove the leading / at the start; otherwise ends up being parsed as root
                        subfilename = stripline[len(includestr):].strip()[1:]
                        out_line = parse_file(os.path.join(here, 'docs', subfilename))
                    else:
                        out_line = line
                    if ':ref:' in out_line:
                        raise RuntimeError('refs not supported')
                    out_data.append(out_line)
        return ''.join(out_data)

    def read_from_files(filenames):
        for filename in filenames:
            filename = os.path.join(here, filename)
            outs.append(parse_file(filename))

    read_from_files([os.path.join(here, 'docs', 'index.rst'),
                     os.path.join(here, 'docs', 'pages', 'understanding', 'whataresignatures.rst'),
                     os.path.join('docs', 'pages', 'usage', 'installation.rst')])

    outs.append("Documentation\n"
                "#############\n"
                "The documentation is available `here <https://signatory.readthedocs.io>`__.")

    read_from_files([os.path.join(here, 'docs', 'pages', 'miscellaneous', 'citation.rst'),
                     os.path.join(here, 'docs', 'pages', 'miscellaneous', 'acknowledgements.rst')])

    with io.open(os.path.join(here, 'README.rst'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(outs))
        
            
if __name__ == '__main__':
    main()
