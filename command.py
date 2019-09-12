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
"""Provides a set of commands for running tests, building from source, building documentation etc.

This is essentially only useful for developers. End users are expected to install via pip.

If you're an end user who's a super keen bean who wants to build from source and run tests themselves then you'll be
interested in the commands defined by this file: first run either
`python command.py install`
or
`python command.py develop`
(the former will install it in the normal way. The second will create a 'build' directory adjacent to this file, put
the compiled parts of the package in there, leave the Python parts of this package where they are, and then add links
so that Python can see this package.)
Then run
`python command.py test`
"""


#### THIS FILE IS NOT INCLUDED IN THE SDIST
# Do not make building from the sdist depend on it 
# Note that we make a distinction between sdist and source:
# The supported way to get the source (including tests etc.) is via GitHub.
# The sdist is treated as a quirk of PyPI.


import argparse
import io
import os
import re
import shlex
import subprocess
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
    install_parser = subparsers.add_parser('install')
    develop_parser = subparsers.add_parser('develop')
    test_parser = subparsers.add_parser('test', parents=[deviceparser])
    benchmark_parser = subparsers.add_parser('benchmark', parents=[deviceparser])
    docs_parser = subparsers.add_parser('docs')
    publish_parser = subparsers.add_parser('publish')
    prepublish_parser = subparsers.add_parser('prepublish', parents=[deviceparser])
    genreadme_parser = subparsers.add_parser('genreadme')

    install_parser.set_defaults(cmd=install)
    develop_parser.set_defaults(cmd=develop)
    test_parser.set_defaults(cmd=test)
    benchmark_parser.set_defaults(cmd=benchmark)
    docs_parser.set_defaults(cmd=docs)
    publish_parser.set_defaults(cmd=publish)
    prepublish_parser.set_defaults(cmd=prepublish)
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

    prepublish_parser.add_argument('-l', '--loc', default='/tmp', dest='directory',
                                   help="Where to place the conda environments used for testing.")

    args = parser.parse_args()

    # Have to do it this was for Python 2/3 compatability
    if hasattr(args, 'cmd'):
        args.cmd(args)
    else:
        # No command was specified
        print("Please enter a command. Use -h to see available commands.")


here = os.path.realpath(os.path.dirname(__file__))


def run_commands(*commands, **kwargs):
    """Runs a collection of commands in a shell.

    Note that it's not super robust - there aren't really reliable ways to do cross-platform shell scripting.
    """

    # For Python 2 compatability.
    stdout = kwargs.pop('stdout', True)
    if kwargs:
        raise ValueError

    print_commands = ['echo {}'.format(shlex.quote("(running) " + command)) for command in commands]
    all_commands = []
    for i in range(len(commands)):
        all_commands.append(print_commands[i])
        if stdout:
            all_commands.append(commands[i])
        else:
            all_commands.append("{} > /dev/null".format(commands[i]))
    # && should work on both Windows and Linux. Not sure about Macs. Still Unix, so probably works?
    subprocess.run(' && '.join(all_commands), shell=True)


def install(args=()):
    """Install from source."""
    run_commands("pip install {}".format(here))
    
    
def develop(args=()):
    """Install from source; will create a 'build' directory adjacent to this file, put the compiled parts of the
    package in there, leave the Python parts of this package where they are, and then add links so that Python can see
    this package."""
    run_commands("python {} develop".format(os.path.join(here, "setup.py")))
    
    
def test(args):
    """Run all tests. Running all tests typically takes about an hour.
    The package 'iisignature' will need to be installed, to test against.
    It can be installed via `pip install iisignature`
    """
    import iisignature  # fail fast here if necessary
    import test.runner
    import torch
    with torch.cuda.device(args.device):
        print('Using device {}'.format(args.device))
        test.runner.main(failfast=args.failfast, times=args.times, names=args.names)


def benchmark(args):
    """Run speed benchmarks."""
    import test.benchmark as bench
    import torch
    with torch.cuda.device(args.device):
        print('Using device {}'.format(args.device))
        if args.type == 'typical':
            runner = bench.BenchmarkRunner.typical(test_esig=args.esig, fns=args.fns)
        elif args.type == 'depths':
            runner = bench.BenchmarkRunner.depths(test_esig=args.esig, fns=args.fns)
        elif args.type == 'channels':
            runner = bench.BenchmarkRunner.channels(test_esig=args.esig, fns=args.fns)
        elif args.type == 'small':
            runner = bench.BenchmarkRunner.small(test_esig=args.esig, fns=args.fns)
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
    import py2annotate  # fail fast here if necessary
    run_commands("sphinx-build -M html {} {}".format(os.path.join(here, "docs"), os.path.join(here, "docs", "_build")))
    webbrowser.open_new_tab('file:///{}'.format(os.path.join(here, 'docs', '_build', 'html', 'index.html')))
    
    
def publish(args=()):
    """Will need twine already installed"""
    run_commands("twine upload {}".format(os.path.join(here, "dist", "*")))


def prepublish(args):
    """Runs tests on all supported configurations to check before publishing."""
    # TODO: update to a proper system
    import metadata
    print("Prepublishing version {}".format(metadata.version))
    run_commands("rm -rf {}".format(os.path.join(here, "dist")))
    genreadme()
    run_commands("python {} sdist".format(os.path.join(here, 'setup.py')))
    device_str = ' -d {}'.format(args.device)
    for pythonv in ['2.7', '3.5', '3.6', '3.7']:
        build_and_test(pythonv, metadata.version, device_str, args.directory)


def build_and_test(pythonv, signatoryv, device_str, directory):
    # Kind of fragile but good enough for now
    # Only works through bash due to the 'conda init bash', 'conda activate'
    run_commands("conda clean -a -y",
                 "conda create -p {directory}/signatory-{pythonv} -y python={pythonv}".format(directory=directory,
                                                                                              pythonv=pythonv),
                 ". ~/miniconda3/etc/profile.d/conda.sh",
                 "conda activate {directory}/signatory-{pythonv}".format(directory=directory, pythonv=pythonv),
                 "conda install -y pytorch=1.2.0 -c pytorch",
                 "pip install --upgrade pip",
                 "pip install {here}/dist/signatory-{signatoryv}.tar.gz".format(here=here, signatoryv=signatoryv),
                 "pip install iisignature",
                 "python {} test -f{}".format(os.path.join(here, "command.py"), device_str),
                 "conda deactivate",
                 "conda env remove -p {directory}/signatory-{pythonv}".format(directory=directory,
                                                                              pythonv=pythonv),
                 "conda clean -a -y",
                 stdout=False)

    
def genreadme(args=()):
    """The readme is generated automatically from the documentation"""
    outs = []
    includestr = '.. include::'
    on = '.. genreadme on'
    off = '.. genreadme off'
    reference = re.compile(r'^\.\. [\w-]+:$')

    def parse_file(filename):
        out_data = []
        with io.open(filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
            skipping = False
            for line in data:
                stripline = line.strip()
                if stripline == on:
                    skipping = False
                    continue
                if stripline == off:
                    skipping = True
                if skipping:
                    continue
                if reference.match(stripline):
                    continue
                if stripline.startswith(includestr):
                    # [1:] to remove the leading / at the start; otherwise ends up being parsed as root
                    subfilename = stripline[len(includestr):].strip()[1:]
                    out_line = parse_file(os.path.join(here, 'docs', subfilename))
                else:
                    out_line = line
                if ':ref:' in data:
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
                "-------------\n"
                "The documentation is available `here <https://signatory.readthedocs.io>`__.")

    read_from_files([os.path.join(here, 'docs', 'pages', 'miscellaneous', 'faq.rst'),
                     os.path.join(here, 'docs', 'pages', 'miscellaneous', 'citation.rst'),
                     os.path.join(here, 'docs', 'pages', 'miscellaneous', 'acknowledgements.rst')])

    with io.open(os.path.join(here, 'README.rst'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(outs))
        
            
if __name__ == '__main__':
    main()
