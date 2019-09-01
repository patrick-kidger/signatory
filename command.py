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
import pprint
import subprocess
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
    subparsers = parser.add_subparsers(dest='command', required=True, help='Which command to run')
    install_parser = subparsers.add_parser('install')
    develop_parser = subparsers.add_parser('develop')
    test_parser = subparsers.add_parser('test', parents=[deviceparser])
    benchmark_parser = subparsers.add_parser('benchmark', parents=[deviceparser])
    docs_parser = subparsers.add_parser('docs')
    publish_parser = subparsers.add_parser('publish')
    prepublish_parser = subparsers.add_parser('prepublish')
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
    test_parser.add_argument('-n', '--nonames', action='store_false', dest='names', help="Don't print names and start "
                                                                                         "time of the tests being run.")
    test_parser.add_argument('-t', '--notimes', action='store_false', dest='times', help="Don't print the overall "
                                                                                         "times of the tests that have "
                                                                                         "been run.")

    args = parser.parse_args()

    args.cmd(args)


here = os.path.realpath(os.path.dirname(__file__))


def run_commands(*commands):
    """Runs a collection of commands in a shell."""
    if not isinstance(commands, (tuple, list)):
        raise ValueError
    print_commands = ["echo {}".format(command) for command in commands]
    all_commands = [command_list[i] for i in range(len(commands)) for command_list in (print_commands, commands)]
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
    import test.speed_comparison as speed
    import torch
    with torch.cuda.device(args.device):
        print('Using device {}'.format(args.device))
        results = speed.run_tests()
    ratios = speed.get_ratios(results)
    pprint.pprint(results)
    print('-----------------------')
    pprint.pprint(ratios)

    
def docs(args=()):
    """Build the documentation. After it has been built then it can be found in ./docs/_build/html/index.html
    The package 'py2annotate' will need to be installed. It can be installed via `pip install py2annotate`
    Note that the documentation is already available online at https://signatory.readthedocs.io
    """
    import py2annotate  # fail fast here if necessary
    run_commands("sphinx-build -M html {}, {}".format(os.path.join(here, "docs"), os.path.join(here, "docs", "_build")))
    
    
def publish(args=()):
    """Will need twine already installed"""
    run_commands("twine upload {}".format(os.path.join(here, "dist", "*")))


def prepublish(args=()):
    """Runs tests on all supported configurations to check before publishing."""
    # TODO: update to a proper system
    import metadata
    run_commands("rm -rf {}".format(os.path.join(here, "dist")))
    genreadme()
    print("Prepublishing version {}".format(metadata.version))
    run_commands("python {} sdist".format(os.path.join(here, 'setup.py')))
    for pythonv in ['2.7', '3.5', '3.6', '3.7']:
        build_and_test(pythonv, metadata.version)


def build_and_test(pythonv, signatoryv):
    # Kind of fragile but good enough for now
    run_commands("conda create --prefix='/tmp/signatory-{pythonv} -y python={pythonv}".format(pythonv=pythonv),
                 "conda activate /tmp/signatory-{pythonv}".format(pythonv=pythonv),
                 "conda install -y pytorch=1.2.0 -c pytorch",
                 "pip install dist/signatory-{signatoryv}.tar.gz".format(signatoryv=signatoryv),
                 "pip install iisignature",
                 "echo version={pythonv}".format(pythonv=pythonv),
                 "python {} test -f".format(os.path.join(here, "command.py")),
                 "conda deactivate",
                 "conda env remove -p /tmp/signatory-{pythonv}".format(pythonv=pythonv))

    
def genreadme(args=()):
    """The readme is generated automatically from the documentation"""
    outs = []
    startstr = ".. currentmodule::"
    includestr = '.. include::'

    def parse_file(filename):
        out_data = []
        with io.open(filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
            skipping = False
            for line in data:
                if startstr in line:
                    skipping = True
                    continue
                if skipping and line.strip() == '':
                    continue
                else:
                    skipping = False
                lstripline = line.lstrip()
                if lstripline.startswith(includestr):
                    # [1:] to remove the leading / at the start; otherwise ends up being parsed as root
                    subfilename = lstripline[len(includestr):].strip()[1:]
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

    read_from_files([os.path.join(here, 'docs', 'fragments', 'title.rst'),
                     os.path.join(here, 'docs', 'pages', 'understanding', 'whataresignatures.rst'),
                     os.path.join('docs', 'pages', 'usage', 'installation.rst')])

    outs.append("Documentation\n"
                "-------------\n"
                "The documentation is available `here <https://signatory.readthedocs.io>`__.")

    read_from_files([os.path.join(here, 'docs', 'pages', 'miscellaneous', 'faq.rst'),
                     os.path.join(here, 'docs' , 'pages', 'miscellaneous' , 'citation.rst'),
                     os.path.join(here, 'docs', 'pages', 'miscellaneous', 'acknowledgements.rst')])

    with io.open(os.path.join(here, 'README.rst'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(outs))
        
            
if __name__ == '__main__':
    main()
