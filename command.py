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


import io
import os
import subprocess
import sys
#### DO NOT IMPORT NON-(STANDARD LIBRARY) MODULES HERE
# Instead, lazily import them inside the command.
# This allows all the commands that don't e.g. require a built version of Signatory to operate without it


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Please pass a command, e.g. python command.py test')
        
    command = sys.argv[1]
    argv = sys.argv[2:]

    if command == 'install':
        install(argv)
    elif command == 'develop':
        develop(argv)
    elif command == 'test':
        test(argv)
    elif command == 'docs':
        argv(docs)
    elif command == 'publish':
        publish(argv)
    elif command == 'genreadme':
        genreadme(argv)
    else:
        raise ValueError("Unrecognised command.")

        
def install(argv):
    """Install from source."""
    subprocess.run("pip install .")   
    
    
def develop(argv):
    """Install from source; will create a 'build' directory adjacent to this file, put the compiled parts of the
    package in there, leave the Python parts of this package where they are, and then add links so that Python can see
    this package."""
    subprocess.run("python setup.py develop")
    
    
def test(argv):
    """Run all tests. Running all tests typically takes about an hour.
    The package 'iisignature' will need to be installed, to test against.
    It can be installed via `pip install iisignature`
    """
    import iisignature  # fail fast here if necessary
    import test.runner
    failfast = '-f' in argv or '--failfast' in argv
    record_test_times = not ('--notimes' in sys.argv)
    test.runner.main(failfast=failfast, record_test_times=record_test_times)
    
    
def docs(argv):
    """Build the documentation. After it has been built then it can be found in ./docs/_build/html/index.html
    The package 'py2annotate' will need to be installed. It can be installed via `pip install py2annotate`
    Note that the documentation is already available online at https://signatory.readthedocs.io
    """
    import py2annotate  # fail fast here if necessary
    subprocess.run("sphinx-build -M html ./docs ./docs/_build")
    
    
def publish(argv):
    """Will need twine already installed"""
    subprocess.run("twine upload dist/*")
    
    
def genreadme(argv):
    """The readme is generated automatically from the documentation"""
    outs = []
    startstr = ".. currentmodule::"
    includestr = '.. include::'
    docdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'docs')

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
                    subfilename = lstripline[len(includestr):].strip()
                    # [1:] to remove the leading / at the start; ends up being parsed as root
                    out_line = parse_file(os.path.join(docdir, subfilename[1:]))
                else:
                    out_line = line
                if ':ref:' in data:
                    raise RuntimeError('refs not supported')
                out_data.append(out_line)
        return ''.join(out_data)

    here = os.path.realpath(os.path.dirname(__file__))

    def read_from_files(filenames):
        for filename in filenames:
            filename = os.path.join(here, filename)
            outs.append(parse_file(filename))

    read_from_files(['docs/fragments/title.rst',
                     'docs/pages/understanding/whataresignatures.rst',
                     'docs/pages/usage/installation.rst'])

    outs.append("Documentation\n"
                "-------------\n"
                "The documentation is available `here <https://signatory.readthedocs.io>`__.")

    read_from_files(['docs/pages/miscellaneous/faq.rst',
                     'docs/pages/miscellaneous/citation.rst',
                     'docs/pages/miscellaneous/acknowledgements.rst'])

    with io.open(os.path.join(here, 'README.rst'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(outs))
        
            
if __name__ == '__main__':
    main()
