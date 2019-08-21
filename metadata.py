import io
import os
import re


project = 'signatory'
author = "Patrick Kidger"
copyright = "2019, {}".format(author)
author_email = "contact@kidger.site"
url = "https://github.com/patrick-kidger/signatory"
license = "Apache-2.0"
python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4"
keywords = "signature"
classifiers = ["Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: Unix",  # TODO: test for:
                                            #   Operating System :: Microsoft :: Windows
                                            #   or
                                            #   Operating System :: OS Independent
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.5",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

here = os.path.realpath(os.path.dirname(__file__))

# for simplicity we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, 'src', project, '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise SystemExit("Unable to find __version__ string.")

with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    readme = f.read()

try:
    with io.open(os.path.join(here, 'docs', 'fragments', 'description.rst'), 'r', encoding='utf-8') as f:
        description = f.read()
except FileNotFoundError:
    description = '< No description when building from source. >'
