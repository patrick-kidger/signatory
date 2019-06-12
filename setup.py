import io
import os
import re
import setuptools
import torch.utils.cpp_extension as cpp

##########################################################################################################

name = 'signatory'
author = "Patrick Kidger"
author_email = "contact@kidger.site"
url = "https://github.com/patrick-kidger/signatory"
license = "Apache-2.0"
keywords = "signature"
classifiers = ["Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: Unix",               # TODO: test for:
                                                         #   Operating System :: Microsoft :: Windows
                                                         #   or
                                                         #   Operating System :: OS Independent
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.5",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",  # TODO: test for:
                                                         #   Programming Language :: Python :: 2
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]
ext_modules = [cpp.CppExtension(name='_impl', sources=['src/pytorchbind.cpp', 'src/signature.cpp'],
                                depends=['src/signature.hpp'])]

##########################################################################################################

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

if readme.split('\n', maxsplit=1)[0].strip()[1:].strip().lower() != name:
    raise SystemExit('Name in README.md does not match given name.')

description = readme.split('\n', maxsplit=2)[1]


top_python_package = os.path.join(here, 'src', name)
init_file = os.path.join(top_python_package, '__init__.py')
# We assert that the src/{name} folder and src/{name}/__init__.py file exist. We have to keep the name in sync with
# whatever we call the package.
if not os.path.isdir(top_python_package):
    raise SystemExit('Top python package does not match given name.')
if not os.path.isfile(init_file):
    raise SystemExit('Top python package does not have an __init__.py file.')


# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
def find_meta(meta, string):
    """
    Extract __*meta*__ from file.
    """
    meta_match = re.search(r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), string, re.M)
    if meta_match:
        return meta_match.group(1)
    raise SystemExit("Unable to find __{meta}__ string.".format(meta=meta))


with io.open(init_file) as f:
    version = find_meta('version', f.read())


setuptools.setup(name=name,
                 version=version,
                 author=author,
                 author_email=author_email,
                 maintainer=author,
                 maintainer_email=author_email,
                 description=description,
                 long_description=readme,
                 long_description_content_type="text/markdown",
                 url=url,
                 license=license,
                 keywords=keywords,
                 classifiers=classifiers,
                 zip_safe=False,
                 python_requires=">=3.5",  # TODO: ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4"
                 packages=[name],
                 ext_package=name,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
