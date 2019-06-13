import io
import os
import setuptools
import torch.utils.cpp_extension as cpp

import genreadme  # builds README file
import metadata


ext_modules = [cpp.CppExtension(name='_impl', sources=['src/pytorchbind.cpp', 'src/signature.cpp'],
                                depends=['src/signature.hpp'])]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    readme = f.read()

with io.open(os.path.join(here, 'docs', 'pages', 'description.rst'), 'r', encoding='utf-8') as f:
    description = f.read()

setuptools.setup(name=metadata.project,
                 version=metadata.version,
                 author=metadata.author,
                 author_email=metadata.author_email,
                 maintainer=metadata.author,
                 maintainer_email=metadata.author_email,
                 description=description,
                 long_description=readme,
                 url=metadata.url,
                 license=license,
                 keywords=metadata.keywords,
                 classifiers=metadata.classifiers,
                 zip_safe=False,
                 python_requires=">=3.5",  # TODO: ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4"
                 packages=[metadata.project],
                 ext_package=metadata.project,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
