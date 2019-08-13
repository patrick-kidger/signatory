import io
import os
import setuptools
import torch.utils.cpp_extension as cpp

import metadata


ext_modules = [cpp.CppExtension(name='_impl',
                                sources=['src/utilities.cpp',
                                         'src/misc.cpp',
                                         'src/tensor_algebra_ops.cpp',
                                         'src/free_lie_algebra_ops.cpp',
                                         'src/signature.cpp',
                                         'src/logsignature.cpp',
                                         'src/pytorchbind.cpp'],
                                depends=['src/utilities.hpp',
                                         'src/misc.hpp',
                                         'src/tensor_algebra_ops.hpp',
                                         'src/free_lie_algebra_ops.hpp',
                                         'src/signature.hpp',
                                         'src/logsignature.hpp'])]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    readme = f.read()

with io.open(os.path.join(here, 'docs', 'fragments', 'description.rst'), 'r', encoding='utf-8') as f:
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
                 python_requires=metadata.python_requires,
                 packages=[metadata.project],
                 ext_package=metadata.project,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
