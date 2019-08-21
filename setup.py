import setuptools
try:
    import torch.utils.cpp_extension as cpp
except ImportError:
    raise ImportError("PyTorch is not installed, and must be installed prior to installing Signatory.")

import metadata


ext_modules = [cpp.CppExtension(name='_impl',
                                sources=['src/free_lie_algebra_ops.cpp',
                                         'src/logsignature.cpp',
                                         'src/lyndon.cpp',
                                         'src/misc.cpp',
                                         'src/pytorchbind.cpp',
                                         'src/signature.cpp',
                                         'src/tensor_algebra_ops.cpp'],
                                depends=['src/free_lie_algebra_ops.hpp',
                                         'src/logsignature.hpp',
                                         'src/lyndon.hpp',
                                         'src/misc.hpp',
                                         'src/signature.hpp',
                                         'src/tensor_algebra_ops.hpp'],
                                extra_compile_args=['-fvisibility=hidden'])]


setuptools.setup(name=metadata.project,
                 version=metadata.version,
                 author=metadata.author,
                 author_email=metadata.author_email,
                 maintainer=metadata.author,
                 maintainer_email=metadata.author_email,
                 description=metadata.description,
                 long_description=metadata.readme,
                 url=metadata.url,
                 license=metadata.license,
                 keywords=metadata.keywords,
                 classifiers=metadata.classifiers,
                 zip_safe=False,
                 python_requires=metadata.python_requires,
                 packages=[metadata.project],
                 ext_package=metadata.project,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
