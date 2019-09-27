.. _miscellaneous-faq:

FAQ and Known Issues
####################

If you have a question and don't find an answer here then do please `open an issue <https://github.com/patrick-kidger/signatory/issues/new>`__.

* What's the difference between Signatory and `iisignature <https://github.com/bottler/iisignature>`__?

The essential difference (and the reason for Signatory's existence) is that iisignature is limited to the CPU, whilst Signatory is for both CPU and GPU. Signatory is also typically faster even on the CPU, especially for larger computations. Other than that, iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides some operations that the other doesn't.

* I get an ``ImportError: DLL load failed: The specified procedure could not be found.`` when I try to import Signatory.

This appears to be caused by using old versions of Python, e.g. ``3.6.6`` instead of ``3.6.9``. Upgrading your version of Python seems to resolve the issue.

* The installation via ``pip`` fails.

This should be pretty uncommon as we provide for all major operating systems and versions of Python. If you're on Linux then it may be a compilation failure, see the next question. In any case, please let us know by `opening an issue <https://github.com/patrick-kidger/signatory/issues/new>`__.

* I can't find prebuilt wheels (i.e. binaries) available for Linux.

This is deliberate. We provide prebuilt wheels for Windows and Mac, and allow Linux users to compile their own. The reason for this is a technical one. Signatory is built on PyTorch, which actually implies that it must be compiled on CentOS 7 or later, which is not currently supported by the `manylinux <https://github.com/pypa/manylinux>`__ project. The only option open to us would be to do what PyTorch themselves do: build on CentOS 7, and then manually try to ensure compatability with earlier versions as well. Fortunately, compilation on Linux is straightforward, and should automatically occur when installing via ``pip``. It is unlikely that the installation will fail to 'just work' via ``pip``. If the installation does fail for you when doing this then do please `open an issue <https://github.com/patrick-kidger/signatory/issues/new>`__.

* Exceptions messages aren't very helpful on a Mac.

This isn't an issue directly to do with Signatory. We use pybind11 to translate C++ exceptions to Python exceptions, and some part of this process breaks down when on a Mac. If you're trying to debug your code then the best (somewhat unhelpful) advice is to try running the problematic code on either Windows or Linux to check what the error message is.
