.. _miscellaneous-faq:

FAQ and Known Issues
####################

If you have a question and don't find an answer here then do please `open an issue <https://github.com/patrick-kidger/signatory/issues/new>`__.

.. _miscellaneous-faq-importing:

Problems with importing or installing Signatory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* I get an ``ImportError: DLL load failed: The specified procedure could not be found.`` when I try to import Signatory.

This appears to be caused by using old versions of Python, e.g. ``3.6.6`` instead of ``3.6.9``. Upgrading your version of Python seems to resolve the issue.

* I get an ``Import Error: ... Symbol not found: ...`` when I try to import Signatory.

This occurs when the version of Python or PyTorch you have installed is different to the version of Python or PyTorch that your copy of Signatory is compiled for. Make sure that you have specified the correct version of PyTorch when downloading Signatory; see :ref:`the installation instructions<usage-installation>`, and that you include the extra ``--no-cache-dir --force-reinstall`` flags as described there.

.. _miscellaneous-faq-other:

Everything else
^^^^^^^^^^^^^^^

* What's the difference between Signatory and `iisignature <https://github.com/bottler/iisignature>`__?

The essential difference (and the reason for Signatory's existence) is that iisignature is limited to the CPU, whilst Signatory is for both CPU and GPU. Signatory is also typically faster even on the CPU, thanks to parallelisation and algorithmic improvements. Other than that, iisignature is NumPy-based, whilst Signatory uses PyTorch. There are also a few differences in the provided functionality; each package provides some operations that the other doesn't.

* Exceptions messages aren't very helpful on a Mac.

This isn't an issue directly to do with Signatory. We use pybind11 to translate C++ exceptions to Python exceptions, and some part of this process breaks down when on a Mac. If you're trying to debug your code then the best (somewhat unhelpful) advice is to try running the problematic code on either Windows or Linux to check what the error message is.
