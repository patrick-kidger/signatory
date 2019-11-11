.. _usage-installation:

Installation
############
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7 and Linux, Mac, Windows.

Requires `PyTorch <http://pytorch.org/>`__.

Install via ``pip install signatory==<SIGNATORY_VERSION>-torch<TORCH_VERSION>``, where ``<SIGNATORY_VERSION>`` is the version of Signatory you would like to download (the most recent version is 1.1.4) and ``<TORCH_VERSION>`` is the version of PyTorch you are using (supported versions are PyTorch 1.2.0 and 1.3.0). Then just ``import signatory`` inside Python.

.. command.readme off (GitHub doesn't support using admonitions this way, and just uses indented text instead.)
.. admonition:: Example

    .. command.readme on

    For example, if you are using PyTorch 1.2.0 and want Signatory 1.1.4, then you should run ``pip install signatory==1.1.4-torch1.2.0``.

.. command.readme off
.. caution::

    .. command.readme on

    Take care **not** to run ``pip install signatory``, as this will likely download the wrong version. This care is needed due to a `limitation of PyTorch <https://github.com/pytorch/pytorch/issues/28754>`__.

.. command.readme insert Installation from source is also possible; please consult the `documentation <https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#usage-install-from-source>`__. This also includes information on how to run the tests and benchmarks.

If you have any problems with installation then check the `FAQ <https://signatory.readthedocs.io/en/latest/pages/miscellaneous/faq.html#miscellaneous-faq-importing>`__. If that doesn't help then feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.

.. command.readme off

.. _usage-install-from-source:

Install from source
^^^^^^^^^^^^^^^^^^^
For most use-cases, the prebuilt binaries available as described above should be sufficient. If installing a binary fails, or you want to try eking out a little extra speed with a specific-to-you compilation, then you'll need to install from source. You'll need to have a C++ compiler installed and known to ``pip``. (This may already be the case; see the notes below.) You must have already installed `PyTorch <http://pytorch.org/>`__, as this is a requirement to run ``setup.py``.

Then run **either**

.. code-block:: bash

    pip install signatory --no-binary signatory

**or**

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    python setup.py install

In either case, the actual compilation should automatically occur for you.

If you chose the first option then you'll get just the files necessary to run Signatory.

If you choose the second option then tests, benchmarking code, and code to build the documentation will also be provided. Subsequent to this,

- Tests can be run, see ``python command.py test --help``. This requires installing `iisignature <https://github.com/bottler/iisignature>`__ and `pytest <https://pytest.org>`__``.
- Speed and memory  benchmarks can be performed, see ``python command.py benchmark --help``. This requires installing `iisignature <https://github.com/bottler/iisignature>`__, `esig <https://pypi.org/project/esig/>`__, and `memory profiler <https://pypi.org/project/memory-profiler/su>`__.
- Documentation can be built via ``python command.py docs``. This requires installing `Sphinx <https://pypi.org/project/Sphinx/>`__, `sphinx_rtd_theme <https://pypi.org/project/sphinx-rtd-theme/>`__ and `py2annotate <https://github.com/patrick-kidger/py2annotate>`__.

.. note::
    
    If on Linux then the commands stated above should probably work.
    
    If on Windows then it is probably first necessary to run a command of the form
    
    .. code-block:: bash
    
        "C:/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
        
    (the exact command will depend on your operating system and version of Visual Studio).
    
    If on a Mac then the installation command may instead look like
    
    .. code-block:: bash
    
        MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

.. note::

    Your C++ compiler must be the same as the one used to compile PyTorch for your platform. (If this is not the case then a large warning will appear during compilation of Signatory, and the installation will probably fail.)

A helpful point of reference for getting this work might be the `official build scripts <https://github.com/patrick-kidger/signatory/blob/master/.github/workflows/build.yml>`__ for Signatory.

.. _usage-all-supported-versions:

All supported versions
^^^^^^^^^^^^^^^^^^^^^^

Signatory aims to support essentially every possible combination of operating system, Python version, and PyTorch version. In particular we provide explicit support for:

+----------------------------------------------+
| **PyTorch 1.2.0 and 1.3.0:**                 |
+------------+----------+----------+-----------+
|            | Linux    | Mac      | Windows   |
+------------+----------+----------+-----------+
| Python 2.7 | ✓        | ✓        | ✗*        |
+------------+----------+----------+-----------+
| Python 3.5 | ✓        | ✓        | ✓         |
+------------+----------+----------+-----------+
| Python 3.6 | ✓        | ✓        | ✓         |
+------------+----------+----------+-----------+
| Python 3.7 | ✓        | ✓        | ✓         |
+------------+----------+----------+-----------+
| \* PyTorch does not support this combination |
+----------------------------------------------+