.. _usage-installation:

Installation
############

.. code-block:: bash

    pip install signatory==<SIGNATORY_VERSION>.<TORCH_VERSION> --no-cache-dir --force-reinstall

where ``<SIGNATORY_VERSION>`` is the version of Signatory you would like to download (the most recent version is |version|) and ``<TORCH_VERSION>`` is the version of PyTorch you are using.

Available for Python 3.7--3.9 on Linux and Windows. Requires `PyTorch <http://pytorch.org/>`__ 1.8.0--1.11.0.

(If you need it, then previous versions of Signatory included support for older versions of Python, PyTorch, and MacOS, see `here <https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#older-versions>`__.)

After installation, just ``import signatory`` inside Python.

Take care **not** to run ``pip install signatory``, as this will likely download the wrong version.

Example:
--------

For example, if you are using PyTorch 1.11.0 and want Signatory 1.2.7, then you should run:

.. code-block:: bash

    pip install signatory==1.2.7.1.11.0 --no-cache-dir --force-reinstall
        
Why you need to specify all of this:
------------------------------------

Yes, this looks a bit odd. This is needed to work around `limitations of PyTorch <https://github.com/pytorch/pytorch/issues/28754>`__ and `pip <https://www.python.org/dev/peps/pep-0440/>`__.

The ``--no-cache-dir --force-reinstall`` flags are because ``pip`` doesn't expect to need to care about versions quite as much as this, so it will sometimes erroneously use inappropriate caches if not told otherwise.

.. command.readme insert Installation from source is also possible; please consult the `documentation <https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#usage-install-from-source>`__. This also includes information on how to run the tests and benchmarks.

If you have any problems with installation then check the `FAQ <https://signatory.readthedocs.io/en/latest/pages/miscellaneous/faq.html#miscellaneous-faq-importing>`__. If that doesn't help then feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.

.. command.readme off

.. _usage-install-from-source:

Install from source
-------------------
For most use-cases, the prebuilt binaries available as described above should be sufficient. However installing from source is also perfectly feasible, and usually not too tricky.

You'll need to have a C++ compiler installed and known to ``pip``, and furthermore this must be the same compiler that PyTorch uses. (This is ``msvc`` on Windows, ``gcc`` on Linux, and ``clang`` on Macs.) You must have already installed `PyTorch <http://pytorch.org/>`__. (You don't have to compile PyTorch itself from source, though!)

Then run **either**

.. code-block:: bash

    pip install signatory==<SIGNATORY_VERSION>.<TORCH_VERSION> --no-binary signatory

(where ``<SIGNATORY_VERSION>`` and ``<TORCH_VERSION>`` are as above.)

**or**

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    python setup.py install

If you chose the first option then you'll get just the files necessary to run Signatory.

If you choose the second option then tests, benchmarking code, and code to build the documentation will also be provided. Subsequent to this,

- | Tests can be run, see ``python command.py test --help``.
  | This requires installing `iisignature <https://github.com/bottler/iisignature>`__ and `pytest <https://pytest.org>`__.
- | Speed and memory  benchmarks can be performed, see ``python command.py benchmark --help``.
  | This requires installing `matplotlib, iisignature <https://github.com/bottler/iisignature>`__, `esig <https://pypi.org/project/esig/>`__, and `memory profiler <https://pypi.org/project/memory-profiler/su>`__.
- | Documentation can be built via ``python command.py docs``.
  | This requires installing `Sphinx <https://pypi.org/project/Sphinx/>`__, `sphinx_rtd_theme <https://pypi.org/project/sphinx-rtd-theme/>`__ and `py2annotate <https://github.com/patrick-kidger/py2annotate>`__.

.. note::
    
    If on Linux then the commands stated above should probably work.
    
    If on Windows then it is probably first necessary to run a command of the form
    
    .. code-block:: bash
    
        "C:/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
        
    (the exact command will depend on your operating system and version of Visual Studio).
    
    If on a Mac then the installation command should instead look like either

    .. code-block:: bash

            MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install signatory==<SIGNATORY_VERSION>.<TORCH_VERSION> --no-binary signatory

    or
    
    .. code-block:: bash
    
        MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

    depending on the choice of installation method.

A helpful point of reference for getting this to work might be the `official build scripts <https://github.com/patrick-kidger/signatory/blob/master/.github/workflows/build.yml>`__ for Signatory.

Older versions
--------------
Older versions of Signatory supported earlier versions of Python and PyTorch. It also included support for MacOS, but this has now been dropped as being difficult to maintain.

The full list of available combinations can seen `on PyPI <https://pypi.org/project/signatory/#history>`__.
