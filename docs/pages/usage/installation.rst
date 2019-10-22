.. _usage-installation:

Installation
############
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7 and Linux, Mac, Windows.

Requires `PyTorch <http://pytorch.org/>`__.

Install via ``pip install signatory_installer``. Then just ``import signatory`` inside Python.

.. note::

    The ``installer`` is because this command actually downloads a script which detects the version of PyTorch that is currently installed, and then downloads the corresponding version of Signatory. If you already know that you are using e.g. PyTorch 1.2.0 and want Signatory 1.1.4, then you can just run ``pip install signatory==1.1.4-torch1.2.0``.

.. genreadme insert install_from_source

.. genreadme off

..
    The FAQ link has to be a direct link, not a reference, so that it works on the GitHub README.
    And furthermore GitHub's READMEs don't like comments, so we have to toggle genreadme either side of this comment.

.. genreadme on

If you have any problems with installation then check the `FAQ <https://signatory.readthedocs.io/en/latest/pages/miscellaneous/faq.html#miscellaneous-faq-importing>`__. If that doesn't help then feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.

.. genreadme off

.. _usage-install-from-source:

Install from source
^^^^^^^^^^^^^^^^^^^
For most use-cases, the prebuilt binaries available as described above should be sufficient. If installing a binary fails, or you want to run the tests yourself, or you want to try eking out a little extra speed with a specific-to-you compilation, then you'll need to install from source. You'll need to be able to compile C++. You must have already installed `PyTorch <http://pytorch.org/>`__, as this is a requirement to run ``setup.py``. Then:

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    python setup.py install
    
Subsequent to this,

- Tests can be run, see ``python command.py test --help``. This requires installing `iisignature <https://github.com/bottler/iisignature>`__.
- Speed and memory  benchmarks can be performed, see ``python command.py benchmark --help``. This requires installing `iisignature <https://github.com/bottler/iisignature>`__, `esig <https://pypi.org/project/esig/>`__, and `memory profiler <https://pypi.org/project/memory-profiler/su>`__.
- Documentation built via ``python command.py docs``. This requires installing `Sphinx <https://pypi.org/project/Sphinx/>`__, `sphinx_rtd_theme <https://pypi.org/project/sphinx-rtd-theme/>`__ and `py2annotate <https://github.com/patrick-kidger/py2annotate>`__.

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