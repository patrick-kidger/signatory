.. currentmodule:: usage-installation

.. _usage-installation2:

Installation
############
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7.

Available for Linux, Mac, Windows.

Requires `PyTorch <http://pytorch.org/>`__. Tested with PyTorch version 1.2.0, but should work with all recent versions.

Install via ``pip install signatory``.

Install from source
^^^^^^^^^^^^^^^^^^^
For most use-cases, the prebuilt binaries available as described above should be sufficient. If you want to run the tests yourself, or perhaps eke out a little extra speed with a specific-to-you compilation then you'll need to install from source. You'll need to be able to compile C++.

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    python command.py install
    python command.py test