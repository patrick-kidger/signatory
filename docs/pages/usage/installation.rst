.. currentmodule:: usage-installation

Installation
------------
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7.

Requires `PyTorch <http://pytorch.org/>`__. Tested with PyTorch version 1.0.1, but should probably work with all recent versions.

Install via ``pip``:

.. code-block:: bash

    pip install signatory


Alternatively install via ``git``:

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    pip install .

Prebuilt wheels are not yet available - you'll have to have the relevant toolchain installed to compile C++. (If you're on Linux this is probably already the case.)