*********
Signatory
*********
Efficient computations of the signature transform for PyTorch, on both CPU and GPU, including backpropagation.

Overview
--------
If you're reading this documentation then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what this 'signature' thing is...

The 'signature transform' is a transformation that takes in a stream of data (often a time series), and returns a collection of statistics about that stream of data. This collection of statistics determines the path essentially uniquely, in an efficient computable way. Furthermore it is rich enough that every continuous function of the input stream may be approximated arbitrarily well by a linear function of its signature; the signature is what we call a 'universal nonlinearity'. If you're doing machine learning then you probably understand why this is such a desirable property!

In principle it's quite similar to the Fourier transform: it's a transformation that can be applied to a stream of data, which extracts certain information. The Fourier transform describes frequencies; the signature most naturally describes *order*. That is, the order of events, potentially in different channels, is a particularly easy thing to understand using the signature.

Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on its use in machine learning, just as a feature transformation, and `this <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at integrating it into neural neural networks.

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

Documentation
-------------
The documentation is available `here <https://signatory.readthedocs.io>`__.

FAQ
---
* What's the difference between Signatory and iisignature_?

The essential difference (and indeed the reason for Signatory's existence) is that iisignature is CPU-only, whilst Signatory is for both CPU and GPU, to provide the speed necessary for machine learning. (In particular this removes the need to copy data back and forth between the CPU and the GPU.) iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides a few operations that the other doesn't.

* I'm only using the CPU. Does it matter whether I use Signatory or iisignature_?

Not particularly!

.. _iisignature: https://github.com/bottler/iisignature

Citation
--------
If you found this library useful in your research, please consider citing

.. code-block:: bibtex

    @article{deepsignatures,
        title={{Deep Signatures}},
        author={{Bonnier, Patric and Kidger, Patrick and Perez Arribas, Imanol and Salvi, Cristopher and Lyons, Terry}},
        journal={arXiv:1905.08494},
        year={2019}
    }

which this project was a spin-off from.

Acknowledgements
----------------
The Python bindings for the C++ code were written with the aid of `pybind11 <https://github.com/pybind/pybind11>`__.

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`__ package. The notes accompanying the iisignature project greatly helped with the implementation of Signatory.