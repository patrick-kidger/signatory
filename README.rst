*********
Signatory
*********
Efficient computations of the signature transform for PyTorch, on both CPU and GPU, including backpropagation.

What are signatures?
--------------------
If you're reading this then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what this 'signature' thing is...

The *signature transform* is a transformation that takes in a stream of data (often a time series), and returns a collection of statistics about that stream of data. This collection of statistics determines the path essentially uniquely, in an efficient computable way. Furthermore it is rich enough that every continuous function of the input stream may be approximated arbitrarily well by a linear function of its signature; the signature is what we call a *universal nonlinearity*. If you're doing machine learning then you probably understand why this is such a desirable property!

In principle it's quite similar to the Fourier transform: it's a transformation that can be applied to a stream of data, which extracts certain information. The Fourier transform describes frequencies; the signature most naturally describes *order*. That is, the order of events, potentially in different channels, is a particularly easy thing to understand using the signature.

Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on its use in machine learning, just as a feature transformation, and `this <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at integrating it into neural networks.

Installation
------------
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7.

Requires `PyTorch <http://pytorch.org/>`__. Tested with PyTorch version 1.2.0, but should work with all recent versions.

Install via ``pip install signatory``.

Only source distributions are available at the moment, so you will need to be able to compile C++. If you are on Linux then this should automatically happen when you run ``pip``. Other operating systems may vary.

Documentation
-------------
The documentation is available `here <https://signatory.readthedocs.io>`__.

FAQ
---
* What's the difference between Signatory and iisignature_?

The essential difference (and the reason for Signatory's existence) is that iisignature is limited to the CPU, whilst Signatory is for both CPU and GPU. This allows Signatory to run *much* faster. (See the next question.) Other than that, iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides a few operations that the other doesn't.

* What's the difference in speed between Signatory and iisignature_?

Depends on your CPU and GPU, really. But to throw some numbers out there: on the CPU, Signatory tends to be about twice as fast. With the GPU, it's roughly 65 times as fast.

* I get an ``ImportError`` when I try to install Signatory.

You probably haven't installed PyTorch. Do that, then run ``pip`` to install Signatory.

* How do I backpropagate through the signature transform?

Just call ``.backward()`` like you normally would in PyTorch!

.. _iisignature: https://github.com/bottler/iisignature

Citation
--------
If you found this library useful in your research, please consider citing

.. code-block:: bibtex

    @misc{signatory,
        title={{Signatory: a library for performing signature and logsignature calculations on the GPU}},
        author={Kidger, Patrick},
        note={https://github.com/patrick-kidger/signatory},
        year={2019}
    }

Acknowledgements
----------------
The Python bindings for the C++ code were written with the aid of `pybind11 <https://github.com/pybind/pybind11>`__.

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`__ package. The notes accompanying the iisignature project greatly helped with the implementation of Signatory.