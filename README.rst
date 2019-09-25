
*********
Signatory
*********
Differentiable computations of the signature and logsignature transforms, on both CPU and GPU.




What are signatures?
####################
If you're reading this then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what this 'signature' thing is...

The *signature transform* is a transformation that takes in a stream of data (often a time series), and returns a collection of statistics about that stream of data, called the *signature*. This collection of statistics determines the path essentially uniquely. Importantly, the signature is rich enough that every continuous function of the input stream may be approximated arbitrarily well by a linear function of its signature; the signature transform is what we call a *universal nonlinearity*. If you're doing machine learning then you probably understand why this is such a desirable property!


Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on the use of the signature transform in machine learning, just as a feature transformation, and `this <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at integrating the signature transform into neural networks.


Installation
############
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7.

Available for Linux, Mac, Windows.

Requires `PyTorch <http://pytorch.org/>`__. Tested with PyTorch version 1.2.0, but should work with all recent versions.

Install via ``pip install signatory`` or ``pip3 install signatory`` as appropriate.

Then just ``import signatory`` inside Python.

Installation from source is also possible; please consult the documentation, and open an issue if you run into any problems.


Documentation
-------------
The documentation is available `here <https://signatory.readthedocs.io>`__.


FAQ
###
* What's the difference between Signatory and iisignature_?

The essential difference (and the reason for Signatory's existence) is that iisignature is limited to the CPU, whilst Signatory is for both CPU and GPU. Signatory is also typically much faster even on the CPU, especially for larger computations. Other than that, iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides some operations that the other doesn't.

* I get an ``ImportError`` when I try to install Signatory.

You probably haven't installed PyTorch. Do that, then run ``pip`` or ``pip3`` to install Signatory.

* The installation via ``pip`` or ``pip3`` fails.

This should be pretty uncommon as we provide for all major operating systems and versions of Python. Please let us know by `opening an issue <https://github.com/patrick-kidger/signatory/issues/new>`.

.. _iisignature: https://github.com/bottler/iisignature


Citation
########
If you found this library useful in your research, please consider citing

.. code-block:: bibtex

    @misc{signatory,
        title={{Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU}},
        author={Kidger, Patrick},
        note={https://github.com/patrick-kidger/signatory},
        year={2019}
    }


Acknowledgements
################

The Python bindings for the C++ code were written with the aid of `pybind11 <https://github.com/pybind/pybind11>`__.

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`__ package. The notes accompanying the iisignature project greatly helped with the implementation of Signatory.