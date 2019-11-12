
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

Requires `PyTorch <http://pytorch.org/>`__ 1.2.0 or 1.3.0.

Installation is pretty simple:

.. code-block:: bash

    pip install signatory==<SIGNATORY_VERSION>-torch<TORCH_VERSION>

where ``<SIGNATORY_VERSION>`` is the version of Signatory you would like to download (the most recent version is 1.1.4) and ``<TORCH_VERSION>`` is the version of PyTorch you are using.


    For example, if you are using PyTorch 1.3.0 and want Signatory 1.1.4, then you should run:

    .. code-block:: bash

        pip install signatory==1.1.4-torch1.3.0

Then just ``import signatory`` inside Python.


    Take care **not** to run ``pip install signatory``, as this will likely download the wrong version. This care is needed due to a `limitation of PyTorch <https://github.com/pytorch/pytorch/issues/28754>`__.

Installation from source is also possible; please consult the `documentation <https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#usage-install-from-source>`__. This also includes information on how to run the tests and benchmarks.

If you have any problems with installation then check the `FAQ <https://signatory.readthedocs.io/en/latest/pages/miscellaneous/faq.html#miscellaneous-faq-importing>`__. If that doesn't help then feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.



Documentation
#############
The documentation is available `here <https://signatory.readthedocs.io>`__.

Example
#######
Usage is straightforward.

.. code-block:: python

    import signatory
    import torch
    # batch size is 1
    # length of input stream is 10
    # number of channels is 2
    x = torch.rand(1, 10, 2)
    # Compute signature to depth 4
    signatory.signature(x, 4)

For further examples, see the `documentation <https://signatory.readthedocs.io/en/latest/pages/examples/examples.html>`__.


Citation
########
If you found this library useful in your research, please consider citing

.. code-block:: bibtex

    @misc{signatory,
        title={{Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU}},
        author={Kidger, Patrick},
        note={\texttt{https://github.com/patrick-kidger/signatory}},
        year={2019}
    }


Acknowledgements
################

The Python bindings for the C++ code were written with the aid of `pybind11 <https://github.com/pybind/pybind11>`__.

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`__ package. The notes accompanying the iisignature project greatly helped with the implementation of Signatory.