
|Signatory|
###########

.. |Signatory| image:: https://raw.githubusercontent.com/patrick-kidger/signatory/master/docs/_static/signatory.png

Differentiable computations of the signature and logsignature transforms, on both CPU and GPU.




What is the signature transform?
################################
The *signature transform* is roughly analogous to the Fourier transform, in that it operates on a stream of data (often a time series). Whilst the Fourier transform extracts information about frequency, the signature transform extracts information about *order* and *area*. Furthermore (and unlike the Fourier transform), order and area represent all possible nonlinear effects: the signature transform is a *universal nonlinearity*, meaning that every continuous function of the input stream may be approximated arbitrary well by a *linear* function of its signature. If you're doing machine learning then you probably understand why this is such a desirable property!

Besides this, the signature transform has many other nice properties -- robustness to missing or irregularly sampled data; optional translation invariance; optional sampling invariance. Furthermore it can be used to encode certain physical quantities, and may be used for data compression.


Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on the use of the signature transform in machine learning, just as a feature transformation, and `this <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__ for a more in-depth look at integrating the signature transform into neural networks.




Installation
############
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7 and Linux, Mac, Windows. Requires `PyTorch <http://pytorch.org/>`__ 1.2.0, 1.3.0, 1.3.1, or 1.4.0.

Install via:

.. code-block:: bash

    pip install signatory==<SIGNATORY_VERSION>.<TORCH_VERSION>

where ``<SIGNATORY_VERSION>`` is the version of Signatory you would like to download (the most recent version is 1.2.0) and ``<TORCH_VERSION>`` is the version of PyTorch you are using.


    For example, if you are using PyTorch 1.3.0 and want Signatory 1.1.4, then you should run:

    .. code-block:: bash

        pip install signatory==1.1.4.1.3.0

    Yes, this looks a bit odd. This is needed to work around limitations of `PyTorch <https://github.com/pytorch/pytorch/issues/28754>`__ and `pip <https://www.python.org/dev/peps/pep-0440/>`__.

    Take care **not** to run ``pip install signatory``, as this will likely download the wrong version.

After installation, just ``import signatory`` inside Python.

Installation from source is also possible; please consult the `documentation <https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#usage-install-from-source>`__. This also includes information on how to run the tests and benchmarks.

If you have any problems with installation then check the `FAQ <https://signatory.readthedocs.io/en/latest/pages/miscellaneous/faq.html#miscellaneous-faq-importing>`__. If that doesn't help then feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.



Documentation
#############
The documentation is available `here <https://signatory.readthedocs.io>`__.

Example
#######
Usage is straightforward. As a simple example,

.. code-block:: python

    import signatory
    import torch
    batch, stream, channels = 1, 10, 2
    depth = 4
    path = torch.rand(batch, stream, channels)
    signature = signatory.signature(path, depth)
    # signature is a PyTorch tensor

For further examples, see the `documentation <https://signatory.readthedocs.io/en/latest/pages/examples/examples.html>`__.


Citation
########
If you found this library useful in your research, please consider citing `the paper <https://arxiv.org/abs/2001.00706>`__.

.. code-block:: bibtex

    @article{signatory,
        title={{Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU}},
        author={Kidger, Patrick and Lyons, Terry},
        journal={arXiv:2001.00706},
        url={https://github.com/patrick-kidger/signatory},
        year={2020}
    }