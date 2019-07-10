*********
Signatory
*********
High-performance computations of the signature transform for PyTorch, on both CPU and GPU, including backpropagation.

What is the signature transform?
--------------------------------
If you're reading this documentation then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what all the fuss is about...

The 'signature transform' is a transformation that does a particularly good job extracting features from streams of data. Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on its use in machine learning, as a feature transformation. Then have a look `here <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at building it into neural network models, as an arbitrary layer anywhere within a neural network. It's pretty cool.

In brief: the signature of a path determines the path essentially uniquely, and does so in an efficient, computable way.  Furthermore, the signature is rich enough that every continuous function of the path may be approximated arbitrarily well by a linear function of its signature; it is what we call a ‘universal nonlinearity’. Now for various reasons this is a mathematical idealisation not borne out in practice (which is why we put them in a neural network and don't just use a simple linear model), but they still work very well!

Installation
------------
Available for Python 2.7, Python 3.5, Python 3.6, Python 3.7.

Requires `PyTorch <http://pytorch.org/>`__. Tested with PyTorch version 1.0.1, but will probably work with other versions as well.

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
* What's the difference between Signatory_ and iisignature_?

The essential difference is that iisignature is NumPy-based and CPU-only. Meanwhile Signatory is for PyTorch and may also run on the GPU, as it is targeted towards machine learning applications.

Empirically Signatory is also about twice as fast (on the CPU) as iisignature on reasonably sized batches. (Signatory is optimised for batched operations, using batches to be `cache-friendly <https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code>`__.)

.. _iisignature: https://github.com/bottler/iisignature
.. _Signatory: https://github.com/patrick-kidger/signatory

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

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`__ package, which was a source of inspiration for Signatory.