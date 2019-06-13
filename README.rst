*********
Signatory
*********
High-performance signature computations in PyTorch, on both CPU and GPU, accelerated via C++ extensions, including backpropagation.

What is the signature transform?
--------------------------------
The 'signature transform' is a transformation that does a particularly good job extracting features from streams of data. See `here <https://arxiv.org/abs/1603.03788>`_ for a primer on its use in machine learning, as a feature transformation. See `here <https://arxiv.org/abs/1905.08494>`_ for a more in-depth look at building it into neural network models , as an arbitrary layer anywhere within a neural network. It's pretty cool.

In brief: the signature of a path determines the path essentially uniquely, and does so in an efficient, computable way.  Furthermore, the signature is rich enough that every continuous function of the path may be approximated arbitrarily well by a linear function of its signature; it is what we call a ‘universal nonlinearity’. Now for various reasons this is a mathematical idealisation not borne out in practice (which is why we put them in a neural network and don't just use a simple linear model), but they still work very well!

Installation
------------
::

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    pip install .

Documentation
-------------
The documentation is available `here <https://signatory.readthedocs.io>`_.

FAQ
---
* What's the difference between ``signatory`` and `iisignature <https://github.com/bottler/iisignature>`_?

The main difference is that ``iisignature`` is NumPy-based and CPU-only; I believe it was written primarily with mathematical research in mind. Meanwhile ``signatory`` is for PyTorch and may also run on the GPU; it is targeted towards machine learning applications.

The two packages also use different embeddings from streams of data into path space; see the next question.

* I get different results when I use the `iisignature <https://github.com/bottler/iisignature>`_ package?

The signature transform is defined on paths; given a stream of data we must decide how to embed it into a path. ``iisignature`` uses a piecewise linear embedding, whilst ``signatory`` uses a piecewise constant embedding. From a data science point of view, both are equally arbitrary, so as long as you pick one and stick with it shouldn't matter.

This embedding was selected for ``signatory`` because signature calculations for this embedding may be done much more rapidly (with provably fewer scalar multiplications); empirically it runs 3-4 times faster on the CPU than ``iisignature`` on reasonably sized batches. (`signatory` is optimised for batched operations, using batches to be `cache-friendly <https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code>`_.)

Citation
--------
If you found this library useful in your research, please consider citing::

    @article{deepsignatures,
        title={{Deep Signatures}},
        author={{Bonnier, Patric and Kidger, Patrick and Perez Arribas, Imanol and Salvi, Cristopher and Lyons, Terry}},
        journal={arXiv:1905.08494},
        year={2019}
    }

which this project was a spin-off from.

Acknowledgements
----------------
The Python bindings for the C++ code were written with the aid of `pybind11 <https://github.com/pybind/pybind11>`_.

For NumPy-based CPU-only signature calculations, you may also be interested in the `iisignature <https://github.com/bottler/iisignature>`_ package, which was a source of inspiration for signatory.