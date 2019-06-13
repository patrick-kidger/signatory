Signatory
---------
High-performance signature computations in PyTorch, on both CPU and GPU, accelerated via C++ extensions, including backpropagation.

What is the signature transform?
--------------------------------
The 'signature transform' is a transformation that does a particularly good job extracting features from streams of data. See [here](https://arxiv.org/abs/1603.03788) for a primer on its use in machine learning, as a feature transformation. See [here](https://arxiv.org/abs/1905.08494) for a more in-depth look at building it into neural network models , as an arbitrary layer anywhere within a neural network. It's pretty cool.

In brief: the signature of a path determines the path essentially uniquely, and does so in an efficient, computable way.  Furthermore, the signature is rich enough that every continuous function of the path may be approximated arbitrarily well by a linear function of its signature; it is what we call a ‘universal nonlinearity’. Now for various reasons this is a mathematical idealisation not borne out in practice (which is why we put them in a neural network and don't just use a simple linear model), but they still work very well!

.. toctree::
    :maxdepth: 2

    installation
    faq
    citation
    acknowledgements