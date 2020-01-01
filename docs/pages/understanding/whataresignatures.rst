.. _understanding-whataresignatures:

What is the signature transform?
################################
The *signature transform* is roughly analogous to the Fourier transform, in that it operates on a stream of data (often a time series). Whilst the Fourier transform extracts information about frequency, the signature transform extracts information about *order* and *area*. Furthermore (and unlike the Fourier transform), order and area represent all possible nonlinear effects: the signature transform is a *universal nonlinearity*, meaning that every continuous function of the input stream may be approximated arbitrary well by a *linear* function of its signature. If you're doing machine learning then you probably understand why this is such a desirable property!

Besides this, the signature transform has many other nice properties -- robustness to missing or irregularly sampled data; optional translation invariance; optional sampling invariance. Furthermore it can be used to encode certain physical quantities, and may be used for data compression.

.. command.readme off

The definition of the signature transform can be a little bit intimidating -

.. admonition:: Definition

    Let :math:`\mathbf x = (x_1, \ldots, x_n)`, where :math:`x_i \in \mathbb R^d`. Linearly interpolate :math:`\mathbf x` into a path :math:`f = (f^1, \ldots, f^d) \colon [0, 1] \to \mathbb R^d`. The signature transform to depth :math:`N` of :math:`\mathbf x` is defined as 

    .. math::

        \mathrm{Sig}(\mathbf x) = \left(\left( \,\underset{0 < t_1 < \cdots < t_k < 1}{\int\cdots\int} \prod_{j = 1}^k \frac{\mathrm d f^{i_j}}{\mathrm dt}(t_j) \mathrm dt_1 \cdots \mathrm dt_k \right)_{1 \leq i_1, \ldots, i_k \leq d}\right)_{1 \leq k \leq N}.

Really understanding the mathematics behind the signature transform is frankly pretty hard, but you probably don't need to understand how it works -- just how to use it.

.. command.readme on

Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on the use of the signature transform in machine learning, just as a feature transformation, and `this <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__ for a more in-depth look at integrating the signature transform into neural networks.

.. command.readme off

Furthermore, efficient ways of computing it are somewhat nontrivial -- but they do exist. Now if only someone had already written a :ref:`package to compute it for you<index>`...

.. note::

    Recall that the signature transform extracts information about both order and area. This is because order and area are actually (in some sense) the same concept. For a (very simplistic) example of this: consider the functions :math:`f(x) = x(1-x)` and :math:`g(x) = x(x-1)` for :math:`x \in [0, 1]`. Then the area of :math:`f` is :math:`\int_0^1 f(x) \mathrm{d} x = \tfrac{1}{6}` whilst the area of :math:`g` is :math:`\int_0^1 g(x) \mathrm{d} x = \tfrac{-1}{6}`. Meanwhile, the graph of :math:`f` goes *up* then *down*, whilst the graph of :math:`g` goes *down* then *up*: the order of the ups and downs corresponds to the area.