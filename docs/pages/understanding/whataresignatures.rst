.. _understanding-whataresignatures:

What are signatures?
####################
If you're reading this then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what this 'signature' thing is...

The *signature transform* is a transformation that takes in a stream of data (often a time series), and returns a collection of statistics about that stream of data, called the *signature*. This collection of statistics determines the path essentially uniquely. Importantly, the signature is rich enough that every continuous function of the input stream may be approximated arbitrarily well by a linear function of its signature; the signature transform is what we call a *universal nonlinearity*. If you're doing machine learning then you probably understand why this is such a desirable property!

.. genreadme off

The definition of the signature transform can be a little bit intimidating -

.. admonition:: Definition

    Let :math:`\mathbf x = (x_1, \ldots, x_n)`, where :math:`x_i \in \mathbb R^d`. Linearly interpolate :math:`\mathbf x` into a path :math:`f = (f^1, \ldots, f^d) \colon [0, 1] \to \mathbb R^d`. The signature of :math:`\mathbf x` is defined as :math:`\mathrm{Sig}(\mathbf x) = \mathrm{Sig}(f)``, where

    .. math::

        \mathrm{Sig}(f) = \left(\left( \,\underset{0 < t_1 < \cdots < t_k < 1}{\int\cdots\int} \prod_{j = 1}^k \frac{\mathrm d f^{i_j}}{\mathrm dt}(t_j) \mathrm dt_1 \cdots \mathrm dt_k \right)_{1 \leq i_1, \ldots, i_k \leq d}\right)_{k\geq 0}.

But if you're just using the signature transform then you don't need to worry about really understanding what all of that means -- just how to use it. Computing it is somewhat nontrivial. Now if only someone had already written a :ref:`package to compute it for you<index>`...

In principle the signature transform is quite similar to the Fourier transform: it is a transformation that can be applied to a stream of data which extracts certain information. The Fourier transform describes frequencies; meanwhile the signature most naturally describes *order* and *area*. The order of events, potentially in different channels, is a particularly easy thing to understand using the signature. Similarly various notions of area are also easy to understand.

.. note::

    It turns out that order and area are actually in some sense the same concept. For a (very simplistic) example of this: consider the functions :math:`f(x) = x(1-x)` and :math:`g(x) = x(x-1)` for :math:`x \in [0, 1]`. Then the area of :math:`f` is :math:`\int_0^1 f(x) \mathrm{d} x = \tfrac{1}{6}` whilst the area of :math:`g` is :math:`\int_0^1 g(x) \mathrm{d} x = \tfrac{-1}{6}`. Meanwhile, the graph of :math:`f` goes *up* then *down*, whilst the graph of :math:`g` goes *down* then *up*: the order of the ups and downs corresponds to the area.

.. genreadme on

Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on the use of the signature transform in machine learning, just as a feature transformation, and `this <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at integrating the signature transform into neural networks.