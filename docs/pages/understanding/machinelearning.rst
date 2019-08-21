.. currentmodule:: understanding-machinelearning

Overview
--------
If you're reading this documentation then it's probably because you already know what the signature transform is, and are looking to use it in your project. But in case you've stumbled across this and are curious what this 'signature' thing is...

The 'signature transform' is a transformation that takes in a stream of data (often a time series), and returns a collection of statistics about that stream of data. This collection of statistics determines the path essentially uniquely, in an efficient computable way. Furthermore it is rich enough that every continuous function of the input stream may be approximated arbitrarily well by a linear function of its signature; the signature is what we call a 'universal nonlinearity'. If you're doing machine learning then you probably understand why this is such a desirable property!

In principle it's quite similar to the Fourier transform: it's a transformation that can be applied to a stream of data, which extracts certain information. The Fourier transform describes frequencies; the signature most naturally describes *order*. That is, the order of events, potentially in different channels, is a particularly easy thing to understand using the signature.

Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on its use in machine learning, just as a feature transformation, and `this <https://arxiv.org/abs/1905.08494>`__ for a more in-depth look at integrating it into neural neural networks.