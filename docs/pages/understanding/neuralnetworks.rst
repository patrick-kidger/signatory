.. _understanding-neuralnetworks:

Neural networks
###############
The universal nonlinearity property (mentioned :ref:`here<understanding-whataresignatures>`) requires the whole, infinite, signature. This doesn't fit in your computer's memory. The solution is simple: truncate the signature to some finite collection of statistics, and then embed it within a nonlinear model, like a neural network. The signature transform now instead acts as a pooling function, doing a provably good job of extracting information.

Have a look at `this <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__ for a more in-depth look at integrating it into neural neural networks.

As a general recommendation:

* The number of terms in signatures can grow rapidly with depth and number of channels, so experiment with what is an acceptable amount of work.

* Place small stream-preserving neural networks before the signature transform; these typically greatly enhance the power of the signature transform. This can be done easily with the :class:`signatory.Augment` class.

* It's often worth augmenting the input stream with an extra 'time' dimension. This can be done easily with the :class:`signatory.Augment` class. (Have a look at Appendix A of `this <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__ for an understanding of what augmenting with time gives you, and when you may or may not want to do it.)