.. _examples-simple:

Simple example
##############
Here's a very simple example on using :func:`signatory.signature`.

.. code-block:: python

    import torch
    import signatory
    # Create a tensor of shape (2, 10, 5)
    # Recall that the order of dimensions is (batch, stream, channel)
    path = torch.rand(2, 10, 5)
    # Take the signature to depth 3
    sig = signatory.signature(path, 3)
    # sig is of shape (2, 155)

In this example, :attr:`path` is a three dimensional tensor, and the returned tensor is two dimensional. The first dimension of :attr:`path` corresponds to the batch dimension, and indeed we can see that this dimension is also in the shape of :attr:`sig`.

The second dimension of :attr:`path` corresponds to the 'stream' dimension, whilst the third dimension corresponds to channels. Mathematically speaking, that means that each batch element of :attr:`path` is interpreted as a sequence of points :math:`x_1, \ldots, x_{10}`, with each :math:`x_i \in \mathbb{R}^5`.

The output :attr:`sig` has batch dimension of size 2, just like the input. Its other dimension is of size 155. This is the number of terms in the depth-3 signature of a path with 5 channels. (This can also be computed with the helper function :func:`signatory.signature_channels`.)