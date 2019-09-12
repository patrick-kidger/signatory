.. currentmodule:: examples-simple

Basic example
-------------
Here's a very simple example on using :func:`signatory.signature`.

.. code-block:: python

    import torch
    import signatory
    # Create a tensor of shape (2, 10, 5)
    path = torch.rand(2, 10, 5)
    # Take the signature to depth 3
    sig = signatory.signature(path, 3)
    # sig is of shape (2, 155)

In this example, :attr:`path` is a three dimensional tensor, and the returned tensor is two dimensional. The first dimension of :attr:`path` corresponsd to the batch dimension, and indeed we can see that this dimension is also in the shape of :attr:`sig`.

The second dimension of :attr:`path` corresponds to the 'stream' dimension, whilst the third dimension corresponds to channels. Mathematically speaking, that means that each batch element of :attr:`path` is interpreted as a sequence of points :math:`x_1, \ldots, x_{10}`, with each :math:`x_i \in \mathbb{R}^5`. We can then linearly interpolate to interpret this as a path :math:`X \colon [0, 1] \to \mathbb{R}^5`.

The output :attr:`sig` has batch dimension of size 2, just like the input. Its other dimension is of size 155. This is the number of terms in the depth-3 signature of a path with 5 channels. (It can be computed with the helper function :func:`signatory.signature_channels`.)


Translation invariance
----------------------
The signature is translation invariant. That is, given some stream of data :math:`x_1, \ldots, x_n` with :math:`x_i \in \mathbb{R}^c`, and some :math:`y \in \mathbb{R}^c`, then the signature of :math:`x_1, \ldots, x_n` is equal to the signature of :math:`x_1 + y, \ldots, x_n + y`.

Sometimes this is desirable, sometimes it isn't. If it isn't desirable, then the simplest solution is to add a 'basepoint'. That is, add a point :math:`0 \in \mathbb{R}^c` to the start of the path. This will allow us to notice any translations, as the signature of :math:`0, x_1, \ldots, x_n` and the signature of :math:`0, x_1 + y, \ldots, x_n + y` will be different.

In code, this can be accomplished very easily by using the :attr:`basepoint` argument. Simply set it to :attr:`True` to add such a basepoint to the path before taking the signature:

.. code-block:: python

    import torch
    import signatory
    path = torch.rand(2, 10, 5)
    sig = signatory.signature(path, 3, basepoint=True)