.. _examples-online:

Computing the signature of an incoming stream of data
#####################################################

Suppose we have the signature of a stream of data :math:`x_1, \ldots, x_{1000}`. Subsequently some more data arrives, say :math:`x_{1001}, \ldots, x_{1007}`. It is possible to calculate the signature of the whole stream of data :math:`x_1, \ldots, x_{1007}` with just this information. It is not necessary to compute the signature of the whole path from the beginning!

In code, this problem can be solved like this:

.. code-block:: python

    import torch
    import signatory

    # Generate a path X
    # Recall that the order of dimensions is (batch, stream, channel)
    X = torch.rand(1, 1000, 5)
    # Calculate its signature to depth 3
    sig_X = signatory.signature(X, 3)

    # Generate some more data for the path
    Y = torch.rand(1, 7, 5)
    # Calculate the signature of the overall path
    final_X = X[:, -1, :]
    sig_XY = signatory.signature(Y, 3, basepoint=final_X, initial=sig_X)

    # This is equivalent to
    XY = torch.cat([X, Y], dim=1)
    sig_XY = signatory.signature(XY, 3)

As can be seen, two pieces of information need to be provided: the final value of :attr:`X` along the stream dimension, and the signature of :attr:`X`. But not :attr:`X` itself.

The first method (using the :attr:`initial` argument) will be much quicker than the second (simpler) method. The first
method efficiently uses just the new information :attr:`Y`, whilst the second method unnecessarily iterates over all of
the old information :attr:`X`.

In particular note that we only needed the last value of :attr:`X`. If memory efficiency is a concern, then by using the first method we can discard the other 999 terms of :attr:`X` without an issue!

.. note::

    If the signature of :attr:`Y` on its own was also of interest, then it is possible to compute this first, and then combine it with :attr:`sig_X` to compute :attr:`sig_XY`. See :ref:`examples-combine`.