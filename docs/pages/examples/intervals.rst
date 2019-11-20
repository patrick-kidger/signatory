.. _examples-intervals:

Computing signatures over multiple intervals of the same path efficiently
#########################################################################

The basic :func:`signatory.signature` function computes the signature of a whole stream of data. Sometimes we have a whole stream of data, and then want to compute the signature of just the data sitting in some subinterval.

Naively, we could just slice it:

.. code-block:: python

    import torch
    import signatory
    # WARNING! THIS IS SLOW AND INEFFICIENT CODE
    path = torch.rand(1, 1000, 5)
    sig1 = signatory.signature(path[:, :40, :], 3)
    sig2 = signatory.signature(path[:, 300:600, :], 3)
    sig3 = signatory.signature(path[:, 400:990, :], 3)
    sig4 = signatory.signature(path[:, 700:, :], 3)
    sig5 = signatory.signature(path, 3)

However in this scenario it is possible to be much more efficient by doing some precomputation, which can then allow for computing such signatures very rapidly. This is done by the :class:`signatory.Path` class.

.. code-block:: python

    import torch
    import signatory

    path = torch.rand(1, 1000, 5)
    path_class = signatory.Path(path, 3)
    sig1 = path_class.signature(0, 40)
    sig2 = path_class.signature(300, 600)
    sig3 = path_class.signature(400, 990)
    sig4 = path_class.signature(700, None)
    sig5 = path_class.signature()

In fact, the :class:`signatory.Path` class supports adding data to it as well:

.. code-block:: python

    import torch
    import signatory

    path1 = torch.rand(1, 1000, 5)
    path_class = signatory.Path(path1, 3)
    # path_class is considering a path of length 1000
    # calculate signatures as normal
    sig1 = path_class.signature(40, None)
    sig2 = path_class.signature(500, 600)
    # more data arrives
    path2 = torch.rand(1, 200, 5)
    path_class.update(path2)
    # path_class is now considering a path of length 1200
    sig3 = path_class.signature(900, 1150)

.. note::

    To be able to compute signatures over intervals like this, then of course :class:`signatory.Path` must hold information about the whole stream of data in memory.

    If only the signature of the whole path is of interest then the main :func:`signatory.signature` function will work fine.

    If the signature of a path for which data continues to arrive (analogous to the use of :meth:`signatory.Path.update` above) is of interest, then see :ref:`examples-online`, which demonstrates how to efficiently use the :func:`signatory.signature` function in this way.

    If the signature on adjacent disjoint intervals is required, and the signature on the union of these intervals is desired, then see :ref:`examples-combine` for how to compute the signature on each of these intervals, and how to efficiently combine them to find the signature on larger intervals. This then avoids the overhead of the :class:`signatory.Path` class.