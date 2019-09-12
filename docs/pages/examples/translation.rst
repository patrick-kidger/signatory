.. _examples-translation:

Translation invariance
######################
The signature is translation invariant. That is, given some stream of data :math:`x_1, \ldots, x_n` with :math:`x_i \in \mathbb{R}^c`, and some :math:`y \in \mathbb{R}^c`, then the signature of :math:`x_1, \ldots, x_n` is equal to the signature of :math:`x_1 + y, \ldots, x_n + y`.

Sometimes this is desirable, sometimes it isn't. If it isn't desirable, then the simplest solution is to add a 'basepoint'. That is, add a point :math:`0 \in \mathbb{R}^c` to the start of the path. This will allow us to notice any translations, as the signature of :math:`0, x_1, \ldots, x_n` and the signature of :math:`0, x_1 + y, \ldots, x_n + y` will be different.

In code, this can be accomplished very easily by using the :attr:`basepoint` argument. Simply set it to :attr:`True` to add such a basepoint to the path before taking the signature:

.. code-block:: python

    import torch
    import signatory
    path = torch.rand(2, 10, 5)
    sig = signatory.signature(path, 3, basepoint=True)