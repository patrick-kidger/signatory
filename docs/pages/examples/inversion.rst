.. _examples-inversion:

Inversion of signatures
#######################

We show below a simple example of signature inversion. A crucial parameter is the :attr:`depth` chosen for the signature: the function :func:`signatory.invert_signature` reconstructs a piecewise linear path with :attr:`depth` + 1 points that approximates the original path the signature was computed on.

.. code-block:: python

    import math
    import torch
    import signatory

    # Create a path consisting in a half circle
    time = torch.linspace(0, 1, 10)
    path = torch.stack([torch.cos(math.pi * time), torch.sin(math.pi * time)]).T.unsqueeze(0)

    # Compute the signature
    depth = 11
    signature = signatory.signature(path, depth)

    # Reconstruct the path by inverting the signature
    reconstructed_path = signatory.invert_signature(signature, depth, path.shape[2], initial_position=path[:, 0, :])


Note that the signature being translation invariant, we have given the first position of the path as argument to :func:`signatory.invert_signature`. Otherwise, the :attr:`reconstructed_path` would begin at zero.

We show below the original path (blue curve) and its reconstruction (orange curve).

.. image:: /_static/inversion/Half_circle_inversion.png
    :width: 400