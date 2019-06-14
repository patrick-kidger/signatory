.. currentmodule:: reference

Reference
---------
At the heart of the package is the ``signatory.signature`` function:

.. autofunction:: signatory.signature

----

The ``signatory.signature`` function is conveniently wrapped into a ``torch.nn.Module``:

.. autoclass:: signatory.Signature

----

The following function is provided as a convenience to count the number of channels (that is, the number of scalar terms) in the signature of a path.

.. autofunction:: signatory.signature_channels

----

Finally the following ``torch.nn.Module`` is provided as a convenience. As described in `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`_, it is often advantageous to augment a path before taking the signature.

.. autoclass:: signatory.Augment