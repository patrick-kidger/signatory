.. currentmodule:: reference

Reference
---------
At the heart of the package is the :func:`signatory.signature` function:

.. autofunction:: signatory.signature

.. autoclass:: signatory.Signature

----

The following functions are provided as a convenience.

.. autofunction:: signatory.signature_channels

.. autofunction:: signatory.extract_term

----

Finally the following :class:`torch.nn.Module` is essentially unrelated to signatures, but is provided as it is often useful in the same context. As described in `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__, it is often advantageous to augment a path before taking the signature.

.. autoclass:: signatory.Augment