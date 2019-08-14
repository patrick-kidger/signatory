.. currentmodule:: utilities

Utilities
---------

The following functions are provided as a convenience.

.. autofunction:: signatory.extract_term

.. autofunction:: signatory.signature_channels

.. autofunction:: signatory.logsignature_channels

----

The following :class:`torch.nn.Module` is essentially unrelated to signatures, but is provided as it is often useful in the same context. As described in `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__, it is often advantageous to augment a path before taking the signature.

.. autoclass:: signatory.Augment

----

Computing logsignatures involves operations in the free Lie algebra, which we represent in terms of either Lyndon words or the Lyndon basis. These operations are aided by the following functions.

.. autofunction:: signatory.lyndon_words

.. autofunction:: signatory.lyndon_brackets