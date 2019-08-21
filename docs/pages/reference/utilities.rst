.. currentmodule:: reference-utilities

Utilities
---------

The following miscellaneous operations are provided as a convenience.

----

This :class:`torch.nn.Module` is essentially unrelated to signatures, but is provided as it is often useful in the same context. As described in `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__, it is often advantageous to augment a path before taking the signature.

.. autoclass:: signatory.Augment

----

Computing logsignatures involves operations in the free Lie algebra, which may be understood in terms of Lyndon words or the Lyndon basis. These operations are aided by the following functions.

.. autofunction:: signatory.lyndon_words

.. autofunction:: signatory.lyndon_brackets