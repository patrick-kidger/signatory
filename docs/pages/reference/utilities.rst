.. _reference-utilities:

Utilities
#########

.. currentmodule:: signatory

The following miscellaneous operations are provided as a convenience.

----

.. autofunction:: signatory.max_parallelism

----

This :class:`torch.nn.Module` is essentially unrelated to signatures, but is provided as it is often useful in the same context. As described in `Deep Signature Transforms -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__, it is often advantageous to augment a path before taking the signature.

.. autoclass:: signatory.Augment

    .. automethod:: signatory.Augment.forward

----

Signatures may be thought of as a sum of coefficients of words. This gives the words in the order that they correspond to the values returned by :func:`signatory.signature`.

.. autofunction:: signatory.all_words

----

Computing logsignatures involves operations in the free Lie algebra, which may be understood in terms of Lyndon words or the Lyndon basis. In particular logsignatures may be thought of as a sum of coefficients of Lyndon words. These next two functions compute these words, and their standard bracketing, in the order that they correspond to the values returned by :func:`signatory.logsignature`.

.. autofunction:: signatory.lyndon_words

.. autofunction:: signatory.lyndon_brackets