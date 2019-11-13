.. _reference-signatures:

Signatures
##########

.. currentmodule:: signatory

At the heart of the package is the :func:`signatory.signature` function.

.. note::

    It comes with quite a lot of optional arguments, but most of them won't need to be used for most use cases. See :ref:`examples-simple` for a straightforward look at how to use it.

.. autofunction:: signatory.signature

.. autoclass:: signatory.Signature

    .. automethod:: signatory.Signature.forward

.. autofunction:: signatory.signature_channels

.. autofunction:: signatory.extract_signature_term

.. autofunction:: signatory.signature_combine

.. autofunction:: signatory.multi_signature_combine