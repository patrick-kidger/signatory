.. _reference-libraryapi:

Library API
###########

.. module:: signatory

For quick reference these are a list of all provided functions, grouped by which reference page they are on.

:ref:`reference-signatures`

.. autosummary::
    :nosignatures:

    signatory.signature
    signatory.Signature
    signatory.signature_channels
    signatory.extract_signature_term
    signatory.signature_combine
    signatory.multi_signature_combine

:ref:`reference-logsignatures`

.. autosummary::
    :nosignatures:

    signatory.logsignature
    signatory.LogSignature
    signatory.logsignature_channels
    signatory.signature_to_logsignature
    signatory.SignatureToLogSignature

:ref:`reference-path`

.. autosummary::
    :nosignatures:

    signatory.Path

:ref:`reference-signatures-inversion`

.. autosummary::
    :nosignatures:

    signatory.invert_signature

:ref:`reference-utilities`

.. autosummary::
    :nosignatures:

    signatory.Augment
    signatory.all_words
    signatory.lyndon_words
    signatory.lyndon_brackets

.. toctree::
    :caption: Reference pages

    /pages/reference/signatures
    /pages/reference/logsignatures
    /pages/reference/path
    /pages/reference/signatures-inversion
    /pages/reference/utilities