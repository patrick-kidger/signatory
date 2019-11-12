.. _reference-path:

Path
####

.. currentmodule:: signatory

.. autoclass:: signatory.Path
    :members:

.. warning::

    If repeatedly making forward and backward passes (for example when training a neural network) and you have a learnt layer before the :class:`signatory.Path`, then make sure to construct a new :class:`signatory.Path` object for each forward pass.

    Reusing the same object between forward passes will mean that signatures aren't computed using the latest information, as the internal buffers will still correspond to the data passed in when the :class:`signatory.Path` object was first constructed.