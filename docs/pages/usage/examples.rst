.. currentmodule:: usage-examples

Examples
--------
In principle a simple augment-signature-linear model is enough to achieve universal approximation:

.. literalinclude:: /../test/example1.py
    :lines: 19-

Whilst in principle this exhibits universal approximation, adding some learnt transformation before the signature transform tends to improve things. See `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__.

.. literalinclude:: /../test/example2.py
    :lines: 19-

The :class:`signatory.Signature` layer can be used multiple times. In this example the first :class:`signatory.Signature` layer is called with :attr:`stream` as True, so that the stream dimension is preserved. The signatures of all intermediate streams are returned as well, so as we still have a stream dimension, it is reasonable to take the signature again.

.. literalinclude:: /../test/example3.py
    :lines: 19-