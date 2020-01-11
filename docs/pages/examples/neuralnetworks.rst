.. _examples-neuralnetworks:

Using signatures in neural networks
###################################

In principle a simple augment-signature-linear model is enough to achieve universal approximation:

.. literalinclude:: /../examples/example1.py
    :language: python
    :start-after: # start-literal-include

Whilst in principle this exhibits universal approximation, adding some learnt transformation before the signature transform tends to improve things. See `Deep Signature Transforms -- Bonnier et al. 2019 <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__. Thus we might improve our model:

.. literalinclude:: /../examples/example2.py
    :language: python
    :start-after: # start-literal-include

The :class:`signatory.Signature` layer can be used multiple times in a neural network. In this next example the first :class:`signatory.Signature` layer is called with :attr:`stream` as True, so that the stream dimension is preserved. This means that the signatures of all intermediate streams are returned as well. So as we still have a stream dimension, it is reasonable to take the signature again.

.. literalinclude:: /../examples/example3.py
    :language: python
    :start-after: # start-literal-include