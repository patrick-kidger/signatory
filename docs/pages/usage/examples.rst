.. currentmodule:: usage-examples

Examples
--------
In principle a simple augment-signature-linear model is enough to achieve universal approximation:

.. code-block:: python

    import signatory
    import torch
    from torch import nn

    class SigNet(nn.Module):
        def __init__(self, in_channels, out_dimension, sig_depth):
            super(SigNet, self).__init__()
            self.augment = signatory.Augment(in_channels=in_channels,
                                             layer_sizes=(),
                                             kernel_size=1,
                                             include_original=True,
                                             include_time=True)
            self.signature = signatory.Signature(depth=sig_depth,
                                                 basepoint=True)
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(in_channels=in_channels + 1,
                                                        depth=sig_depth)
            self.linear = torch.nn.Linear(sig_channels,
                                          out_dimension)

        def forward(self, inp):
            # inp is a three dimensional tensor of shape (batch, stream, in_channels)
            x = self.augment(inp)
            if x.size(1) <= 1:
                raise RuntimeError("Given an input with too short a stream to take the"
                                   " signature")
            # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
            # as time has been added as a value
            y = self.signature(x)
            # y is a two dimensional tensor of shape (batch, terms), corresponding to
            # the terms of the signature
            z = self.linear(y)
            # z is a two dimensional tensor of shape (batch, out_dimension)
            return z

Whilst in principle this exhibits universal approximation, adding some learnt transformation before the signature transform tends to improve things. See `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`__.

.. code-block:: python

    import signatory
    import torch
    from torch import nn

    class SigNet2(nn.Module):
        def __init__(self, in_channels, out_dimension, sig_depth):
            super(SigNet2, self).__init__()
            self.augment = signatory.Augment(in_channels=in_channels,
                                             layer_sizes=(8, 8, 2),
                                             kernel_size=4,
                                             include_original=True,
                                             include_time=True)
            self.signature = signatory.Signature(depth=sig_depth,
                                                 basepoint=True)
            # +3 because signatory.Augment is used to add time, and 2 other channels,
            # as well
            sig_channels = signatory.signature_channels(in_channels=in_channels + 3,
                                                        depth=sig_depth)
            self.linear = torch.nn.Linear(sig_channels,
                                          out_dimension)

        def forward(self, inp):
            # inp is a three dimensional tensor of shape (batch, stream, in_channels)
            x = self.augment(inp)
            if x.size(1) <= 1:
                raise RuntimeError("Given an input with too short a stream to take the"
                                   " signature")
            # x in a three dimensional tensor of shape (batch, stream, in_channels + 3)
            y = self.signature(x)
            # y is a two dimensional tensor of shape (batch, sig_channels),
            # corresponding to the terms of the signature
            z = self.linear(y)
            # z is a two dimensional tensor of shape (batch, out_dimension)
            return z

The :class:`signatory.Signature` layer can be used multiple times. In this example the first :class:`signatory.Signature` layer is called with :attr:`stream` as True, so that the stream dimension is preserved. The signatures of all intermediate streams are returned as well, so as we still have a stream dimension, it is reasonable to take the signature again.

.. code-block:: python

    import signatory
    import torch
    from torch import nn

    class SigNet3(nn.Module):
        def __init__(self, in_channels, out_dimension, sig_depth):
            super(SigNet3, self).__init__()
            self.augment1 = signatory.Augment(in_channels=in_channels,
                                              layer_sizes=(8, 8, 4),
                                              kernel_size=4,
                                              include_original=True,
                                              include_time=True)
            self.signature1 = signatory.Signature(depth=sig_depth,
                                                  basepoint=True,
                                                  stream=True)

            # +5 because self.augment1 is used to add time, and 2 other
            # channels, as well
            sig_channels1 = signatory.signature_channels(in_channels=in_channels + 5,
                                                         depth=sig_depth)
            self.augment2 = signatory.Augment(in_channels=sig_channels1,
                                              layer_sizes=(8, 8, 4),
                                              kernel_size=4,
                                              include_original=False,
                                              include_time=False)
            self.signature2 = signatory.Signature(depth=sig_depth,
                                                  basepoint=True,
                                                  stream=False)

            # 4 because that's the final layer size in self.augment2
            sig_channels2 = signatory.signature_channels(in_channels=4,
                                                         depth=sig_depth)
            self.linear = torch.nn.Linear(sig_channels2, out_dimension)

        def forward(self, inp):
            # inp is a three dimensional tensor of shape (batch, stream, in_channels)
            a = self.augment1(inp)
            if a.size(1) <= 1:
                raise RuntimeError("Given an input with too short a stream to take the"
                                   " signature")
            # a in a three dimensional tensor of shape (batch, stream, in_channels + 5)
            b = self.signature1(a)
            # b is a three dimensional tensor of shape (batch, stream, sig_channels1)
            c = self.augment2(b)
            if c.size(1) <= 1:
                raise RuntimeError("Given an input with too short a stream to take the"
                                   " signature")
            # c is a three dimensional tensor of shape (batch, stream, 4)
            d = self.signature2(c)
            # d is a two dimensional tensor of shape (batch, sig_channels2)
            e = self.linear(d)
            # e is a two dimensional tensor of shape (batch, out_dimension)
            return e