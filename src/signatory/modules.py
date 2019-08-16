import torch
from torch import nn
from torch.nn import functional as F

# noinspection PyUnreachableCode
if False:
    from typing import Any, Callable, Tuple, Union

from . import backend
# noinspection PyProtectedMember, PyUnresolvedReferences
from ._impl import _make_lyndon_info


class Signature(nn.Module):
    """Module wrapper around the :func:`signatory.signature` function.

    Arguments:
        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

    Called with a single argument :attr:`path` of type :class:`torch.Tensor`.
    """

    def __init__(self, depth, stream=False, basepoint=False, **kwargs):
        # type: (int, bool, Union[bool, torch.Tensor], **Any) -> None
        super(Signature, self).__init__(**kwargs)
        self.depth = depth
        self.stream = stream
        self.basepoint = basepoint

    def forward(self, path):
        # type: (torch.Tensor) -> torch.Tensor
        return backend.signature(path, self.depth, self.stream, self.basepoint)

    def extra_repr(self):
        return 'depth={depth}, stream={stream}, basepoint={basepoint}'.format(depth=self.depth, stream=self.stream,
                                                                              basepoint=str(self.basepoint)[:6])


class LogSignature(nn.Module):
    """Module wrapper around the :func:`signatory.logsignature` function.

    Calling this module on an input `path` with the same number of channels as the last input `path` it was called with
    will be faster than the corresponding :func:`signatory.logsignature` function, as this module caches the result of
    certain computations which depend only on this value. (For larger numbers of channels, this speedup will be
    substantial.)

    Arguments:
        depth (int): as :func:`signatory.logsignature`.

        stream (bool, optional): as :func:`signatory.logsignature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.logsignature`.

        mode (str, optional): as :func:`signatory.logsignature`.

    Called with a single argument :attr:`path` of type :class:`torch.Tensor`.
    """

    def __init__(self, depth, stream=False, basepoint=False, mode="brackets", **kwargs):
        # type: (int, bool, Union[bool, torch.Tensor], str, **Any) -> None
        super(LogSignature, self).__init__(**kwargs)
        self.depth = depth
        self.stream = stream
        self.basepoint = basepoint
        self.mode = mode

        self.path_size = None
        self.lyndon_info = None

    def forward(self, path):
        # type: (torch.Tensor) -> torch.Tensor
        if self.path_size != path.size(-1):
            self.lyndon_info = _make_lyndon_info(path.size(-1), self.depth)

        # don't call logsignature itself because that (deliberately) doesn't expose a lyndon_info argument.
        # noinspection PyProtectedMember, PyUnresolvedReferences
        return backend._LogSignatureFunction.apply(path, self.depth, self.stream, self.basepoint, self.mode,
                                                   self.lyndon_info)

    def extra_repr(self):
        return ('depth={depth}, stream={stream}, basepoint={basepoint}, mode{mode}'
                .format(depth=self.depth, stream=self.stream, basepoint=str(self.basepoint)[:6], mode=self.mode))


class Augment(nn.Module):
    r"""Augmenting the stream before feeding it into a signature is often useful; the hope is to obtain higher-order
    information in the signature. One way to do this is in a data-dependent way is to apply a feedforward neural network
    to sections of the stream, so as to obtain another stream; on this stream the signature is then applied.

    The input path is expected to be a three-dimensional tensor, with dimensions :math:`(N, L, C)`, where :math:`N` is
    the batch size, :math:`L` is the length of the input sequence, and :math:`C` denotes the number of channels. Thus
    each batch element is interpreted as a stream of data :math:`(x_1, \ldots, x_L)`, where each
    :math:`x_i \in \mathbb{R}^C`. (This is the same as :class:`torch.nn.Conv1d`, for example.)

    Then this stream may be 'augmented' via some function

    .. math::
        \phi \colon \mathbb{R}^{C \times k} \to \mathbb{R}^{\widehat{C}}

    giving a stream of data

    .. math::
        \left(\phi(x_1, ... x_k), \ldots, \phi(x_{n - k + 1}, \ldots, x_n)\right),

    which is essentially a three-dimensional tensor with dimensions :math:`(N, L - k + 1, \widehat{C})`.

    Thus this essentially operates as a one dimensional convolution, except that a whole network is swept across the
    input, rather than just a single convolutional layer.

    Both the original stream and time can be specifically included in the augmentation. (This usually tends to give
    better empirical results.) For example, if both :attr:`include_original` is True and :attr:`include_time` is True,
    then each :math:`\phi(x_i, ... x_{k + i - 1})` is of the form

    .. math::
        \left(\frac{i}{T}, x_i, \varphi(x_i, ... x_{k + i - 1})\right).

    where :math:`T` is a constant appropriately chosen so that the first entry moves between :math:`0` and :math:`1` as
    :math:`i` varies. (Specifically, :math:`T = L - k + 1 + 2 \times \text{padding}`.)

    For further details see `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`_. (This module
    is here more for convenience: it doesn't directly relate to the signature transform; it's just useful to have around
    when you are using the signature transform.)

    Arguments:
        in_channels (int): Number of channels :math:`C` in the input stream.

        layer_sizes (tuple of int): Specifies the sizes of the layers of the feedforward neural network to apply to
            the stream. The final value of this tuple specifies the number of channels in the augmented stream,
            corresponding to the value :math:`\widehat{C}` in the preceding discussion.

        kernel_size (int): Specifies the size of the kernel to slide over the stream, corresponding to the value
            :math:`k` in the preceding discussion.

        stride (int, optional): Defaults to 1. How far to move along the input stream before re-applying the
            feedforward neural network. Thus the output stream is given by

            .. math::
                (\phi(x_1, \ldots, x_k),
                 \phi(x_{1 + \text{stride}}, \ldots, x_{k + 2 \times \text{stride}}),
                 \phi(x_{1 + 2 \times \text{stride}}, \ldots, x_{k + 2 \times \text{stride}}),
                 \ldots)

        padding (int, optional): Defaults to 0. How much zero padding to add to either end of the the input stream
            before sweeping the feedforward neural network along it.

        dilation (int, optional): The spacing between input elements given to the feedforward neural network.
            Defaults to 1. Harder to describe; see the equivalent argument for :class:`torch.nn.Conv1d`.

        bias (bool, optional): Defaults to True. Whether to use biases in the neural network.

        activation (callable, optional): Defaults to ReLU. The activation function to use in the feedforward neural
            network.

        include_original (bool, optional): Defaults to True. Whether or not to include the original stream
            (pre-augmentation) in the augmented stream.

        include_time (bool, optional): Defaults to True. Whether or not to also augment the stream with a 'time' value.
            These are values in :math:`[0, 1]` corresponding to how far along the stream dimension the element is.

    .. note::

        Thus the resulting stream of data has shape :math:`(N, L, \text{out_channels})`, where in pseudocode:

        .. code-block:: python

            out_channels = layer_sizes[-1]
            if include_original:
                out_channels += in_channels
            if include_time:
                out_channels += 1

    """

    def __init__(self,
                 in_channels,            # type: int
                 layer_sizes,            # type: Tuple[int, ...]
                 kernel_size,            # type: int
                 stride=1,               # type: int
                 padding=0,              # type: int
                 dilation=1,             # type: int
                 bias=True,              # type: bool
                 activation=F.relu,      # type: Callable[[torch.Tensor], torch.Tensor]
                 include_original=True,  # type: bool
                 include_time=True,      # type: bool
                 **kwargs                # type: Any
                 ):
        # type: (...) -> None
        super(Augment, self).__init__(**kwargs)

        if isinstance(layer_sizes, int):
            layer_sizes = (layer_sizes,)

        self.in_channels = in_channels
        self.layer_sizes = layer_sizes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.activation = activation
        self.include_original = include_original
        self.include_time = include_time

        self.convs = nn.ModuleList()
        if layer_sizes:
            self.convs.append(nn.Conv1d(in_channels=in_channels,
                                        out_channels=layer_sizes[0],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        bias=bias))
            last_layer_channels = layer_sizes[0]
            for augment_channel in layer_sizes[1:]:
                # These pointwise convolutions correspond to sliding a standard feedforward network across the input.
                self.convs.append(nn.Conv1d(in_channels=last_layer_channels,
                                            out_channels=augment_channel,
                                            kernel_size=1,
                                            bias=bias))
                last_layer_channels = augment_channel

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        if len(x.shape) != 3:
            raise RuntimeError('Argument x should have three dimensions, (batch, stream, channel). Given shape'
                               '{shape} dimensions with {x}.'.format(shape=x.shape, x=x))
        len_truncated = x.size(1) - self.kernel_size + 1
        pieces = []
        if self.include_original:
            truncated_x = x.narrow(1, self.kernel_size - 1, len_truncated)
            pieces.append(truncated_x)

        if self.include_time:
            time = torch.linspace(0, 1, len_truncated, dtype=torch.float, device=x.device)
            time.unsqueeze_(1)
            time = time.expand(x.size(0), len_truncated, 1)
            pieces.append(time)

        if self.layer_sizes:
            augmented_x = self.convs[0](x.transpose(1, 2))
            for conv in self.convs[1:]:
                augmented_x = self.activation(augmented_x)
                augmented_x = conv(augmented_x)
            augmented_x.transpose_(1, 2)
            pieces.append(augmented_x)
        return torch.cat(pieces, dim=2)  # concatenate along channel axis

    def extra_repr(self):
        return ('activation={activation}, include_original={include_original}, include_time={include_time}'
                ).format(activation=self.activation, include_original=self.include_original,
                         include_time=self.include_time)
