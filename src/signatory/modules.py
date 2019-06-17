import torch
from torch import nn
from torch.nn import functional as F
import warnings

# noinspection PyUnreachableCode
if False:
    from typing import Any, Callable, Tuple, Union

from . import backend


class Signature(nn.Module):
    """Module wrapper around the signatory.signature function. See :func:`signatory.signature`.

    Arguments:
        depth: as :func:`signatory.signature`.

        basepoint: as :func:`signatory.signature`.

        stream: as :func:`signatory.signature`.

        flatten: as :func:`signatory.signature`.

    Called with a single argument :attr:`path` of type :class:`torch.Tensor`.
    """

    def __init__(self, depth, basepoint=False, stream=False, flatten=True, **kwargs):
        # type: (int, bool, bool, bool, **Any) -> None
        if not isinstance(depth, int) or depth < 1:
            raise ValueError('Depth must be an integer greater than or equal to one. Given {depth} of type '
                             '{tdepth}'.format(depth=depth, tdepth=type(depth)))
        super(Signature, self).__init__(**kwargs)
        self.depth = depth
        self.basepoint = basepoint
        self.stream = stream
        self.flatten = flatten

    def forward(self, path):
        # type: (torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        if path.size(1) == 1:
            warnings.warn('{clsname} called on path with only one channel; the signature is now just the moments of the'
                          ' path, so there is no interesting information from cross terms.'
                          .format(clsname=self.__class__.__name__))
        return backend.signature(path, self.depth, self.basepoint, self.stream, self.flatten)

    def extra_repr(self):
        return 'depth={depth}'.format(depth=self.depth)


class Augment(nn.Module):
    r"""Augmenting the stream before feeding it into a signature is crucial to obtain higher-order information in the
    signature. One way to do this is in a data-dependent way is to apply a feedforward neural network to sections of the
    stream, so as to obtain another stream; on this stream the signature is then applied.

    The input path is expected to be a three-dimensional tensor, with dimensions :math:`(N, C, L)`, where
    :math:`N` is the batch size, :math:`C` denotes the number of channels, and :math:`L` is the length of the input
    sequence. Thus each batch element is interpreted as a stream of data :math:`(x_1, \ldots, x_L)`, where each
    :math:`x_i \in \mathbb{R}^C`. (This is the same as :class:`torch.nn.Conv1d`, for example.)

    Then this stream may be 'augmented' via some function

    .. math::
        \phi \colon \mathbb{R}^{C \times k} \to \mathbb{R}^{\widehat{C}}

    giving a stream of data

    .. math::
        \left(\phi(x_1, ... x_k), \ldots, \phi(x_{n - k + 1}, \ldots, x_n)\right),

    which is essentially a three-dimensional tensor with dimensions :math:`(N, \widehat{C}, L - k + 1)`.

    Thus this essentially operates as a one dimensional convolution, except that a whole network is swept across the
    input, rather than just a single convolutional layer.

    Both the original stream and time can be specifically included in the augmentation. (This usually tends to give
    better empirical results.) For example, if both :attr:`include_original` is True and :attr:`include_time` is True,
    then each :math:`\phi(x_i, ... x_{k + i - 1})` is of the form

    .. math::
        \left(\frac{i}{T}, x_i, \varphi(x_i, ... x_{k + i - 1})\right).

    where :math:`T` is a constant appropriately chosen so that the first entry moves between :math:`0` and :math:`1` as
    :math:`i` varies. (Specifically, :math:`T = L - k + 1 + 2 \times \text{padding}`)

    For further details see `Deep Signatures -- Bonnier et al. 2019 <https://arxiv.org/abs/1905.08494>`_. (Thus this
    module is here more for convenience: it doesn't directly relate to the signature transform; it's just useful to have
    around when you are using the signature transform.)

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

        Thus the resulting stream of data has size ``out_channel``, where in pseudocode:

        .. code-block:: python

            out_channel = layer_sizes[-1]
            if include_original:
                out_channel += in_channels
            if include_time:
                out_channel += 1
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
            raise RuntimeError('Argument x should have three dimensions, (batch, channnel, stream). Given shape'
                               '{shape} dimensions with {x}.'.format(shape=x.shape, x=x))
        pieces = []
        if self.include_original:
            truncated_x = x.narrow(2, self.kernel_size - 1, x.size(2) - self.kernel_size + 1)
            pieces.append(truncated_x)

        if self.include_time:
            time = torch.linspace(0, 1, x.size(2) - self.kernel_size + 1, dtype=torch.float, device=x.device)
            time = time.expand(x.size(0), 1, -1)
            pieces.append(time)

        if self.layer_sizes:
            augmented_x = self.convs[0](x)
            for conv in self.convs[1:]:
                augmented_x = self.activation(augmented_x)
                augmented_x = conv(augmented_x)
            pieces.append(augmented_x)
        return torch.cat(pieces, dim=1)  # concatenate along channel axis

    def extra_repr(self):
        return ('activation={activation}, include_original={include_original}, include_time={include_time}'
                ).format(activation=self.activation, include_original=self.include_original,
                         include_time=self.include_time)
