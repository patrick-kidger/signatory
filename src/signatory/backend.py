import torch
from torch import autograd

# noinspection PyProtectedMember, PyUnresolvedReferences
from ._impl import (_LogSignatureMode,
                    _signature_forward,
                    _signature_backward,
                    _logsignature_forward,
                    _logsignature_backward)

# noinspection PyUnreachableCode
if False:
    from typing import Union


def _forward(ctx, path, depth, stream, basepoint, fn_forward, extra_args=()):
    ctx.basepoint = basepoint

    if basepoint is True:
        basepoint_value = torch.zeros((path.shape[0], path.shape[2]), dtype=path.dtype, device=path.device)
    elif isinstance(basepoint, torch.Tensor):
        basepoint_value = basepoint
        basepoint = True
    else:
        basepoint_value = torch.Tensor()

    result, backwards_info = fn_forward(path, depth, stream, basepoint, basepoint_value, *extra_args)
    if ctx.requires_grad:
        ctx.backwards_info = backwards_info
        ctx.save_for_backward(result)

    return result


def _backward(ctx, grad_result, fn_backward):
    # Just to check that the result of the forward pass hasn't been modified in-place. (Which would make the result
    # of the backwards calculation be incorrect!) The reason we don't use the tensor itself is because another
    # handle to the same information is already saved in ctx.backwards_info.
    _ = ctx.saved_tensors
    # Actually the check doesn't work at the moment: https://github.com/pytorch/pytorch/issues/24413

    grad_path, grad_basepoint_value = fn_backward(grad_result, ctx.backwards_info)
    if not isinstance(ctx.basepoint, torch.Tensor):
        grad_basepoint_value = None
    return grad_path, None, None, grad_basepoint_value


class _SignatureFunction(autograd.Function):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint):
        return _forward(ctx, path, depth, stream, basepoint, _signature_forward)

    @staticmethod
    @autograd.function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        return _backward(ctx, grad_result, _signature_backward)


class _LogSignatureFunction(_SignatureFunction):
    @staticmethod
    def forward(ctx, path, depth, stream, basepoint, mode):
        if mode == "expand":
            mode = _LogSignatureMode.Expand
        elif mode == "brackets":
            mode = _LogSignatureMode.Brackets
        elif mode == "words":
            mode = _LogSignatureMode.Words
        else:
            raise ValueError("Invalid values for argument 'mode'. Valid values are 'expand', 'brackets', or 'words'.")

        return _forward(ctx, path, depth, stream, basepoint, _logsignature_forward, (mode,))

    @staticmethod
    @autograd.function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_result):
        return (*_backward(ctx, grad_result, _logsignature_backward), None)


def signature(path, depth, stream=False, basepoint=False):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor]) -> torch.Tensor
    r"""Applies the signature transform to a stream of data.

    The input :attr:`path` is expected to be a three-dimensional tensor, with dimensions :math:`(N, L, C)`, where
    :math:`N` is the batch size, :math:`L` is the length of the input sequence, and :math:`C` denotes the number of
    channels. Thus each batch element is interpreted as a stream of data :math:`(x_1, \ldots, x_L)`, where each
    :math:`x_i \in \mathbb{R}^C`. (This is the same as :class:`torch.nn.Conv1d`, for example.)

    If :attr:`basepoint` is True then an additional point :math:`x_0 = 0 \in \mathbb{R}^C` is prepended to the path.
    (Alternatively it can be a :class:`torch.Tensor1` of shape :math:`(N, C)` specifying the point to prepend.)

    Each path is then lifted to a piecewise linear path :math:`X \colon [0, 1] \to \mathbb{R}^C`, and the signature
    transform of this path is then computed. This (by the definition of the signature transform) gives a sequence of
    tensors of shape

    .. math::
        (N, C), (N, C, C), \ldots (N, C, \ldots, C),

    where the final tensor has :attr:`depth` many dimensions of size :math:`C`. These are then flattened down and
    concatenated to give a single tensor of shape

    .. math::
        (N, C + C^2 + \cdots + C^\text{depth}).

    (This value may be computed via the :func:`signatory.signature_channels` function.)

    If :attr:`stream` is True then  the signatures of all paths :math:`(x_1, \ldots, x_j)`, for :math:`j=2, \ldots, L`,
    are computed. (Or :math:`(x_0, \ldots, x_j)`, for :math:`j=1, \ldots, L` if :attr:`basepoint` is provided. In
    neither case is the signature of the path of a single element computed, as that isn't defined.)

    Examples:
        If :attr:`stream` is False then the returned tensor will have shape

        .. math::
            (N, C + C^2 + \cdots + C^d).

        If :attr:`basepoint` is True and :attr:`stream` is True then the returned tensor will
        have shape

        .. math::
            (N, L, C + C^2 + \cdots + C^d),

        as the stream dimension is now preserved. See also the 'Returns' section below.

    Arguments:
        path (:class:`torch.Tensor`): The batch of input paths to apply the signature transform to.

        depth (int): The depth to truncate the signature at.

        stream (bool, optional): Defaults to False. If False then the signature transform of the whole path is computed.
            If True then the signature of all intermediate paths are also computed.

        basepoint (bool or :class:`torch.Tensor`, optional): Defaults to False. If True, then the input paths will have
            an additional point at the origin prepended to the start of the sequence. (If this is False then the
            signature transform is invariant to translations of the path, which may or may not be desirable.)
            Alternatively it may be a :class:`torch.Tensor` specifying the point to prepend, in which case it should
            have shape :math:`(N, C)`

    Returns:
        A :class:`torch.Tensor`. Given an input :class:`torch.Tensor` of shape :math:`(N, L, C)`, and input arguments
        :attr:`depth`, :attr:`basepoint`, :attr:`stream`, then the return value is, in pseudocode:

        .. code-block:: python

            if stream:
                if basepoint is True or isinstance(basepoint, torch.Tensor):
                    return torch.Tensor of shape (N, L, C + C^2 + ... + C^depth)
                else:
                    return torch.Tensor of shape (N, L - 1, C + C^2 + ... + C^depth)
            else:
                return torch.Tensor of shape (N, C + C^2 + ... + C^depth)

    """
    # noinspection PyUnresolvedReferences
    return _SignatureFunction.apply(path, depth, stream, basepoint)


def logsignature(path, depth, stream=False, basepoint=False, mode="brackets"):
    # type: (torch.Tensor, int, bool, Union[bool, torch.Tensor], str) -> torch.Tensor
    """Applies the logsignature transform to a stream of data.

    Applies the signatures to the stream of data, then takes the logarithm (in the truncated tensor algebra) of the
    result. (It is *not* taking the pointwise logarithm: the high-dimensional Euclidean space in which the signature
    lives has an interpretation as a "truncated tensor algebra", on which a particular notion of "logarithm" is defined.
    See the "Returns" section below for a little more detail.)

    In particular the :attr`modes` argument determines how the logsignature is represented.

    Arguments:
        path (:class:`torch.Tensor`): as :func:`signatory.signature`.

        depth (int): as :func:`signatory.signature`.

        stream (bool, optional): as :func:`signatory.signature`.

        basepoint (bool or :class:`torch.Tensor`, optional): as :func:`signatory.signature`.

        mode (str, optional): How the output should be presented. Valid values are "words", "brackets", or
            "expand". Precisely what each of these options mean is described in the "Returns" section below. As a rule
            of thumb: use "words" for new projects (as it is the fastest); use "brackets" if you want compatibility with
            other projects (such as `iisignature <https://github.com/bottler/iisignature>`__, which defaults to this).
            The mode "expand" is mostly only interesting for mathematicians.

    Returns:
        A :class:`torch.Tensor`. If :code:`mode == "expand"` then it will be of the same shape as the returned tensor
        from :func:`signatory.signature`. If :code:`mode in ("words", "brackets")` then it will again be of the
        same shape, except that the channel dimension will instead be of size
        :code:`logsignature_channels(in_channels, depth)`. (i.e. much smaller, which is the whole point of using the
        logsignature over the signature in the first place.)

        We now go on to explain what the different values for :attr:`mode` mean. This discussion is the "Returns"
        section a the value of :attr:`mode` essentially just determines how the output is represented; the
        mathematical meaning is the same in all cases. We start with an explanation for the reader who is not familiar
        with notions such as free Lie algebras and Lyndon words. Which is most people! For the more mathematically
        inclined reader, we have an more in-depth explanation later.

        First the low-math explanation.

        The signatures computed by the :func:`signatory.signature` function are a large collection of numbers; that is,
        they live in a high-dimensional Euclidean space (formally called the "truncated tensor algebra"). This space has
        a a particular notion of "logarithm", and when called with :code:`mode == "expand"` then the logsignature will
        also be displayed as a member of the truncated tensor algebra.

        However it turns out that this is space-inefficient: the logsignature may be represented by a smaller collection
        of numbers, that is, it's really a member of a smaller-dimensional Euclidean space (formally it is a "free Lie
        algebra"). Its representation in the high-dimensional truncated tensor algebra is simply a projection from this
        small space into the larger space. As such this probably isn't a useful value for :attr:`mode` when doing
        machine learning; you instead want a representation in the smaller space..

        A representation in this smaller space is what will be returned if :code:`mode in ("words", "brackets")`.
        (The keen-eyed reader will notice that since the logarithm is bijective, then this also means that the signature
        itself is also using more space than necessary to represent itself -- as it can just be represented by its
        corresponding logsignature, which lives in a smaller dimensional space. This is completely correct. However in
        some sense this larger representation is the whole point of the signature: the dependency between its terms is
        nonlinear, which give lots of interesting features to learn machine learning models on. The same larger
        representation for the logsignature only has linear dependencies amongst its terms, so the same statement isn't
        true there.)

        The smaller space has multiple different possible bases. That is, there are multiple ways to represent elements
        of this smaller space. We offer three different bases; all of them are just linear transformations of each
        other.

        A popular basis is called the "Lyndon basis", and this is the basis that is used if :code:`mode == "brackets"`.
        This is thus the default choice, as it is the result that is typically expected.

        However it turns out that there is a more computationally efficient basis which may be used: this is what is
        used if :code:`mode == "words"`. Thus this is the recommended choice for new projects.

        Now the high-math explanation.

        The explanation will now assume that you are familiar with Lyndon words and the Lyndon basis.

        As already explained, :code:`mode == "expand"` represents the logsignature in the truncated tensor algebra, so
        that the numbers you see are the coefficients of all possible words.

        Next, :code:`mode == "brackets"` represents the logsignature as coefficients of the Lyndon basis. The order of
        the coefficients corresponds to the lexicographic ordering of the foliage of the elements of the Lyndon basis.

        Now it turns out that the coefficients in the Lyndon basis may be computed by calculated by (a) computing the
        logsignature in the truncated tensor algebra, (b) extracting the coefficents of all Lyndon words, and (c)
        applying a unitriangular (i.e. triangular with 1 on the diagonal) linear transformation to them. In particular,
        this transformation is clearly invertible (as you'd expect, given that we're computing a basis), so we see that
        the coefficients of all the Lyndon words also contain the same information; the Lyndon words may be interpreted
        as forming a basis for the free Lie algebra. (And this basis in terms of the Lyndon words is *not* the same as
        the Lyndon basis!) The representation in this basis is what is computed when :code:`mode == "words"`. The order
        of the coefficients corresponds to the lexicographic ordering of the Lyndon words.
    """
    # noinspection PyUnresolvedReferences
    return _LogSignatureFunction.apply(path, depth, stream, basepoint, mode)
