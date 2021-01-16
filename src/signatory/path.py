# Copyright 2019 Patrick Kidger. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""Provides the Path class, a high-level object capable of giving signatures over intervals."""

import bisect
import copy
import numpy as np
import torch
from torch import autograd
from torch.autograd import function as autograd_function

from . import signature_module as smodule
from . import logsignature_module as lmodule
from . import impl

from typing import Dict, List, Tuple, Union


class _BackwardShortcut(autograd.Function):
    @staticmethod
    def forward(ctx, signature, depth, scalar_term, *path_pieces):
        if len(path_pieces) == 0:
            raise ValueError('path_pieces must have nonzero length')

        # Record the tensors upon which the backward calculation depends
        save_for_backward = [signature]
        save_for_backward.extend(path_pieces)
        ctx.save_for_backward(*save_for_backward)
        ctx.depth = depth
        ctx.scalar_term = scalar_term

        return signature

    @staticmethod
    @autograd_function.once_differentiable  # Our backward function uses in-place operations for memory efficiency
    def backward(ctx, grad_signature):
        # Test for any in-place changes
        # This isn't perfect. If any of the stored tensors do not own their own storage (which is possible, as we don't
        # get to control these tensors) then this check will always pass, even if they've been modified in-place.
        # (PyTorch bug 24413)
        saved_tensors = ctx.saved_tensors
        signature = saved_tensors[0]
        path_pieces = saved_tensors[1:]

        length = 0
        for piece in path_pieces:
            length += piece.size(-3)
        p = path_pieces[0]
        path_increments = torch.empty(length - 1, p.size(-2), p.size(-1), device=p.device, dtype=p.dtype)
        torch.sub(p[1:], p[:-1], out=path_increments[:p.size(0) - 1])
        prev_piece = p
        next_path_increment = p.size(0) - 1
        for piece in path_pieces[1:]:
            torch.sub(piece[0], prev_piece[-1], out=path_increments[next_path_increment])
            next_path_increment += 1
            next_next_path_increment = next_path_increment + piece.size(0) - 1
            torch.sub(piece[1:], piece[:-1], out=path_increments[next_path_increment:next_next_path_increment])
            next_path_increment = next_next_path_increment
            prev_piece = piece
        # The above is basically the same as:
        # path = torch.cat(path_pieces, dim=0)
        # path_increments = path[1:] - path[:-1]
        # Except it doesn't waste time copying values like torch.cat would

        grad_path, _, _ = impl.signature_backward(grad_signature,
                                                  signature,
                                                  path_increments,
                                                  ctx.depth,
                                                  False,  # stream
                                                  False,  # basepoint
                                                  False,  # inverse
                                                  False,  # initial
                                                  ctx.scalar_term)

        result = [None, None, None]
        start = 0
        end = 0
        for elem in path_pieces:
            end += elem.size(-3)  # stream dimension
            result.append(grad_path[start:end])
            start = end

        return tuple(result)


# This wires up a shortcut through the backward operation.
# The already-computed signature is just returned during the forward operation.
# And the backward operation through signature is not computed in favour of shortcutting through path_pieces. (Which
# is assumed to be the path which has this signature!)
def _backward_shortcut(signature, path_pieces, depth, scalar_term):
    # (batch, stream, channel) to (stream, batch, channel)
    path_pieces = [path_piece.transpose(0, 1) for path_piece in path_pieces]
    # .detach() so that no gradients are taken through this argument
    return _BackwardShortcut.apply(signature.detach(), depth, scalar_term, *path_pieces)


class Path:
    """Calculates signatures and logsignatures on intervals of an input path.

    By doing some precomputation, it can rapidly calculate the signature or logsignature over any slice of the input
    path. This is particularly useful if you need the signature or logsignature of a path over many different intervals:
    using this class will be much faster than computing the signature or logsignature of each sub-path each time.

    May be efficiently sliced and indexed along its batch dimension via [] syntax. This will return a new Path without
    copying the underlying data.

    Arguments:
        path (torch.Tensor): As :func:`signatory.signature`.

        depth (int): As :func:`signatory.signature`.

        basepoint (bool or torch.Tensor, optional): As :func:`signatory.signature`.

        remember_path (bool, optional): Defaults to True. Whether to record the :attr:`path` argument that this was
            called with. If True, then it will be accessible as the :code:`.path` attribute. This argument may be set to
            False if it is known that the underlying path is no longer of interest and should not kept in memory just
            because it was passed as an argument here.

        scalar_term (bool, optional): Defaults to False. Whether to include the scalar '1' when calling the
            :meth:`signatory.Path.signature` method; see also the equivalent argument for :func:`signatory.signature`.
    """

    # !! If you change this, make sure to adjust __eq__ and __copy__ accordingly.
    __slots__ = ('_remember_path', '_scalar_term', '_depth', '_signature', '_inverse_signature', '_path',
                 '_length', '_signature_length', '_lengths', '_signature_lengths', '_batch_size', '_channels',
                 '_device', '_signature_channels', '_logsignature_channels', '_end',
                 '_signature_to_logsignature_instances')

    def __init__(self, path: torch.Tensor, depth: int, basepoint: Union[bool, torch.Tensor] = False,
                 remember_path: bool = True, scalar_term: bool = False, **kwargs):
        self._remember_path: bool = remember_path
        self._scalar_term: bool = scalar_term
        self._depth: int = depth

        self._signature: List[torch.Tensor] = []
        self._inverse_signature: List[torch.Tensor] = []

        self._path: List[torch.Tensor] = []

        self._length: int = 0
        self._signature_length: int = 0
        self._lengths: List[int] = []
        self._signature_lengths: List[int] = []

        self._batch_size: int = path.size(-3)
        self._channels: int = path.size(-1)
        self._device: torch.device = path.device
        self._signature_channels: int = smodule.signature_channels(self._channels, self._depth, self._scalar_term)
        self._logsignature_channels: int = lmodule.logsignature_channels(self._channels, self._depth)

        self._end: Union[bool, torch.Tensor] = basepoint

        self._signature_to_logsignature_instances: Dict[Tuple[int, int, str, bool],
                                                        lmodule.SignatureToLogSignature] = {}

        if remember_path:
            use_basepoint, basepoint_value = smodule.interpret_basepoint(basepoint, path.size(0), path.size(2),
                                                                         path.dtype, path.device)
            if use_basepoint:
                self._length += 1
                self._lengths.append(1)
                self._path.append(basepoint_value.unsqueeze(-2))  # unsqueeze a stream dimension

        self._update(path, None, None)

        super(Path, self).__init__(**kwargs)

    def __copy__(self):
        # Represents copying everything except tensors; i.e. all the cheap stuff to copy.

        # Stupid hackery necessary to "call super().__copy__", which doesn't actually exist
        copy_method = type(self).__copy__
        try:
            del type(self).__copy__
            new_path = copy.copy(self)
        finally:
            type(self).__copy__ = copy_method

        for attr_name in self.__slots__:
            if attr_name not in ('_end', '_signature_to_logsignature_instances'):
                attr_value = getattr(self, attr_name)
                # Many of these objects are immutable so the copy isn't actually important
                setattr(new_path, attr_name, copy.copy(attr_value))
        if not isinstance(self._end, torch.Tensor):
            # copying a bool in this case... completely unnecessary but consistent with what we do above.
            new_path._end = copy.copy(self._end)
        # Not really necessary because SignatureToLogSignature has only immutable state, but that's an implementation
        # detail that we shouldn't rely on.
        new_path._signature_to_logsignature_instances = copy.deepcopy(self._signature_to_logsignature_instances)

        return new_path

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        for attr_name in self.__slots__:
            if attr_name not in ('_signature', '_inverse_signature', '_path', '_end',
                                 '_signature_to_logsignature_instances'):
                if getattr(self, attr_name) != getattr(other, attr_name):
                    return False
        if not isinstance(self._end, type(other._end)) or not isinstance(other._end, type(self._end)):
            return False
        if isinstance(self._end, torch.Tensor):
            if (self._end != other._end).any():
                return False
        else:
            if self._end != other._end:
                return False
        for attr_name in ('_signature', '_inverse_signature', '_path'):
            self_value = getattr(self, attr_name)
            other_value = getattr(self, attr_name)
            if len(self_value) != len(other_value):
                return False
            for self_tensor, other_tensor in zip(self_value, other_value):
                if (self_tensor != other_tensor).any():
                    return False
        return True

    def __ne__(self, other):
        return not self == other

    def signature(self, start: Union[int, None] = None, end: Union[int, None] = None) -> torch.Tensor:
        """Returns the signature on a particular interval.

        Arguments:
            start (int or None, optional): Defaults to the start of the path. The start point of the interval to 
                calculate the signature on.

            end (int or None, optional): Defaults to the end of the path. The end point of the interval to calculate
                the signature on.

        Returns:
            The signature on the interval :code:`[start, end]`.

            In the simplest case, when :attr:`path` and :attr:`depth` are the arguments that this class was initialised
            with (and :attr:`basepoint` was not passed), then this function returns a value equal to
            :code:`signatory.signature(path[start:end], depth)`.

            In general, let :code:`p = torch.cat(self.path, dim=1)`, so that it is all given paths (including those
            :attr:`path` from both initialistion and :meth:`signatory.Path.update`, and any :attr:`basepoint`)
            concatenated together. Then this function will return a value equal to
            :code:`signatory.signature(p[start:end], depth)`.
        """

        # Record for error messages if need be
        old_start = start
        old_end = end

        # Interpret start and end in the same way as slicing behaviour
        if start is None:
            start = 0
        if end is None:
            end = self._length
        if start < -self._length:
            start = -self._length
        elif start > self._length:
            start = self._length
        if end < -self._length:
            end = -self._length
        elif end > self._length:
            end = self._length
        if start < 0:
            start += self._length
        if end < 0:
            end += self._length

        # Check that start and end are valid
        if end - start == 1:
            # Friendlier help message for a common mess-up.
            raise ValueError("start={}, end={} is interpreted as {}, {} for path of length {}, which "
                             "does not describe a valid interval. The given start and end differ by only one, but "
                             "a single point is not sufficent to define a path."
                             .format(old_start, old_end, start, end, self._length))
        if end - start < 2:
            raise ValueError("start={}, end={} is interpreted as {}, {} for path of length {}, which "
                             "does not describe a valid interval.".format(old_start, old_end, start, end, self._length))

        # Find the signature on [:end]
        sig_end = end - 2
        index_sig_end, sig_end = self._locate(self._signature_lengths, sig_end)
        signature = self._signature[index_sig_end][:, sig_end, :]

        # If start takes its minimum value then we've got the correct signature
        # Otherwise we need to apply the inverse signature of the preceding part of the path
        if start != 0:
            # Find the inverse signature on [:start]
            sig_start = start - 1
            index_sig_start, sig_start = self._locate(self._signature_lengths, sig_start)
            inverse_sig_at_start = self._inverse_signature[index_sig_start][:, sig_start, :]

            # Find the signature on [start:end]
            signature = smodule.multi_signature_combine([inverse_sig_at_start, signature], self._channels, self.depth,
                                                        scalar_term=self._scalar_term)

        # Find path[start:end]
        path_pieces = []
        index_end, end = self._locate(self._lengths, end)
        index_start, start = self._locate(self._lengths, start)
        if index_start == index_end:
            path_pieces.append(self.path[index_start][:, start:end, :])
        else:
            path_pieces.append(self.path[index_start][:, start:, :])
            for path_piece in self.path[index_start + 1:index_end]:
                path_pieces.append(path_piece)
            if end != 0:
                # self.path[index_end] is off-the-end if end == 0
                # and the path we'd append here is of zero length
                path_pieces.append(self.path[index_end][:, :end, :])

        # We know that we're only returning the signature on [start:end], and that there is no dependence on the region
        # [0:start]. But if we were to compute the backwards operation naively then this information wouldn't be used.
        #
        # What's returned would be treated as inverse_sig[0:start] \otimes sig[0:end] and we'd backprop through the
        # whole [0:start] region unnecessarily. We'd end up doing a whole lot of work to find that there's a zero
        # gradient on path[0:start].
        # (Or actually probably find that there's some very small gradient due to floating point errors...)
        #
        # This obviously isn't desirable if start takes a large value - lots of unnecessary work - so here we insert a
        # custom backwards that shortcuts that whole procedure.
        return _backward_shortcut(signature, path_pieces, self._depth, self._scalar_term)

    @staticmethod
    def _locate(lengths, index):
        lengths_index = bisect.bisect_right(lengths, index)
        if lengths_index > 0:
            index -= lengths[lengths_index - 1]
        return lengths_index, index

    def logsignature(self, start: Union[int, None] = None, end: Union[int, None] = None,
                     mode: str = "words") -> torch.Tensor:
        """Returns the logsignature on a particular interval.

        Arguments:
            start (int or None, optional): As :meth:`signatory.Path.signature`.

            end (int or None, optional): As :meth:`signatory.Path.signature`.

            mode (str, optional): As :func:`signatory.logsignature`.

        Returns:
            The logsignature on the interval :attr:`[start, end]`. See the documentation for
            :meth:`signatory.Path.signature`.
        """
        signature = self.signature(start, end)
        try:
            signature_to_logsignature_instance = self._signature_to_logsignature_instances[(self._channels,
                                                                                            self._depth,
                                                                                            mode,
                                                                                            self._scalar_term)]
        except KeyError:
            signature_to_logsignature_instance = lmodule.SignatureToLogSignature(self._channels, self._depth,
                                                                                 stream=False, mode=mode,
                                                                                 scalar_term=self._scalar_term)
            self._signature_to_logsignature_instances[(self._channels,
                                                       self._depth,
                                                       mode,
                                                       self._scalar_term)] = signature_to_logsignature_instance
        return signature_to_logsignature_instance(signature)

    def update(self, path: torch.Tensor) -> None:
        """Concatenates the given path onto the path already stored.

        This means that the signature of the new overall path can now be asked for via :meth:`signatory.Path.signature`.
        Furthermore this will be dramatically faster than concatenating the two paths together and then creating a new
        Path object: the 'concatenation' occurs implicitly, without actually involving any recomputation or reallocation
        of memory.

        Arguments:
            path (torch.Tensor): The path to concatenate on. As :func:`signatory.signature`.
        """
        if path.size(-3) != self._batch_size:
            raise ValueError("Cannot append a path with different number of batch elements to what has already been "
                             "used.")
        if path.size(-1) != self._channels:
            raise ValueError("Cannot append a path with different number of channels to what has already been used.")
        initial = self._signature[-1][:, -1, :]
        inverse_initial = self._inverse_signature[-1][:, -1, :]
        self._update(path, initial, inverse_initial)

    def _update(self, path, initial, inverse_initial):
        signature = smodule.signature(path, self._depth, stream=True, basepoint=self._end, initial=initial,
                                      scalar_term=self._scalar_term)
        inverse_signature = smodule.signature(path, self._depth, stream=True, basepoint=self._end, inverse=True,
                                              initial=inverse_initial, scalar_term=self._scalar_term)
        self._signature.append(signature)
        self._inverse_signature.append(inverse_signature)

        if self.remember_path:
            self._path.append(path)
        self._end = path[:, -1, :].clone()  # clone to use new memory so the old can be freed

        self._length += path.size(-2)
        self._signature_length += signature.size(-2)
        self._lengths.append(self._length)
        self._signature_lengths.append(self._signature_length)

    @property
    def remember_path(self) -> bool:
        """Whether this Path remembers the :attr:`path` argument it was called with."""
        return self._remember_path

    @property
    def path(self) -> List[torch.Tensor]:
        """The path(s) that this Path was created with."""
        if self.remember_path:
            return self._path
        else:
            raise RuntimeError('This Path object has not retained a reference to the path it was called with. The Path '
                               'object must have been initialised with `remember_path=True`.')

    @property
    def depth(self) -> int:
        """The depth that Path has calculated the signature to."""
        return self._depth

    def size(self, index: Union[int, None] = None) -> Union[int, torch.Size]:
        """The size of the input path. As :meth:`torch.Tensor.size`.

        Arguments:
            index (int or None, optional): As :meth:`torch.Tensor.size`.

        Returns:
            As :meth:`torch.Tensor.size`.
        """
        if index is None:
            return self.shape
        else:
            return self.shape[index]

    @property
    def shape(self) -> torch.Size:
        """The shape of the input path. As :attr:`torch.Tensor.shape`."""
        return torch.Size([self._batch_size, self._length, self._channels])

    # Method not property for consistency with signature_channels and logsignature_channels
    def channels(self) -> int:
        """The number of channels of the input stream."""
        return self._channels

    def signature_size(self, index: Union[int, None] = None) -> Union[int, torch.Size]:
        """The size of the signature of the path. As :meth:`torch.Tensor.size`.

        Arguments:
            index (int or None, optional): As :meth:`torch.Tensor.size`.

        Returns:
            As :meth:`torch.Tensor.size`.
        """
        if index is None:
            return self.signature_shape
        else:
            return self.signature_shape[index]

    @property
    def signature_shape(self) -> torch.Size:
        """The shape of the signature of the path. As :attr:`torch.Tensor.shape`."""
        return torch.Size([self._batch_size, self._signature_length, self._signature_channels])

    # Method not property for consistency with signatory.signature_channels
    def signature_channels(self) -> int:
        """The number of signature channels; as :func:`signatory.signature_channels`."""
        return self._signature_channels

    def logsignature_size(self, index: Union[int, None] = None) -> Union[int, torch.Size]:
        """The size of the logsignature of the path. As :meth:`torch.Tensor.size`.

        Arguments:
            index (int or None, optional): As :meth:`torch.Tensor.size`.

        Returns:
            As :meth:`torch.Tensor.size`.
        """
        if index is None:
            return self.logsignature_shape
        else:
            return self.logsignature_shape[index]

    @property
    def logsignature_shape(self) -> torch.Size:
        """The shape of the logsignature of the path. As :attr:`torch.Tensor.shape`."""
        return torch.Size([self._batch_size, self._signature_length, self._logsignature_channels])

    # Method not property for consistency with signatory.signature_channels
    def logsignature_channels(self) -> int:
        """The number of logsignature channels; as :func:`signatory.logsignature_channels`."""
        return self._logsignature_channels

    def _getitem_inplace(self, item):
        # Have to make sure we only allow things that preserve the batch dimension. As a special case we allow integers
        # by turning them into slices.
        not_valid = True
        if isinstance(item, int):
            item = slice(item, item + 1)
            not_valid = False
        elif isinstance(item, slice):
            not_valid = False
        elif isinstance(item, torch.Tensor):
            if item.ndimension() == 1:
                not_valid = False
        elif isinstance(item, np.ndarray):
            if item.ndim == 1:
                not_valid = False
        elif isinstance(item, list):
            if all([isinstance(elem, int) for elem in item]):
                not_valid = False
        if not_valid:
            raise IndexError("Only integers, slices, one dimensional Tensors, one dimensional numpy arrays, and lists "
                             "of integers, are valid indices.")
        new_batch_size = self._signature[0][item].size(0)
        if new_batch_size == 0:
            raise IndexError("Index corresponds to a batch of size zero, which is disallowed.")

        new_signature = [tensor[item] for tensor in self._signature]
        new_inverse_signature = [tensor[item] for tensor in self._inverse_signature]
        new_path = [tensor[item] for tensor in self._path]
        if isinstance(self._end, torch.Tensor):
            new_end = self._end[item]
        else:
            new_end = self._end

        # Only assign them after we're certain they've all been created successfully, lest we leave self in a
        # half-modified state.
        # Probably overkill as I think if an error is going to be thrown as it will always be thrown on the
        # new_signature line before we assign anything, but it doesn't hurt to be sure.
        self._signature = new_signature
        self._inverse_signature = new_inverse_signature
        self._path = new_path
        self._end = new_end
        self._batch_size = new_batch_size

    def shuffle(self):
        """Randomly permutes the Path along its batch dimension, and returns as a new Path. Returns a tuple of the
        new Path object, and the random permutation that produced it."""
        new_path = copy.copy(self)  # shallow copy
        _, perm = new_path.shuffle_()
        return new_path, perm

    def shuffle_(self):
        """In place version of :meth:`signatory.Path.shuffle`."""
        perm = torch.randperm(self.size(-3), device=self._device)
        self._getitem_inplace(perm)
        return self, perm

    def __getitem__(self, item):
        new_path = copy.copy(self)  # shallow copy
        new_path._getitem_inplace(item)
        return new_path
