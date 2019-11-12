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
"""Provides a helpers for functional reimplementation of parts of Signatory using iisignature."""


import torch


def iisignature_signature_or_logsignature(fn, path, depth, stream, basepoint, inverse):

    # This is a bit of an unholy mess. Turns out that it's quite hard to reproduce the effect of some of the arguments
    # of signatory.signature(...) or signatory.logsignature(...) using more naive methods.
    # It's almost like that's why those arguments exist in the first place.
    #
    # That said, both the signature and logsignature functions are similar enough that we can factor out all of this
    # code for both of them.

    batch_size, input_stream, input_channels = path.shape
    device = path.device
    dtype = path.dtype

    # We begin by assembling the input path.
    # First we consider the path that generated 'initial'
    iisignature_path_pieces = []
    # Then we add on the basepoint
    if isinstance(basepoint, torch.Tensor) or basepoint is True:
        if basepoint is True:
            iisignature_basepoint = torch.zeros(batch_size, 1, input_channels, device=device, dtype=dtype)
        else:
            iisignature_basepoint = basepoint.unsqueeze(1)
        iisignature_path_pieces.append(iisignature_basepoint)
    # Then we add on the actual path
    iisignature_path_pieces.append(path)

    # Now flip everything if inverse is used
    if inverse:
        iisignature_path_pieces_reversed = []
        for tensor in reversed(iisignature_path_pieces):
            iisignature_path_pieces_reversed.append(tensor.flip(1))
        iisignature_path_pieces = iisignature_path_pieces_reversed

    iisignature_path = torch.cat(iisignature_path_pieces, dim=1)

    # Now actually compute some signatures or logsignatures
    if stream:
        signature_length = input_stream - 1
        if isinstance(basepoint, torch.Tensor) or basepoint is True:
            signature_length += 1
        results = []
        if inverse:
            for i in range(signature_length):
                results.append(fn(iisignature_path[:, i:], depth))
            result = torch.stack(results, dim=1).flip(1)
        else:
            for i in range(iisignature_path.size(1) - signature_length + 1, iisignature_path.size(1) + 1):
                results.append(fn(iisignature_path[:, :i], depth))
            result = torch.stack(results, dim=1)
    else:
        result = fn(iisignature_path, depth)

    return result
