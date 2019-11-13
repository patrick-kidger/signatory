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
"""A simple example."""


# start-literal-include
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
        self.signature = signatory.Signature(depth=sig_depth)
        # +1 because signatory.Augment is used to add time as well
        sig_channels = signatory.signature_channels(channels=in_channels + 1,
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
        y = self.signature(x, basepoint=True)
        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        z = self.linear(y)
        # z is a two dimensional tensor of shape (batch, out_dimension)
        return z
