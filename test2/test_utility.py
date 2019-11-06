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
"""Tests some of the utility-style functions within the package."""


import iisignature
import signatory

import helpers as h


def _iisignature_convert(ii_elem):
    outstr = ''
    for character in ii_elem:
        if character == '1':
            outstr += '0'
        elif character == '2':
            outstr += '1'
        elif character == '3':
            outstr += '2'
        elif character == '4':
            outstr += '3'
        elif character == '5':
            outstr += '4'
        elif character == '6':
            outstr += '5'
        elif character == '7':
            outstr += '6'
        elif character == '8':
            outstr += '7'
        elif character == '9':
            outstr += '8'
        elif character == '?':
            outstr += '9'
        else:
            outstr += character
    return outstr


def test_lyndon_brackets(iisignature_prepare):
    """Tests the lyndon_brackets function"""
    for channels in range(2, 11):  # iisignature supports channels with unique symbols in the range 2 to 10 inclusive
        for depth in range(1, 6):
            iisignature_brackets = iisignature.basis(iisignature_prepare(channels, depth))
            signatory_brackets = signatory.lyndon_brackets(channels, depth)
            for ii_elem, sig_elem in zip(iisignature_brackets, signatory_brackets):
                with h.Information(channels=channels, depth=depth, ii_elem=ii_elem, sig_elem=sig_elem):
                    assert sig_elem == eval(_iisignature_convert(ii_elem))


def test_lyndon_words(iisignature_prepare):
    """Tests the lyndon_words function"""
    for channels in range(2, 11):  # iisignature supports channels with unique symbols in the range 2 to 10 inclusive
        for depth in range(1, 6):
            iisignature_brackets = iisignature.basis(iisignature_prepare(channels, depth))
            signatory_words = signatory.lyndon_words(channels, depth)
            for ii_elem, sig_elem in zip(iisignature_brackets, signatory_words):
                ii_elem_new = ii_elem.replace('[', '').replace(']', '').replace(',', '')
                ii_elem_new = _iisignature_convert(ii_elem_new)
                sig_elem_new = ''.join(str(i) for i in sig_elem)
                with h.Information(channels=channels, depth=depth, ii_elem=ii_elem, sig_elem=sig_elem,
                                   ii_elem_new=ii_elem_new, sig_elem_new=sig_elem_new):
                    assert sig_elem_new == ii_elem_new


def test_lyndon_amount():
    """Tests that the lyndon_words and lyndon_brackets functions gives the same number of elements"""
    for channels in range(1, 10):
        for depth in range(1, 6):
            len_words = len(signatory.lyndon_words(channels, depth))
            len_brackets = len(signatory.lyndon_brackets(channels, depth))
            with h.Information(channels=channels, depth=depth, len_words=len_words, len_brackets=len_brackets):
                assert len_words == len_brackets


def test_signature_channels():
    """Tests the signature_channels function"""
    for channels in range(1, 16):
        for depth in range(1, 15):
            result = signatory.signature_channels(channels, depth)
            sum_over = sum(channels ** i for i in range(1, depth + 1))
            with h.Information(channels=channels, depth=depth, result=result, sum_over=sum_over):
                assert result == sum_over


def test_logsignature_channels():
    """Tests the logsignature_channels function"""
    for channels in range(1, 10):
        for depth in range(1, 6):
            result = signatory.logsignature_channels(channels, depth)
            from_words = len(signatory.lyndon_words(channels, depth))
            with h.Information(channels=channels, depth=depth, result=result, from_words=from_words):
                assert result == from_words
