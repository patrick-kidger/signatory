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


import iisignature
import signatory
import torch


with_grad = object()
without_grad = object()


expand_mode = 'expand'
words_mode = 'words'
brackets_mode = 'brackets'
all_modes = (expand_mode, words_mode, brackets_mode)


class Information(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)
        super(Information, self).__init__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            pieces = []
            for key, val in self.kwargs:
                pieces.append("{}: {}".format(key, val))
            raise exc_type('\n'.join(pieces))


def diff(arg1, arg2):
    if not arg1.allclose(arg2):
        diff = arg1 - arg2
        max_diff = torch.max(torch.abs(diff))
        assert False, 'diff={diff}\nmax_diff={max_diff}\narg1={arg1}arg2={arg2}'.format(diff=diff, max_diff=max_diff,
                                                                                        arg1=arg1, arg2=arg2)


_iisignature_prepare_cache = {}


def iisignature_prepare(channels, depth):
    try:
        return _iisignature_prepare_cache[(channels, depth)]
    except KeyError:
        prepared = iisignature.prepare(channels, depth)
        _iisignature_prepare_cache[(channels, depth)] = prepared
        return prepared


_signatory_logsignature_classes_cache = set()


def cache_signatory_logsignature_instances(*args, **kwargs):
    s = signatory.LogSignature(*args, **kwargs)
    _signatory_logsignature_classes_cache.add(s)
    return s
