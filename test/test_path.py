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
"""Tests the Path class."""


import gc
import random
import signatory
import torch
from torch import autograd
import weakref

import utils_testing as utils


class TestPath(utils.EnhancedTestCase):
    def test_signature(self):
        for c in utils.ConfigIter(inverse=False, stream=False, N=(1, 2), depth=(1, 2, 4)):
            path_obj = signatory.Path(c.path, c.depth, basepoint=c.basepoint)

            if c.basepoint is True:
                new_path = [torch.zeros(c.N, 1, c.C, dtype=torch.double, device=c.device), c.path]
            elif c.basepoint is False:
                new_path = [c.path]
            else:  # isinstance(self.basepoint, torch.Tensor) == True
                new_path = [c.basepoint.unsqueeze(1), c.path]
            for _ in range(random.choice([0, 0, 1, 2, 3])):
                length = random.choice([1, 2, 3])
                extra_path = torch.rand(c.N, length, c.C, device=c.device, dtype=torch.double)
                path_obj.update(extra_path)
                new_path.append(extra_path)

            basepointed_path = torch.cat(new_path, dim=1)

            for start in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
                for end in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
                    try:
                        sig = path_obj.signature(start, end)
                    except ValueError:
                        try:
                            c.signature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                        except ValueError:
                            continue
                        else:
                            self.fail(c.fail(start=start, end=end))
                    try:
                        true_sig = c.signature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                    except ValueError:
                        self.fail(c.fail(start=start, end=end))
                    if not true_sig.allclose(sig):
                        self.fail(c.diff_fail({'start': start, 'end': end}, sig=sig, true_sig=true_sig))

                    if (path_obj.signature_size(-3), path_obj.signature_size(-1)) != true_sig.shape:
                        self.fail(c.fail(path_signature_shape=path_obj.signature_shape,
                                         true_signature_shape=true_sig.shape))

                    if path_obj.signature_channels() != true_sig.size(-1):
                        self.fail(c.fail(path_signature_channels=path_obj.signature_channels(),
                                         true_signature_channels=true_sig.size(-1)))

                    if (path_obj.size(-3), path_obj.size(-1)) != basepointed_path.shape:
                        self.fail(c.fail(path_shape=path_obj.shape,
                                         true_shape=basepointed_path.shape))

                    if path_obj.channels() != basepointed_path.size(-1):
                        self.fail(c.fail(path_channels=path_obj.channels(),
                                         true_channels=basepointed_path.size(-1)))

    def test_logsignature(self):
        for c in utils.ConfigIter(mode=utils.all_modes, inverse=False, stream=False, N=(1, 2), depth=(1, 2, 4)):
            path_obj = signatory.Path(c.path, c.depth, basepoint=c.basepoint)

            if c.basepoint is True:
                new_path = [torch.zeros(c.N, 1, c.C, dtype=torch.double, device=c.device), c.path]
            elif c.basepoint is False:
                new_path = [c.path]
            else:  # isinstance(self.basepoint, torch.Tensor) == True
                new_path = [c.basepoint.unsqueeze(1), c.path]
            for _ in range(random.choice([0, 0, 1, 2, 3])):
                length = random.choice([1, 2, 3])
                extra_path = torch.rand(c.N, length, c.C, device=c.device, dtype=torch.double)
                path_obj.update(extra_path)
                new_path.append(extra_path)

            basepointed_path = torch.cat(new_path, dim=1)

            for start in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
                for end in range(-2 * path_obj.size(1), 2 * path_obj.size(1)):
                    try:
                        logsig = path_obj.logsignature(start, end, mode=c.signatory_mode)
                    except ValueError:
                        try:
                            c.logsignature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                        except ValueError:
                            continue
                        else:
                            self.fail(c.fail(start=start, end=end))
                    try:
                        true_logsig = c.logsignature(store=False, path=basepointed_path[:, start:end, :], basepoint=False)
                    except ValueError:
                        self.fail(c.fail(start=start, end=end))
                    if not true_logsig.allclose(logsig):
                        self.fail(c.diff_fail({'start': start, 'end': end}, logsig=logsig, true_logsig=true_logsig))

                    if (path_obj.logsignature_size(-3), path_obj.logsignature_size(-1)) != true_logsig.shape:
                        self.fail(c.fail(path_logsignature_shape=path_obj.logsignature_shape,
                                         true_logsignature_shape=true_logsig.shape))

                    if path_obj.logsignature_channels() != true_logsig.size(-1):
                        self.fail(c.fail(path_logsignature_channels=path_obj.logsignature_channels(),
                                         true_logsignature_channels=true_logsig.size(-1)))

    def test_gradient_signature(self):
        def gradchecked(path, depth, basepoint, update, start, end):
            path_obj = signatory.Path(path, depth, basepoint=basepoint)
            if update is not None:
                path_obj.update(update)
            return path_obj.signature(start, end)

        for c in utils.ConfigIter(inverse=False,
                                  stream=False,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            base_length = c.path.size(1)
            if isinstance(c.basepoint, torch.Tensor) or c.basepoint:
                base_length += 1
            for update_length in (False, 1, 2, 3):
                if update_length:
                    length = base_length + update_length
                    update = lambda: torch.rand(c.N, update_length, c.C, dtype=torch.double, device=c.device,
                                                requires_grad=True)
                else:
                    length = base_length
                    update = lambda: None

                for start in range(0, length + 1):
                    for end in range(start + 2, length + 1):
                        try:
                            autograd.gradcheck(gradchecked, (c.path, c.depth, c.basepoint, update(), start, end))
                        except Exception:
                            self.fail(c.fail(base_length=base_length, length=length, update_length=update_length))

    def test_gradient_logsignature(self):
        def gradchecked(path, depth, basepoint, update, start, end, mode):
            path_obj = signatory.Path(path, depth, basepoint=basepoint)
            if update is not None:
                path_obj.update(update)
            return path_obj.logsignature(start, end, mode=mode)

        for c in utils.ConfigIter(mode=utils.all_modes,
                                  inverse=False,
                                  stream=False,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            base_length = c.path.size(1)
            if isinstance(c.basepoint, torch.Tensor) or c.basepoint:
                base_length += 1
            for update_length in (False, 1, 2, 3):
                if update_length:
                    length = base_length + update_length
                    update = lambda: torch.rand(c.N, update_length, c.C, dtype=torch.double, device=c.device,
                                                requires_grad=True)
                else:
                    length = base_length
                    update = lambda: None

                for start in range(0, length + 1):
                    for end in range(start + 2, length + 1):
                        try:
                            autograd.gradcheck(gradchecked, (c.path, c.depth, c.basepoint, update(), start, end,
                                                             c.signatory_mode))
                        except Exception:
                            self.fail(c.fail(base_length=base_length, length=length, update_length=update_length,
                                             start=start, end=end))

    def test_ctx_dies_signature(self):
        for c in utils.ConfigIter(inverse=False,
                                  stream=False,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            path_obj = signatory.Path(c.path, c.depth, basepoint=c.basepoint)
            # TODO: this can fail if length is 2
            signatory_out = path_obj.signature(1, None)
            ctx = signatory_out.grad_fn
            ref = weakref.ref(ctx)
            del ctx
            del signatory_out
            gc.collect()
            self.assertIsNone(ref(), c.fail())

    def test_ctx_dies_logsignature(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  inverse=False,
                                  stream=False,
                                  requires_grad=True,
                                  size=utils.random_size(5)):
            path_obj = signatory.Path(c.path, c.depth, basepoint=c.basepoint)
            # TODO: this can fail if length is 2
            signatory_out = path_obj.logsignature(1, None, mode=c.signatory_mode)
            ctx = signatory_out.grad_fn
            ref = weakref.ref(ctx)
            del ctx
            del signatory_out
            gc.collect()
            self.assertIsNone(ref(), c.fail())
