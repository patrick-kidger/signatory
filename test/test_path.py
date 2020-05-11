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


import copy
import gc
import pytest
import random
import torch
import warnings
import weakref

from helpers import helpers as h
from helpers import validation as v


tests = ['Path']
depends = ['signature', 'logsignature']
signatory = v.validate_tests(tests, depends)


def _update_lengths_update_grads(maxlength):
    update_lengths = []
    update_grads = []
    num = int(torch.randint(low=0, high=3, size=(1,)))
    for _ in range(num):
        update_lengths.append(int(torch.randint(low=1, high=maxlength, size=(1,))))
        update_grads.append(random.choice([True, False]))
    return update_lengths, update_grads


def test_path():
    """Tests that Path behaves correctly."""
    # Test small edge cases thoroughly
    for device in h.get_devices():
        for batch_size in (1, 2):
            for input_stream, basepoints in zip((1, 2), ((True, h.without_grad, h.with_grad),
                                                         (False, True, h.without_grad, h.with_grad))):
                for input_channels in (1, 2):
                    for depth in (1, 2):
                        for scalar_term in (True, False):
                            for path_grad in (False, True):
                                basepoint = random.choice(basepoints)
                                update_lengths, update_grads = _update_lengths_update_grads(3)
                                _test_path(device, path_grad, batch_size, input_stream, input_channels, depth,
                                           basepoint, update_lengths, update_grads, scalar_term, extrarandom=False,
                                           which='all')

    # Randomly test larger cases
    for _ in range(50):
        device = random.choice(h.get_devices())
        batch_size = random.choice((1, 2, 5))
        input_stream = random.choice([3, 6, 10])
        input_channels = random.choice([1, 2, 6])
        depth = random.choice([1, 2, 4, 6])
        basepoint = random.choice([False, True, h.without_grad, h.with_grad])
        path_grad = random.choice([False, True])
        update_lengths, update_grads = _update_lengths_update_grads(10)
        scalar_term = random.choice([False, True])
        _test_path(device, path_grad, batch_size, input_stream, input_channels, depth,
                   basepoint, update_lengths, update_grads, scalar_term, extrarandom=True, which='random')

    # Do at least one large test
    for device in h.get_devices():
        _test_path(device, path_grad=True, batch_size=5, input_stream=10, input_channels=6, depth=6,
                   basepoint=True, update_lengths=[5, 6], update_grads=[False, True], scalar_term=False,
                   extrarandom=False, which='none')


def _randint(value):
    return torch.randint(low=0, high=value, size=(1,)).item()


def _test_path(device, path_grad, batch_size, input_stream, input_channels, depth, basepoint, update_lengths,
               update_grads, scalar_term, extrarandom, which):
    path = h.get_path(batch_size, input_stream, input_channels, device, path_grad)
    basepoint = h.get_basepoint(batch_size, input_channels, device, basepoint)
    path_obj = signatory.Path(path, depth, basepoint=basepoint, scalar_term=scalar_term)

    if isinstance(basepoint, torch.Tensor):
        full_path = torch.cat([basepoint.unsqueeze(1), path], dim=1)
    elif basepoint is True:
        full_path = torch.cat([torch.zeros(batch_size, 1, input_channels, device=device, dtype=torch.double), path],
                              dim=1)
    else:
        full_path = path

    if not path_grad and not (isinstance(basepoint, torch.Tensor) and basepoint.requires_grad):
        backup_path_obj = copy.deepcopy(path_obj)

        # derived objects to test
        copy_path_obj = copy.copy(path_obj)
        shuffle_path_obj1, perm1 = path_obj.shuffle()
        shuffle_path_obj2, perm2 = copy.deepcopy(path_obj).shuffle_()
        getitem1 = _randint(batch_size)
        getitem_path_obj1 = path_obj[getitem1]  # integer

        all_derived = [(copy_path_obj, slice(None)),
                       (shuffle_path_obj1, perm1),
                       (shuffle_path_obj2, perm2),
                       (getitem_path_obj1, getitem1)]

        start = _randint(batch_size)
        end = _randint(batch_size)
        getitem2 = slice(start, end)
        getitem3 = torch.randint(low=0, high=batch_size, size=(_randint(int(1.5 * batch_size)),))
        getitem4 = torch.randint(low=0, high=batch_size, size=(_randint(int(1.5 * batch_size)),)).numpy()
        getitem5 = torch.randint(low=0, high=batch_size, size=(_randint(int(1.5 * batch_size)),)).tolist()
        try:
            getitem_path_obj2 = path_obj[getitem2]  # slice, perhaps a 'null' slice
        except IndexError as e:
            if start >= end:
                pass
            else:
                pytest.fail(str(e))
        else:
            all_derived.append((getitem_path_obj2, getitem2))
        try:
            getitem_path_obj3 = path_obj[getitem3]  # 1D tensor
        except IndexError as e:
            if len(getitem3) == 0:
                pass
            else:
                pytest.fail(str(e))
        else:
            all_derived.append((getitem_path_obj3, getitem3))
        try:
            getitem_path_obj4 = path_obj[getitem4]  # array
        except IndexError as e:
            if len(getitem4) == 0:
                pass
            else:
                pytest.fail(str(e))
        else:
            all_derived.append((getitem_path_obj4, getitem4))
        try:
            getitem_path_obj5 = path_obj[getitem5]  # list
        except IndexError as e:
            if len(getitem5) == 0:
                pass
            else:
                pytest.fail(str(e))
        else:
            all_derived.append((getitem_path_obj5, getitem5))

        if which == 'random':
            all_derived = [random.choice(all_derived)]
        elif which == 'none':
            all_derived = []

        for derived_path_obj, derived_index in all_derived:
            # tests that the derived objects do what they claim to do
            _test_derived(path_obj, derived_path_obj, derived_index, extrarandom)
            # tests that the derived objects are consistent wrt themselves
            full_path_ = full_path[derived_index]
            if isinstance(derived_index, int):
                full_path_ = full_path_.unsqueeze(0)
            _test_path_obj(full_path_.size(0), input_channels, device, derived_path_obj, full_path_, depth,
                           update_lengths, update_grads, scalar_term, extrarandom)
        # tests that the changes to the derived objects have not affected the original
        assert path_obj == backup_path_obj

    # finally test the original object
    _test_path_obj(batch_size, input_channels, device, path_obj, full_path, depth, update_lengths, update_grads,
                   scalar_term, extrarandom)


def _test_path_obj(batch_size, input_channels, device, path_obj, full_path, depth, update_lengths, update_grads,
                   scalar_term, extrarandom):
    # First of all test a Path with no updates
    _test_signature(path_obj, full_path, depth, scalar_term, extrarandom)
    _test_logsignature(path_obj, full_path, depth, extrarandom)
    _test_equality(path_obj)
    assert path_obj.depth == depth

    if len(update_lengths) > 1:
        # Then test Path with variable amounts of updates
        for length, grad in zip(update_lengths, update_grads):
            new_path = torch.rand(batch_size, length, input_channels, dtype=torch.double, device=device,
                                  requires_grad=grad)
            path_obj.update(new_path)
            full_path = torch.cat([full_path, new_path], dim=1)

        _test_signature(path_obj, full_path, depth, scalar_term, extrarandom)
        _test_logsignature(path_obj, full_path, depth, extrarandom)
        _test_equality(path_obj)
        assert path_obj.depth == depth


def _test_signature(path_obj, full_path, depth, scalar_term, extrarandom):
    def candidate(start=None, end=None):
        return path_obj.signature(start, end)

    def true(start, end):
        return signatory.signature(full_path[:, start:end], depth, scalar_term=scalar_term)

    def extra(true_signature):
        assert (path_obj.signature_size(-3), path_obj.signature_size(-1)) == true_signature.shape
        assert path_obj.signature_channels() == true_signature.size(-1)
        assert path_obj.shape == full_path.shape
        assert path_obj.channels() == full_path.size(-1)

    _test_operation(path_obj, candidate, true, extra, '_BackwardShortcutBackward', extrarandom)


def _test_logsignature(path_obj, full_path, depth, extrarandom):
    if extrarandom:
        if random.choice([True, False, False]):
            modes = h.all_modes
        else:
            modes = (h.expand_mode, h.words_mode)
    else:
        modes = h.all_modes

    for mode in modes:

        def candidate(start=None, end=None):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on "
                                                          "the GPU.", category=UserWarning)
                return path_obj.logsignature(start, end, mode=mode)

        def true(start, end):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on "
                                                          "the GPU.", category=UserWarning)
                return signatory.logsignature(full_path[:, start:end], depth, mode=mode)

        def extra(true_logsignature):
            if mode != h.expand_mode:
                assert (path_obj.logsignature_size(-3),
                        path_obj.logsignature_size(-1)) == true_logsignature.shape
                assert path_obj.logsignature_channels() == true_logsignature.size(-1)

        _test_operation(path_obj, candidate, true, extra, '_SignatureToLogsignatureFunctionBackward', extrarandom)


def _test_equality(path_obj):
    assert path_obj == path_obj
    assert not (path_obj != path_obj)
    shuffled_path_obj, perm = path_obj.shuffle()
    assert shuffled_path_obj == path_obj[perm]
    assert not (shuffled_path_obj != path_obj[perm])


def _test_derived(path_obj, derived_path_obj, derived_index, extrarandom):
    def candidate(start=None, end=None):
        return torch.cat(derived_path_obj.path, dim=-2)

    def true(start, end):
        return torch.cat(path_obj.path, dim=-2)[derived_index]

    def extra(true_path):
        pass

    _test_operation(path_obj, candidate, true, extra, None, extrarandom)

    def candidate(start=None, end=None):
        return derived_path_obj.signature(start, end)

    def true(start, end):
        return path_obj.signature(start, end)[derived_index]

    def extra(true_signature):
        pass

    _test_operation(path_obj, candidate, true, extra, '_BackwardShortcutBackward', extrarandom)

    if extrarandom:
        if random.choice([True, False, False]):
            modes = h.all_modes
        else:
            modes = (h.expand_mode, h.words_mode)
    else:
        modes = h.all_modes

    for mode in modes:

        def candidate(start=None, end=None):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on "
                                                          "the GPU.", category=UserWarning)
                return derived_path_obj.logsignature(start, end, mode=mode)

        def true(start, end):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="The logsignature with mode='brackets' has been requested on "
                                                          "the GPU.", category=UserWarning)
                return path_obj.logsignature(start, end, mode=mode)[derived_index]

        def extra(true_logsignature):
            pass

        _test_operation(path_obj, candidate, true, extra, '_SignatureToLogsignatureFunctionBackward', extrarandom)


def _boundaries(length):
    yield -length - 1
    yield -length
    yield -1
    yield 0
    yield 1
    yield length - 1
    yield length
    yield None


def _start_end(length, extrarandom):
    for start in _boundaries(length):
        for end in _boundaries(length):
            if (not extrarandom) or random.choice([True, False]):
                yield start, end
    for _ in range(5):
        start = int(torch.randint(low=-length, high=length, size=(1,)))
        end = int(torch.randint(low=-length, high=length, size=(1,)))
        yield start, end


def _test_operation(path_obj, candidate, true, extra, backward_name, extrarandom):
    # We perform multiple tests here.
    # Test #1: That the memory usage is consistent
    # Test #2: That the backward 'ctx' is correctly garbage collected
    # Test #3: The forward accuracy of a particular operation
    # Test #4: The backward accuracy of the same operation

    def one_iteration(start, end):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
        try:
            tensor = candidate(start, end)
        except ValueError as e:
            try:
                true(start, end)
            except ValueError:
                return 0
            else:
                pytest.fail(str(e))
        try:
            true_tensor = true(start, end)
        except ValueError as e:
            pytest.fail(str(e))
        h.diff(tensor, true_tensor)  # Test #3

        extra(true_tensor)  # Any extra tests

        if tensor.requires_grad:
            grad = torch.rand_like(tensor)
            tensor.backward(grad)
            path_grads = []
            for path in path_obj.path:
                if path.grad is None:
                    path_grads.append(None)
                else:
                    path_grads.append(path.grad.clone())
                    path.grad.zero_()
            true_tensor.backward(grad)
            for path, path_grad in zip(path_obj.path, path_grads):
                if path_grad is None:
                    assert (path.grad is None) or (path.grad.nonzero().numel() == 0)
                else:
                    h.diff(path.grad, path_grad)  # Test #4
                    path.grad.zero_()
            ctx = tensor.grad_fn
            assert type(ctx).__name__ == backward_name
            ref = weakref.ref(ctx)
            del ctx
            del tensor
            gc.collect()
            assert ref() is None  # Test #2
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()
        else:
            return 0

    # Computations involving the start or not operate differently, so we take the max over both
    memory_used = max(one_iteration(0, None), one_iteration(1, None))
    for start, end in _start_end(path_obj.size(1), extrarandom):
        # This one seems to be a bit inconsistent with how much memory is used on each run, so we give some
        # leeway by doubling
        assert one_iteration(start, end) <= 2 * memory_used
