import signatory
import torch
import unittest

import utils


class TestSignatureArguments(unittest.TestCase):
    def test_minimal_axes(self):
        for c in utils.ConfigIter(basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 2, 4), (4, 4, 1), (1, 2, 1)),
                                  requires_grad=True):
            try:
                c.signature()
                c.signature_backward()
            except Exception:
                self.fail(c.fail())

        for c in utils.ConfigIter(basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)),
                                  requires_grad=True):
            try:
                c.signature()
                c.signature_backward()
            except Exception:
                self.fail(c.fail())

    def test_too_small_axes(self):
        for c in utils.ConfigIter(basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 1, 4), (4, 0, 4), (4, 4, 0), (1, 1, 1)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.signature()

        for c in utils.ConfigIter(basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 0, 4), (4, 4, 0)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.signature()

    def test_arguments(self):
        for c in utils.ConfigIter(L=(1, 2, 3, 4),
                                  requires_grad=True):
            if not c.has_basepoint() and c.L == 1:
                with self.assertRaises(ValueError):
                    c.signature()
            else:
                try:
                    c.signature()
                except Exception:
                    self.fail(c.fail())


class TestSignatureShapes(unittest.TestCase):
    @staticmethod
    def correct_shape(size, depth, stream, basepoint):
        N, L, C = size
        if stream:
            if isinstance(basepoint, torch.Tensor) or basepoint:
                return N, L, signatory.signature_channels(C, depth)
            else:
                return N, L - 1, signatory.signature_channels(C, depth)
        else:
            return N, signatory.signature_channels(C, depth)

    def test_shapes(self):
        for c in utils.ConfigIter(requires_grad=True):
            signatory_out = c.signature()
            correct_shape = self.correct_shape(c.size, c.depth, c.stream, c.basepoint)
            self.assertEqual(signatory_out.shape, correct_shape, c.fail())


class TestLogSignatureArguments(unittest.TestCase):
    def test_minimal_axes(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 2, 4), (4, 4, 1), (1, 2, 1)),
                                  requires_grad=True):
            try:
                c.logsignature()
                c.logsignature_backward()
            except Exception:
                self.fail(c.fail())

        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((1, 4, 4), (4, 1, 4), (4, 4, 1), (1, 1, 1)),
                                  requires_grad=True):
            try:
                c.logsignature()
                c.logsignature_backward()
            except Exception:
                self.fail(c.fail())

    def test_too_small_axes(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=False,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 1, 4), (4, 0, 4), (4, 4, 0), (1, 1, 1)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.logsignature()

        for c in utils.ConfigIter(mode=utils.all_modes,
                                  basepoint=True,
                                  depth=(1, 2, 3),
                                  size=((0, 4, 4), (4, 0, 4), (4, 4, 0)),
                                  requires_grad=True):
            with self.assertRaises(ValueError):
                c.logsignature()

    def test_arguments(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  L=(1, 2, 3, 4),
                                  requires_grad=True):
            if not c.has_basepoint() and c.L == 1:
                with self.assertRaises(ValueError):
                    c.logsignature()
            else:
                try:
                    c.logsignature()
                except Exception:
                    self.fail(c.fail())


class TestLogSignatureShapes(unittest.TestCase):
    @staticmethod
    def correct_shape(size, depth, stream, basepoint, mode):
        N, L, C = size
        if mode == "expand":
            channel_fn = signatory.signature_channels
        else:
            channel_fn = signatory.logsignature_channels
        if stream:
            if isinstance(basepoint, torch.Tensor) or basepoint:
                return N, L, channel_fn(C, depth)
            else:
                return N, L - 1, channel_fn(C, depth)
        else:
            return N, channel_fn(C, depth)

    def test_shapes(self):
        for c in utils.ConfigIter(mode=utils.all_modes,
                                  requires_grad=True):
            signatory_out = c.logsignature()
            correct_shape = self.correct_shape(c.size, c.depth, c.stream, c.basepoint, c.signatory_mode)
            self.assertEqual(signatory_out.shape, correct_shape, c.fail())
