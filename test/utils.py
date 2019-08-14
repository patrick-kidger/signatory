"""Provides a wrapper around signatory and iisignature functions so that they can be compared in a consistent way.

This works because signatory.signature, signatory.logsignature, iisignature.sig, iisignature.logsig,
iisignature.sigbackprop, iisignature.logsigbackprop all except more or less the same sorts of arguments.
"""

import collections as co
import functools as ft
import iisignature
import signatory
import torch


try:
    # Python 2
    stringtype = basestring
except NameError:
    # Python 3
    stringtype = str


with_grad = object()
without_grad = object()

expand = object()
brackets = object()
words = object()

all_modes = (expand, brackets, words)


class Config(object):
    """Represents a particular set of inputs to signatory and iisignature functions; also provides methods to actually
    call these functions in an appropriate, comparable, manner.
    """

    def __init__(self, mode, stream, size, depth, prep, basepoint, requires_grad):
        self.signature_or_logsignature = None
        if mode is expand:
            self.signatory_mode = "expand"
            self.iisignature_mode = "x"
            self.using_logsignature()
        elif mode is brackets:
            self.signatory_mode = "brackets"
            self.iisignature_mode = "d"
            self.using_logsignature()
        elif mode is words:
            self.signatory_mode = "words"  # must apply transform to get to brackets
            self.iisignature_mode = "d"
            self.using_logsignature()
        elif mode is None:
            self.signatory_mode = None
            self.iisignature_mode = None
        else:
            raise RuntimeError

        self.stream = stream
        self.size = size
        N, L, C = size
        self.N = N
        self.L = L
        self.C = C
        self.path = torch.rand(size, requires_grad=requires_grad, dtype=torch.double)
        self.depth = depth
        self.prep = prep
        self.basepoint_size = size[0], size[2]

        if basepoint is with_grad:
            basepoint = torch.rand(self.basepoint_size, requires_grad=True, dtype=torch.double)
        elif basepoint is without_grad:
            basepoint = torch.rand(self.basepoint_size, requires_grad=False, dtype=torch.double)

        self.basepoint = basepoint

    def has_basepoint(self):
        return isinstance(self.basepoint, torch.Tensor) or self.basepoint

    @property
    def signatory_out(self):
        try:
            return self._signatory_out
        except AttributeError:
            raise AttributeError("Must call signature() or logsignature() first.")

    @signatory_out.setter
    def signatory_out(self, val):
        self._signatory_out = val

    @property
    def grad(self):
        try:
            return self._grad
        except AttributeError:
            raise AttributeError("Must call signature_backward() or logsignature_backward() first.")

    @grad.setter
    def grad(self, val):
        self._grad = val

    @property
    def signatory_grad(self):
        try:
            return self._signatory_grad
        except AttributeError:
            raise AttributeError("Must call signature_backward() og logsignature_backward() first.")

    @signatory_grad.setter
    def signatory_grad(self, val):
        self._signatory_grad = val

    @property
    def basepointed_path(self):
        try:
            return self._basepointed_path
        except AttributeError:
            raise AttributeError("Must call sig() or logsig() first.")

    @basepointed_path.setter
    def basepointed_path(self, val):
        self._basepointed_path = val

    @property
    def iisignature_grad(self):
        try:
            return self._iisignature_grad
        except AttributeError:
            raise AttributeError("Must call sig_backward() or logsig_backward() first.")

    @iisignature_grad.setter
    def iisignature_grad(self, val):
        self._iisignature_grad = val

    def using_signature(self):
        if self.signature_or_logsignature is False:
            raise RuntimeError("Calling signature when already called logsignature, or if mode is not None.")
        self.signature_or_logsignature = True

    def using_logsignature(self):
        if self.signature_or_logsignature is True:
            raise RuntimeError("Calling logsignature when already called signature")
        self.signature_or_logsignature = False

    def signature(self):
        """Calls signatory.signature"""
        self.using_signature()
        signatory_out = signatory.signature(self.path, self.depth, self.stream, self.basepoint)
        self.signatory_out = signatory_out
        return signatory_out

    def signature_backward(self, grad=None):
        """Calls backwards on the result of signatory.signature"""
        self.using_signature()
        if grad is None:
            grad = torch.rand_like(self.signatory_out)
        self.grad = grad
        self.signatory_out.backward(grad)
        if isinstance(self.basepoint, torch.Tensor) and self.basepoint.requires_grad:
            # get the gradient on the basepoint as well
            signatory_grad = torch.cat([self.basepoint.grad.unsqueeze(1), self.path.grad], dim=1)
        else:
            signatory_grad = self.path.grad
        self.signatory_grad = signatory_grad
        return signatory_grad

    def sig(self):
        """Calls iisignature.sig"""
        self.using_signature()
        if self.basepoint is True:
            basepointed_path = torch.cat([torch.zeros(self.N, 1, self.C, dtype=torch.double), self.path.detach()],
                                         dim=1)
        elif self.basepoint is False:
            basepointed_path = self.path.detach()
        else:  # isinstance(self.basepoint, torch.Tensor) == True
            basepointed_path = torch.cat([self.basepoint.unsqueeze(1).detach(), self.path.detach()], dim=1)
        self.basepointed_path = basepointed_path
        return torch.tensor(iisignature.sig(basepointed_path, self.depth, 2 if self.stream else 0), dtype=torch.double)

    def sig_backward(self):
        """Calls iisignature.sigbackprop"""
        self.using_signature()
        if self.stream:
            raise RuntimeError("iisignature.sigbackprop does not support stream=True")
        iisignature_grad = iisignature.sigbackprop(self.grad, self.basepointed_path, self.depth)
        if self.basepoint is True or (isinstance(self.basepoint, torch.Tensor) and not self.basepoint.requires_grad):
            # if we don't have a gradient through the basepoint then discard the corresponding basepoint
            # part of the iisig_backward result
            iisignature_grad = iisignature_grad[:, 1:, :]
        iisignature_grad = torch.tensor(iisignature_grad, dtype=torch.double)
        self.iisignature_grad = iisignature_grad
        return iisignature_grad

    def logsignature(self):
        """Calls signatory.logsignature"""
        self.using_logsignature()
        signatory_out = signatory.logsignature(self.path, self.depth, self.stream, self.basepoint, self.signatory_mode)
        self.signatory_out = signatory_out
        return signatory_out

    def logsignature_backward(self, grad=None):
        """Calls backwards on the result of signatory.logsignature"""
        self.using_logsignature()
        if grad is None:
            grad = torch.rand_like(self.signatory_out)
        self.grad = grad
        self.signatory_out.backward(grad)
        if isinstance(self.basepoint, torch.Tensor) and self.basepoint.requires_grad:
            # get the gradient on the basepoint as well
            signatory_grad = torch.cat([self.basepoint.grad.unsqueeze(1), self.path.grad], dim=1)
        else:
            signatory_grad = self.path.grad
        self.signatory_grad = signatory_grad
        return signatory_grad

    def logsig(self):
        """Calls iisignature.logsig"""
        self.using_logsignature()
        if self.stream:
            raise RuntimeError("iisignature.logsig does not support stream=True")
        if self.basepoint is True:
            basepointed_path = torch.cat([torch.zeros(self.N, 1, self.C, dtype=torch.double), self.path.detach()],
                                         dim=1)
        elif self.basepoint is False:
            basepointed_path = self.path.detach()
        else:  # isinstance(self.basepoint, torch.Tensor) == True
            basepointed_path = torch.cat([self.basepoint.unsqueeze(1).detach(), self.path.detach()], dim=1)
        self.basepointed_path = basepointed_path
        return torch.tensor(iisignature.logsig(basepointed_path, self.prep(), self.iisignature_mode),
                            dtype=torch.double)

    def logsig_backward(self):
        """Calls iisignature.logsigbackprop"""
        self.using_logsignature()
        if self.stream:
            raise RuntimeError("iisignature.logsigbackprop does not support stream=True")
        iisignature_grad = iisignature.logsigbackprop(self.grad, self.basepointed_path, self.prep(),
                                                      self.iisignature_mode)
        if self.basepoint is True or (isinstance(self.basepoint, torch.Tensor) and not self.basepoint.requires_grad):
            # if we don't have a gradient through the basepoint then discard the corresponding basepoint
            # part of the iisig_backward result
            iisignature_grad = iisignature_grad[:, 1:, :]
        iisignature_grad = torch.tensor(iisignature_grad, dtype=torch.double)
        self.iisignature_grad = iisignature_grad
        return iisignature_grad

    def fail(self, **kwargs):
        """Returns a string indicating this current collection of arguments."""

        returnval = ("\n"
                     "mode={mode}\n"
                     "stream={stream}\n"
                     "size={size}\n"
                     "path.requires_grad={requires_grad}\n"
                     "depth={depth}\n"
                     "basepoint={basepoint}"
                     .format(mode=self.signatory_mode, stream=self.stream, size=self.size,
                             requires_grad=self.path.requires_grad, depth=self.depth, basepoint=self.basepoint))
        for key, value in kwargs.items():
            returnval += '\n{key}={value}'.format(key=key, value=value)

        return returnval

    @staticmethod
    def diff(arg1, arg2):
        """Calculates the difference between arguments and returns these in a value suitable for integrating with
        self.fail.

        See also diff_fail.

        Example: (where self is an instance of unittest.TestCase)
            >>> for c in ConfigIter():
            >>>     ...
            >>>     if test_failed:
            >>>         self.fail(c.fail(**c.diff(arg1, arg2), arg1=arg1, arg2=arg2))
        """
        diff = arg1 - arg2
        maxdiff = torch.max(torch.abs(diff))
        return co.OrderedDict([('maxdiff', maxdiff), ('diff', diff)])

    def diff_fail(self, **kwargs):
        """Wraps diff and fail together."""

        if len(kwargs) != 2:
            raise RuntimeError("diff_fail only accepts two keyword arguments")
        arg1_name, arg2_name = kwargs.keys()
        arg1_val, arg2_val = kwargs.values()
        out_dict = self.diff(arg1_val, arg2_val)
        out_dict[arg1_name] = arg1_val
        out_dict[arg2_name] = arg2_val
        return self.fail(**out_dict)


class ConfigIter(object):
    """Iterates over a prescibed collection of inputs."""

    def __init__(self, *,
                 stream=(True, False),
                 basepoint=None,
                 N=(1, 2, 3, 10),        # |
                 L=(2, 3, 4, 10),        # | what sizes to iterate over: (N[i], L[j], C[k]) for all i, j, k.
                 C=(1, 2, 3, 6),         # |
                 depth=(1, 2, 3, 4, 6),  # what depths to iterate over
                 size=None,              # what sizes to iterate over; supersedes (N, L, C) is passed
                 mode=None,              # what logsignature modes to operate in, if using logsignatures
                 requires_grad=False):   # set to True if wanting to do backwards calls

        if basepoint is None:
            if requires_grad:
                basepoint = (False, True, with_grad, without_grad)
            else:
                basepoint = (False, True, without_grad)
        elif basepoint in (False, True, with_grad, without_grad):
            basepoint = (basepoint,)

        if stream in (True, False):
            stream = (stream,)

        if isinstance(mode, stringtype) or mode in (expand, brackets, words, None):
            mode = (mode,)

        self.stream = stream
        self.N = N
        self.L = L
        self.C = C
        self.depth = depth
        self.size = size
        self.mode = mode
        self.requires_grad = requires_grad
        self.basepoint = basepoint

    def size_iter(self):
        if self.size is not None:
            for size in self.size:
                yield size
        else:
            for C in self.C:  # deliberately the slowest varying for iisignature.prepare efficiency
                for N in self.N:
                    for L in self.L:
                        yield (N, L, C)

    def __iter__(self):
        for depth in self.depth:
            for size in self.size_iter():
                # Note that we only depend on size[2], which is the slowest varying size: as such wrapping this into
                # a lambda gives an efficiency boost over just doing passing the result of self.prepare(...), as it's
                # only recalculated (because of the lru_cache) precisely when it needs to be.
                prepare = lambda: self.prepare(size[2], depth)
                for mode in self.mode:
                    for stream in self.stream:
                        for basepoint in self.basepoint:
                            yield Config(mode, stream, size, depth, prepare, basepoint, self.requires_grad)

    @ft.lru_cache(maxsize=1)
    def prepare(self, channels, depth):
        return iisignature.prepare(channels, depth)
