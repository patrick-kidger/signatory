import os
import torch  # must be imported before anything from the implementation of the pytorch version of torchtest

from .signature import signature


with open(os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION'), 'r') as f:
    __version__ = f.read()

del torch
del os
del f
