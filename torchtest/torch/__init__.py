# must be imported before anything from the implementation of the pytorch version of torchtest
import torch
from torchtest.torch._impl import d_sigmoid
del torch
