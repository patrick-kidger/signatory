import torch  # must be imported before anything from signatory
del torch

from .signature import (signature,
                        signature_channels)


__version__ = "0.1"
