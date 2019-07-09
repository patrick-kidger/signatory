import torch  # must be imported before anything from signatory

from .backend import (signature,
                      signature_channels,
                      extract_term)
from .modules import (Signature,
                      Augment)


__version__ = "0.2.0"

del torch
