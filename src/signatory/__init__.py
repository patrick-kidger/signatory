import torch  # must be imported before anything from signatory

from .backend import (signature,
                      logsignature,
                      signature_channels,
                      logsignature_channels,
                      extract_term)
from .modules import (Signature,
                      LogSignature,
                      Augment)


__version__ = "0.2.0"

del torch
