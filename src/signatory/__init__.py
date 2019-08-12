import torch  # must be imported before anything from signatory

from .backend import (signature,
                      logsignature)
from .modules import (Signature,
                      LogSignature,
                      Augment)
from .utility import (signature_channels,
                      logsignature_channels,
                      extract_term)


__version__ = "0.3.0"

del torch
