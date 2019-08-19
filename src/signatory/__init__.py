import torch  # must be imported before anything from signatory

from .augment import Augment
from .logsignature_module import (logsignature,
                                  LogSignature,
                                  logsignature_channels)
from .signature_module import (signature,
                               Signature,
                               signature_channels,
                               extract_signature_term)
from .lyndon import (lyndon_words,
                     lyndon_brackets)


__version__ = "1.0.0"

del torch
