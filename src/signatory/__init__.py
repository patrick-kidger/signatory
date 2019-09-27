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


import torch  # must be imported before anything from signatory

from .augment import Augment
from .logsignature_module import (logsignature,
                                  LogSignature,
                                  Logsignature,  # alias for LogSignature
                                  logsignature_channels)
from .path import Path
from .signature_module import (signature,
                               Signature,
                               signature_channels,
                               extract_signature_term,
                               signature_combine)
from .utility import (lyndon_words,
                      lyndon_brackets,
                      all_words)


__version__ = "1.1.3"

del torch
