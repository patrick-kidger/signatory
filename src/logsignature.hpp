/* Copyright 2019 Patrick Kidger. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */
 // Here we handle the computation of the logsignature.


#ifndef SIGNATORY_LOGSIGNATURE_HPP
#define SIGNATORY_LOGSIGNATURE_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <tuple>      // std::tuple

#include "misc.hpp"

namespace signatory {
    // Modes for the return value of logsignature
    // See signatory.logsignature for further documentation
    enum class LogSignatureMode { Expand, Brackets, Words };

    // Makes a LyndonInfo PyCapsule
    py::object make_lyndon_info(int64_t channels, s_size_type depth, LogSignatureMode mode);

    // See signatory.signature_to_logsignature for documentation
    std::tuple<torch::Tensor, py::object>
    signature_to_logsignature_forward(torch::Tensor signature, int64_t input_channel_size, s_size_type depth,
                                      bool stream, LogSignatureMode mode, py::object lyndon_info_capsule,
                                      bool scalar_term);

    // See signatory.signature_to_logsignature for documentation
    torch::Tensor signature_to_logsignature_backward(torch::Tensor grad_logsignature,
                                                     torch::Tensor signature,
                                                     int64_t input_channel_size,
                                                     s_size_type depth,
                                                     bool stream,
                                                     LogSignatureMode mode,
                                                     py::object lyndon_info_capsule,
                                                     bool scalar_term);
}  // namespace signatory

#endif //SIGNATORY_LOGSIGNATURE_HPP
