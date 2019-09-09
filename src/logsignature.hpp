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
    // Makes a LyndonInfo PyCapsule
    py::object make_lyndon_info(int64_t channels, s_size_type depth, LogSignatureMode mode);

    // See signatory.logsignature for documentation
    std::tuple<torch::Tensor, py::object>
    logsignature_forward(torch::Tensor path, s_size_type depth, bool stream, bool basepoint,
                         torch::Tensor basepoint_value, bool inverse, LogSignatureMode mode,
                         py::object lyndon_info_capsule);

    // See signatory.logsignature for documentation
    std::tuple<torch::Tensor, torch::Tensor>
    logsignature_backward(torch::Tensor grad_logsignature, py::object backwards_info_capsule);
}  // namespace signatory

#endif //SIGNATORY_LOGSIGNATURE_HPP
