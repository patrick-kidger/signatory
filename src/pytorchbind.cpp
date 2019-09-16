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
 // Provides the bindings into PyTorch


#include <torch/extension.h>  // to get the pybind11 stuff

#include "logsignature.hpp"  // signatory::logsignature_forward, signatory::logsignature_backward,
                             // signatory::make_lyndon_info
#include "misc.hpp"          // signatory::LogSignatureMode, signatory::signature_channels
#include "signature.hpp"     // signatory::signature_forward, signatory::signature_backward,
#include "lyndon.hpp"        // signatory::lyndon_words, signatory::lyndon_brackets,
                             // signatory::lyndon_words_to_basis_transform
#include "tensor_algebra_ops.hpp"  // signatory::tensor_algebra_mult_forward, signatory::tensor_algebra_mult_backward

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logsignature_forward",
          &signatory::logsignature_forward);
    m.def("logsignature_backward",
          &signatory::logsignature_backward);
    m.def("make_lyndon_info",
        &signatory::make_lyndon_info);
    py::enum_<signatory::LogSignatureMode>(m, "LogSignatureMode")
            .value("Expand", signatory::LogSignatureMode::Expand)
            .value("Brackets", signatory::LogSignatureMode::Brackets)
            .value("Words", signatory::LogSignatureMode::Words);
    m.def("signature_forward",
          &signatory::signature_forward);
    m.def("signature_backward",
          &signatory::signature_backward);
    m.def("signature_channels",
          &signatory::signature_channels);
    m.def("lyndon_words",
          &signatory::lyndon_words,
          py::return_value_policy::move);
    m.def("lyndon_brackets",
          &signatory::lyndon_brackets,
          py::return_value_policy::move);
    m.def("lyndon_words_to_basis_transform",
          &signatory::lyndon_words_to_basis_transform,
          py::return_value_policy::move);
    m.def("tensor_algebra_mult_forward",
          &signatory::tensor_algebra_mult_forward);
    m.def("tensor_algebra_mult_backward",
        &signatory::tensor_algebra_mult_backward);
}