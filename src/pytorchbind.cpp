#include <torch/extension.h>  // to get the pybind11 stuff

#include "logsignature.hpp"  // signatory::logsignature_forward, signatory::logsignature_backward,
                             // signatory::make_lyndon_info
#include "misc.hpp"          // signatory::LogSignatureMode, signatory::signature_channels
#include "signature.hpp"     // signatory::signature_forward, signatory::signature_backward,
#include "lyndon.hpp"     // signatory::lyndon_words, signatory::lyndon_brackets,
                             // signatory::lyndon_words_to_basis_transform

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
          &signatory::signature_backward,
          py::arg("grad_out"), py::arg("backwards_info_capsule"), py::arg("clone")=true);
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
}