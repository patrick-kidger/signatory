#include <torch/extension.h>  // to get the pybind11 stuff

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<signatory::LogSignatureMode>(m, "_LogSignatureMode")
            .value("Expand", signatory::LogSignatureMode::Expand)
            .value("Brackets", signatory::LogSignatureMode::Brackets)
            .value("Words", signatory::LogSignatureMode::Words);
    m.def("_signature_channels",
          &signatory::signature_channels);
    m.def("_signature_forward",
          &signatory::signature_forward);
    m.def("_signature_backward",
          &signatory::signature_backward,
          // need to specify default argument
          py::arg("grad_out"), py::arg("backwards_info_capsule"), py::arg("clone") = true);
    m.def("_logsignature_forward",
        &signatory::logsignature_forward);
    m.def("_logsignature_backward",
        &signatory::logsignature_backward);
}