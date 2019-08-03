#include <torch/extension.h>  // to get the pybind11 stuff

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<signatory::Mode>(m, "_Mode")
            .value("Expand", signatory::Mode::Expand)
            .value("Duval", signatory::Mode::Duval)
            .value("Lex", signatory::Mode::Lex)
            .value("Lyndon", signatory::Mode::Lyndon);
    m.def("_signature_channels",
          &signatory::signature_channels);
    m.def("_signature_forward",
          &signatory::signature_forward);
    m.def("_signature_backward",
          &signatory::signature_backward);
    m.def("_logsignature_forward",
        &signatory::logsignature_forward);
    m.def("_logsignature_backward",
        &signatory::logsignature_backward);
}