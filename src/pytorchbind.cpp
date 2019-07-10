#include <torch/extension.h>  // to get the pybind11 stuff

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_signature_channels",
          &signatory::signature_channels);
    m.def("_signature_forward",
          &signatory::signature_forward);
    m.def("_signature_backward",
          &signatory::signature_backward);
}