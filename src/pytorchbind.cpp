#include <torch/extension.h>  // to get the pybind11 stuff
#include <fstream>  // std::ifstream
#include <ios>      // std::streamsize
#include <limits>   // std::numeric_limits
#include <string>   // std::string, std::getline

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<signatory::SigSpec>(m, "_SigSpec").def(py::init<torch::Tensor, int, bool, bool, bool>());
    m.def("signature_channels",
          &signatory::signature_channels,
          "Computes the number of output channels from a signature call.");
    m.def("_signature_forward",
          &signatory::signature_forward,
          "Computes the forwards pass through a signature.");
    m.def("_signature_backward",
          &signatory::signature_backward,
          "Computes the backward pass through a signature.");
}