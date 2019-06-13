#include <torch/extension.h>  // to get the pybind11 stuff
#include <fstream>  // std::ifstream
#include <ios>      // std::streamsize
#include <limits>   // std::numeric_limits
#include <string>   // std::string, std::getline

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::ifstream readme_file {"../README.md"};
    // skip first line
    readme_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // read the second line
    std::string readme;
    std::getline(readme_file, readme);
    readme_file.close();

    m.doc() = readme;
    py::class_<signatory::SigSpec>(m, "_SigSpec").def(py::init<torch::Tensor, int, bool, bool, bool>());
    m.def("signature_channels",
          &signatory::signature_channels,
          "Computes the number of output channels from a signature call.\n"
          "\n"
          "Arguments:\n"
          "    input_channels (int): The number of channels in the input; that is,\n"
          "        the dimension of the space that the input path resides in.\n"
          "    depth (int): The depth of the signature that is being computed.\n"
          "\n"
          "Returns:\n"
          "    An int specifying the number of channels in the signature of the path.",
          py::arg("input_channels"), py::arg("depth"));
    m.def("_signature_forward",
          &signatory::signature_forward,
          "Computes the forwards pass through a signature.");
    m.def("_signature_backward",
          &signatory::signature_backward,
          "Computes the backward pass through a signature.");
}