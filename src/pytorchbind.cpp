#include <torch/extension.h>  // to get the pybind11 stuff
#include <fstream>  // std::ifstream
#include <ios>      // std::streamsize
#include <limits>   // std::numeric_limits
#include <string>   // std::string, std::getline

#include "signature.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::ifstream readme_file {"../README.rst"};
    // skip first two lines
    readme_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    readme_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // read the third line
    std::string readme;
    std::getline(readme_file, readme);
    readme_file.close();

    // TODO: update docs
    m.doc() = readme;
    m.def("signature_channels",
          &signatory::signature_channels,
          "Computes the number of output channels from a signature call.");
    m.def("signature",
          &signatory::signature,
          "Computes the signature of a path.");
    m.def("signature_backward",
          &signatory::signature_backward,
          "Computes the backward pass.");
}