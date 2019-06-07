#include "extension.hpp"  // to get the pybind11 stuff
#include <fstream>  // std::ifstream
#include <ios>      // std::streamsize
#include <limits>   // std::numeric_limits
#include <string>   // std::string, std::getline

#include "signature.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::ifstream readme_file {"../README.rst"};
    // skip first two lines
    readme_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    readme_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // read the third line
    std::string readme;
    std::getline(readme_file, readme);
    readme_file.close();
    m.doc() = readme;

    m.def("signature", &signatory::signature, "Computes the signature of a path.", py::arg("path"), py::arg("depth"),
          py::arg("basepoint")=false, py::arg("stream")=false, py::arg("flatten")=true);
}