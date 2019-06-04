#include "extension.hpp"  // to get the pybind11 stuff
#include "test.cpp"

PYBIND11_MODULE(EXTENSION_NAME, m) {
    m.doc() = "Testing stuff";
    m.def("d_sigmoid", &d_sigmoid, "The derivative of a sigmoid");
}