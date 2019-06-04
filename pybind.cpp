#include "src/extension.hpp"
#include "src/test.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d_sigmoid", &d_sigmoid, "d sigmoid");
}