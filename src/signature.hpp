#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP

#include <torch/extension.h>
#include <Python.h>   // PyCapsule
#include <cstdint>    // int64_t
#include <tuple>      // std::tuple


namespace signatory {
    int64_t signature_channels(int64_t input_channels, int depth);

    std::tuple<torch::Tensor, py::object>
    signature_forward(torch::Tensor path, int depth, bool stream, bool basepoint, torch::Tensor basepoint_value);

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule);
}
#endif //TORCHTEST_SRC_SIGNATURE_HPP