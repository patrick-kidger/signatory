#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP

#include <torch/extension.h>
#include <Python.h>   // PyCapsule
#include <cstdint>    // int64_t
#include <tuple>      // std::tuple
#include <type_traits>  // std::make_signed


namespace signatory {
    // signed-ness is important because we'll sometimes iterate downwards
    using size_type = std::make_signed<std::vector<torch::Tensor>::size_type>::type;
    using u_size_type = std::make_unsigned<size_type>::type;

    // Modes for the return value of logsignature
    enum class LogSignatureMode { Expand, Brackets, Words };

    int64_t signature_channels(int64_t input_channels, size_type depth);

    std::tuple<torch::Tensor, py::object>
    signature_forward(torch::Tensor path, size_type depth, bool stream, bool basepoint, torch::Tensor basepoint_value);

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule, bool clone=true);

    std::tuple<torch::Tensor, py::object>
    logsignature_forward(torch::Tensor path, size_type depth, bool stream, bool basepoint,
                         torch::Tensor basepoint_value, LogSignatureMode mode);

    std::tuple<torch::Tensor, torch::Tensor>
    logsignature_backward(torch::Tensor grad_logsig, py::object backwards_info_capsule);
}
#endif //TORCHTEST_SRC_SIGNATURE_HPP