#ifndef SIGNATORY_LOGSIGNATURE_HPP
#define SIGNATORY_LOGSIGNATURE_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <tuple>      // std::tuple

#include "misc.hpp"

namespace signatory {
    std::tuple<torch::Tensor, py::object>
    logsignature_forward(torch::Tensor path, size_type depth, bool stream, bool basepoint,
                         torch::Tensor basepoint_value, LogSignatureMode mode);

    std::tuple<torch::Tensor, torch::Tensor>
    logsignature_backward(torch::Tensor grad_logsignature, py::object backwards_info_capsule);
}  // namespace signatory

#endif //SIGNATORY_LOGSIGNATURE_HPP
