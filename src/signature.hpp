#ifndef SIGNATORY_SIGNATURE_HPP
#define SIGNATORY_SIGNATURE_HPP

namespace signatory {
    std::tuple<torch::Tensor, py::object>
    signature_forward(torch::Tensor path, s_size_type depth, bool stream, bool basepoint, torch::Tensor basepoint_value);

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule, bool clone=true);
}  // namespace signatory

#endif //SIGNATORY_SIGNATURE_HPP
