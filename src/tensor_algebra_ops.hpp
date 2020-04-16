/* Copyright 2019 Patrick Kidger. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */
 // Here we handle computing operations in the tensor algebra, like multiplication and exponentiation.
 // The most common object to pass around is a std::vector<torch::Tensor>, which corresponds to a general member of the
 // tensor algebra.
 // For the member 1 + \sum_i=1^n A_i of the tensor algebra, where A_i is of size (c, ..., c),
 //                                                                                \-------/
 //                                                                                 i times
 // this should be represented as a std::vector {A_1, ..., A_n}, where each A_i is a torch::Tensor of shape (c, ..., c).
 //                                                                                                          \-------/
 //                                                                                                           i times
 // (In particular unless otherwise noted, the 1 in the scalar part is implicit.)
 //
 // Furthermore all tensors should typically be of shape (batch, channels) unless otherwise noted.


#ifndef SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
#define SIGNATORY_TENSOR_ALGEBRA_OPS_HPP

#include <torch/extension.h>
#include <utility>  // std::pair

#include "misc.hpp"


namespace signatory {
    // Note that ta_ops operations do not perform any checking that the information passed is in a valid state.
    namespace ta_ops {
        // Computes a multiplication in the tensor algebra.
        // 'arg1' and 'arg2' are both general members of the tensor algebra.
        // If inverse==false then arg1 is modified to hold arg1 \otimes arg2.
        // If inverse==true then arg1 is modified to hold arg2 \otimes arg1.
        void mult(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2, bool inverse);

        // Backwards through mult(..., /*inverse=*/false).
        // 'arg1' and 'arg2' should be as mult was called with. (Not as it returns).
        // If add_not_copy==false then the gradient through arg2 will be copied into grad_arg2.
        // If add_not_copy==true then the gradient through arg2 will be added onto grad_arg2.
        template<bool add_not_copy>
        void mult_backward(std::vector<torch::Tensor>& grad_arg1,
                           std::vector<torch::Tensor>& grad_arg2,
                           const std::vector<torch::Tensor>& arg1,
                           const std::vector<torch::Tensor>& arg2);

        // Computes a restricted exponential in the tensor algebra.
        //
        // That is, it computes the exponential of 'in', and places the result in 'out'. It is restricted because 'in'
        // can only be a member of the lowest nonscalar level of the tensor algebra.
        // (We don't compute the exponential of an arbitrary element of the tensor algebra.)
        //
        // 'in' should be a tensor of shape (batch, stream, channel)
        // 'out' should already be of the appropriate size corresponding to the depth.
        // 'reciprocals' should all the reciprocals out to depth 'out.size()' already computed.
        void restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, torch::Tensor reciprocals);

        // Backwards through the restricted exponential.
        // 'in' should be as passed to restricted_exp.
        // 'out' should be as returned from restricted_exp.
        // 'grad_in' will have the gradients from this operation copied into it.
        // 'grad_out' is the input gradient to this function, and will be modified in-place.
        void restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                     torch::Tensor in, const std::vector<torch::Tensor>& out,
                                     torch::Tensor reciprocals);

        // Computes a fused multiply-exponentiate.
        // 'next' should be a member of the lowest nonscalar level of the tensor algebra.
        // 'prev' should be a general member of the tensor algebra.
        // If sigspec.inverse == false then 'prev' is modified to hold prev \otimes \exp(next)
        // If sigspec.inverse == true then 'prev' is modified to hold \exp(next) \otimes prev
        void mult_fused_restricted_exp(torch::Tensor next, std::vector<torch::Tensor>& prev, bool inverse,
                                       torch::Tensor reciprocals, int64_t batch_threads=1);

        // Backwards through the fused multiply-exponentiate.
        // 'grad_next' will have the gradient from this operation copied in to it.
        // 'grad_prev' is the input gradient to this function, and will be modified in-place.
        // 'next' should be as passed to mult_fused_restricted_exp
        // 'prev' should as passed to mult_fused_restricted_exp
        void mult_fused_restricted_exp_backward(torch::Tensor grad_next,
                                                std::vector<torch::Tensor>& grad_prev,
                                                torch::Tensor next,
                                                const std::vector<torch::Tensor>& prev,
                                                bool inverse,
                                                torch::Tensor reciprocals);

        // Computes the logarithm in the tensor algebra
        // 'output_vector' and 'input_vector' are both members of the tensor algebra, with assumed scalar values 1.
        // They are assumed to have equal values to each other when passed.
        // Then 'output_vector' is modified to be log(input_vector).
        void log(std::vector<torch::Tensor>& output_vector, const std::vector<torch::Tensor>& input_vector,
                 torch::Tensor reciprocals);

        // Computes the backwards pass through compute_log
        // 'input_vector' is as passed to log.
        // 'grad_output_vector' is the input gradient, and will be modified in-place.
        // 'grad_input_vector' is the output gradient, and will have the result of this operation added on to it.
        void log_backward(std::vector<torch::Tensor>& grad_output_vector,
                          std::vector<torch::Tensor>& grad_input_vector,
                          const std::vector<torch::Tensor>& input_vector,
                          torch::Tensor reciprocals);
    }  // namespace signatory::ta_ops

    // See signatory.signature_combine
    torch::Tensor signature_combine_forward(std::vector<torch::Tensor> sigtensors, int64_t input_channels,
                                            s_size_type depth, bool scalar_term);

    // See signatory.signature_combine
    std::vector<torch::Tensor> signature_combine_backward(torch::Tensor grad_out,
                                                          std::vector<torch::Tensor> sigtensors,
                                                          int64_t input_channels,
                                                          s_size_type depth,
                                                          bool scalar_term);
}  // namespace signatory

#endif //SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
