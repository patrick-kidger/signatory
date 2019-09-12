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
 // Here we handle computing operations in the tensor algebra, like division and exponentiation.
 // Note that in particular we do _not_ handle multiplication. We never actually need it in the process of computing the
 // signature. (We use a much more efficient fused multiply-exponentiate instead)


#ifndef SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
#define SIGNATORY_TENSOR_ALGEBRA_OPS_HPP

#include <utility>  // std::pair

#include "misc.hpp"


namespace signatory {
    namespace ta_ops {
        // Computes a restricted exponential.
        // That is, it computes the exponential of 'in', and places the result in 'out'. It is restricted because 'in'
        // can only be a member of the lowest nonscalar level of the tensor algebra.
        // (We don't compute the exponential of an arbitrary element of the tensor algebra.)
        void restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const misc::SigSpec& sigspec);

        // Backwards through the restricted exponential.
        // 'in' should be as passed to restricted_exp.
        // 'out' should be as returned from restricted_exp.
        // 'grad_in' will have the gradients from this operation copied into it.
        // 'grad_out' is the input gradient to this function, and will be modified in-place.
        void restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                     torch::Tensor in, const std::vector<torch::Tensor>& out,
                                     const misc::SigSpec& sigspec);

        // Computes a fused multiply-exponentiate.
        // 'next' should be a member of the lowest nonscalar level of the tensor algebra.
        // 'prev' should be a general member of the tensor algebra.
        // If sigspec.inverse == false then 'prev' is modified to hold prev \otimes \exp(next)
        // If sigspec.inverse == true then 'prev' is modified to hold \exp(next) \otimes prev
        void mult_fused_restricted_exp(torch::Tensor next, std::vector<torch::Tensor>& prev,
                                       const misc::SigSpec& sigspec);

        // Backwards through the fused multiply-exponentiate.
        // 'grad_next' will have the gradient from this operation copied in to it.
        // 'grad_prev' is the input gradient to this function, and will be modified in-place.
        // 'next' should be as passed to mult_fused_restricted_exp
        // 'prev' should as passed to mult_fused_restricted_exp
        void mult_fused_restricted_exp_backward(torch::Tensor grad_next,
                                                std::vector<torch::Tensor> grad_prev,
                                                torch::Tensor next,
                                                const std::vector<torch::Tensor>& prev,
                                                const misc::SigSpec& sigspec);

        // Computes (sort of) multiplication in the tensor algebra.
        // 'arg1' is assumed to be a member of the tensor algebra, with assumed scalar value 'scalar_term_value'.
        // 'arg2' is assumed to be a member of the tensor algebra, with assumed scalar value zero.
        // Then 'arg1' is modified to hold arg1 \otimes arg2 for some of its terms; its highest 'top_terms_to_skip' many
        // terms are left unchanged. Thus the result ends up being a weird hybrid of what was passed in, and the result
        // of an actual multiplication.
        void mult_partial(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                          torch::Scalar scalar_term_value, s_size_type top_terms_to_skip, const misc::SigSpec& sigspec);

        // Backwards through mult_partial.
        // 'arg1', 'arg2', 'scalar_value_term', 'top_terms_to_skip' should be as in the forward call to mult_partial.
        // 'grad_arg1' is the input gradient, and will be modified in-place.
        // 'grad_arg2' is the output gradient, and will have the result of this operation added on to it.
        void mult_partial_backward(std::vector<torch::Tensor>& grad_arg1,
                                   std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1,
                                   const std::vector<torch::Tensor>& arg2,
                                   torch::Scalar scalar_value_term,
                                   s_size_type top_terms_to_skip,
                                   const misc::SigSpec& sigspec);

        // Computes the logarithm in the tensor algebra
        // 'output_vector' and 'input_vector' are both members of the tensor algebra, with assumed scalar values 1.
        // They are assumed to have equal values to each other when passed.
        // Then 'output_vector' is modified to be log(input_vector).
        void log(std::vector<torch::Tensor>& output_vector, const std::vector<torch::Tensor>& input_vector,
                 const misc::SigSpec& sigspec);

        // Computes the backwards pass through compute_log
        // 'input_vector' is as passed to log.
        // 'grad_output_vector' is the input gradient, and will be modified in-place.
        // 'grad_input_vector' is the output gradient, and will have the result of this operation added on to it.
        void log_backward(std::vector<torch::Tensor>& grad_output_vector,
                          std::vector<torch::Tensor>& grad_input_vector,
                          const std::vector<torch::Tensor>& input_vector,
                          const misc::SigSpec& sigspec);
    }  // namespace signatory::ta_ops

    // Computes arg1_inp \otimes arg2_inp in the tensor algebra.
    // 'arg1_inp' and 'arg2_inp' are tensors representing members of the tensor algebra.
    // 'input_channels' and 'depth' are the corresponding numbers of channels and depths for these tensors. These must
    // be consistent with the sizes of 'arg1_inp' and 'arg2_inp'.
    // Both arguments are left unmodified. The result of the operation is returned.
    torch::Tensor tensor_algebra_mult_forward(torch::Tensor arg1_inp, torch::Tensor arg2_inp, int64_t input_channels,
                                              s_size_type depth);

    // The corresponding backward operation for tensor_algebra_mult_forward.
    // 'grad' should be the gradient on the output of tensor_algebra_mult_forward. All other arguments should be as they
    // were inputted to tensor_algebra_mult_forward.
    std::pair<torch::Tensor, torch::Tensor>
    tensor_algebra_mult_backward(torch::Tensor grad, torch::Tensor arg1_inp, torch::Tensor arg2_inp,
                                 int64_t input_channels, s_size_type depth);
}  // namespace signatory

#endif //SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
