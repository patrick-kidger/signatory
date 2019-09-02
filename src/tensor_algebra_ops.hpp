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


#ifndef SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
#define SIGNATORY_TENSOR_ALGEBRA_OPS_HPP

#include "misc.hpp"


namespace signatory {
    namespace ta_ops {
        // TODO: Handle exponentials in a cheaper way? It's a symmetric tensor so we can save a lot of work
        // Computes the exponential of the 'in' tensor. Each higher-order tensor is placed in 'out'.
        // It is 'restricted' in the sense that it only computes the exponential of a tensor belonging to the lowest
        // level of the tensor algebra, not the exponential of an arbitrary element of the tensor algebra.
        void compute_restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const misc::SigSpec& sigspec);

        // Computes the backwards pass through the restricted exponential. 'in' should be the input to the forward pass
        // of the exponential, but 'out' should be the result of the forward pass of the exponential. (I.e. what it
        // holds after the function has been called - recall that the function operates with 'out' as an out-argument.)
        // Argument 'grad_out' should have the gradient on the output of the forward pass, and has in-place changes
        // occurring to it.
        // Argument 'grad_in' will have the gradients resulting from this operation placed into it, overwriting whatever
        // is current present.
        void compute_restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                             torch::Tensor in, const std::vector<torch::Tensor>& out,
                                             const misc::SigSpec& sigspec);

        // Computes the tensor product of two members of the tensor algebra. It's not completely generic, as it imposes
        // that the scalar value of both members of the tensor algebra must be 1. (As we don't store the scalar value,
        // as for signatures - the most common use of this function - it is always 1.)
        //
        // that is, it computes arg1 \otimes arg2
        //
        // if rightret==false then it returns the result in arg1, and arg2 is left unchanged
        // if rightret==true then it returns the result in arg2, and arg1 is left unchanged
        void compute_mult(std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2, bool rightret,
                          const misc::SigSpec& sigspec);

        // Computes the backward pass for the tensor product operation of two members of the tensor algebra.
        //
        // Note that both 'arg1' and 'arg2' should be the inputs that were used in the forward pass of the
        // multiplication. In particular neither of them should be the result of the forward pass of the multiplication
        // (As compute_mult returns its result via one of its input arguments.)
        //
        // Argument 'grad_arg2' is the input gradient, and will be modified in-place according to the multiplication.
        // Argument 'grad_arg1' is the output gradient. If add_not_copy==true then the result of this operation is
        // added on to it. If add_not_copy==false then the result of this operation is placed into it directly,
        // overwriting whatever is already present.
        void compute_mult_backward(std::vector<torch::Tensor>& grad_arg1, std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                   bool add_not_copy, const misc::SigSpec& sigspec);

        // Computes the tensor 'divison' of two members of the tensor algebra. It's not completely generic, as it
        // imposes that the scalar value of both members of the tensor algebra must be 1. (As we don't store the scalar
        // value, as for signatures - the most common use of this function - it is always 1.)
        //
        // that is, it computes arg1 \otimes -arg2
        //
        // it returns the result in arg1 and leaves arg2 unchanged.
        void compute_div(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                         const misc::SigSpec& sigspec);

        /* No compute_div_backward because it's never used */

        // Computes the partial tensor product of two members of the tensor algebra, where arg1 is assumed to have
        // scalar_term_value as the value of its scalar term, arg2 is assumed to have zero as the value of its scalar
        // term, and the multiplication is only computed for a particular set of terms: the result ends up being a
        // weird hybrid of what was passed in, and the result of an actual multiplication.
        //
        // The logsignature computation only uses certain terms of certain forms, hence this weird collection of
        // restrictions.
        //
        // The result is returned in arg1, and arg2 is left unchanged.
        void compute_mult_partial(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                  torch::Scalar scalar_term_value, s_size_type top_terms_to_skip,
                                  const misc::SigSpec& sigspec);

        // Backward pass through compute_mult_partial. Is somewhat simplified compared to the naive implementation.
        // grad_arg1 is the input gradient, and will be modified in-place.
        // grad_arg2 is the output gradient. (Note that this is the other way around to compute_mult_backward...)
        // arg1, arg2, sigspec, top_terms_to_skip should be as in the forward call to compute_mult_partial.
        void compute_mult_partial_backward(std::vector<torch::Tensor>& grad_arg1,
                                           std::vector<torch::Tensor>& grad_arg2,
                                           const std::vector<torch::Tensor>& arg1,
                                           const std::vector<torch::Tensor>& arg2,
                                           torch::Scalar scalar_value_term,
                                           s_size_type top_terms_to_skip,
                                           const misc::SigSpec& sigspec);

        // The coefficient of a term in the power series of the logarithm
        torch::Scalar log_coefficient_at_depth(s_size_type depth_index, const misc::SigSpec& sigspec);

        // Computes the logarithm in the tensor algebra
        // output_vector is assumed to be initialised with a copy of input_vector.
        void compute_log(std::vector<torch::Tensor>& output_vector,
                         const std::vector<torch::Tensor>& input_vector,
                         const misc::SigSpec& sigspec);

        // Computes the backwards pass through compute_log
        void compute_log_backward(std::vector<torch::Tensor>& grad_output_vector,
                                  std::vector<torch::Tensor>& grad_input_vector,
                                  const std::vector<torch::Tensor>& input_vector,
                                  const misc::SigSpec& sigspec);
    }  // namespace signatory::ta_ops
}  // namespace signatory

#endif //SIGNATORY_TENSOR_ALGEBRA_OPS_HPP
