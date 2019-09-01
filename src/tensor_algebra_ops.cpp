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


#include <torch/extension.h>
#include <vector>     // std::vector

#include "misc.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    namespace ta_ops {
        namespace detail {
            // This is the loop that's used inside many of the forward operations in the tensor algebra
            template<bool invert=false>
            void compute_multdiv_inner(torch::Tensor tensor_at_depth_to_calculate,
                                       const std::vector<torch::Tensor>& arg1,
                                       const std::vector<torch::Tensor>& arg2,
                                       s_size_type depth_to_calculate,
                                       const misc::SigSpec& sigspec) {
                for (s_size_type j = 0, k = depth_to_calculate - 1; j < depth_to_calculate; ++j, --k) {
                    /* loop invariant: j + k = depth_to_calculate - 1 */
                    torch::Tensor view_out = tensor_at_depth_to_calculate.view({sigspec.batch_size,
                                                                                arg1[j].size(channel_dim),
                                                                                arg2[k].size(channel_dim)});
                    view_out.addcmul_(arg2[k].unsqueeze(channel_dim - 1),      /* += (this tensor times */
                                      arg1[j].unsqueeze(channel_dim),          /*     this tensor times */
                                      (invert && misc::is_even(k)) ? -1 : 1);  /*     this scalar)      */
                }
            }

            // This is the loop that's used inside many of the backward operations in the tensor algebra
            // No template argument as we only ever need to perform this with invert=false
            void compute_multdiv_inner_backward(torch::Tensor grad_tensor_at_depth_to_calculate,
                                                std::vector<torch::Tensor>& grad_arg1,
                                                std::vector<torch::Tensor>& grad_arg2,
                                                const std::vector<torch::Tensor> arg1,
                                                const std::vector<torch::Tensor> arg2,
                                                s_size_type depth_to_calculate,
                                                const misc::SigSpec& sigspec) {
                for (s_size_type j = depth_to_calculate - 1, k = 0; j >= 0; --j, ++k) {
                    /* loop invariant: j + k = depth_to_calculate - 1 */
                    torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({sigspec.batch_size,
                                                                                     arg1[j].size(channel_dim),
                                                                                     arg2[k].size(channel_dim)});

                    grad_arg1[j].unsqueeze(channel_dim).baddbmm_(out_view, arg2[k].unsqueeze(channel_dim));
                    grad_arg2[k].unsqueeze(channel_dim - 1).baddbmm_(arg1[j].unsqueeze(channel_dim - 1), out_view);
                }
            }

            // This performs part of the logarithm computation
            void compute_log_partial(std::vector<torch::Tensor>& logsignature_vector,
                                     const std::vector<torch::Tensor>& signature_vector,
                                     s_size_type lower_depth_index,
                                     const misc::SigSpec& sigspec) {
                for (s_size_type depth_index = sigspec.depth - 3; depth_index >= lower_depth_index; --depth_index) {
                    compute_mult_partial(logsignature_vector, signature_vector,
                                         log_coefficient_at_depth(depth_index, sigspec),
                                         depth_index + 1, sigspec);
                }
            }
        }  // namespace signatory::ta_ops::detail
        void compute_restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const misc::SigSpec& sigspec) {
            out[0].copy_(in);
            for (s_size_type i = 0; i < sigspec.depth - 1; ++i) {
                torch::Tensor view_out = out[i + 1].view({sigspec.batch_size,
                                                          in.size(channel_dim),
                                                          out[i].size(channel_dim)});
                torch::mul_out(view_out, out[i].unsqueeze(channel_dim - 1), in.unsqueeze(channel_dim));
                out[i + 1] *= sigspec.reciprocals[i];
            }
        }

        void compute_restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                             torch::Tensor in, const std::vector<torch::Tensor>& out,
                                             const misc::SigSpec& sigspec) {
            if (sigspec.depth >= 2) {
                // grad_out is a vector of length sigspec.depth.
                // grad_out[sigspec.depth - 1] doesn't need any gradients added on to it.
                // (Hence the strange starting index for i)
                for (s_size_type i = sigspec.depth - 2; i >= 0; --i) {
                    grad_out[i + 1] *= sigspec.reciprocals[i];
                    torch::Tensor view_grad_out = grad_out[i + 1].view({sigspec.batch_size,
                                                                        in.size(channel_dim),
                                                                        out[i].size(channel_dim)});

                    grad_in.unsqueeze(channel_dim).baddbmm_(view_grad_out, out[i].unsqueeze(channel_dim));
                    grad_out[i].unsqueeze(channel_dim - 1).baddbmm_(in.unsqueeze(channel_dim - 1), view_grad_out);
                }
                grad_in += grad_out[0];
            }
            else {  // implies depth == 1
                grad_in.copy_(grad_out[0]);
            }
        }

        void compute_mult(std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2, bool rightret,
                          const misc::SigSpec& sigspec) {
            for (s_size_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = (rightret ? arg2 : arg1)[depth_to_calculate];

                detail::compute_multdiv_inner(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, sigspec);

                tensor_at_depth_to_calculate += (rightret ? arg1 : arg2)[depth_to_calculate];
            }
        }

        void compute_mult_backward(std::vector<torch::Tensor>& grad_arg1, std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                   bool add_not_copy, const misc::SigSpec& sigspec) {
            for (s_size_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth; ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg2[depth_to_calculate];

                if (add_not_copy) {
                    grad_arg1[depth_to_calculate] += grad_tensor_at_depth_to_calculate;
                }
                else {
                    grad_arg1[depth_to_calculate].copy_(grad_tensor_at_depth_to_calculate);
                }

                detail::compute_multdiv_inner_backward(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1,
                                                       arg2, depth_to_calculate, sigspec);
            }
        }

        void compute_div(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                         const misc::SigSpec& sigspec) {
            for (s_size_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg1[depth_to_calculate];

                detail::compute_multdiv_inner</*invert=*/true>(tensor_at_depth_to_calculate, arg1, arg2,
                                                               depth_to_calculate, sigspec);

                if (misc::is_even(depth_to_calculate)) {
                    tensor_at_depth_to_calculate -= arg2[depth_to_calculate];
                }
                else {
                    tensor_at_depth_to_calculate += arg2[depth_to_calculate];
                }
            }
        }

        void compute_mult_partial(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                  torch::Scalar scalar_term_value, s_size_type top_terms_to_skip,
                                  const misc::SigSpec& sigspec) {
            for (s_size_type depth_to_calculate = sigspec.depth - top_terms_to_skip - 1; depth_to_calculate >= 0;
                 --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg1[depth_to_calculate];

                // corresponding to the zero scalar assumed to be associated with arg2
                tensor_at_depth_to_calculate.zero_();

                detail::compute_multdiv_inner(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, sigspec);

                tensor_at_depth_to_calculate.add_(arg2[depth_to_calculate], scalar_term_value);
            }
        }

        void compute_mult_partial_backward(std::vector<torch::Tensor>& grad_arg1,
                                           std::vector<torch::Tensor>& grad_arg2,
                                           const std::vector<torch::Tensor>& arg1,
                                           const std::vector<torch::Tensor>& arg2,
                                           torch::Scalar scalar_value_term,
                                           s_size_type top_terms_to_skip,
                                           const misc::SigSpec& sigspec) {
            for (s_size_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth - top_terms_to_skip;
                 ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg1[depth_to_calculate];

                grad_arg2[depth_to_calculate].add_(grad_tensor_at_depth_to_calculate, scalar_value_term);

                detail::compute_multdiv_inner_backward(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1,
                                                       arg2, depth_to_calculate, sigspec);

                grad_tensor_at_depth_to_calculate.zero_();
            }
        }

        torch::Scalar log_coefficient_at_depth(s_size_type depth_index, const misc::SigSpec& sigspec) {
            return ((misc::is_even(depth_index) ? -1 : 1) * sigspec.reciprocals[depth_index]).item();
        }

        void compute_log(std::vector<torch::Tensor>& output_vector,
                         const std::vector<torch::Tensor>& input_vector,
                         const misc::SigSpec& sigspec) {
            output_vector[0].copy_(input_vector[0] * log_coefficient_at_depth(sigspec.depth - 2, sigspec));
            for (s_size_type depth_index = sigspec.depth - 3; depth_index >= 0; --depth_index) {
                compute_mult_partial(output_vector,
                                     input_vector,
                                     /*scalar_value_term=*/log_coefficient_at_depth(depth_index, sigspec),
                                     /*top_terms_to_skip=*/depth_index + 1,
                                     sigspec);
            }
            compute_mult_partial(output_vector, input_vector, /*scalar_value_term=*/1, /*top_terms_to_skip=*/0,
                                 sigspec);
        }

        void compute_log_backward(std::vector<torch::Tensor>& grad_output_vector,
                                  std::vector<torch::Tensor>& grad_input_vector,
                                  const std::vector<torch::Tensor>& input_vector,
                                  const misc::SigSpec& sigspec) {
            // Will have the logarithm progressively computed in it
            std::vector<torch::Tensor> scratch_vector;
            scratch_vector.reserve(input_vector.size());
            for (const auto& elem : input_vector) {
                scratch_vector.push_back(elem.clone());
            }

            // Used as extra scratch space prior to pushing into...
            std::vector<torch::Tensor> copy_vector;
            copy_vector.reserve(scratch_vector.size());

            // ...this, which records all the partially-computed logarithms
            std::vector<std::vector<torch::Tensor>> record_vector;
            record_vector.reserve(sigspec.depth - 1);

            // Compute the logarithm forwards and remember every intermediate tensor
            scratch_vector[0] *= log_coefficient_at_depth(sigspec.depth - 2, sigspec);
            for (s_size_type depth_index = sigspec.depth - 3; depth_index >= 0; --depth_index) {
                copy_vector.clear();
                for (const auto& elem : scratch_vector) {
                    copy_vector.push_back(elem.clone());
                }
                record_vector.push_back(copy_vector);
                compute_mult_partial(scratch_vector,
                                     input_vector,
                                     /*scalar_value_term=*/log_coefficient_at_depth(depth_index, sigspec),
                                     /*top_terms_to_skip=*/depth_index + 1,
                                     sigspec);
            }
            record_vector.push_back(scratch_vector);

            // Now actually perform the backwards operation
            s_size_type backward_index = record_vector.size() - 1;
            compute_mult_partial_backward(grad_output_vector, grad_input_vector, record_vector[backward_index],
                                          input_vector, 1, 0, sigspec);

            for (s_size_type depth_index = 0; depth_index < sigspec.depth - 2; ++depth_index) {
                --backward_index;
                compute_mult_partial_backward(grad_output_vector, grad_input_vector, record_vector[backward_index],
                                              input_vector, log_coefficient_at_depth(depth_index, sigspec),
                                              depth_index + 1, sigspec);
            }

            grad_input_vector[0].add_(grad_output_vector[0], log_coefficient_at_depth(sigspec.depth - 2, sigspec));
        }
    }  // namespace signatory::ta_ops
}  // namespace signatory