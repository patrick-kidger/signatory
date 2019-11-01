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
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "misc.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    namespace ta_ops {
        namespace detail {
            // This is the loop that's used inside some of the forward operations in the tensor algebra
            // It corresponds to the noncommutative part of these operations.
            void mult_inner(torch::Tensor tensor_at_depth,
                            const std::vector<torch::Tensor>& arg1,
                            const std::vector<torch::Tensor>& arg2,
                            s_size_type depth_index) {
                for (s_size_type j = 0, k = depth_index - 1; j < depth_index; ++j, --k) {
                    /* loop invariant: j + k = depth_index - 1 */
                    torch::Tensor out_view = tensor_at_depth.view({arg1[j].size(batch_dim),
                                                                   arg1[j].size(channel_dim),
                                                                   arg2[k].size(channel_dim)});
                    out_view.addcmul_(arg2[k].unsqueeze(channel_dim - 1),  /* += (this tensor times */
                                      arg1[j].unsqueeze(channel_dim));     /*     this tensor)      */
                }
            }

            // This is the loop that's used inside some of the backward operations in the tensor algebra
            void mult_inner_backward(torch::Tensor grad_tensor_at_depth,
                                     std::vector<torch::Tensor>& grad_arg1,
                                     std::vector<torch::Tensor>& grad_arg2,
                                     const std::vector<torch::Tensor> arg1,
                                     const std::vector<torch::Tensor> arg2,
                                     s_size_type depth_index) {
                for (s_size_type j = depth_index - 1, k = 0; j >= 0; --j, ++k) {
                    /* loop invariant: j + k = depth_index - 1 */
                    torch::Tensor out_view = grad_tensor_at_depth.view({arg1[j].size(batch_dim),
                                                                        arg1[j].size(channel_dim),
                                                                        arg2[k].size(channel_dim)});

                    grad_arg1[j].unsqueeze(channel_dim).baddbmm_(out_view, arg2[k].unsqueeze(channel_dim));
                    grad_arg2[k].unsqueeze(channel_dim - 1).baddbmm_(arg1[j].unsqueeze(channel_dim - 1), out_view);
                }
            }

            bool is_even(s_size_type index) {
                return (index % 2) == 0;
            }

            // The coefficient of a term in the power series of the logarithm
            torch::Scalar log_coefficient_at_depth(s_size_type depth_index, torch::Tensor reciprocals) {
                return ((is_even(depth_index) ? -1 : 1) * reciprocals[depth_index]).item();
            }

            // Computes (sort of) multiplication in the tensor algebra.
            // 'arg1' is assumed to be a member of the tensor algebra, with assumed scalar value 'scalar_term_value'.
            // 'arg2' is assumed to be a member of the tensor algebra, with assumed scalar value zero.
            // Then 'arg1' is modified to hold arg1 \otimes arg2 for some of its terms; its highest 'top_terms_to_skip'
            // many terms are left unchanged. Thus the result ends up being a weird hybrid of what was passed in, and
            // the result of an actual multiplication.
            void mult_partial(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                              torch::Scalar scalar_term_value, s_size_type top_terms_to_skip) {
                auto depth = arg1.size();
                for (s_size_type depth_index = depth - top_terms_to_skip - 1; depth_index >= 0; --depth_index) {
                    torch::Tensor tensor_at_depth = arg1[depth_index];

                    // corresponding to the zero scalar assumed to be associated with arg2
                    tensor_at_depth.zero_();

                    detail::mult_inner(tensor_at_depth, arg1, arg2, depth_index);

                    tensor_at_depth.add_(arg2[depth_index], scalar_term_value);
                }
            }

            // Backwards through mult_partial.
            // 'arg1', 'arg2', 'scalar_value_term', 'top_terms_to_skip' should be as in the forward call to
            // mult_partial.
            // 'grad_arg1' is the input gradient, and will be modified in-place.
            // 'grad_arg2' is the output gradient, and will have the result of this operation added on to it.
            void mult_partial_backward(std::vector<torch::Tensor>& grad_arg1,
                                       std::vector<torch::Tensor>& grad_arg2,
                                       const std::vector<torch::Tensor>& arg1,
                                       const std::vector<torch::Tensor>& arg2,
                                       torch::Scalar scalar_value_term,
                                       s_size_type top_terms_to_skip) {
                s_size_type depth = arg1.size();
                for (s_size_type depth_index = 0; depth_index < depth - top_terms_to_skip; ++depth_index) {
                    torch::Tensor grad_tensor_at_depth = grad_arg1[depth_index];

                    grad_arg2[depth_index].add_(grad_tensor_at_depth, scalar_value_term);

                    detail::mult_inner_backward(grad_tensor_at_depth, grad_arg1, grad_arg2, arg1, arg2, depth_index);

                    grad_tensor_at_depth.zero_();
                }
            }
        }  // namespace signatory::ta_ops::detail

        void mult(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2, bool inverse) {
            auto& arg_a = inverse ? arg2 : arg1;
            auto& arg_b = inverse ? arg1 : arg2;

            auto depth = arg_a.size();
            for (s_size_type depth_index = depth - 1; depth_index >= 0; --depth_index) {
                torch::Tensor tensor_at_depth = arg1[depth_index];  // not arg_a or arg_b
                detail::mult_inner(tensor_at_depth, arg_a, arg_b, depth_index);
                tensor_at_depth += arg2[depth_index];  // not arg_a or arg_b
            }
        }

        template<bool add_not_copy>
        void mult_backward(std::vector<torch::Tensor>& grad_arg1,
                           std::vector<torch::Tensor>& grad_arg2,
                           const std::vector<torch::Tensor>& arg1,
                           const std::vector<torch::Tensor>& arg2) {
            s_size_type depth = arg1.size();
            for (s_size_type depth_index = 0; depth_index < depth; ++depth_index) {
                torch::Tensor grad_tensor_at_depth = grad_arg1[depth_index];
                if (add_not_copy) {
                    grad_arg2[depth_index] += grad_tensor_at_depth;
                }
                else {
                    grad_arg2[depth_index].copy_(grad_tensor_at_depth);
                }
                detail::mult_inner_backward(grad_tensor_at_depth, grad_arg1, grad_arg2, arg1, arg2, depth_index);
            }
        }
        template void mult_backward</*add_not_copy=*/false>(std::vector<torch::Tensor>& grad_arg1,
                                                            std::vector<torch::Tensor>& grad_arg2,
                                                            const std::vector<torch::Tensor>& arg1,
                                                            const std::vector<torch::Tensor>& arg2);
        template void mult_backward</*add_not_copy=*/true>(std::vector<torch::Tensor>& grad_arg1,
                                                           std::vector<torch::Tensor>& grad_arg2,
                                                           const std::vector<torch::Tensor>& arg1,
                                                           const std::vector<torch::Tensor>& arg2);

        void restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, torch::Tensor reciprocals) {
            int64_t batch_size = in.size(batch_dim);
            int64_t input_channel_size = in.size(channel_dim);
            out[0].copy_(in);
            for (s_size_type i = 0; i < static_cast<s_size_type>(out.size()) - 1; ++i) {
                torch::Tensor view_out = out[i + 1].view({batch_size,
                                                          input_channel_size,
                                                          out[i].size(channel_dim)});
                torch::mul_out(view_out, out[i].unsqueeze(channel_dim - 1), in.unsqueeze(channel_dim));
                out[i + 1] *= reciprocals[i];
            }
        }

        void restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                     torch::Tensor in, const std::vector<torch::Tensor>& out,
                                     torch::Tensor reciprocals) {
            // Pull out the first pass of the for loop below. Note the use of bmm_out over baddbmm_.
            // The alternative to pulling this out is to call grad_in.zero_() before the loop, but that involves
            // touching the data, which takes extra time.
            int64_t batch_size = in.size(batch_dim);
            int64_t input_channel_size = in.size(channel_dim);
            s_size_type depth = out.size();
            if (depth > 1) {
                grad_out[depth - 1] *= reciprocals[depth - 2];
                torch::Tensor view_grad_out = grad_out[depth - 1].view({batch_size,
                                                                        input_channel_size,
                                                                        out[depth - 2].size(channel_dim)});
                torch::Tensor grad_in_unsqueeze = grad_in.unsqueeze(channel_dim);

                torch::bmm_out(/*out=*/grad_in_unsqueeze, view_grad_out, out[depth - 2].unsqueeze(channel_dim));
                grad_out[depth - 2].unsqueeze(channel_dim - 1).baddbmm_(in.unsqueeze(channel_dim - 1), view_grad_out);

                // grad_out is a vector of length depth.
                // grad_out[depth - 1] doesn't need any gradients added on to it.
                // grad_out[depth - 2] is pulled out above
                // Thus the strange starting index for i
                for (s_size_type i = depth - 3; i >= 0; --i) {
                    grad_out[i + 1] *= reciprocals[i];
                    torch::Tensor view_grad_out = grad_out[i + 1].view({batch_size,
                                                                        input_channel_size,
                                                                        out[i].size(channel_dim)});
                    grad_in.unsqueeze(channel_dim).baddbmm_(view_grad_out, out[i].unsqueeze(channel_dim));
                    grad_out[i].unsqueeze(channel_dim - 1).baddbmm_(in.unsqueeze(channel_dim - 1), view_grad_out);
                }
                grad_in += grad_out[0];
            }
            else {  // depth == 1
                grad_in.copy_(grad_out[0]);
            }
        }

        void mult_fused_restricted_exp(torch::Tensor next, std::vector<torch::Tensor>& prev, bool inverse,
                                       torch::Tensor reciprocals) {
            int64_t batch_size = next.size(batch_dim);
            int64_t input_channel_size = next.size(channel_dim);
            s_size_type depth = prev.size();

            // We're going to need to know the new increment, divided by every depth up to the maximum depth
            // We precompute them here as we're going to need them several times.
            torch::Tensor next_divided = next.unsqueeze(0) * reciprocals.unsqueeze(1).unsqueeze(2);

            int64_t left_channel_dim;
            int64_t right_channel_dim;
            if (inverse) {
                left_channel_dim = channel_dim - 1;
                right_channel_dim = channel_dim;
            }
            else {
                left_channel_dim = channel_dim;
                right_channel_dim = channel_dim - 1;
            }

            for (s_size_type depth_index = depth - 1; depth_index >= 1; --depth_index) {
                torch::Tensor scratch = prev[0] + next_divided[depth_index - 1];
                for (s_size_type j = 1, k = depth_index - 2; j < depth_index; ++j, --k) {
                    auto old_scratch_size = scratch.size(channel_dim);
                    torch::Tensor prev_view;
                    if (inverse) {
                        prev_view = prev[j].view({batch_size,
                                                  input_channel_size,
                                                  old_scratch_size});
                    }
                    else {
                        prev_view = prev[j].view({batch_size,
                                                  old_scratch_size,
                                                  input_channel_size});
                    }
                    scratch = prev_view.addcmul(scratch.unsqueeze(left_channel_dim),
                                                next_divided[k].unsqueeze(right_channel_dim));
                    scratch = scratch.view({batch_size, old_scratch_size * input_channel_size});
                }
                torch::Tensor prev_view;
                if (inverse) {
                    prev_view = prev[depth_index].view({batch_size,
                                                        input_channel_size,
                                                        scratch.size(channel_dim)});
                }
                else {
                    prev_view = prev[depth_index].view({batch_size,
                                                        scratch.size(channel_dim),
                                                        input_channel_size});
                }
                prev_view.addcmul_(scratch.unsqueeze(left_channel_dim), next.unsqueeze(right_channel_dim));
            }
            prev[0] += next;
        }

        void mult_fused_restricted_exp_backward(torch::Tensor grad_next,
                                                std::vector<torch::Tensor>& grad_prev,
                                                torch::Tensor next,
                                                const std::vector<torch::Tensor>& prev,
                                                bool inverse,
                                                torch::Tensor reciprocals) {
            // If you're reading this function and trying to understand it...
            // ...then good luck.
            // Seriously though, it's a backwards through quite a complicated operation, so there isn't much getting
            // around the fact that it's going to be a bit involved.

            int64_t batch_size = next.size(batch_dim);
            int64_t input_channel_size = next.size(channel_dim);
            s_size_type depth = prev.size();

            // First of all we recompute the forward pass and record all the intermediate tensors that were used and
            // discarded. We call these 'scratches'.
            std::vector<std::vector<torch::Tensor>> all_scratches;
            all_scratches.reserve(depth - 1);

            torch::Tensor next_divided = next.unsqueeze(0) * reciprocals.unsqueeze(1).unsqueeze(2);

            int64_t left_channel_dim;
            int64_t right_channel_dim;
            if (inverse) {
                left_channel_dim = channel_dim - 1;
                right_channel_dim = channel_dim;
            }
            else {
                left_channel_dim = channel_dim;
                right_channel_dim = channel_dim - 1;
            }

            for (s_size_type depth_index = depth - 1; depth_index >= 1; --depth_index) {
                all_scratches.emplace_back();
                std::vector<torch::Tensor>& scratches = all_scratches.back();
                scratches.reserve(depth_index);
                torch::Tensor scratch = prev[0] + next_divided[depth_index - 1];
                scratches.push_back(scratch);
                for (s_size_type j = 1, k = depth_index - 2; j < depth_index; ++j, --k) {
                    auto old_scratch_size = scratch.size(channel_dim);
                    torch::Tensor prev_view;
                    if (inverse) {
                        prev_view = prev[j].view({batch_size,
                                                  input_channel_size,
                                                  old_scratch_size});
                    }
                    else {
                        prev_view = prev[j].view({batch_size,
                                                  old_scratch_size,
                                                  input_channel_size});
                    }
                    scratch = prev_view.addcmul(scratch.unsqueeze(left_channel_dim),
                                                next_divided[k].unsqueeze(right_channel_dim));
                    scratch = scratch.view({batch_size, old_scratch_size * input_channel_size});
                    scratches.push_back(scratch);
                }
            }

            // Now we actually do the gradient operations

            torch::Tensor grad_next_divided = torch::zeros_like(next_divided);

            // Allocate memory for gradient through the scratches

            std::vector<std::vector<torch::Tensor>> all_grad_scratches;
            all_grad_scratches.reserve(all_scratches.size());
            for (const auto& scratches : all_scratches) {
                all_grad_scratches.emplace_back();
                all_grad_scratches.reserve(scratches.size());
                std::vector<torch::Tensor>& grad_scratches = all_grad_scratches.back();
                for (const auto& elem : scratches) {
                    grad_scratches.push_back(torch::empty_like(elem));
                }
            }

            // Now do the actual backward operation

            grad_next.copy_(grad_prev[0]);
            for (s_size_type depth_index = 1, back_index = all_scratches.size() - 1;
                 depth_index < depth;
                 ++depth_index, --back_index) {
                const std::vector<torch::Tensor>& grad_scratches = all_grad_scratches[back_index];
                const std::vector<torch::Tensor>& scratches = all_scratches[back_index];

                torch::Tensor grad_scratch = grad_scratches.back();
                torch::Tensor scratch = scratches.back();

                torch::Tensor grad_prev_view;
                if (inverse) {
                    grad_prev_view = grad_prev[depth_index].view({batch_size,
                                                                  input_channel_size,
                                                                  scratch.size(channel_dim)});
                    torch::Tensor out = grad_scratch.unsqueeze(channel_dim - 1);
                    torch::bmm_out(/*out=*/out,
                                   next.unsqueeze(channel_dim - 1),
                                   grad_prev_view);
                    grad_next.unsqueeze(channel_dim).baddbmm_(grad_prev_view, scratch.unsqueeze(channel_dim));
                }
                else {
                    grad_prev_view = grad_prev[depth_index].view({batch_size,
                                                                  scratch.size(channel_dim),
                                                                  input_channel_size});
                    torch::Tensor out = grad_scratch.unsqueeze(channel_dim);
                    torch::bmm_out(/*out=*/out,
                                   grad_prev_view,
                                   next.unsqueeze(channel_dim));
                    grad_next.unsqueeze(channel_dim - 1).baddbmm_(scratch.unsqueeze(channel_dim - 1), grad_prev_view);
                }

                for (s_size_type j = depth_index - 1, k = 0; j >= 1; --j, ++k) {
                    torch::Tensor grad_scratch = grad_scratches[j];
                    torch::Tensor grad_old_scratch = grad_scratches[j - 1];
                    torch::Tensor old_scratch = scratches[j - 1];
                    torch::Tensor next_divided_narrow = next_divided[k];
                    torch::Tensor grad_next_divided_narrow = grad_next_divided[k];

                    grad_prev[j] += grad_scratch;

                    torch::Tensor grad_scratch_view;
                    if (inverse) {
                        grad_scratch_view = grad_scratch.view({batch_size,
                                                               input_channel_size,
                                                               old_scratch.size(channel_dim)});
                        torch::Tensor out = grad_old_scratch.unsqueeze(channel_dim - 1);
                        torch::bmm_out(/*out=*/out,
                                       next_divided_narrow.unsqueeze(channel_dim - 1),
                                       grad_scratch_view);
                        grad_next_divided_narrow.unsqueeze(channel_dim).baddbmm_(grad_scratch_view,
                                                                                 old_scratch.unsqueeze(channel_dim));
                    }
                    else {
                        grad_scratch_view = grad_scratch.view({batch_size,
                                                               old_scratch.size(channel_dim),
                                                               input_channel_size});
                        torch::Tensor out = grad_old_scratch.unsqueeze(channel_dim);
                        torch::bmm_out(/*out=*/out,
                                       grad_scratch_view,
                                       next_divided_narrow.unsqueeze(channel_dim));
                        grad_next_divided_narrow.unsqueeze(channel_dim - 1).baddbmm_(old_scratch.unsqueeze(channel_dim - 1),
                                                                                     grad_scratch_view);
                    }
                }
                torch::Tensor grad_next_divided_narrow = grad_next_divided[depth_index - 1];
                grad_next_divided_narrow += grad_scratches[0];
                grad_prev[0] += grad_scratches[0];
            }

            // Finally the do the backward from next_divided into next

            // In principle when depth == 1 then the code below should be a no-op, but BLAS throws an error here
            if (depth > 1) {
                torch::Tensor grad_next_divided_view = grad_next_divided.view({depth - 1,
                                                                               batch_size * input_channel_size});
                torch::Tensor grad_next_view = grad_next.view({batch_size * input_channel_size});
                grad_next_view.unsqueeze(0).addmm_(reciprocals.unsqueeze(0), grad_next_divided_view);
            }
        }

        void log(std::vector<torch::Tensor>& output_vector, const std::vector<torch::Tensor>& input_vector,
                 torch::Tensor reciprocals) {
            s_size_type depth = input_vector.size();
            if (depth == 1) {
                output_vector[0].copy_(input_vector[0]);
                return;
            }
            output_vector[0].copy_(input_vector[0] * detail::log_coefficient_at_depth(depth - 2, reciprocals));
            for (s_size_type depth_index = depth - 3; depth_index >= 0; --depth_index) {
                detail::mult_partial(output_vector,
                                     input_vector,
                                     /*scalar_value_term=*/detail::log_coefficient_at_depth(depth_index,
                                                                                            reciprocals),
                                     /*top_terms_to_skip=*/depth_index + 1);
            }
            detail::mult_partial(output_vector, input_vector, /*scalar_value_term=*/1, /*top_terms_to_skip=*/0);
        }

        void log_backward(std::vector<torch::Tensor>& grad_output_vector,
                          std::vector<torch::Tensor>& grad_input_vector,
                          const std::vector<torch::Tensor>& input_vector,
                          torch::Tensor reciprocals) {
            s_size_type depth = input_vector.size();
            if (depth == 1) {
                grad_input_vector[0].copy_(grad_output_vector[0]);
                return;
            }

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
            record_vector.reserve(depth - 1);

            // Compute the logarithm forwards and remember every intermediate tensor
            scratch_vector[0] *= detail::log_coefficient_at_depth(depth - 2, reciprocals);
            for (s_size_type depth_index = depth - 3; depth_index >= 0; --depth_index) {
                copy_vector.clear();
                for (const auto& elem : scratch_vector) {
                    copy_vector.push_back(elem.clone());
                }
                record_vector.push_back(copy_vector);
                detail::mult_partial(scratch_vector,
                                     input_vector,
                                     /*scalar_value_term=*/detail::log_coefficient_at_depth(depth_index, reciprocals),
                                     /*top_terms_to_skip=*/depth_index + 1);
            }
            record_vector.push_back(scratch_vector);

            // Now actually perform the backwards operation
            s_size_type backward_index = record_vector.size() - 1;
            detail::mult_partial_backward(grad_output_vector,
                                          grad_input_vector,
                                          record_vector[backward_index],
                                          input_vector,
                                          /*scalar_value_term=*/1,
                                          /*top_terms_to_skip=*/0);

            for (s_size_type depth_index = 0; depth_index < depth - 2; ++depth_index) {
                --backward_index;
                detail::mult_partial_backward(grad_output_vector,
                                              grad_input_vector,
                                              record_vector[backward_index],
                                              input_vector,
                                              /*scalar_value_term=*/detail::log_coefficient_at_depth(depth_index,
                                                                                                     reciprocals),
                                              /*top_terms_to_skip=*/depth_index + 1);
            }

            grad_input_vector[0].add_(grad_output_vector[0],
                                      detail::log_coefficient_at_depth(depth - 2, reciprocals));
        }
    }  // namespace signatory::ta_ops

    torch::Tensor signature_combine_forward(std::vector<torch::Tensor> sigtensors,  // copy not reference as we modify it
                                            int64_t input_channels,
                                            s_size_type depth) {
        misc::checkargs_channels_depth(input_channels, depth);
        if (sigtensors.size() == 0) {
            throw std::invalid_argument("sigtensors must be of nonzero length.");
        }
        int64_t expected_signature_channels = signature_channels(input_channels, depth);
        if (sigtensors[0].ndimension() != 2) {
            throw std::invalid_argument("An element of sigtensors is not two-dimensional. Every element must have "
                                        "two dimensions, corresponding to "
                                        "(batch, signature_channels(input_channels, depth))");
        }
        int64_t batch_size = sigtensors[0].size(batch_dim);
        for (auto& elem : sigtensors) {
            if (elem.ndimension() != 2) {
                throw std::invalid_argument("An element of sigtensors is not two-dimensional. Every element must have "
                                            "two dimensions, corresponding to "
                                            "(batch, signature_channels(input_channels, depth))");
            }
            if (elem.size(batch_dim) != batch_size) {
                throw std::invalid_argument("Not every element of sigtensors has the same number of batch dimensions.");
            }
            if (elem.size(channel_dim) != expected_signature_channels) {
                throw std::invalid_argument("An element of sigtensors did not have the right number of channels.");
            }
            // No sense keeping track of gradients when we have a custom backwards (and we're doing inplace operations)
            elem = elem.detach();
        }

        torch::Tensor out = sigtensors[0].clone();
        std::vector<torch::Tensor> out_vector;
        misc::slice_by_term(out, out_vector, input_channels, depth);
        for (u_size_type sigtensor_index = 1; sigtensor_index < sigtensors.size(); ++sigtensor_index) {
            std::vector<torch::Tensor> sigtensor_vector;
            misc::slice_by_term(sigtensors[sigtensor_index], sigtensor_vector, input_channels, depth);
            ta_ops::mult(out_vector, sigtensor_vector, /*inverse=*/false);
        }
        return out;
    }

    std::vector<torch::Tensor> signature_combine_backward(torch::Tensor grad_out,
                                                          std::vector<torch::Tensor> sigtensors,  // copy not reference as we modify it
                                                          int64_t input_channels,
                                                          s_size_type depth) {
        grad_out = grad_out.detach();
        for (auto& elem : sigtensors) {
            elem = elem.detach();
        }

        // Allocate memory for the output gradients
        std::vector<torch::Tensor> grad_sigtensors;
        grad_sigtensors.reserve(sigtensors.size());
        grad_sigtensors.emplace_back();  // we'll fill in the first slot at the very end
        for (s_size_type sigtensors_index = 1;
             sigtensors_index < static_cast<s_size_type>(sigtensors.size());
             ++sigtensors_index) {
            grad_sigtensors.push_back(torch::empty_like(sigtensors[sigtensors_index]));
        }

        // Recompute the inputs to each tensor multiplication
        std::vector<std::vector<torch::Tensor>> scratch_vector_vector;
        auto reserve_amount = sigtensors.size();
        if (reserve_amount < 2) {
            reserve_amount = 0;
        }
        else {
            reserve_amount -= 2;
        }
        scratch_vector_vector.reserve(reserve_amount);
        torch::Tensor scratch = sigtensors[0];  // no clone necessary here, we're going to do it in the loop below
        // -1 to the size because we don't need to store the final output
        for (u_size_type sigtensor_index = 1; sigtensor_index < sigtensors.size() - 1; ++sigtensor_index) {
            scratch = scratch.clone();
            std::vector<torch::Tensor> scratch_vector;
            misc::slice_by_term(scratch, scratch_vector, input_channels, depth);

            std::vector<torch::Tensor> sigtensor_vector;
            misc::slice_by_term(sigtensors[sigtensor_index], sigtensor_vector, input_channels, depth);
            ta_ops::mult(scratch_vector, sigtensor_vector, /*inverse=*/false);

            scratch_vector_vector.push_back(scratch_vector);
        }

        // Allocate memory for the gradient when computing backward through the tensor multiplications
        torch::Tensor grad_scratch = grad_out.clone();
        std::vector<torch::Tensor> grad_scratch_vector;
        misc::slice_by_term(grad_scratch, grad_scratch_vector, input_channels, depth);

        for (s_size_type sigtensors_index = sigtensors.size() - 1; sigtensors_index >= 2; --sigtensors_index) {
            // Recompute the inputs of each multiplication
            std::vector<torch::Tensor> sigtensor_vector;
            misc::slice_by_term(sigtensors[sigtensors_index], sigtensor_vector, input_channels, depth);

            // Actually perform the backward operation
            std::vector<torch::Tensor> grad_sigtensor_vector;
            misc::slice_by_term(grad_sigtensors[sigtensors_index], grad_sigtensor_vector, input_channels, depth);
            ta_ops::mult_backward</*add_not_copy=*/false>(grad_scratch_vector, grad_sigtensor_vector,
                                                          // -1 because we're getting the input to this operation, so we
                                                          // need to look one step into the past
                                                          // -1 again because we don't store this input for the very
                                                          // first operation (we don't need to for that one), and it's
                                                          // pulled out as a special case below.
                                                          scratch_vector_vector[sigtensors_index - 2],
                                                          sigtensor_vector);
        }
        if (sigtensors.size() > 1) {
            // sigtensors_index == 1
            // This iteration pulled out because we don't need to do the final division
            std::vector<torch::Tensor> sigtensor_vector;
            misc::slice_by_term(sigtensors[1], sigtensor_vector, input_channels, depth);
            std::vector<torch::Tensor> first_sigtensor_vector;
            misc::slice_by_term(sigtensors[0], first_sigtensor_vector, input_channels, depth);
            std::vector<torch::Tensor> grad_sigtensor_vector;
            misc::slice_by_term(grad_sigtensors[1], grad_sigtensor_vector, input_channels, depth);
            ta_ops::mult_backward</*add_not_copy=*/false>(grad_scratch_vector, grad_sigtensor_vector,
                                                          first_sigtensor_vector, sigtensor_vector);
        }
        // Fill in the gradient for the very first sigtensor.
        grad_sigtensors[0] = grad_scratch;

        return grad_sigtensors;
    }
}  // namespace signatory