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
#include <type_traits>  // std::is_same
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "misc.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    namespace ta_ops {

        /************************************************
         * Forward and backward computations for 'mult' *
         ************************************************/

        namespace detail {
            // This is the loop that's used inside the forward operation of tensor multiplication in the tensor algebra
            // It corresponds to the noncommutative part of this operation.
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

        /**********************************************************
         * Forward and backward computations for 'restricted_exp' *
         **********************************************************/

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

        /*********************************************************************
         * Forward and backward computations for 'mult_fused_restricted_exp' *
         *********************************************************************/

        /* Okay, buckle up.
         *
         * This next bit of the code is very complicated.
         *
         * The mathematical operation it corresponds to is pretty involved. And we have to describe this on the CPU and
         * on the GPU, for both the forward and backward operations. (It's really the backward operation that looks
         * particularly bad.)
         *
         * These operations represent the hot loop for signature computations, so it's important that these be as
         * efficient as possible.
         */

        namespace detail {
            // Adapted from https://stackoverflow.com/a/21028912/12254339
            // Substitutes value initialisation for default initialisation, which for basic types means that they're not
            // initialised, which is a performance improvement.
            template <typename T, typename A=std::allocator<T>>
            class default_init_allocator : public A {
                using a_t = std::allocator_traits<A>;
            public:
                template <typename U>
                struct rebind {
                    using other = default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
                };

                using A::A;

                template <typename U>
                void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
                    ::new(static_cast<void*>(ptr)) U;
                }
                template <typename U, typename...Args>
                void construct(U* ptr, Args&&... args) {
                    a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
                }
            };

            void mult_fused_restricted_exp_cuda(torch::Tensor next, std::vector<torch::Tensor>& prev, bool inverse,
                                                torch::Tensor reciprocals) {
                // We haven't tried writing custom GPU code. But if we did it would go here. Instead this is
                // specified in terms of the higher-level PyTorch Tensors.

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
            // That's the forward operation, written in terms of high-level PyTorch tensors. That wasn't so bad, was it?

            // This describes the forward operation for a single batch element on the CPU.
            // No parallelisation is performed in this computation at the moment. We could probably add it on but it
            // probably won't give a huge advantage.
            // We already parallelise over the batch and stream dimensions. So to see a speedup from parallelisation
            // here, we'd need to be computing signatures of just a few short paths, to high depths. Still, worth
            // thinking about.
            template <typename scalar_t, bool inverse>
            void mult_fused_restricted_exp_cpu_inner(torch::TensorAccessor<scalar_t, 2> next_a,
                                                     std::vector<torch::TensorAccessor<scalar_t, 2>>& prev_a,
                                                     torch::TensorAccessor<scalar_t, 1> reciprocals_a,
                                                     int64_t batch_index,
                                                     std::vector<scalar_t, default_init_allocator<scalar_t>>& next_divided,
                                                     std::vector<scalar_t, default_init_allocator<scalar_t>>& new_scratch,
                                                     std::vector<scalar_t, default_init_allocator<scalar_t>>& old_scratch) {
                int64_t input_channel_size = next_a.size(1);  // 1 is the channel dimension
                s_size_type depth = prev_a.size();

                int64_t next_divided_index = 0;
                for (int64_t reciprocal_index = 0; reciprocal_index < reciprocals_a.size(0); ++reciprocal_index) {
                    for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                        next_divided[next_divided_index] = reciprocals_a[reciprocal_index] *
                                                           next_a[batch_index][channel_index];
                        ++next_divided_index;
                    }
                }

                for (s_size_type depth_index = depth - 1; depth_index >= 1; --depth_index) {
                    int64_t scratch_size = input_channel_size;

                    int64_t next_divided_index_part = (depth_index - 1) * input_channel_size;

                    for (int64_t scratch_index = 0; scratch_index < input_channel_size; ++scratch_index) {
                        new_scratch[scratch_index] = prev_a[0][batch_index][scratch_index] +
                                                     next_divided[next_divided_index_part + scratch_index];
                    }

                    for (s_size_type j = 1, k = depth_index - 2; j < depth_index; ++j, --k) {
                        old_scratch.swap(new_scratch);
                        int64_t next_divided_index_part2 = k * input_channel_size;
                        for (int64_t old_scratch_index = 0; old_scratch_index < scratch_size; ++old_scratch_index) {
                            for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                                int64_t new_scratch_index;
                                if (inverse) {
                                    new_scratch_index = channel_index * scratch_size + old_scratch_index;
                                }
                                else {
                                    new_scratch_index = old_scratch_index * input_channel_size + channel_index;
                                }
                                new_scratch[new_scratch_index] = prev_a[j][batch_index][new_scratch_index] +
                                                                 old_scratch[old_scratch_index] *
                                                                 next_divided[next_divided_index_part2 + channel_index];
                            }
                        }

                        scratch_size *= input_channel_size;
                    }

                    for (int64_t new_scratch_index = 0; new_scratch_index < scratch_size; ++new_scratch_index) {
                        for (int64_t next_index = 0; next_index < input_channel_size; ++next_index) {
                            int64_t prev_a_index;
                            if (inverse) {
                                prev_a_index = next_index * scratch_size + new_scratch_index;
                            }
                            else {
                                prev_a_index = new_scratch_index * input_channel_size + next_index;
                            }
                            prev_a[depth_index][batch_index][prev_a_index] += new_scratch[new_scratch_index] *
                                                                 next_a[batch_index][next_index];
                        }
                    }
                }

                for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                    prev_a[0][batch_index][channel_index] += next_a[batch_index][channel_index];
                }

                // This corresponds to whether the triangle number of index 'depth' is odd.
                // In this case we have performed an odd number of swaps above, so we perform one more here to leave the
                // memory size unchanged.
                auto depth_mod = depth % 4;
                if (depth_mod == 0 || depth_mod == 3) {
                    old_scratch.swap(new_scratch);
                }
            }

            // This basically just parallelises over the batch elements, calling mult_fused_restricted_exp_cpu_inner on
            // each one.
            template <typename scalar_t>
            void mult_fused_restricted_exp_cpu(torch::Tensor next, std::vector<torch::Tensor>& prev, bool inverse,
                                               torch::Tensor reciprocals, int64_t batch_threads) {
                // Convert from Tensors to TensorAccessors
                auto next_a = next.accessor<scalar_t, 2>();
                std::vector<torch::TensorAccessor<scalar_t, 2>> prev_a;
                prev_a.reserve(prev.size());
                for (auto elem : prev) {
                    prev_a.push_back(elem.accessor<scalar_t, 2>());
                }
                auto reciprocals_a = reciprocals.accessor<scalar_t, 1>();

                int64_t batch_size = next.size(batch_dim);
                int64_t input_channel_size = next.size(channel_dim);
                s_size_type depth = prev.size();

                // commented out because of what I think is an MSVC bug?
                #pragma omp parallel /*default(none)*/ \
                                     if(batch_threads > 1) \
                                     num_threads(batch_threads) \
                                     shared(batch_size, next_a, prev_a, inverse, reciprocals_a, input_channel_size, \
                                            depth)
                {
                    // Allocate scratch space outside of the hot loop
                    std::vector<scalar_t, default_init_allocator<scalar_t>> next_divided (reciprocals_a.size(0) * input_channel_size);
                    std::vector<scalar_t, default_init_allocator<scalar_t>> old_scratch;
                    std::vector<scalar_t, default_init_allocator<scalar_t>> new_scratch;
                    // Figure out how large each vector is going to get by the end of the computation.
                    if (depth > 1) {
                        if ((depth % 2) == 0) {
                            old_scratch.resize(pow(input_channel_size, depth - 2));
                            new_scratch.resize(old_scratch.size() * input_channel_size);
                        }
                        else {
                            new_scratch.resize(pow(input_channel_size, depth - 2));
                            old_scratch.resize(new_scratch.size() * input_channel_size);
                        }
                    }

                    #pragma omp for schedule(static)
                    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                        // Actually do the computation.
                        // Note that we pass in the 2-dimensional TensorAccessors and batch_index and let
                        // mult_fused_restricted_exp_cpu_inner reduce them to 1-dimensional TensorAccessors. This gives
                        // a small speedup over creating the 1-dimensional TensorAccessors out here and passing just
                        // those in.
                        if (inverse) {
                            mult_fused_restricted_exp_cpu_inner<scalar_t, /*inverse=*/true>(next_a,
                                                                                            prev_a,
                                                                                            reciprocals_a,
                                                                                            batch_index,
                                                                                            next_divided,
                                                                                            new_scratch,
                                                                                            old_scratch);
                        }
                        else {
                            mult_fused_restricted_exp_cpu_inner<scalar_t, /*inverse=*/false>(next_a,
                                                                                             prev_a,
                                                                                             reciprocals_a,
                                                                                             batch_index,
                                                                                             next_divided,
                                                                                             new_scratch,
                                                                                             old_scratch);
                        }
                    }
                }

            }

            // If you're reading this function and trying to understand it...
            // ...then good luck.
            // Seriously though, it's a backward through a very complicated operation, so there isn't much getting
            // around the fact that it's going to be a bit involved.
            void mult_fused_restricted_exp_backward_cuda(torch::Tensor grad_next,
                                                         std::vector<torch::Tensor>& grad_prev,
                                                         torch::Tensor next,
                                                         const std::vector <torch::Tensor>& prev,
                                                         bool inverse,
                                                         torch::Tensor reciprocals) {
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

                // Allocate memory for the gradient through next_divided

                torch::Tensor grad_next_divided = torch::zeros_like(next_divided);

                // Allocate memory for the gradient through the scratches

                std::vector<std::vector<torch::Tensor>> all_grad_scratches;
                all_grad_scratches.reserve(all_scratches.size());
                for (const auto& scratches : all_scratches) {
                    all_grad_scratches.emplace_back();
                    std::vector<torch::Tensor>& grad_scratches = all_grad_scratches.back();
                    grad_scratches.reserve(scratches.size());
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
                        grad_next.unsqueeze(channel_dim - 1).baddbmm_(scratch.unsqueeze(channel_dim - 1),
                                                                      grad_prev_view);
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
                    grad_next_divided[depth_index - 1] += grad_scratches[0];
                    grad_prev[0] += grad_scratches[0];
                }

                // Finally the do the backward from next_divided into next

                if (depth > 1) {
                    // In principle when depth == 1 then the code below should be a no-op, but BLAS throws an error here
                    torch::Tensor grad_next_divided_view = grad_next_divided.view({depth - 1,
                                                                                   batch_size * input_channel_size});
                    torch::Tensor grad_next_view = grad_next.view({batch_size * input_channel_size});
                    grad_next_view.unsqueeze(0).addmm_(reciprocals.unsqueeze(0), grad_next_divided_view);
                }
            }

            // Alright, buckle your seatbelts. We're going to do the CPU implementation now.

            template <typename scalar_t, typename accessor>
            int64_t mvsize(const std::vector<scalar_t, accessor>& obj) {
                return obj.size();
            }

            template <typename scalar_t>
            int64_t mvsize(torch::TensorAccessor<scalar_t, 1> obj) {
                return obj.size(0);
            }

            template <typename scalar_t, typename accessor>
            scalar_t mvindex(const std::vector<scalar_t, accessor>& matrix, int64_t index, int64_t out_index,
                             int64_t vector_index) {
                return matrix[index];
            }

            template <typename scalar_t, typename accessor, typename accessor2>
            scalar_t mvindex(const std::vector<std::vector<scalar_t, accessor>, accessor2>& matrix, int64_t index,
                             int64_t out_index, int64_t vector_index) {
                return matrix[vector_index][out_index];
            }

            template <typename scalar_t>
            scalar_t mvindex(torch::TensorAccessor<scalar_t, 1> matrix, int64_t index, int64_t out_index,
                             int64_t vector_index) {
                return matrix[index];
            }

            /* Performs either matrix-vector multiplication or transpose(vector)-matrix multiplication.
             *
             * 'out', and vector' should each either be std::vector<scalar_t> or torch::TensorAccessor<scalar_t, 1>
             * 'matrix' should be either a std::vector<scalar_t> or torch::TensorAccessor<scalar_t, 1>
             *
             * It must be such that the size of 'out', multiplied by the size of 'vector', is equal to the size of
             * 'matrix'.
             *
             * It performs matrix-vector multiplication if flip==false, and transpose(vector)-matrix multiplication if
             * flip==true.
             *
             * It adds the result on to what is already in 'out' if add==true, and stores it in 'out' if add==false.
             *
             *
             * It would be nice to instead delegate this to a BLAS implementation, but I don't know how to access
             * whatever implementation of BLAS that an arbitrary version of PyTorch might ship with.
             */
            template <typename scalar_t, bool flip, bool add, typename T, typename T2, typename T3>
            void mv(T& out, const T2& matrix, const T3& vector) {
                int64_t out_size = mvsize<scalar_t>(out);
                int64_t vector_size = mvsize<scalar_t>(vector);
                if (flip) {
                    int64_t index = 0;
                    for (/*index initialised above*/; index < out_size;/*index increment below*/) {
                        if (add) {
                            out[index] += vector[0] * mvindex<scalar_t>(matrix, index, /*out_index=*/index,
                                                                        /*vector_index=*/0);
                        }
                        else {
                            out[index] = vector[0] * mvindex<scalar_t>(matrix, index, /*out_index=*/index,
                                                                       /*vector_index=*/0);
                        }
                        ++index;
                    }
                    for (int64_t vector_index = 1; vector_index < vector_size; ++vector_index) {
                        for (int64_t out_index = 0; out_index < out_size; ++out_index) {
                            out[out_index] += vector[vector_index] * mvindex<scalar_t>(matrix, index, out_index,
                                                                                       vector_index);
                            ++index;
                        }
                    }
                }
                else {
                    int64_t index = 0;
                    for (int64_t out_index = 0; out_index < out_size; ++out_index) {
                        if (add) {
                            out[out_index] += vector[0] * mvindex<scalar_t>(matrix, index, out_index,
                                                                            /*vector_index=*/0);
                        }
                        else {
                            out[out_index] = vector[0] * mvindex<scalar_t>(matrix, index, out_index,
                                                                           /*vector_index=*/0);
                        }
                        ++index;
                        for (int64_t vector_index = 1; vector_index < vector_size; ++vector_index) {
                            out[out_index] += vector[vector_index] * mvindex<scalar_t>(matrix, index, out_index,
                                                                                       vector_index);
                            ++index;
                        }
                    }
                }
            }

            template <typename scalar_t, bool inverse>
            void
            mult_fused_restricted_exp_backward_cpu_inner(torch::TensorAccessor<scalar_t, 2> grad_next_a,
                                                         std::vector<torch::TensorAccessor<scalar_t, 2>>& grad_prev_a,
                                                         torch::TensorAccessor<scalar_t, 2> next_a,
                                                         const std::vector<torch::TensorAccessor<scalar_t, 2>>& prev_a,
                                                         torch::TensorAccessor<scalar_t, 1> reciprocals_a,
                                                         int64_t batch_index) {
                int64_t input_channel_size = next_a.size(1);  // 1 is the channel dimension
                s_size_type depth = prev_a.size();

                std::vector<std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>> all_scratches;
                all_scratches.reserve(depth - 1);

                std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>
                        next_divided (reciprocals_a.size(0),
                                      std::vector<scalar_t, default_init_allocator<scalar_t>> (input_channel_size));
                for (int64_t reciprocal_index = 0; reciprocal_index < reciprocals_a.size(0); ++reciprocal_index) {
                    for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                        next_divided[reciprocal_index][channel_index] = reciprocals_a[reciprocal_index] *
                                                                        next_a[batch_index][channel_index];
                    }
                }

                if (depth > 1) {
                    std::vector<scalar_t, default_init_allocator<scalar_t>> new_scratch;
                    std::vector<scalar_t, default_init_allocator<scalar_t>> old_scratch;
                    if ((depth % 2) == 0) {
                        old_scratch.reserve(pow(input_channel_size, depth - 2));
                        new_scratch.reserve(old_scratch.size() * input_channel_size);
                    }
                    else {
                        new_scratch.reserve(pow(input_channel_size, depth - 2));
                        old_scratch.reserve(new_scratch.size() * input_channel_size);
                    }

                    for (s_size_type depth_index = depth - 1; depth_index >= 1; --depth_index) {
                        all_scratches.emplace_back();
                        std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>&
                                scratches = all_scratches.back();
                        scratches.reserve(depth_index);

                        new_scratch.resize(input_channel_size);
                        for (int64_t scratch_index = 0; scratch_index < input_channel_size; ++scratch_index) {
                            new_scratch[scratch_index] = prev_a[0][batch_index][scratch_index] +
                                                         next_divided[depth_index - 1][scratch_index];
                        }

                        scratches.push_back(new_scratch);

                        for (s_size_type j = 1, k = depth_index - 2; j < depth_index; ++j, --k) {
                            old_scratch.swap(new_scratch);
                            new_scratch.resize(old_scratch.size() * input_channel_size);
                            for (int64_t old_scratch_index = 0;
                                 old_scratch_index < static_cast<int64_t>(old_scratch.size());
                                 ++old_scratch_index) {
                                for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                                    int64_t new_scratch_index;
                                    if (inverse) {
                                        new_scratch_index = channel_index * old_scratch.size() + old_scratch_index;
                                    }
                                    else {
                                        new_scratch_index = old_scratch_index * input_channel_size + channel_index;
                                    }
                                    new_scratch[new_scratch_index] = prev_a[j][batch_index][new_scratch_index] +
                                                                     old_scratch[old_scratch_index] *
                                                                     next_divided[k][channel_index];
                                }
                            }

                            scratches.push_back(new_scratch);
                        }
                    }
                }

                // Allocate memory for the gradient through next_divided

                std::vector<std::vector<scalar_t>> grad_next_divided;
                grad_next_divided.reserve(next_divided.size());
                for (s_size_type next_divided_index = 0;
                     next_divided_index < static_cast<s_size_type>(next_divided.size());
                     ++next_divided_index)
                {
                    // Deliberately initialising to zero here, rather than empty. It would just be rather a faff to
                    // write code that pulls out the first iteration we use this, and set rather than add in that
                    // iteration.
                    grad_next_divided.push_back(std::vector<scalar_t> (input_channel_size, 0));
                }

                // Allocate memory for the gradient through the scratches

                std::vector<std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>> all_grad_scratches;
                all_grad_scratches.reserve(all_scratches.size());
                for (const auto& scratches : all_scratches) {
                    all_grad_scratches.emplace_back();
                    std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>& grad_scratches =
                            all_grad_scratches.back();
                    grad_scratches.reserve(scratches.size());
                    for (const auto& elem : scratches) {
                        grad_scratches.push_back(std::vector<scalar_t, default_init_allocator<scalar_t>> (elem.size()));
                    }
                }

                // Do the backward computation

                for (int64_t index = 0; index < grad_prev_a[0].size(1); ++index) {
                    grad_next_a[batch_index][index] = grad_prev_a[0][batch_index][index];
                }
                for (s_size_type depth_index = 1, back_index = all_scratches.size() - 1;
                     depth_index < depth;
                     ++depth_index, --back_index) {
                    std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>& grad_scratches =
                            all_grad_scratches[back_index];
                    const std::vector<std::vector<scalar_t, default_init_allocator<scalar_t>>>& scratches =
                            all_scratches[back_index];

                    std::vector<scalar_t, default_init_allocator<scalar_t>>& grad_scratch = grad_scratches.back();
                    const std::vector<scalar_t, default_init_allocator<scalar_t>>& scratch = scratches.back();

                    mv<scalar_t, /*flip=*/inverse, /*add=*/false>(grad_scratch,
                                                                  grad_prev_a[depth_index][batch_index],
                                                                  next_a[batch_index]);
                    auto grad_next_a_at_batch = grad_next_a[batch_index];
                    mv<scalar_t, /*flip=*/!inverse, /*add=*/true>(grad_next_a_at_batch,
                                                                  grad_prev_a[depth_index][batch_index],
                                                                  scratch);

                    for (s_size_type j = depth_index - 1, k = 0; j >= 1; --j, ++k) {
                        const std::vector<scalar_t, default_init_allocator<scalar_t>>& grad_scratch =
                                grad_scratches[j];
                        std::vector<scalar_t, default_init_allocator<scalar_t>>& grad_old_scratch =
                                grad_scratches[j - 1];
                        const std::vector<scalar_t, default_init_allocator<scalar_t>>& old_scratch =
                                scratches[j - 1];
                        const std::vector<scalar_t, default_init_allocator<scalar_t>>& next_divided_narrow =
                                next_divided[k];
                        std::vector<scalar_t>& grad_next_divided_narrow = grad_next_divided[k];

                        for (s_size_type index = 0; index < static_cast<s_size_type>(grad_scratch.size()); ++index) {
                            grad_prev_a[j][batch_index][index] += grad_scratch[index];
                        }

                        mv<scalar_t, /*flip=*/inverse, /*add=*/false>(grad_old_scratch, grad_scratch,
                                                                      next_divided_narrow);
                        mv<scalar_t, /*flip=*/!inverse, /*add=*/true>(grad_next_divided_narrow, grad_scratch,
                                                                      old_scratch);
                    }
                    for (s_size_type index = 0;
                         index < static_cast<s_size_type>(grad_next_divided[depth_index - 1].size());
                         ++index) {
                        grad_next_divided[depth_index - 1][index] += grad_scratches[0][index];
                        grad_prev_a[0][batch_index][index] += grad_scratches[0][index];
                    }
                }

                if (depth > 1) {
                    auto grad_next_a_at_batch = grad_next_a[batch_index];
                    mv<scalar_t, /*flip=*/true, /*add=*/true>(grad_next_a_at_batch, grad_next_divided,
                                                              reciprocals_a);
                }
            }

            template <typename scalar_t>
            void mult_fused_restricted_exp_backward_cpu(torch::Tensor grad_next,
                                                        std::vector<torch::Tensor>& grad_prev,
                                                        torch::Tensor next,
                                                        const std::vector<torch::Tensor>& prev,
                                                        bool inverse,
                                                        torch::Tensor reciprocals) {
                auto grad_next_a = grad_next.accessor<scalar_t, 2>();

                std::vector<torch::TensorAccessor<scalar_t, 2>> grad_prev_a;
                grad_prev_a.reserve(grad_prev.size());
                for (auto elem : grad_prev) {
                    grad_prev_a.push_back(elem.accessor<scalar_t, 2>());
                }

                auto next_a = next.accessor<scalar_t, 2>();

                std::vector<torch::TensorAccessor<scalar_t, 2>> prev_a;
                prev_a.reserve(prev.size());
                for (auto elem : prev) {
                    prev_a.push_back(elem.accessor<scalar_t, 2>());
                }

                auto reciprocals_a = reciprocals.accessor<scalar_t, 1>();

                int64_t batch_size = next.size(batch_dim);
                #pragma omp parallel for default(none) \
                                         shared(batch_size, grad_next_a, grad_prev_a, next_a, prev_a, inverse, \
                                                reciprocals_a)
                for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                    if (inverse) {
                        mult_fused_restricted_exp_backward_cpu_inner<scalar_t,
                                                                     /*inverse=*/true>(grad_next_a,
                                                                                       grad_prev_a,
                                                                                       next_a,
                                                                                       prev_a,
                                                                                       reciprocals_a,
                                                                                       batch_index);
                    }
                    else {
                        mult_fused_restricted_exp_backward_cpu_inner<scalar_t,
                                                                     /*inverse=*/false>(grad_next_a,
                                                                                        grad_prev_a,
                                                                                        next_a,
                                                                                        prev_a,
                                                                                        reciprocals_a,
                                                                                        batch_index);
                    }
                }
            }
        }  // namespace signatory::ta_ops::detail

        void mult_fused_restricted_exp(torch::Tensor next, std::vector<torch::Tensor>& prev, bool inverse,
                                       torch::Tensor reciprocals, int64_t batch_threads) {
            if (next.is_cuda()) {
                detail::mult_fused_restricted_exp_cuda(next, prev, inverse, reciprocals);
            }
            else{
                AT_DISPATCH_FLOATING_TYPES(next.scalar_type(), "mult_fused_restricted_exp_cpu", ([&] {
                    detail::mult_fused_restricted_exp_cpu<scalar_t>(next, prev, inverse, reciprocals, batch_threads);
                }));
            }
        }

        void mult_fused_restricted_exp_backward(torch::Tensor grad_next,
                                                std::vector<torch::Tensor>& grad_prev,
                                                torch::Tensor next,
                                                const std::vector<torch::Tensor>& prev,
                                                bool inverse,
                                                torch::Tensor reciprocals) {
            if (grad_next.is_cuda()) {
                detail::mult_fused_restricted_exp_backward_cuda(grad_next, grad_prev, next, prev, inverse, reciprocals);
            }
            else{
                AT_DISPATCH_FLOATING_TYPES(grad_next.scalar_type(), "mult_fused_restricted_exp_backward_cpu", ([&] {
                    detail::mult_fused_restricted_exp_backward_cpu<scalar_t>(grad_next, grad_prev, next, prev, inverse,
                                                                             reciprocals);
                }));
            }
        }

        /***********************************************
         * Forward and backward computations for 'log' *
         ***********************************************/

        namespace detail {
            // The coefficient of a term in the power series of the logarithm
            torch::Scalar log_coefficient_at_depth(s_size_type depth_index, torch::Tensor reciprocals) {
                return ((((depth_index % 2) == 0) ? -1 : 1) * reciprocals[depth_index]).item();
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

                    mult_inner(tensor_at_depth, arg1, arg2, depth_index);

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

                    mult_inner_backward(grad_tensor_at_depth, grad_arg1, grad_arg2, arg1, arg2, depth_index);

                    grad_tensor_at_depth.zero_();
                }
            }
        }  // namespace signatory::ta_ops::detail

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

    /*************************************************************
     * Forward and backward computations for 'signature_combine' *
     *************************************************************/

    torch::Tensor signature_combine_forward(std::vector<torch::Tensor> sigtensors, // copy not reference as we modify it
                                            int64_t input_channels,
                                            s_size_type depth,
                                            bool scalar_term) {
        // Perform a bunch of argument checking

        misc::checkargs_channels_depth(input_channels, depth);
        if (sigtensors.size() == 0) {
            throw std::invalid_argument("sigtensors must be of nonzero length.");
        }
        int64_t expected_signature_channels = signature_channels(input_channels, depth, scalar_term);
        if (sigtensors[0].ndimension() != 2) {
            throw std::invalid_argument("An element of sigtensors is not two-dimensional. Every element must have "
                                        "two dimensions, corresponding to "
                                        "(batch, signature_channels(input_channels, depth, scalar_term))");
        }

        py::gil_scoped_release release;

        int64_t batch_size = sigtensors[0].size(batch_dim);
        for (auto& elem : sigtensors) {
            if (elem.ndimension() != 2) {
                throw std::invalid_argument("An element of sigtensors is not two-dimensional. Every element must have "
                                            "two dimensions, corresponding to "
                                            "(batch, signature_channels(input_channels, depth, scalar_term))");
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

        // Actually do the computation

        torch::Tensor out;
        torch::Tensor out_with_scalar = sigtensors[0].clone();
        if (scalar_term) {
            out = out_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/1,
                                         /*length=*/out_with_scalar.size(channel_dim) - 1);
        }
        else {
            out = out_with_scalar;
        }
        std::vector<torch::Tensor> out_vector;
        misc::slice_by_term(out, out_vector, input_channels, depth);
        for (s_size_type sigtensor_index = 1;
             sigtensor_index < static_cast<s_size_type>(sigtensors.size());
             ++sigtensor_index) {
            std::vector<torch::Tensor> sigtensor_vector;
            torch::Tensor sigtensor = sigtensors[sigtensor_index];
            if (scalar_term) {
                sigtensor = sigtensor.narrow(/*dim=*/channel_dim, /*start=*/1,
                                             /*length=*/sigtensor.size(channel_dim) - 1);
            }
            misc::slice_by_term(sigtensor, sigtensor_vector, input_channels, depth);
            ta_ops::mult(out_vector, sigtensor_vector, /*inverse=*/false);
        }
        return out_with_scalar;
    }

    std::vector<torch::Tensor> signature_combine_backward(torch::Tensor grad_out,
                                                          // copy not reference as we modify it
                                                          std::vector<torch::Tensor> sigtensors,
                                                          int64_t input_channels,
                                                          s_size_type depth,
                                                          bool scalar_term) {

        py::gil_scoped_release release;

        grad_out = grad_out.detach();
        for (auto& elem : sigtensors) {
            elem = elem.detach();
        }

        // Allocate memory for the output gradients
        std::vector<torch::Tensor> grad_sigtensors;
        std::vector<torch::Tensor> grad_sigtensors_with_scalars;
        grad_sigtensors.reserve(sigtensors.size());
        grad_sigtensors_with_scalars.reserve(sigtensors.size());
        grad_sigtensors.emplace_back();  // we'll fill in the first slot at the very end
        grad_sigtensors_with_scalars.emplace_back();
        for (s_size_type sigtensors_index = 1;
             sigtensors_index < static_cast<s_size_type>(sigtensors.size());
             ++sigtensors_index) {
            torch::Tensor grad_sigtensor_with_scalar = torch::empty_like(sigtensors[sigtensors_index]);
            torch::Tensor grad_sigtensor;
            if (scalar_term) {
                grad_sigtensor_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/0, /*length=*/1).zero_();
                grad_sigtensor = grad_sigtensor_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/1,
                                                                   /*length=*/grad_sigtensor_with_scalar.size(channel_dim) - 1);
            }
            else {
                grad_sigtensor = grad_sigtensor_with_scalar;
            }
            grad_sigtensors.push_back(grad_sigtensor);
            grad_sigtensors_with_scalars.push_back(grad_sigtensor_with_scalar);
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
        if (scalar_term) {
            scratch = scratch.narrow(/*dim=*/channel_dim, /*start=*/1, /*length=*/scratch.size(channel_dim) - 1);
        }
        // -1 to the size because we don't need to store the final output
        for (s_size_type sigtensor_index = 1;
             sigtensor_index < static_cast<s_size_type>(sigtensors.size()) - 1;
             ++sigtensor_index) {
            scratch = scratch.clone();
            std::vector<torch::Tensor> scratch_vector;
            misc::slice_by_term(scratch, scratch_vector, input_channels, depth);

            std::vector<torch::Tensor> sigtensor_vector;
            torch::Tensor sigtensor = sigtensors[sigtensor_index];
            if (scalar_term) {
                sigtensor = sigtensor.narrow(/*dim=*/channel_dim, /*start=*/1,
                                             /*length=*/sigtensor.size(channel_dim) - 1);
            }
            misc::slice_by_term(sigtensor, sigtensor_vector, input_channels, depth);
            ta_ops::mult(scratch_vector, sigtensor_vector, /*inverse=*/false);

            scratch_vector_vector.push_back(scratch_vector);
        }

        // Allocate memory for the gradient when computing backward through the tensor multiplications
        torch::Tensor grad_scratch_with_scalar = grad_out.clone();
        torch::Tensor grad_scratch;
        if (scalar_term) {
            grad_scratch_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/0, /*length=*/1).zero_();
            grad_scratch = grad_scratch_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/1,
                                                           /*length=*/grad_out.size(channel_dim) - 1);
        }
        else {
            grad_scratch = grad_scratch_with_scalar;
        }
        std::vector<torch::Tensor> grad_scratch_vector;
        misc::slice_by_term(grad_scratch, grad_scratch_vector, input_channels, depth);

        // Actually do the computation
        for (s_size_type sigtensors_index = sigtensors.size() - 1; sigtensors_index >= 2; --sigtensors_index) {
            // Recompute the inputs of each multiplication
            std::vector<torch::Tensor> sigtensor_vector;
            torch::Tensor sigtensor = sigtensors[sigtensors_index];
            if (scalar_term) {
                sigtensor = sigtensor.narrow(/*dim=*/channel_dim, /*start=*/1,
                                             /*length=*/sigtensor.size(channel_dim) - 1);
            }
            misc::slice_by_term(sigtensor, sigtensor_vector, input_channels, depth);

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
            // Correponds to sigtensors_index == 1
            // This iteration pulled out because we don't need to do the final division
            std::vector<torch::Tensor> sigtensor_vector;
            torch::Tensor sigtensor_one = sigtensors[1];
            if (scalar_term) {
                sigtensor_one = sigtensor_one.narrow(/*dim=*/channel_dim, /*start=*/1,
                                                     /*length=*/sigtensor_one.size(channel_dim) - 1);
            }
            misc::slice_by_term(sigtensor_one, sigtensor_vector, input_channels, depth);
            std::vector<torch::Tensor> first_sigtensor_vector;
            torch::Tensor sigtensor_zero = sigtensors[0];
            if (scalar_term) {
                sigtensor_zero = sigtensor_zero.narrow(/*dim=*/channel_dim, /*start=*/1,
                                                       /*length=*/sigtensor_zero.size(channel_dim) - 1);
            }
            misc::slice_by_term(sigtensor_zero, first_sigtensor_vector, input_channels, depth);
            std::vector<torch::Tensor> grad_sigtensor_vector;
            misc::slice_by_term(grad_sigtensors[1], grad_sigtensor_vector, input_channels, depth);
            ta_ops::mult_backward</*add_not_copy=*/false>(grad_scratch_vector, grad_sigtensor_vector,
                                                          first_sigtensor_vector, sigtensor_vector);
        }
        // Fill in the gradient for the very first sigtensor.
        grad_sigtensors_with_scalars[0] = grad_scratch_with_scalar;
        grad_sigtensors[0] = grad_scratch;  // unnecessary, I'm pretty sure. TODO: remove this + shorten grad_sigtensors by one?

        return grad_sigtensors_with_scalars;
    }
}  // namespace signatory