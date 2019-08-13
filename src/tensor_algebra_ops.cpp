#include <torch/extension.h>
#include <vector>     // std::vector

#include "misc.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    namespace ta_ops {
        namespace detail {
            template<bool invert=false>
            void compute_multdiv_inner(torch::Tensor tensor_at_depth_to_calculate,
                                       const std::vector<torch::Tensor>& arg1,
                                       const std::vector<torch::Tensor>& arg2,
                                       size_type depth_to_calculate,
                                       const misc::SigSpec& sigspec) {
                for (size_type j = 0, k = depth_to_calculate - 1; j < depth_to_calculate; ++j, --k) {
                    /* loop invariant: j + k = depth_to_calculate - 1 */
                    torch::Tensor view_out = tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                arg2[k].size(0),
                                                                                sigspec.batch_size});
                    view_out.addcmul_(arg2[k].unsqueeze(0),              /* += (this tensor times */
                                      arg1[j].unsqueeze(1),              /*     this tensor times */
                                      (invert && misc::is_even(k)) ? -1 : 1);  /*     this scalar)      */
                }
            }

            // No template argument as we only ever need to perform this with invert=false
            void compute_multdiv_inner_backward(torch::Tensor grad_tensor_at_depth_to_calculate,
                                                std::vector<torch::Tensor>& grad_arg1,
                                                std::vector<torch::Tensor>& grad_arg2,
                                                const std::vector<torch::Tensor> arg1,
                                                const std::vector<torch::Tensor> arg2,
                                                size_type depth_to_calculate,
                                                const misc::SigSpec& sigspec) {
                for (size_type j = depth_to_calculate - 1, k = 0; j >= 0; --j, ++k) {
                    /* loop invariant: j + k = depth_to_calculate - 1 */
                    torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                     arg2[k].size(0),
                                                                                     sigspec.batch_size});
                    /* This is just a batch matrix-multiply with the batch dimension in last place
                       It's not totally clear what the optimal way of writing this operation is, but this is at least
                       faster than transposing and using .baddbmm_ (not surprising, given the transposing) */
                    grad_arg1[j] += (out_view * arg2[k].unsqueeze(0)).sum(/*dim=*/1);
                    grad_arg2[k] += (out_view * arg1[j].unsqueeze(1)).sum(/*dim=*/0);
                }
            }

            void compute_log_partial(std::vector<torch::Tensor>& logsignature_vector,
                                     const std::vector<torch::Tensor>& signature_vector,
                                     size_type lower_depth_index,
                                     const misc::SigSpec& sigspec) {
                for (size_type depth_index = sigspec.depth - 3; depth_index >= lower_depth_index; --depth_index) {
                    compute_mult_partial(logsignature_vector, signature_vector,
                                         log_coefficient_at_depth(depth_index, sigspec),
                                         depth_index + 1, sigspec);
                }
            }
        }  // namespace signatory::ta_ops::detail
        void compute_restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const misc::SigSpec& sigspec) {
            out[0].copy_(in);
            for (size_type i = 0; i < sigspec.depth - 1; ++i) {
                torch::Tensor view_out = out[i + 1].view({in.size(0), out[i].size(0), sigspec.batch_size});
                torch::mul_out(view_out, out[i].unsqueeze(0), in.unsqueeze(1));
                out[i + 1] *= sigspec.reciprocals[i];
            }
        }

        void compute_restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                             torch::Tensor in, const std::vector<torch::Tensor>& out,
                                             const misc::SigSpec& sigspec) {
            if (sigspec.depth >= 2) {
                // grad_out is a vector of length sigspec.depth.
                // grad_out[sigspec.depth - 1] doesn't need any gradients added on to it.
                // grad_out[sigspec.depth - 2] has the computation done in the pulled-out first iteration
                // grad_out[sigspec.depth - 3] and below are handled in the for loop. (Hence the strange starting index
                //   for i)

                // first iteration is pulled out so that grad_in uses copy instead of += the first time around
                grad_out[sigspec.depth - 1] *= sigspec.reciprocals[sigspec.depth - 2];
                torch::Tensor view_grad_out = grad_out[sigspec.depth - 1].view({in.size(0),
                                                                                out[sigspec.depth - 2].size(0),
                                                                                sigspec.batch_size});
                grad_out[sigspec.depth - 2] += (view_grad_out * in.unsqueeze(1)).sum(/*dim=*/0);
                grad_in.copy_((view_grad_out * out[sigspec.depth - 2].unsqueeze(0)).sum(/*dim=*/1));

                for (size_type i = sigspec.depth - 3; i >= 0; --i) {
                    grad_out[i + 1] *= sigspec.reciprocals[i];
                    view_grad_out = grad_out[i + 1].view({in.size(0), out[i].size(0), sigspec.batch_size});

                    // This is just a batch matrix-multiply with the batch dimension in last place
                    // It's not totally clear what the optimal way of writing this operation is, but this is at least
                    // faster than transposing and using .baddbmm_ (not surprising, given the transposing)
                    grad_in += (view_grad_out * out[i].unsqueeze(0)).sum(/*dim=*/1);
                    grad_out[i] += (view_grad_out * in.unsqueeze(1)).sum(/*dim=*/0);
                }
                grad_in += grad_out[0];
            }
            else {  // implies depth == 1
                grad_in.copy_(grad_out[0]);
            }
        }

        void compute_mult(std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2, bool rightret,
                          const misc::SigSpec& sigspec) {
            for (size_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = (rightret ? arg2 : arg1)[depth_to_calculate];

                detail::compute_multdiv_inner(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, sigspec);

                tensor_at_depth_to_calculate += (rightret ? arg1 : arg2)[depth_to_calculate];
            }
        }

        void compute_mult_backward(std::vector<torch::Tensor>& grad_arg1, std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                   bool add_not_copy, const misc::SigSpec& sigspec) {
            for (size_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth; ++depth_to_calculate) {
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
            for (size_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
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
                                  torch::Scalar scalar_term_value, size_type top_terms_to_skip,
                                  const misc::SigSpec& sigspec) {
            for (size_type depth_to_calculate = sigspec.depth - top_terms_to_skip - 1; depth_to_calculate >= 0;
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
                                           size_type top_terms_to_skip,
                                           const misc::SigSpec& sigspec) {
            for (size_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth - top_terms_to_skip;
                 ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg1[depth_to_calculate];

                grad_arg2[depth_to_calculate].add_(grad_tensor_at_depth_to_calculate, scalar_value_term);

                detail::compute_multdiv_inner_backward(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1,
                                                       arg2, depth_to_calculate, sigspec);

                grad_tensor_at_depth_to_calculate.zero_();
            }
        }

        torch::Scalar log_coefficient_at_depth(size_type depth_index, const misc::SigSpec& sigspec) {
            return ((misc::is_even(depth_index) ? -1 : 1) * sigspec.reciprocals[depth_index]).item();
        }

        void compute_log(std::vector<torch::Tensor>& output_vector,
                         const std::vector<torch::Tensor>& input_vector,
                         const misc::SigSpec& sigspec) {
            detail::compute_log_partial(output_vector, input_vector, /*lower_depth_index=*/0, sigspec);
            compute_mult_partial(output_vector, input_vector, /*scalar_value_term=*/1,
                    /*top_terms_to_skip=*/0, sigspec);
        }

        void compute_log_backward(std::vector<torch::Tensor>& grad_output_vector,
                                  std::vector<torch::Tensor>& grad_input_vector,
                                  std::vector<torch::Tensor>& scratch_vector,
                                  const std::vector<torch::Tensor>& input_vector,
                                  torch::Tensor scratch,
                                  torch::Tensor scratch_init,
                                  const misc::SigSpec& sigspec) {
            scratch.copy_(scratch_init);
            scratch *= log_coefficient_at_depth(sigspec.depth - 2, sigspec);

            detail::compute_log_partial(scratch_vector, input_vector, 0, sigspec);
            compute_mult_partial_backward(grad_output_vector, grad_input_vector, scratch_vector,
                                          input_vector, 1, 0, sigspec);

            for (size_type depth_index = 0; depth_index < sigspec.depth - 2; ++depth_index) {
                scratch.copy_(scratch_init);
                scratch *= log_coefficient_at_depth(sigspec.depth - 2, sigspec);

                /* Yuck, this is O(depth^2). Sadly I don't see a way to compute this without either that or saving
                 * intermediate results, which is in some sense even worse. */
                detail::compute_log_partial(scratch_vector, input_vector, depth_index + 1, sigspec);

                compute_mult_partial_backward(grad_output_vector, grad_input_vector, scratch_vector,
                                              input_vector, log_coefficient_at_depth(depth_index, sigspec),
                                              depth_index + 1, sigspec);
            }
        }
    }  // namespace signatory::ta_ops
}  // namespace signatory