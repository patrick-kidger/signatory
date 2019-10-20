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
#include <cmath>      // std::lround
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <tuple>      // std::tie, std::tuple
#include <vector>     // std::vector

#include "misc.hpp"
#include "signature.hpp"
#include "tensor_algebra_ops.hpp"
#

namespace signatory {
    namespace detail {
        // Takes the path and basepoint and returns the path increments
        torch::Tensor compute_path_increments(torch::Tensor path, bool basepoint, torch::Tensor basepoint_value,
                                              bool inverse) {
            int64_t num_increments {path.size(stream_dim) - 1};
            // The difference between these cases: basepoint/no basepoint + inverse/no inverse are basically just
            // niceties.
            // Essentially all that's going on is that if basepoint is passed then the basepoint is concatenated on to
            // the path.
            // All that's going on if inverse is passed is just to multiply everything by -1.
            // We break it up into special cases like this because doing either of the above operations naively involves
            // unnecessary extra operations.
            if (basepoint) {
                if (inverse) {
                    torch::Tensor path_increments = torch::empty_like(path);
                    path_increments[0].copy_(basepoint_value);
                    path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments).copy_(
                            path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments));
                    path_increments -= path;
                    return path_increments;
                }
                else {
                    torch::Tensor path_increments = path.clone();
                    path_increments[0] -= basepoint_value;
                    path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments) -=
                            path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments);
                    return path_increments;
                }
            }
            else {
                if (inverse) {
                    return path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments) -
                           path.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments);
                }
                else {
                    return path.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments) -
                           path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments);
                }
            }
        }

        // Computes the backward pass through the path increments operation.
        // Returns the gradients for the original path, and for the basepoint.
        std::tuple<torch::Tensor, torch::Tensor>
        compute_path_increments_backward(torch::Tensor grad_path_increments, bool basepoint, bool inverse,
                                         torch::TensorOptions opts) {
            int64_t batch_size {grad_path_increments.size(batch_dim)};
            int64_t input_stream_size {grad_path_increments.size(stream_dim)};
            int64_t input_channel_size {grad_path_increments.size(channel_dim)};
            if (!basepoint) {
                ++input_stream_size;
            }

            int64_t num_increments{input_stream_size - 1};
            if (basepoint) {
                if (inverse) {
                    torch::Tensor grad_path = torch::empty_like(grad_path_increments);
                    grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments).copy_(
                            grad_path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments));
                    grad_path[-1].zero_();
                    grad_path -= grad_path_increments;
                    return std::tuple<torch::Tensor, torch::Tensor>
                           {grad_path, grad_path_increments[0]};
                }
                else {
                    torch::Tensor grad_path = grad_path_increments.clone();
                    grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments)
                            -= grad_path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments);
                    return std::tuple<torch::Tensor, torch::Tensor>
                           {grad_path, -grad_path_increments[0]};
                }
            }
            else {
                if (inverse) {
                    torch::Tensor grad_path = torch::empty({input_stream_size, batch_size, input_channel_size}, opts);
                    grad_path[-1].zero_();
                    grad_path.narrow(/*dim=*/stream_dim,
                                     /*start=*/0,
                                     /*len=*/num_increments).copy_(grad_path_increments);
                    grad_path.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments) -= grad_path_increments;
                    // no second return value in this case
                    return std::tuple<torch::Tensor, torch::Tensor> {grad_path, torch::empty({0}, opts)};

                }
                else {
                    torch::Tensor grad_path = torch::empty({input_stream_size, batch_size, input_channel_size}, opts);
                    grad_path[0].zero_();
                    grad_path.narrow(/*dim=*/stream_dim, /*start=*/1,
                                     /*len=*/num_increments).copy_(grad_path_increments);
                    grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments) -= grad_path_increments;
                    // no second return value in this case
                    return std::tuple<torch::Tensor, torch::Tensor> {grad_path, torch::empty({0}, opts)};
                }
            }
        }

        struct bool_wrapper { bool value; };
    }  // namespace signatory::detail

    void signature_checkargs(torch::Tensor path, s_size_type depth, bool basepoint, torch::Tensor basepoint_value,
                             bool initial, torch::Tensor initial_value) {
        if (path.ndimension() == 2) {
            // Friendlier help message for a common mess-up.
            throw std::invalid_argument("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                                        "corresponding to (batch, stream, channel) respectively. If you just want "
                                        "the signature or logsignature of a single path then wrap it in a single "
                                        "batch dimension by replacing e.g. `signature(path, depth)` with "
                                        "`signature(path.unsqueeze(0), depth).squeeze(0)`.");
        }
        if (path.ndimension() != 3) {
            throw std::invalid_argument("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                                        "corresponding to (batch, stream, channel) respectively.");
        }
        if (path.size(batch_dim) == 0 || path.size(stream_dim) == 0 || path.size(channel_dim) == 0) {
            throw std::invalid_argument("Argument 'path' cannot have dimensions of size zero.");
        }
        if (!basepoint && path.size(stream_dim) == 1) {
            throw std::invalid_argument("Argument 'path' must have stream dimension of size at least 2. (Need at "
                                        "least this many points to define a path.)");
        }
        if (depth < 1) {
            throw std::invalid_argument("Argument 'depth' must be an integer greater than or equal to one.");
        }
        if (!path.is_floating_point()) {
            throw std::invalid_argument("Argument 'path' must be of floating point type.");
        }
        torch::TensorOptions path_opts = misc::make_opts(path);
        if (basepoint) {
            if (basepoint_value.ndimension() != 2) {
                throw std::invalid_argument("Argument 'basepoint' must be a 2-dimensional tensor, corresponding to "
                                            "(batch, channel) respectively.");
            }
            if (basepoint_value.size(channel_dim) != path.size(channel_dim) ||
                basepoint_value.size(batch_dim) != path.size(batch_dim)) {
                throw std::invalid_argument("Arguments 'basepoint' and 'path' must have dimensions of the same "
                                            "size.");
            }
            if (path_opts != misc::make_opts(basepoint_value)) {
                throw std::invalid_argument("Argument 'basepoint' does not have the same dtype or device as "
                                            "'path'.");
            }
        }
        if (initial) {
            if (initial_value.ndimension() != 2) {
                throw std::invalid_argument("Argument 'initial' must be a 2-dimensional tensor, corresponding to "
                                            "(batch, signature_channels) respectively.");
            }
            if (initial_value.size(channel_dim) != signature_channels(path.size(channel_dim), depth) ||
                initial_value.size(batch_dim) != path.size(batch_dim)) {
                throw std::invalid_argument("Argument 'initial' must have correctly sized batch and channel "
                                            "dimensions.");
            }
            if (path_opts != misc::make_opts(initial_value)) {
                throw std::invalid_argument("Argument 'initial' does not have the same dtype or device as 'path'.");
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor>
    signature_forward(torch::Tensor path, s_size_type depth, bool stream, bool basepoint, torch::Tensor basepoint_value,
                      bool inverse, bool initial, torch::Tensor initial_value, bool open_mp_parallelise) {
        signature_checkargs(path, depth, basepoint, basepoint_value, initial, initial_value);

        // No sense keeping track of gradients when we have a dedicated backwards function (and in-place operations mean
        // that in any case one cannot autograd through this function)
        path = path.detach();
        basepoint_value = basepoint_value.detach();
        initial_value = initial_value.detach();

        int64_t batch_size = path.size(batch_dim);
        int64_t input_channel_size = path.size(channel_dim);
        int64_t output_stream_size = path.size(stream_dim) - (basepoint ? 0 : 1);
        int64_t output_channel_size = signature_channels(path.size(channel_dim), depth);
        torch::TensorOptions opts = misc::make_opts(path);
        torch::Tensor reciprocals = misc::make_reciprocals(depth, opts);

        torch::Tensor path_increments = detail::compute_path_increments(path, basepoint, basepoint_value, inverse);

        // We allocate memory for certain things upfront.
        // We want to construct things in-place wherever possible. Signatures get large; this saves a lot of time.

        torch::Tensor first_term;
        torch::Tensor signature;
        std::vector<torch::Tensor> signature_by_term;
        std::vector<torch::Tensor> signature_by_term_at_stream;
        if (stream) {
            // if stream == true then we want to store all intermediate results
            signature = torch::empty({output_stream_size, batch_size, output_channel_size}, opts);
            first_term = signature[0];
            misc::slice_by_term(signature, signature_by_term, input_channel_size, depth);
        }
        else {
            signature = torch::empty({batch_size, output_channel_size}, opts);
            first_term = signature;
        }
        misc::slice_by_term(first_term, signature_by_term_at_stream, input_channel_size, depth);

        // compute the first term
        if (initial) {
            first_term.copy_(initial_value);
            ta_ops::mult_fused_restricted_exp(path_increments[0],
                                              signature_by_term_at_stream,
                                              inverse,
                                              reciprocals);
        }
        else {
            ta_ops::restricted_exp(path_increments[0],
                                   signature_by_term_at_stream,
                                   reciprocals);
        }

        if (stream) {
            for (int64_t stream_index = 1; stream_index < output_stream_size; ++stream_index) {
                signature[stream_index].copy_(signature[stream_index - 1]);
                misc::slice_at_stream(signature_by_term, signature_by_term_at_stream, stream_index);
                ta_ops::mult_fused_restricted_exp(path_increments[stream_index],
                                                  signature_by_term_at_stream,
                                                  inverse,
                                                  reciprocals);
            }
        }
        else {
            if (open_mp_parallelise && open_mp) {
                int64_t nthreads = omp_get_max_threads();
                std::vector<std::vector<torch::Tensor>> omp_results(nthreads);
                // There's no guarantee that we actually get the maximum number of threads, so we have to check
                // which ones actually get used.
                // This also serves as a check that start < end, in the block below
                std::vector<detail::bool_wrapper> omp_used(nthreads, {false});
                // As for why we bother with the wrapper: std::vector<bool> is special-cased from other vectors and is
                // basically broken. https://stackoverflow.com/questions/670308/alternative-to-vectorbool

                #pragma omp parallel default(none) shared(omp_results, omp_used, path_increments, inverse, reciprocals,\
                                                          output_stream_size, batch_size, output_channel_size, \
                                                          input_channel_size, depth, opts)
                {
                    int64_t start = 1 + ((output_stream_size - 1) * omp_get_thread_num()) / omp_get_num_threads();
                    int64_t end = 1 + ((output_stream_size - 1) * (1 + omp_get_thread_num())) / omp_get_num_threads();
                    if (start < end) {
                        std::vector<torch::Tensor> omp_signature_by_term_at_stream;
                        torch::Tensor omp_signature = torch::empty({batch_size, output_channel_size}, opts);
                        misc::slice_by_term(omp_signature, omp_signature_by_term_at_stream, input_channel_size, depth);
                        ta_ops::restricted_exp(path_increments[start], omp_signature_by_term_at_stream, reciprocals);
                        for (int64_t stream_index = start + 1; stream_index < end; ++stream_index) {
                            ta_ops::mult_fused_restricted_exp(path_increments[stream_index],
                                                              omp_signature_by_term_at_stream,
                                                              inverse,
                                                              reciprocals);
                        }
                        omp_results[omp_get_thread_num()] = std::move(omp_signature_by_term_at_stream);
                        omp_used[omp_get_thread_num()] = {true};
                    }
                }
                for (int64_t thread_index = 0; thread_index < nthreads; ++thread_index) {
                    if (omp_used[thread_index].value) {
                        ta_ops::mult(signature_by_term_at_stream, omp_results[thread_index], inverse);
                    }
                    // there is no else{break;} block because it need not be true that the used threads are
                    // contiguously indexed, because of the start < end condition above.
                }
            }
            else {
                for (int64_t stream_index = 1; stream_index < output_stream_size; ++stream_index) {
                    ta_ops::mult_fused_restricted_exp(path_increments[stream_index],
                                                      signature_by_term_at_stream,
                                                      inverse,
                                                      reciprocals);
                }
            }
        }

        return std::tuple<torch::Tensor, torch::Tensor> {signature, path_increments};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_signature, torch::Tensor signature, torch::Tensor path_increments,
                       s_size_type depth, bool stream, bool basepoint, bool inverse, bool initial) {
        grad_signature = grad_signature.detach();
        signature = signature.detach();
        path_increments = path_increments.detach();

        torch::TensorOptions opts = misc::make_opts(signature);
        torch::Tensor reciprocals = misc::make_reciprocals(depth, opts);
        int64_t output_stream_size = path_increments.size(stream_dim);
        int64_t input_channel_size = path_increments.size(channel_dim);

        std::vector<torch::Tensor> signature_by_term;
        misc::slice_by_term(signature, signature_by_term, input_channel_size, depth);

        // When computing the signature we essentially did a lot of computations of the form
        // A \otimes exp(b),
        // where A is a generic member of the tensor algebra, and b is a member of the lowest nonscalar part of the
        // tensor algebra.
        // Then signature_by_term_at_stream represents A.
        // grad_signature_by_term_at_stream represents the gradient on A \otimes exp(b).
        // Note the asymmetry.
        std::vector<torch::Tensor> grad_signature_by_term_at_stream;
        std::vector<torch::Tensor> signature_by_term_at_stream;

        // There's some differences between the stream==true and stream==false cases.
        // The essential difference is that in the stream==true case, we have recorded a lot more information, which we
        // can just use. In the stream==false case this information must be recomputed.

        torch::Tensor grad_signature_at_stream;
        if (stream) {
            grad_signature_at_stream = grad_signature[-1];
        }
        else {
            grad_signature_at_stream = grad_signature;
        }
        // make sure not to leak changes
        grad_signature_at_stream = grad_signature_at_stream.clone();

        misc::slice_by_term(grad_signature_at_stream, grad_signature_by_term_at_stream, input_channel_size, depth);

        if (stream) {
            // if stream then we already know the signature of x_1, ... x_k because we saved it as our result,
            // and we don't need to worry about recomputing it (c.f. the else branch below).
            if (output_stream_size < 2) {
                // However if output_stream_size is so small that we never even enter the for loop below. In
                // which case signature_by_term_at_stream isn't set. We fix that here for the sake of
                // restricted_exp_backward after the for loop, which requires it to be set.
                misc::slice_at_stream(signature_by_term, signature_by_term_at_stream, 0);
            }
        }
        else {
            // We're going to recompute the signature, as we need it to perform the gradient computations.
            // In particular we compute it backwards (which is possible via a particular reversibility property of the
            // signature), in the sense that given some input path x_1, ... x_n we compute the signature of
            // x_1, ... x_k for all k: during the forward pass we did this for k going from 2 to n. During this backward
            // pass we do it for k going from n to 2.
            // In particular we clone the signature here as we're going to modify it in-place during these computations
            // and we don't want to leak changes to the original output.
            misc::slice_by_term(signature.clone(), signature_by_term_at_stream, input_channel_size, depth);
        }


        torch::Tensor grad_path_increments = torch::empty_like(path_increments);

        for (int64_t stream_index = output_stream_size - 1; stream_index >= 1; --stream_index) {
            torch::Tensor grad_next = grad_path_increments[stream_index];
            torch::Tensor next = path_increments[stream_index];

            if (stream) {
                // Just look up signature_by_term_at_stream because we saved it for output
                misc::slice_at_stream(signature_by_term, signature_by_term_at_stream, stream_index - 1);
            }
            else {
                // Recompute signature_by_term_at_stream
                ta_ops::mult_fused_restricted_exp(-next, signature_by_term_at_stream, inverse, reciprocals);
            }

            ta_ops::mult_fused_restricted_exp_backward(grad_next, grad_signature_by_term_at_stream, next,
                                                       signature_by_term_at_stream, inverse, reciprocals);

            if (stream) {
                // If stream then gradients may well have accumulated on the signatures of the partial paths, so
                // add those on here.
                grad_signature_at_stream += grad_signature[stream_index - 1];
            }
        }

        torch::Tensor grad_next = grad_path_increments[0];
        torch::Tensor next = path_increments[0];
        if (initial) {
            if (stream) {
                // We're using memory we own if stream==false, but we're using memory we don't own if stream==true. So
                // we have to clone here before we modify it.
                for (auto& elem : signature_by_term_at_stream) {
                    elem = elem.clone();
                }
            }
            // Recover initial_value in signature_by_term_at_stream
            ta_ops::mult_fused_restricted_exp(-next, signature_by_term_at_stream, inverse, reciprocals);
            // grad_signature_by_term_at_stream is using the same memory as grad_signature_at_stream, which represents
            // the gradient through initial_value.
            ta_ops::mult_fused_restricted_exp_backward(grad_next, grad_signature_by_term_at_stream, next,
                                                       signature_by_term_at_stream, inverse, reciprocals);
        }
        else {
            ta_ops::restricted_exp_backward(grad_next, grad_signature_by_term_at_stream, next,
                                            signature_by_term_at_stream, reciprocals);
        }

        // Find the gradient on the path from the gradient on the path increments.
        torch::Tensor grad_path;
        torch::Tensor grad_basepoint_value;
        std::tie(grad_path, grad_basepoint_value) = detail::compute_path_increments_backward(grad_path_increments,
                                                                                             basepoint,
                                                                                             inverse,
                                                                                             opts);

        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
               {grad_path, grad_basepoint_value, grad_signature_at_stream};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    signature_backward_custom(torch::Tensor grad_signature, torch::Tensor signature, torch::Tensor path,
                              s_size_type depth, bool stream, bool basepoint, torch::Tensor basepoint_value,
                              bool inverse, bool initial) {
        path = path.detach();
        basepoint_value = basepoint_value.detach();
        torch::Tensor path_increments = detail::compute_path_increments(path, basepoint, basepoint_value, inverse);
        return signature_backward(grad_signature, signature, path_increments, depth, stream, basepoint, inverse,
                initial);
    }
}  // namespace signatory
