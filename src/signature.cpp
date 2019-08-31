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
#include <tuple>      // std::tie, std::tuple
#include <vector>     // std::vector

#include "misc.hpp"
#include "pycapsule.hpp"
#include "signature.hpp"
#include "tensor_algebra_ops.hpp"

// TODO: try using threads for the words->basis transform?
// TODO: perf test with 'words' for logsignatures - we care about the ML performance, not the mathematical one

// TODO: add the examples to the tests
// TODO: check for interrupts
// TODO: rationalise backwards_info. Can we combine out_vector and signature_vector?
// TODO: rename out_* to signature_*
// TODO: add Python-level handling of (... x stream x channel) format
// TODO: signature_jacobian, logsignature_jacobian
// TODO: tensorflow
// TODO: custom CUDA kernels for all of this would be lovely...
// TODO: support torchscript? https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
// TODO: concatenating onto an already existing signature. A class that takes data and spits out signatures?


namespace signatory {
    namespace detail {
        // Takes the path and basepoint and returns the path increments
        torch::Tensor compute_path_increments(torch::Tensor path, torch::Tensor basepoint_value,
                                              const misc::SigSpec& sigspec) {
            int64_t num_increments {sigspec.input_stream_size - 1};
            if (sigspec.basepoint) {
                torch::Tensor path_increments = path.clone();
                path_increments.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/1) -= basepoint_value;
                path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments) -=
                        path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments);
                return path_increments;
            }
            else {
                return path.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments) -
                       path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments);
            }
        }

        // Computes the backward pass through the path increments operation.
        // Returns the gradients for the original path, and for the basepoint.
        std::tuple<torch::Tensor, torch::Tensor>
        compute_path_increments_backward(torch::Tensor grad_path_increments, const misc::SigSpec& sigspec) {
            int64_t num_increments{sigspec.input_stream_size - 1};
            if (sigspec.basepoint) {
                torch::Tensor grad_path = grad_path_increments.clone();
                grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments)
                        -= grad_path_increments.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments);
                return {grad_path, -grad_path_increments.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/1).squeeze(0)};
            }
            else {
                torch::Tensor grad_path = torch::empty({sigspec.input_stream_size,
                                                        sigspec.batch_size,
                                                        sigspec.input_channels},
                                                       sigspec.opts);
                grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/1).zero_();
                grad_path.narrow(/*dim=*/stream_dim, /*start=*/1, /*len=*/num_increments).copy_(grad_path_increments);
                grad_path.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/num_increments) -= grad_path_increments;
                // no second return value in this case
                return {grad_path, torch::empty({0}, sigspec.opts)};
            }
        }
    }  // namespace signatory::detail

    std::tuple<torch::Tensor, py::object>
    signature_forward(torch::Tensor path, s_size_type depth, bool stream, bool basepoint, torch::Tensor basepoint_value)
    {
        // No sense keeping track of gradients when we have a dedicated backwards function (and in-place operations mean
        // that in any case one cannot autograd through this function)
        path = path.detach();
        basepoint_value = basepoint_value.detach();
        misc::checkargs(path, depth, basepoint, basepoint_value);

        // Convert from (batch, stream, channel) to (stream, batch, channel), which is the representation we use
        // internally.
        // having 'path' have non-monotonically-decreasing strides doesn't slow things down very much, as 'path' is only
        // really used to compute 'path_increments' below, and the extra speed from a more efficient internal
        // representation more than compensates
        path = path.transpose(0, 1);
        if (!path.is_floating_point()) {
            path = path.to(torch::kFloat32);
        }
        if (basepoint) {
            // basepoint_value has dimensions (batch, channel) so we don't need to switch anything
            basepoint_value = basepoint_value.to(path.dtype());
        }

        misc::SigSpec sigspec{path, depth, stream, basepoint};

        torch::Tensor path_increments = detail::compute_path_increments(path, basepoint_value, sigspec);

        // We allocate memory for certain things upfront.
        //
        // This is motivated by wanting to construct things in-place in 'out', to save a copy. (Which can actually
        // take quite a lot of time; signatures can get quite large!)
        //
        // There is some asymmetry between the stream==true and the stream==false cases. This is because of the
        // fundamental difference that when stream==true we want to preserve intermediate results, whilst in the
        // stream==false case we would prefer to write over them.
        //
        // This basically means that in the stream==true case we're computing 'to the right', by continuing to
        // put things in new memory that corresponds to later in the stream, where as in the stream==false case
        // we're computing 'to the left', by computing the exponential of our current position in the stream, and
        // multiplying it on to what we have so far.
        //
        // I am realising that sometimes necessary complexity is hard to explain.

        torch::Tensor out;                          // We create tensors for the memory.
        std::vector<torch::Tensor> out_vector;      // And slice up their corresponding storage into a vector of
        // tensors. This is only used if stream==true, so that we can
        // slice them up by term prior to slicing them up by stream
        // index. (In the stream==false case we don't need to slice them
        // by stream index, so we don't need the intermediate vector
        // either.

        std::vector<torch::Tensor> stream_vector;   // The signature computed so far.
        std::vector<torch::Tensor> scratch_vector;  // Extra space. If stream==true this is where the signature for
        // the next step is computed into; stream_vector is subsequently
        // multiplied onto it. If stream==false then this is where the
        // exponential for the next increment is computed into prior to
        // multiplying it onto stream_vector.
        if (stream) {
            // if stream == true then we want to store all intermediate results
            out = torch::empty({sigspec.output_stream_size,
                                sigspec.batch_size,
                                sigspec.output_channels},
                               sigspec.opts);
            // and the first term is just the first part of that tensor.
            torch::Tensor first_term = out.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/1).squeeze(0);

            // slice up into terms by depth:
            // first_term is put into scratch_vector, as it's what we (will) have computed so far.
            // out is put into out_vector
            misc::slice_by_term(first_term, scratch_vector, sigspec);
            misc::slice_by_term(out, out_vector, sigspec);
        }
        else {
            // if stream == false then we only want the final result, so we have a smaller tensor in this case
            out = torch::empty({sigspec.batch_size, sigspec.output_channels}, sigspec.opts);

            // however we still also need some scratch space to compute the exponential of a particular increment in
            torch::Tensor scratch = torch::empty({sigspec.batch_size, sigspec.output_channels}, sigspec.opts);

            // slice up into terms by depth:
            // scratch is put into scratch_vector, as it's where we'll compute exponentials
            // out is put into stream_vector, as it's where we'll compute the final result.
            misc::slice_by_term(scratch, scratch_vector, sigspec);
            misc::slice_by_term(out, stream_vector, sigspec);
        }

        // compute the first term
        ta_ops::compute_restricted_exp(path_increments.narrow(/*dim=*/stream_dim, /*start=*/0, /*len=*/1).squeeze(0),
                                       stream ? scratch_vector : stream_vector, sigspec);

        for (int64_t stream_index = 1; stream_index < sigspec.output_stream_size; ++stream_index) {
            if (stream) {
                // what we have computed so far is in scratch_vector
                // so move it into stream_vector as it's now what we're basing our next calculation off.
                stream_vector = std::move(scratch_vector);
                // and now split up the memory for the next scratch_vector from the memory we have stored in
                // out_vector
                misc::slice_at_stream(out_vector, scratch_vector, stream_index);
            }

            // first compute the exponential of the increment and put it in scratch_vector
            ta_ops::compute_restricted_exp(path_increments.narrow(/*dim=*/stream_dim,
                                                                  /*start=*/stream_index,
                                                                  /*len=*/1).squeeze(0),
                                           scratch_vector,
                                           sigspec);
            // multiply on what we have so far in stream_vector onto scratch_vector, to calculate the signature for
            // the path up to this next time step.
            // if stream==true then return this value in scratch vector, so we don't overwrite our intermediate
            // signature stored in stream_vector.
            // if stream==false then just return this value in stream_vector
            ta_ops::compute_mult(stream_vector, scratch_vector, /*rightret=*/stream, sigspec);
        }

        py::object backwards_info_capsule = misc::wrap_capsule<misc::BackwardsInfo>(std::move(sigspec),
                                                                                    std::move(out_vector),
                                                                                    out,
                                                                                    path_increments);

        // TODO: uncomment when 24413 is fixed
        // We have to do the transpose in the Python side to avoid PyTorch bug 24413.
        // https://github.com/pytorch/pytorch/issues/24413
//        torch::Tensor out = misc::transpose(out, sigspec);

        return {out, backwards_info_capsule};
    }

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule, bool clone) {
        misc::BackwardsInfo* backwards_info = misc::unwrap_capsule<misc::BackwardsInfo>(backwards_info_capsule);

        // Unpacked backwards_info
        const misc::SigSpec& sigspec = backwards_info->sigspec;
        const std::vector<torch::Tensor>& out_vector = backwards_info->out_vector;
        torch::Tensor out = backwards_info->out;
        torch::Tensor path_increments = backwards_info->path_increments;

        // TODO: remove when 24413 is fixed. Here we undo the transposing that autograd has done for us in the
        //  pulled-out transposes
        grad_out = misc::transpose_reverse(grad_out, sigspec);

        // Check arguments
        misc::checkargs_backward(grad_out, sigspec);

        // Transpose and clone. (Clone so we don't leak changes through grad_out.)
        grad_out = misc::transpose_reverse(grad_out, sigspec);
        if (!grad_out.is_floating_point()) {
            grad_out = grad_out.to(torch::kFloat32);
        }
        if (clone) {
            // This is provided as an option specifically for logsignature_backward: we control the input to
            // signature_backward in this case, so we know that we don't need to clone.
            grad_out = grad_out.clone();
        }

        // Here we spend a lot of time faffing around with memory management. There are surprisingly many
        // differences between the stream==true and stream==false cases; they handle their memory quite differently.
        //
        // You might notice how a lot of this doesn't quite resemble a straightforward backwards implementation of
        // the forward pass. That's because we can use particular reversibility properties of signatures to save a
        // lot of memory (O(1) rather than O(length of stream) in the stream==false case) in the backwards
        // computation.
        // Consequently the code here is a little involved. (Makes sense really, we'd just delegate to autograd if
        // there wasn't a good reason to do it by hand!)

        std::vector<torch::Tensor> grad_out_vector;          // Only used if stream==true. Used as a halfway house
        // house to hold the sliced-by-terms pieces of
        // grad_out. Not needed if stream==false because we
        // directly slice the terms in grad_prev_stream_vector
        // and grad_next_stream_vector, as we don't have a
        // stream dimension to slice along.

        std::vector<torch::Tensor> stream_vector;            // The stream_vector from the forwards pass; as the
        // computation progresses it will work its way
        // backwards through the values it took. (If
        // stream==true it will do this by recalling the values
        // from memory, as they were the return values from the
        // forward pass. If stream==false then it will
        // recompute these in the order they are required by
        // using a particular reversibility property of
        // signatures.)

        std::vector<torch::Tensor> scratch_vector;           // Where we'll compute the exponential of a path
        // element. (Which we did compute in the forward pass
        // but chose not to save, as it's an O(stream size)
        // amount of easy-to-compute data that may never be
        // used...) This is needed for some of the backward
        // functions, and in the stresm==false case to
        // recompute stream_vector via the reversibility
        // property of signatures.

        std::vector<torch::Tensor> grad_prev_stream_vector;  // The gradient of a previous-to-particular timestep
        std::vector<torch::Tensor> grad_next_stream_vector;  // The gradient of a particular time step

        torch::Tensor scratch = torch::empty({sigspec.batch_size,
                                              sigspec.output_channels},
                                             sigspec.opts);
        misc::slice_by_term(scratch, scratch_vector, sigspec);

        // Populate our memory vectors with the scratch memory and with the gradient we've been given
        if (sigspec.stream) {
            misc::slice_by_term(grad_out, grad_out_vector, sigspec);

            misc::slice_at_stream(out_vector, stream_vector, -1);
            misc::slice_at_stream(grad_out_vector, grad_prev_stream_vector, -1);
        }
        else {
            torch::Tensor grad_scratch = torch::empty({sigspec.batch_size,
                                                       sigspec.output_channels},
                                                      sigspec.opts);

            misc::slice_by_term(grad_scratch, grad_next_stream_vector, sigspec);

            // Clone to avoid overwriting what's in out (not necessary in the stream==true case because we don't
            // overwrite the memory then; we don't need to recompute our way back along the path with compute_div.)
            misc::slice_by_term(out.clone(), stream_vector, sigspec);
            misc::slice_by_term(grad_out, grad_prev_stream_vector, sigspec);
        }

        // grad_path_increments is what we want to compute throughout the for loop.
        torch::Tensor grad_path_increments = torch::zeros({sigspec.output_stream_size,
                                                           sigspec.batch_size,
                                                           sigspec.input_channels},
                                                          sigspec.opts);
        for (int64_t stream_index = sigspec.output_stream_size - 1; stream_index > 0; --stream_index) {
            // Recompute the exponential of a path increment and put it in scratch_vector
            ta_ops::compute_restricted_exp(path_increments.narrow(/*dim=*/stream_dim,
                                                                  /*start=*/stream_index,
                                                                  /*len=*/1).squeeze(0),
                                           scratch_vector,
                                           sigspec);
            if (sigspec.stream) {
                // Get the value of stream_vector from memory
                misc::slice_at_stream(out_vector, stream_vector, stream_index - 1);

                // Set grad_next_stream_vector to grad_prev_stream_vector, and set grad_prev_stream_vector to the
                // gradient that was inputted to the backward pass for this stream index.
                grad_next_stream_vector = std::move(grad_prev_stream_vector);
                misc::slice_at_stream(grad_out_vector, grad_prev_stream_vector, stream_index - 1);
            }
            else {
                // Recompute the value of stream_vector by dividing by the exponential of the path increment, which
                // conveniently we already know
                ta_ops::compute_div(stream_vector, scratch_vector, sigspec);

                // Set grad_next_stream_vector to grad_prev_stream_vector, and then we'll overwrite the contents of
                // grad_prev_stream_vector in a moment.
                grad_prev_stream_vector.swap(grad_next_stream_vector);
            }

            // Now actually do the computations
            ta_ops::compute_mult_backward(grad_prev_stream_vector, grad_next_stream_vector, stream_vector,
                                          scratch_vector, /*add_not_copy=*/sigspec.stream, sigspec);
            ta_ops::compute_restricted_exp_backward(grad_path_increments.narrow(/*dim=*/stream_dim,
                                                                                /*start=*/stream_index,
                                                                                /*len=*/1).squeeze(0),
                                                    grad_next_stream_vector,
                                                    path_increments.narrow(/*dim=*/stream_dim,
                                                                           /*start=*/stream_index,
                                                                           /*len=*/1).squeeze(0),
                                                    scratch_vector,
                                                    sigspec);
        }

        // Another minor implementation detail that differs from a naive backwards implementation: we can use
        // grad_prev_stream_vector and stream_vector here, rather than doing one more iteration of the first part of
        // the above for loop, just to get the same values in grad_next_stream_vector and scratch_vector.
        // (And we don't want to do another compute_mult_backward either.)
        ta_ops::compute_restricted_exp_backward(grad_path_increments.narrow(/*dim=*/stream_dim,
                                                                            /*start=*/0,
                                                                            /*len=*/1).squeeze(0),
                                                grad_prev_stream_vector,
                                                path_increments.narrow(/*dim=*/stream_dim,
                                                                       /*start=*/0,
                                                                       /*len=*/1).squeeze(0),
                                                stream_vector,
                                                sigspec);

        // Find the gradient on the path from the gradient on the path increments.
        torch::Tensor grad_path;
        torch::Tensor grad_basepoint_value;
        std::tie(grad_path, grad_basepoint_value) = detail::compute_path_increments_backward(grad_path_increments,
                                                                                             sigspec);
        // convert from (stream, batch, channel) to (batch, stream, channel)
        grad_path = grad_path.transpose(0, 1);
        // Note that we don't need to transpose grad_basepoint_value as it's already correct as (batch, channel)
        return {grad_path, grad_basepoint_value};
    }
}  // namespace signatory
