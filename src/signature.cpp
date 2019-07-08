#include <torch/extension.h>
#include <Python.h>   // PyCapsule
#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::tie, std::tuple
#include <vector>     // std::vector

#include "signature.hpp"


// TODO: rename arguments to sensible things
// TODO: Finish this
// TODO: write extracting terms in the signature
// TODO: more tests: forward correctness + custom basepoints
// TODO: check on streams of length 1
// TODO: numpy, tensorflow
// TODO: CUDA
// TODO: support torchscript? https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
// TODO: concatenating onto an already existing signature. A class that takes data and spits out signatures?
// TODO: check that the right things are being put in the sdist/bdist
// TODO: support other dimensions just 'going along for the ride' (c.f. iisignature)
// TODO: profile for memory leaks, just in case! (Being able to call .backward() twice would indicate a leak, I think)

#define BACKWARDS_INFO_CAPSULE_NAME "signatory.BackwardsInfoCapsule"

namespace signatory {
    namespace detail {
        // Encapsulates all the things that aren't tensors
        struct SigSpec {
            SigSpec(torch::Tensor path, int depth, bool stream, bool basepoint) :
                    opts{torch::TensorOptions().dtype(path.dtype()).device(path.device())},
                    input_stream_size{path.size(0)},
                    input_channels{path.size(1)},
                    batch_size{path.size(2)},
                    output_stream_size{path.size(0) - (basepoint ? 0 : 1)},
                    output_channels{signature_channels(path.size(1), depth)},
                                                                              // if depth == 1 then we don't care here
                                                                              // we just want to avoid throwing an error
                    reciprocals{torch::ones({depth - 1}, opts) / torch::linspace(2, depth, depth == 1 ? 2 : depth - 1,
                                                                                 opts)},
                    n_output_dims{stream ? 3 : 2},
                    depth{depth},
                    stream{stream},
                    basepoint{basepoint}
            {};

            torch::TensorOptions opts;
            int64_t input_stream_size;
            int64_t input_channels;
            int64_t batch_size;
            int64_t output_stream_size;
            int64_t output_channels;
            torch::Tensor reciprocals;
            int n_output_dims;
            int depth;
            bool stream;
            bool basepoint;
        };

        void slice_by_term(torch::Tensor tensor, std::vector<torch::Tensor>& tensor_vector, int tensor_dim,
                           const SigSpec& sigspec) {
            int64_t current_memory_pos = 0;
            int64_t current_memory_length = sigspec.input_channels;
            for (int i = 0; i < sigspec.depth; ++i) {
                tensor_vector.push_back(tensor.narrow(/*dim=*/tensor_dim,
                                                      /*start=*/current_memory_pos,
                                                      /*len=*/current_memory_length));
                current_memory_pos += current_memory_length;
                current_memory_length *= sigspec.input_channels;
            }
        }
        
        void slice_at_stream(std::vector<torch::Tensor> arg_vector, std::vector<torch::Tensor>& ret_vector,
                             int stream_index) {
            for (auto arg_vector_elem : arg_vector) {
                ret_vector.push_back(arg_vector_elem.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0));
            }
        }
        
        void checkargs(torch::Tensor path, int depth, bool basepoint, torch::Tensor basepoint_value) {
            if (path.ndimension() != 3) {
                throw std::invalid_argument("path must be a 3-dimensional tensor, corresponding to "
                                            "(batch, stream, channel) respectively.");
            }
            if (path.size(0) == 0 || path.size(2) == 0) {
                throw std::invalid_argument("path cannot have dimensions of size zero.");
            }
            if (path.size(1) < 2) {
                throw std::invalid_argument("The stream dimension of path must be of size at least 2 to define a "
                                            "path.");
            }
            if (depth < 1) {
                throw std::invalid_argument("depth must be an integer greater than or equal to one.");
            }
            if (basepoint) {
                if (basepoint_value.ndimension() != 2) {
                    throw std::invalid_argument("basepoint must be a 2-dimensional tensor, corresponding to "
                                                "(batch, channel) respectively.");
                }
                // basepoint_value has dimensions (batch, channel)
                // path has dimensions (batch, stream, channel)
                if (basepoint_value.size(0) == path.size(0) || basepoint_value.size(1) == path.size(2)) {
                    throw std::invalid_argument("basepoint and path must have dimensions of the same size.");
                }
            }
        }

        torch::Tensor compute_path_increments(torch::Tensor path, torch::Tensor basepoint_value,
                                              const SigSpec& sigspec) {
            int64_t num_increments {sigspec.input_stream_size - 1};
            if (sigspec.basepoint) {
                torch::Tensor path_increments = path.clone();
                path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1) -= basepoint_value;
                path_increments.narrow(/*dim=*/0, /*start=*/1, /*len=*/num_increments) -=
                        path.narrow(/*dim=*/0, /*start=*/0, /*len=*/num_increments);
                return path_increments;
            }
            else {
                return path.narrow(/*dim=*/0, /*start=*/1, /*len=*/num_increments) -
                       path.narrow(/*dim=*/0, /*start=*/0, /*len=*/num_increments);
            }
        }

        // TODO: Handle exponentials in a cheaper way? It's a symmetric tensor so we can save ~n! amount of work...
        //       That's a lot of work.
        void compute_exp(torch::Tensor inp, std::vector<torch::Tensor>& out_vector, const SigSpec& sigspec) {
            out_vector[0].copy_(inp);
            for (int i = 0; i < sigspec.depth - 1; ++i) {
                torch::Tensor out_vector_view = out_vector[i + 1].view({inp.size(0),
                                                                        out_vector[i].size(0),
                                                                        sigspec.batch_size});
                torch::mul_out(out_vector_view, out_vector[i].unsqueeze(0), inp.unsqueeze(1));
                out_vector[i + 1] *= sigspec.reciprocals[i];
            }
        }

        void compute_mult(const std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2,
                          const SigSpec& sigspec) {
            for (int depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0 ; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg2[depth_to_calculate];
                for (int j = 0, k = depth_to_calculate - 1; j < depth_to_calculate; ++j, --k) {
                    // loop invariant:
                    // j + k = depth_to_calculate - 1
                    // TODO
//                    torch::Tensor out_view = tensor_at_depth_to_calculate.view({arg1[j].size(0),
//                                                                                arg2[k].size(0),
//                                                                                sigspec.batch_size});
                    torch::Tensor out_view = tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                arg2[k].size(0),
                                                                                sigspec.batch_size});
                    //~TODO
                    torch::addcmul_out(/*out=*/out_view,
                                       out_view,               // add this tensor
                                       arg2[k].unsqueeze(0),   // to (this tensor
                                       arg1[j].unsqueeze(1));  // times this tensor)
                                       // TODO: try doing the factorial part of the exponential here?
                }
                tensor_at_depth_to_calculate += arg1[depth_to_calculate];
            }
        }

        struct BackwardsInfo{
            BackwardsInfo(torch::Tensor out, std::vector<torch::Tensor> out_vector, torch::Tensor path_increments,
                          SigSpec sigspec) :
                    out{out},
                    out_vector{out_vector},
                    path_increments{path_increments},
                    sigspec{sigspec}
            {};

            torch::Tensor out;
            std::vector<torch::Tensor> out_vector;
            torch::Tensor path_increments;
            SigSpec sigspec;
        };

        void BackwardsInfoCapsuleDestructor(PyObject* capsule) {
            delete static_cast<BackwardsInfo*>(PyCapsule_GetPointer(capsule, BACKWARDS_INFO_CAPSULE_NAME));
        }

        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec) {
            if (sigspec.stream) {
                if (grad_out.ndimension() != 3) {
                    throw std::invalid_argument("grad_out must be a 3-dimensional tensor, corresponding to "
                                                "(batch, stream, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != sigspec.output_stream_size ||
                    grad_out.size(2) != sigspec.output_channels) {
                    throw std::invalid_argument("grad_out has the wrong size.");
                }
            }
            else {
                if (grad_out.ndimension() != 2) {
                    throw std::invalid_argument("grad_out must be a 2-dimensional tensor, corresponding to "
                                                "(batch, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != sigspec.output_channels) {
                    throw std::invalid_argument("grad_out has the wrong size.");
                }
            }
        }

        void compute_mult_backward(std::vector<torch::Tensor>& grad_arg1, std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                   const SigSpec& sigspec) {
            for (int depth_to_calculate = 0; depth_to_calculate < sigspec.depth; ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg2[depth_to_calculate];
                grad_arg1[depth_to_calculate] += grad_tensor_at_depth_to_calculate;
                for (int j = depth_to_calculate - 1, k = 0; j >= 0; --j, ++k) {
                    // loop invariant:
                    // j + k = depth_to_calculate - 1
                    // TODO
//                    torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({arg1[j].size(0),
//                                                                                     arg2[k].size(0),
//                                                                                     sigspec.batch_size});
                    torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                     arg2[k].size(0),
                                                                                     sigspec.batch_size});
                    //~TODO
                    grad_arg1[j] += (out_view * arg2[k].unsqueeze(0)).sum(/*dim=*/1);
                    grad_arg2[k] += (out_view * arg1[j].unsqueeze(1)).sum(/*dim=*/0);
                }
            }
        }

        void compute_exp_backward(torch::Tensor grad_inp, std::vector<torch::Tensor>& grad_out_vector,
                                  torch::Tensor inp, const std::vector<torch::Tensor>& out_vector,
                                  const SigSpec& sigspec) {
            for (int i = sigspec.depth - 2; i >= 0; --i) {
                grad_out_vector[i + 1] *= sigspec.reciprocals[i];

                torch::Tensor grad_out_vector_view = grad_out_vector[i + 1].view({inp.size(0),
                                                                                  out_vector[i].size(0),
                                                                                  sigspec.batch_size});
                grad_out_vector[i] += (grad_out_vector_view * inp.unsqueeze(1)).sum(/*dim=*/0);
                grad_inp += (grad_out_vector_view * out_vector[i].unsqueeze(0)).sum(/*dim=*/1);
            }
            grad_inp += grad_out_vector[0];
        }

        std::tuple<torch::Tensor, torch::Tensor>
        compute_path_increments_backward(torch::Tensor grad_path_increments, const SigSpec& sigspec) {
            int64_t num_increments{sigspec.input_stream_size - 1};
            if (sigspec.basepoint) {
                torch::Tensor grad_path = grad_path_increments.clone();
                grad_path.narrow(/*dim=*/0, /*start=*/0, /*len=*/num_increments)
                        -= grad_path_increments.narrow(/*dim=*/0, /*start=*/1, /*len=*/num_increments);
                return {grad_path, -grad_path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1)};
            }
            else {
                torch::Tensor grad_path = torch::empty({sigspec.input_stream_size,
                                                        sigspec.input_channels,
                                                        sigspec.batch_size},
                                                       sigspec.opts);
                grad_path.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).zero_();
                grad_path.narrow(/*dim=*/0, /*start=*/1, /*len=*/num_increments).copy_(grad_path_increments);
                grad_path.narrow(/*dim=*/0, /*start=*/0, /*len=*/num_increments) -= grad_path_increments;
                // no second return value in this case
                return {grad_path, torch::empty({0}, sigspec.opts)};
            }
        }
    }  // namespace signatory::detail

    int64_t signature_channels(int64_t input_channels, int depth) {
        if (input_channels < 1) {
            throw std::invalid_argument("input_channels must be at least 1");
        }
        if (depth < 1) {
            throw std::invalid_argument("depth must be at least 1");
        }

        if (input_channels == 1) {
            return depth;
        }
        else {
            return input_channels * ((pow(input_channels, depth) - 1) / (input_channels - 1));
        }
    }

    std::tuple<torch::Tensor, py::object>
    signature_forward(torch::Tensor path, int depth, bool stream, bool basepoint, torch::Tensor basepoint_value) {
        detail::checkargs(path, depth, basepoint, basepoint_value);

        // convert from (batch, stream, channel) to (stream, channel, batch), which is the representation we use
        // internally for speed (fewer cache misses).
        // having 'path' have non-monotonically-decreasing strides doesn't slow things down very much, as 'path' is only
        // really used to compute 'path_increments' below, and the extra speed from a more efficient internal
        // representation more than compensates
        path = path.transpose(0, 1).transpose(1, 2);
        if (!path.is_floating_point()) {
            path = path.to(torch::kFloat32);
        }

        detail::SigSpec sigspec{path, depth, stream, basepoint};

        torch::Tensor path_increments = detail::compute_path_increments(path, basepoint_value, sigspec);

        // We allocate memory for certain things upfront.
        // This is motivated by wanting to construct things in-place in 'out', to save a copy. (Which can actually take
        // quite a lot of time; signatures can get quite large!)
        // We also split up the memory based on how deep into the signature it is, and store this split in vectors.
        // This means we don't have to faff around in the hot loop figuring out what memory we're storing results in.
        torch::Tensor out;                              // Where we store the result that we return
        std::vector<torch::Tensor> out_vector;          // The overall result; sliced into terms by depth.
                                                        // Only used if stream == true.
        torch::Tensor scratch;                          // Scratch space.
                                                        // Only used if stream == false
        std::vector<torch::Tensor> prev_stream_vector;  // The signature computed so far; sliced into terms by depth
        std::vector<torch::Tensor> next_stream_vector;  // Where the signature computed at the next step is put

        prev_stream_vector.reserve(depth);
        next_stream_vector.reserve(depth);
        if (stream) {
            // if stream == true then we want to store all intermediate results
            out = torch::empty({sigspec.output_stream_size,
                                sigspec.output_channels,
                                sigspec.batch_size},
                               sigspec.opts);
            // and the first term is just the first part of that tensor.
            torch::Tensor first_term = out.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0);

            // slice up into terms by depth:
            // first_term is put into next_stream_vector, as it's what we (will) have computed so far.
            // out is put into out_vector
            detail::slice_by_term(first_term, next_stream_vector, 0, sigspec);
            detail::slice_by_term(out, out_vector, 1, sigspec);
        }
        else {
            // if stream == false then we only want the final result, so we have a smaller tensor in this case
            out = torch::empty({sigspec.output_channels,
                                sigspec.batch_size},
                               sigspec.opts);
            // and we start off by computing the first term in that bit of memory
            torch::Tensor first_term = out;

            // however we also need some scratch space to compute the exponential of a particular increment in
            scratch = torch::empty({sigspec.output_channels, sigspec.batch_size}, sigspec.opts);

            // slice up into terms by depth:
            // first_term is put into next_stream_vector, as it's what we (will) have computed so far.
            // scratch is put into prev_stream_vector
            detail::slice_by_term(first_term, next_stream_vector, 0, sigspec);
            detail::slice_by_term(scratch, prev_stream_vector, 0, sigspec);
        }

        // actually compute the first term
        detail::compute_exp(path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0), next_stream_vector,
                            sigspec);

        for (int stream_index = 1; stream_index < sigspec.output_stream_size; ++stream_index) {
            // what we have computed so far is in next_stream_vector
            // so move it into prev_stream_vector as it's now what we're basing our next calculation off.
            if (stream) {
                prev_stream_vector = next_stream_vector;
                // and now split up the memory for next_stream_vector from the memory we have stored in out_vector
                next_stream_vector.clear();
                detail::slice_at_stream(out_vector, next_stream_vector, stream_index);
            }
            else {
                // meanwhile we don't care about whatever was in prev_stream_vector (it served its purpose in the
                // previous calculation), so now we'll overwrite its data - so its memory can now be claimed by
                // next_stream_vector.
                prev_stream_vector.swap(next_stream_vector);
            }

            // actually do the calculation!
            detail::compute_exp(path_increments.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0),
                                next_stream_vector, sigspec);
            detail::compute_mult(prev_stream_vector, next_stream_vector, sigspec);
        }
        torch::Tensor out_with_transposes;
        if (stream) {
            // convert from (stream, channel, batch) to (batch, stream, channel)
            out_with_transposes = out.transpose(1, 2).transpose(0, 1);
        }
        else{
            if ((sigspec.output_stream_size % 2) == 0) {
                out = scratch;
            }
            // convert from (channel, batch) to (batch, channel)
            out_with_transposes = out.transpose(0, 1);
        }
        return {out_with_transposes,
                py::reinterpret_steal<py::object>(PyCapsule_New(new detail::BackwardsInfo{out,
                                                                                          out_vector,
                                                                                          path_increments,
                                                                                          sigspec},
                                                                BACKWARDS_INFO_CAPSULE_NAME,
                                                                detail::BackwardsInfoCapsuleDestructor))};
    }

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule) {
        // Unwrap the PyCapsule
        auto backwards_info = static_cast<detail::BackwardsInfo*>(PyCapsule_GetPointer(backwards_info_capsule.ptr(),
                                                                                       BACKWARDS_INFO_CAPSULE_NAME));
        torch::Tensor out = backwards_info->out;
        std::vector<torch::Tensor> out_vector = backwards_info->out_vector;
        torch::Tensor path_increments = backwards_info->path_increments;
        detail::SigSpec sigspec = backwards_info->sigspec;

        // Check arguments
        detail::checkargs_backward(grad_out, sigspec);

        // Transpose and clone. (Clone so we don't leak changes through grad_out.)
        if (sigspec.stream) {
            // convert from (batch, stream, channel) to (stream, channel, batch)
            grad_out = grad_out.transpose(0, 1).transpose(1, 2);
        }
        else {
            // convert from (batch, channel) to (channel, batch)
            grad_out = grad_out.transpose(0, 1);
        }
        if (!grad_out.is_floating_point()) {
            grad_out = grad_out.to(torch::kFloat32);
        }
        grad_out = grad_out.clone();

        // Faff around with memory management
        std::vector<torch::Tensor> grad_out_vector;
        std::vector<torch::Tensor> prev_stream_vector;
        std::vector<torch::Tensor> next_stream_vector;
        std::vector<torch::Tensor> grad_prev_stream_vector;
        std::vector<torch::Tensor> grad_next_stream_vector;

        prev_stream_vector.reserve(sigspec.depth);
        next_stream_vector.reserve(sigspec.depth);
        grad_prev_stream_vector.reserve(sigspec.depth);
        grad_next_stream_vector.reserve(sigspec.depth);

        // TODO: condense memory management that's both in and out of for loop?
        if (sigspec.stream) {
            grad_out_vector.reserve(sigspec.depth);

            // slice_by_term(out, out_vector, 1, sigspec)   (Don't need to call; we already know it from forwards.)
            detail::slice_by_term(grad_out, grad_out_vector, 1, sigspec);

            detail::slice_at_stream(out_vector, next_stream_vector, -1);
            detail::slice_at_stream(grad_out_vector, grad_next_stream_vector, -1);

            if (sigspec.output_stream_size != 1) {
                detail::slice_at_stream(out_vector, prev_stream_vector, -2);
                detail::slice_at_stream(grad_out_vector, grad_prev_stream_vector, -2);
            }  // if sigspec.output_stream_size == 1 then we never enter the for loop below, so not setting
               // prev_stream_vector doesn't matter. (And indeed we can't set it, because the tensors aren't long enough
               // to index at -2, because they're only length 1.
        }
        else {
            torch::Tensor scratch = torch::empty({sigspec.output_channels,
                                                  sigspec.batch_size},
                                                 sigspec.opts);
            torch::Tensor grad_scratch = torch::zeros({sigspec.output_channels,
                                                       sigspec.batch_size},
                                                      sigspec.opts);

            // clone to avoid blatting what's in out when we recompute our way along the path with exps in the for loop.
            detail::slice_by_term(out.clone(), next_stream_vector, 0, sigspec);
            detail::slice_by_term(grad_out, grad_next_stream_vector, 0, sigspec);

            detail::slice_by_term(scratch, prev_stream_vector, 0, sigspec);
            detail::slice_by_term(grad_scratch, grad_prev_stream_vector, 0, sigspec);

            detail::compute_exp(-path_increments.narrow(/*dim=*/0,
                                                        /*start=*/-1,
                                                        /*len=*/1).squeeze(0),
                                prev_stream_vector,
                                sigspec);
            detail::compute_mult(next_stream_vector, prev_stream_vector, sigspec);
        }

        // Now actually do the computations
        torch::Tensor grad_path_increments = torch::zeros({sigspec.output_stream_size,
                                                           sigspec.input_channels,
                                                           sigspec.batch_size},
                                                          sigspec.opts);
        // TODO
        detail::compute_exp(path_increments.narrow(/*dim=*/0,
                                    /*start=*/-1,
                                    /*len=*/1).squeeze(0),
                            next_stream_vector,
                            sigspec);
        //~TODO
        for (int stream_index = sigspec.output_stream_size - 1; stream_index > 0; --stream_index) {
            detail::compute_mult_backward(grad_prev_stream_vector, grad_next_stream_vector, prev_stream_vector,
                                          next_stream_vector, sigspec);
            detail::compute_exp_backward(grad_path_increments.narrow(/*dim=*/0,
                                                                     /*start=*/stream_index,
                                                                     /*len=*/1).squeeze(0),
                                         grad_next_stream_vector,
                                         path_increments.narrow(/*dim=*/0,
                                                                /*start=*/stream_index,
                                                                /*len=*/1).squeeze(0),
                                         next_stream_vector,
                                         sigspec);

            // And now back to memory management
            if (sigspec.stream) {
                next_stream_vector = prev_stream_vector;
                prev_stream_vector.clear();
                detail::slice_at_stream(out_vector, prev_stream_vector, stream_index - 2);

                grad_next_stream_vector = grad_prev_stream_vector;
                grad_prev_stream_vector.clear();
                detail::slice_at_stream(grad_out_vector, grad_prev_stream_vector, stream_index - 2);
            }
            else {
                next_stream_vector.swap(prev_stream_vector);
                // TODO: optimise away some of these computations when we get to the very end? We don't need to compute
                //       any prev stuff any more then

                // If stream=False then we haven't saved some of the intermediate results. (We could, but that would
                // use a lot of memory, and the operation is otherwise uses O(1) in stream size amount of memory, so
                // we definitely shouldn't just use O(stream size) without asking.)
                // So to go backwards we have to recompute some of the intermediate results, which is possible when
                // knowing the final result and the original path increments.
                detail::compute_exp(-path_increments.narrow(/*dim=*/0,
                                            /*start=*/stream_index - 1,
                                            /*len=*/1).squeeze(0),
                                    prev_stream_vector,
                                    sigspec);
                detail::compute_mult(next_stream_vector, prev_stream_vector, sigspec);

                grad_next_stream_vector.swap(grad_prev_stream_vector);
                for (auto elem : grad_prev_stream_vector) {
                    elem.zero_();
                }
            }
            // TODO (this one doesn't actually affect anything for the final time around)
            detail::compute_exp(path_increments.narrow(/*dim=*/0,
                                        /*start=*/stream_index - 1,
                                        /*len=*/1).squeeze(0),
                                next_stream_vector,
                                sigspec);
            //~TODO
        }
        detail::compute_exp_backward(grad_path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                     grad_next_stream_vector,
                                     path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                     next_stream_vector,
                                     sigspec);
        
        torch::Tensor grad_path;
        torch::Tensor grad_basepoint_value;
        std::tie(grad_path, grad_basepoint_value) = detail::compute_path_increments_backward(grad_path_increments,
                                                                                             sigspec);
        // convert from (stream, channel, batch) to (batch, stream, channel)
        grad_path = grad_path.transpose(1, 2).transpose(0, 1);
        return {grad_path, grad_basepoint_value};
    }
}  // namespace signatory