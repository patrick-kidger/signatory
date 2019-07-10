#include <torch/extension.h>
#include <Python.h>   // PyCapsule
#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::tie, std::tuple
#include <vector>     // std::vector

#include "signature.hpp"


// TODO: numpy, tensorflow
// TODO: CUDA?
// TODO: support torchscript? https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
// TODO: concatenating onto an already existing signature. A class that takes data and spits out signatures?
// TODO: check that the right things are being put in the sdist/bdist
// TODO: profile for memory leaks, just in case!

namespace signatory {
    namespace detail {
        /*****************************************************/
        /* Stuff that's used for both forwards and backwards */
        /*****************************************************/

        constexpr auto backwards_info_capsule_name = "signatory.BackwardsInfoCapsule";

        // Encapsulates all the things that aren't tensors
        struct SigSpec {
            SigSpec(torch::Tensor path, int depth, bool stream, bool basepoint) :
                    opts{torch::TensorOptions().dtype(path.dtype()).device(path.device())},
                    input_stream_size{path.size(0)},
                    input_channels{path.size(1)},
                    batch_size{path.size(2)},
                    output_stream_size{path.size(0) - (basepoint ? 0 : 1)},
                    output_channels{signature_channels(path.size(1), depth)},
                    reciprocals{torch::ones({depth - 1}, opts)},
                    n_output_dims{stream ? 3 : 2},
                    depth{depth},
                    basepoint{basepoint}
            {
                if (depth > 1) {
                    reciprocals /= torch::linspace(2, depth, depth - 1, opts);
                }  // and reciprocals will be empty - of size 0 - if depth == 1.
            };

            torch::TensorOptions opts;
            int64_t input_stream_size;
            int64_t input_channels;
            int64_t batch_size;
            int64_t output_stream_size;
            int64_t output_channels;
            torch::Tensor reciprocals;
            int n_output_dims;
            int depth;
            bool basepoint;
        };

        // Argument 'in' is assumed to be a tensor for which one dimension has size equal to sigspec.output_channels
        // It is sliced up along that dimension, specified by 'dim', and the resulting tensors placed into 'out'.
        // Each resulting tensor corresponds to one of the (tensor, not scalar) terms in the signature.
        void slice_by_term(torch::Tensor in, std::vector<torch::Tensor>& out, int dim, const SigSpec& sigspec) {
            int64_t current_memory_pos = 0;
            int64_t current_memory_length = sigspec.input_channels;
            for (int i = 0; i < sigspec.depth; ++i) {
                out.push_back(in.narrow(/*dim=*/dim,
                                        /*start=*/current_memory_pos,
                                        /*len=*/current_memory_length));
                current_memory_pos += current_memory_length;
                current_memory_length *= sigspec.input_channels;
            }
        }

        // Argument 'in' is assumed to be a tensor for which its first dimension corresponds to the stream dimension.
        // Its slices along a particular index of that dimension are put in 'out'.
        void slice_at_stream(std::vector<torch::Tensor> in, std::vector<torch::Tensor>& out, int stream_index) {
            for (auto elem : in) {
                out.push_back(elem.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0));
            }
        }

        // TODO: Handle exponentials in a cheaper way? It's a symmetric tensor so we can save ~n! amount of work...
        //       That's a lot of work.
        // Computes the exponential of the 'in' tensor. Each higher-order tensor is placed in 'out'.
        void compute_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const SigSpec& sigspec) {
            out[0].copy_(in);
            for (int i = 0; i < sigspec.depth - 1; ++i) {
                torch::Tensor view_out = out[i + 1].view({in.size(0), out[i].size(0), sigspec.batch_size});
                torch::mul_out(view_out, out[i].unsqueeze(0), in.unsqueeze(1));
                out[i + 1] *= sigspec.reciprocals[i];
            }
        }

        template<bool div>
        int should_invert(int index) {  // small helper function for division
            if (div) {
                return (index % 2) == 0;
            }
            else {
                return false;
            }
        }

        // Computes the tensor product of two members of the tensor algebra.
        // if div==false then it computes arg1 \otimes arg2
        // if div==true then it computes arg1 \otimes -arg2
        // if rightret==false then it returns the result in arg1
        // if rightret==true then it returns the result in arg2
        template<bool rightret, bool div=false>
        void compute_mult(std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2,
                          const SigSpec& sigspec) {
            for (int depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0 ; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = (rightret ? arg2 : arg1)[depth_to_calculate];
                for (int j = 0, k = depth_to_calculate - 1; j < depth_to_calculate; ++j, --k) {
                    // loop invariant:
                    // j + k = depth_to_calculate - 1
                    torch::Tensor view_out = tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                arg2[k].size(0),
                                                                                sigspec.batch_size});

                    torch::addcmul_out(view_out,                                        // Output.
                                       view_out,                                        // Add this tensor
                                       arg2[k].unsqueeze(0),                            // to (this tensor
                                       arg1[j].unsqueeze(1),                            // times this tensor
                                       should_invert<div>(rightret ? j : k) ? -1 : 1);  // times this scalar).
                    // Could also just do
                    // view_out += arg2[k].unsqueeze(0) * arg1[j].unsqueeze(1)
                    // but that definitely creates a large intermediate tensor for the product. In principle addcmul_out
                    // could do something cleverer (possibly unlikely), so we use it anyway.
                }

                if (should_invert<div>(depth_to_calculate)) {
                    tensor_at_depth_to_calculate -= (rightret ? arg1 : arg2)[depth_to_calculate];
                }
                else {
                    tensor_at_depth_to_calculate += (rightret ? arg1 : arg2)[depth_to_calculate];
                }
            }
        }

        /***************************************/
        /* Stuff that's only used for forwards */
        /***************************************/

        // Checks the arguments for the forwards pass
        void checkargs(torch::Tensor path, int depth, bool basepoint, torch::Tensor basepoint_value) {
            if (path.ndimension() != 3) {
                throw std::invalid_argument("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                                            "corresponding to (batch, stream, channel) respectively.");
            }
            if (path.size(0) == 0 || path.size(2) == 0) {
                throw std::invalid_argument("Argument 'path' cannot have dimensions of size zero.");
            }
            if (path.size(1) < 2) {
                throw std::invalid_argument("Argument 'path' must have stream dimension of size at least 2. (Need at "
                                            "least this many points to define a path.)");
            }
            if (depth < 1) {
                throw std::invalid_argument("Argument 'depth' must be an integer greater than or equal to one.");
            }
            if (basepoint) {
                if (basepoint_value.ndimension() != 2) {
                    throw std::invalid_argument("Argument 'basepoint' must be a 2-dimensional tensor, corresponding to "
                                                "(batch, channel) respectively.");
                }
                // basepoint_value has dimensions (batch, channel)
                // path has dimensions (batch, stream, channel)
                if (basepoint_value.size(0) != path.size(0) || basepoint_value.size(1) != path.size(2)) {
                    throw std::invalid_argument("Arguments 'basepoint' and 'path' must have dimensions of the same "
                                                "size.");
                }
            }
        }

        // Takes the path and basepoint and returns the path increments
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

        // Retains information needed for the backwards pass.
        struct BackwardsInfo{
            BackwardsInfo(torch::Tensor out, std::vector<torch::Tensor> out_vector, torch::Tensor path_increments,
                          SigSpec sigspec, bool stream) :
                out{out},
                out_vector{out_vector},
                path_increments{path_increments},
                sigspec{sigspec},
                stream{stream}
            {};

            torch::Tensor out;
            std::vector<torch::Tensor> out_vector;
            torch::Tensor path_increments;
            SigSpec sigspec;
            bool stream;
        };

        // Frees the memory consumed retaining information for the backwards pass. The BackwardsInfo object is wrapped
        // into a PyCapsule.
        void BackwardsInfoCapsuleDestructor(PyObject* capsule) {
            delete static_cast<BackwardsInfo*>(PyCapsule_GetPointer(capsule, backwards_info_capsule_name));
        }

        /****************************************/
        /* Stuff that's only used for backwards */
        /****************************************/

        // Checks the arguments for the backwards pass. Only grad_out is really checked to make sure it is as expected
        // The objects we get from the PyCapsule-wrapped BackwardsInfo object are just assumed to be correct.
        template<bool stream>
        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec) {
            if (stream) {
                if (grad_out.ndimension() != 3) {
                    throw std::invalid_argument("Argument 'grad_out' must be a 3-dimensional tensor, with dimensions "
                                                "corresponding to (batch, stream, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != sigspec.output_stream_size ||
                    grad_out.size(2) != sigspec.output_channels) {
                    throw std::invalid_argument("Argument 'grad_out' has the wrong size.");
                }
            }
            else {
                if (grad_out.ndimension() != 2) {
                    throw std::invalid_argument("Argument 'grad_out' must be a 2-dimensional tensor, with dimensions"
                                                "corresponding to (batch, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != sigspec.output_channels) {
                    throw std::invalid_argument("Argument 'grad_out' has the wrong size.");
                }
            }
        }

        // Computes the backward pass for the tensor product operation of two members of the tensor algebra.
        // Note that both 'arg1' and 'arg2' should be the inputs that were used in the forward pass of the
        // multiplication. In particular neither of them should be the result of the forward pass of the multiplication
        // (As compute_mult returns its result via one of its input arguments.)
        // Argument 'grad_arg2' is the input gradient, and will be modified in-place according to the multiplication.
        // Argument 'grad_arg1' is the output gradient. If add_not_copy==true then the result of this operation is
        // added on to it. If add_not_copy==false then the result of this operation is placed into it directly,
        // overwriting whatever is already present.
        template<bool add_not_copy>
        void compute_mult_backward(std::vector<torch::Tensor>& grad_arg1, std::vector<torch::Tensor>& grad_arg2,
                                   const std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                   const SigSpec& sigspec) {
            for (int depth_to_calculate = 0; depth_to_calculate < sigspec.depth; ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg2[depth_to_calculate];
                if (add_not_copy) {
                    grad_arg1[depth_to_calculate] += grad_tensor_at_depth_to_calculate;
                }
                else {
                    grad_arg1[depth_to_calculate].copy_(grad_tensor_at_depth_to_calculate);
                }
                for (int j = depth_to_calculate - 1, k = 0; j >= 0; --j, ++k) {
                    // loop invariant:
                    // j + k = depth_to_calculate - 1
                    torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({arg1[j].size(0),
                                                                                     arg2[k].size(0),
                                                                                     sigspec.batch_size});
                    // TODO: This is just a batch matrix-multiply where the batch dimension is last instead of first,
                    //       so profile this against transposing and using that.
                    grad_arg1[j] += (out_view * arg2[k].unsqueeze(0)).sum(/*dim=*/1);
                    grad_arg2[k] += (out_view * arg1[j].unsqueeze(1)).sum(/*dim=*/0);
                }
            }
        }

        // Computes the backwards pass through the exponential. 'in' should be the input to the forward pass of the
        // exponential, but 'out' should be the result of the forward pass of the exponential. (I.e. what it holds
        // after the function has been called - recall that the function operates with 'out' as an out-argument.)
        // Argument 'grad_out' should have the gradient on the output of the forward pass, and has in-place changes
        // occuring to it.
        // Argument 'grad_in' will have the gradients resulting from this operation placed into it, overwriting whatever
        // is current present.
        void compute_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
                                  torch::Tensor in, const std::vector<torch::Tensor>& out,
                                  const SigSpec& sigspec) {
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

                for (int i = sigspec.depth - 3; i >= 0; --i) {
                    grad_out[i + 1] *= sigspec.reciprocals[i];
                    view_grad_out = grad_out[i + 1].view({in.size(0), out[i].size(0), sigspec.batch_size});
                    // TODO: This is just a batch matrix-multiply where the batch dimension is last instead of first,
                    //       so profile this against transposing and using that.
                    grad_out[i] += (view_grad_out * in.unsqueeze(1)).sum(/*dim=*/0);
                    grad_in += (view_grad_out * out[i].unsqueeze(0)).sum(/*dim=*/1);
                }
                grad_in += grad_out[0];
            }
            else {  // implies depth == 1
                grad_in.copy_(grad_out[0]);
            }
        }

        // Computes the backward pass through the path increments operation.
        // Returns the gradients for the original path, and for the basepoint.
        std::tuple<torch::Tensor, torch::Tensor>
        compute_path_increments_backward(torch::Tensor grad_path_increments, const SigSpec& sigspec) {
            int64_t num_increments{sigspec.input_stream_size - 1};
            if (sigspec.basepoint) {
                torch::Tensor grad_path = grad_path_increments.clone();
                grad_path.narrow(/*dim=*/0, /*start=*/0, /*len=*/num_increments)
                        -= grad_path_increments.narrow(/*dim=*/0, /*start=*/1, /*len=*/num_increments);
                return {grad_path, -grad_path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0)};
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

        /***********************************************************/
        /* The templated implementations of forwards and backwards */
        /***********************************************************/

        template<bool stream> // Honestly moving certain boolean arguments to templates might have gotten a little out
                              // of hand. It's only a very minor performance improvement.
                              // It started with wanting to make the 'div' argument of compute_mult a template argument
                              // (because it's in a hot loop, not that it's still going to affect much), and, well...
        std::tuple<torch::Tensor, py::object>
        signature_forward_impl(torch::Tensor path, int depth, bool basepoint, torch::Tensor basepoint_value) {
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
            if (basepoint) {
                // (batch, channel) to (channel, batch)
                basepoint_value = basepoint_value.transpose(0, 1);
                basepoint_value = basepoint_value.to(path.dtype());
            }

            detail::SigSpec sigspec{path, depth, stream, basepoint};

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

            stream_vector.reserve(depth);
            scratch_vector.reserve(depth);
            if (stream) {
                out_vector.reserve(depth);

                // if stream == true then we want to store all intermediate results
                out = torch::empty({sigspec.output_stream_size,
                                    sigspec.output_channels,
                                    sigspec.batch_size},
                                   sigspec.opts);
                // and the first term is just the first part of that tensor.
                torch::Tensor first_term = out.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0);

                // slice up into terms by depth:
                // first_term is put into scratch_vector, as it's what we (will) have computed so far.
                // out is put into out_vector
                detail::slice_by_term(first_term, scratch_vector, 0, sigspec);
                detail::slice_by_term(out, out_vector, 1, sigspec);
            }
            else {
                // if stream == false then we only want the final result, so we have a smaller tensor in this case
                out = torch::empty({sigspec.output_channels, sigspec.batch_size}, sigspec.opts);

                // however we still also need some scratch space to compute the exponential of a particular increment in
                torch::Tensor scratch = torch::empty({sigspec.output_channels, sigspec.batch_size}, sigspec.opts);

                // slice up into terms by depth:
                // scratch is put into scratch_vector, as it's where we'll compute exponentials
                // out is put into stream_vector, as it's where we'll compute the final result.
                detail::slice_by_term(scratch, scratch_vector, 0, sigspec);
                detail::slice_by_term(out, stream_vector, 0, sigspec);
            }

            // compute the first term
            detail::compute_exp(path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                stream ? scratch_vector : stream_vector, sigspec);

            for (int stream_index = 1; stream_index < sigspec.output_stream_size; ++stream_index) {
                if (stream) {
                    // what we have computed so far is in scratch_vector
                    // so move it into stream_vector as it's now what we're basing our next calculation off.
                    stream_vector = std::move(scratch_vector);
                    // and now split up the memory for the next scratch_vector from the memory we have stored in
                    // out_vector
                    scratch_vector.clear();
                    detail::slice_at_stream(out_vector, scratch_vector, stream_index);
                }

                // first compute the exponential of the increment and put it in scratch_vector
                detail::compute_exp(path_increments.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0),
                                    scratch_vector, sigspec);
                // multiply on what we have so far in stream_vector onto scratch_vector, to calculate the signature for
                // the path up to this next time step.
                // if stream==true then return this value in scratch vector, so we don't overwrite our intermediate
                // signature stored in stream_vector.
                // if stream==false then just return this value in stream_vector
                detail::compute_mult</*rightret=*/stream>(stream_vector, scratch_vector, sigspec);
            }

            torch::Tensor out_with_transposes;
            if (stream) {
                // convert from (stream, channel, batch) to (batch, stream, channel)
                out_with_transposes = out.transpose(1, 2).transpose(0, 1);
            }
            else{
                // convert from (channel, batch) to (batch, channel)
                out_with_transposes = out.transpose(0, 1);
            }
            return {out_with_transposes,
                    py::reinterpret_steal<py::object>(PyCapsule_New(new detail::BackwardsInfo{out,
                                                                                              out_vector,
                                                                                              path_increments,
                                                                                              sigspec,
                                                                                              stream},
                                                                    detail::backwards_info_capsule_name,
                                                                    detail::BackwardsInfoCapsuleDestructor))};
        }

        template<bool stream>
        std::tuple<torch::Tensor, torch::Tensor>
        signature_backward_impl(torch::Tensor grad_out, BackwardsInfo* backwards_info) {
            // Unpacked backwards_info
            torch::Tensor out = backwards_info->out;
            std::vector<torch::Tensor> out_vector = backwards_info->out_vector;
            torch::Tensor path_increments = backwards_info->path_increments;
            detail::SigSpec sigspec = backwards_info->sigspec;

            // Check arguments
            detail::checkargs_backward<stream>(grad_out, sigspec);

            // Transpose and clone. (Clone so we don't leak changes through grad_out.)
            if (stream) {
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

            stream_vector.reserve(sigspec.depth);
            scratch_vector.reserve(sigspec.depth);
            grad_prev_stream_vector.reserve(sigspec.depth);
            grad_next_stream_vector.reserve(sigspec.depth);

            torch::Tensor scratch = torch::empty({sigspec.output_channels,
                                                  sigspec.batch_size},
                                                 sigspec.opts);
            detail::slice_by_term(scratch, scratch_vector, 0, sigspec);

            // Populate our memory vectors with the scratch memory and with the gradient we've been given
            if (stream) {
                grad_out_vector.reserve(sigspec.depth);
                detail::slice_by_term(grad_out, grad_out_vector, 1, sigspec);

                detail::slice_at_stream(out_vector, stream_vector, -1);
                detail::slice_at_stream(grad_out_vector, grad_prev_stream_vector, -1);
            }
            else {
                torch::Tensor grad_scratch = torch::empty({sigspec.output_channels,
                                                           sigspec.batch_size},
                                                          sigspec.opts);

                detail::slice_by_term(grad_scratch, grad_next_stream_vector, 0, sigspec);

                // Clone to avoid overwriting what's in out (not necessary in the stream==true case because we don't
                // overwrite the memory then.)
                detail::slice_by_term(out.clone(), stream_vector, 0, sigspec);
                detail::slice_by_term(grad_out, grad_prev_stream_vector, 0, sigspec);
            }

            // grad_path_increments is what we want to compute throughout the for loop.
            torch::Tensor grad_path_increments = torch::empty({sigspec.output_stream_size,
                                                               sigspec.input_channels,
                                                               sigspec.batch_size},
                                                              sigspec.opts);
            for (int stream_index = sigspec.output_stream_size - 1; stream_index > 0; --stream_index) {
                // Recompute the exponential of a path increment and put it in scratch_vector
                detail::compute_exp(path_increments.narrow(/*dim=*/0,
                                                           /*start=*/stream_index,
                                                           /*len=*/1).squeeze(0),
                                    scratch_vector,
                                    sigspec);
                if (stream) {
                    // Get the value of stream_vector from memory
                    stream_vector.clear();
                    detail::slice_at_stream(out_vector, stream_vector, stream_index - 1);

                    // Set grad_next_stream_vector to grad_prev_stream_vector, and set grad_prev_stream_vector to the
                    // gradient that was inputted to the backward pass for this stream index.
                    grad_next_stream_vector = std::move(grad_prev_stream_vector);
                    grad_prev_stream_vector.clear();
                    detail::slice_at_stream(grad_out_vector, grad_prev_stream_vector, stream_index - 1);
                }
                else {
                    // Recompute the value of stream_vector by dividing by the exponential of the path increment, which
                    // conveniently we already know
                    detail::compute_mult</*rightret=*/false, /*div=*/true>(stream_vector, scratch_vector, sigspec);

                    // Set grad_next_stream_vector to grad_prev_stream_vector, and then we'll overwrite the contents of
                    // grad_prev_stream_vector in a moment.
                    grad_prev_stream_vector.swap(grad_next_stream_vector);
                }

                // Now actually do the computations
                detail::compute_mult_backward</*add_not_copy=*/stream>(grad_prev_stream_vector, grad_next_stream_vector,
                                                                       stream_vector, scratch_vector, sigspec);
                detail::compute_exp_backward(grad_path_increments.narrow(/*dim=*/0,
                                                                         /*start=*/stream_index,
                                                                         /*len=*/1).squeeze(0),
                                             grad_next_stream_vector,
                                             path_increments.narrow(/*dim=*/0,
                                                                    /*start=*/stream_index,
                                                                    /*len=*/1).squeeze(0),
                                             scratch_vector,
                                             sigspec);
            }

            // Another minor implementation detail that differs from a naive backwards implementation: we can use
            // grad_prev_stream_vector and stream_vector here, rather than doing one more iteration of the first part of
            // the above for loop, just to get the same values in grad_next_stream_vector and scratch_vector.
            // (And we don't want to do another compute_mult_backward either.)
            detail::compute_exp_backward(grad_path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                         grad_prev_stream_vector,
                                         path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                         stream_vector,
                                         sigspec);

            // Find the gradient on the path from the gradient on the path increments.
            torch::Tensor grad_path;
            torch::Tensor grad_basepoint_value;
            std::tie(grad_path, grad_basepoint_value) = detail::compute_path_increments_backward(grad_path_increments,
                                                                                                 sigspec);
            // convert from (stream, channel, batch) to (batch, stream, channel)
            grad_path = grad_path.transpose(1, 2).transpose(0, 1);
            if (sigspec.basepoint) {
                grad_basepoint_value = grad_basepoint_value.transpose(0, 1);
            }
            return {grad_path, grad_basepoint_value};
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
        if (stream) {
            return detail::signature_forward_impl</*stream=*/true>(path, depth, basepoint, basepoint_value);
        }
        else {
            return detail::signature_forward_impl</*stream=*/false>(path, depth, basepoint, basepoint_value);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule) {
        // Unwrap the PyCapsule
        auto backwards_info = static_cast<detail::BackwardsInfo*>(
                PyCapsule_GetPointer(backwards_info_capsule.ptr(), detail::backwards_info_capsule_name));
        if (backwards_info->stream) {
            return detail::signature_backward_impl</*stream=*/true>(grad_out, backwards_info);
        }
        else {
            return detail::signature_backward_impl</*stream=*/false>(grad_out, backwards_info);
        }
    }
}  // namespace signatory