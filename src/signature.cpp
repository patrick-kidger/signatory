#include <torch/extension.h>
#include <Python.h>   // PyCapsule
#include <algorithm>  // std::stable_sort
#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::tie, std::tuple
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "signature.hpp"  // signatory::depth_type, signatory::Mode

// TODO: documentation: when to use signature / logsignature, time augmentation vs stream
// TODO: check logsignature_backwards; sort out projection
// TODO: signature_jacobian, logsignature_jacobian
// TODO: logsignature tests. Make sure memory doesn't get blatted tests.
// TODO: figure out how to mark in-place operations on the output as preventing backwards

// TODO: numpy, tensorflow
// TODO: CUDA?
// TODO: support torchscript? https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
// TODO: concatenating onto an already existing signature. A class that takes data and spits out signatures?
// TODO: check that the right things are being put in the sdist/bdist
// TODO: profile for memory leaks, just in case!

namespace signatory {
    namespace detail {
        /****************************************/
        /* Stuff that's used in multiple places */
        /****************************************/

        // Encapsulates all the things that aren't tensors
        struct SigSpec {
            SigSpec(torch::Tensor path, depth_type depth, bool stream, bool basepoint) :
                    opts{torch::TensorOptions().dtype(path.dtype()).device(path.device())},
                    input_stream_size{path.size(0)},
                    input_channels{path.size(1)},
                    batch_size{path.size(2)},
                    output_stream_size{path.size(0) - (basepoint ? 0 : 1)},
                    output_channels{signature_channels(path.size(1), depth)},
                    n_output_dims{stream ? 3 : 2},
                    depth{depth},
                    reciprocals{torch::ones({depth - 1}, opts)},
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
            int64_t n_output_dims;
            depth_type depth;
            torch::Tensor reciprocals;
            bool basepoint;
        };

        // Argument 'in' is assumed to be a tensor for which one dimension has size equal to sigspec.output_channels
        // It is sliced up along that dimension, specified by 'dim', and the resulting tensors placed into 'out'.
        // Each resulting tensor corresponds to one of the (tensor, not scalar) terms in the signature.
        void slice_by_term(torch::Tensor in, std::vector<torch::Tensor>& out, int64_t dim, const SigSpec& sigspec) {
            int64_t current_memory_pos = 0;
            int64_t current_memory_length = sigspec.input_channels;
            for (int64_t i = 0; i < sigspec.depth; ++i) {
                out.push_back(in.narrow(/*dim=*/dim,
                                        /*start=*/current_memory_pos,
                                        /*len=*/current_memory_length));
                current_memory_pos += current_memory_length;
                current_memory_length *= sigspec.input_channels;
            }
        }

        // Argument 'in' is assumed to be a tensor for which its first dimension corresponds to the stream dimension.
        // Its slices along a particular index of that dimension are put in 'out'.
        void slice_at_stream(std::vector<torch::Tensor> in, std::vector<torch::Tensor>& out, int64_t stream_index) {
            for (auto elem : in) {
                out.push_back(elem.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0));
            }
        }

        torch::Tensor stream_transpose(torch::Tensor tensor, bool stream) {
            if (stream) {
                // convert from (stream, channel, batch) to (batch, stream, channel)
                return tensor.transpose(1, 2).transpose(0, 1);
            }
            else{
                // convert from (channel, batch) to (batch, channel)
                return tensor.transpose(0, 1);
            }
        }

        torch::Tensor stream_transpose_reverse(torch::Tensor tensor, bool stream) {
            if (stream) {
                // convert from (batch, stream, channel) to (stream, channel, batch)
                return tensor.transpose(0, 1).transpose(1, 2);
            }
            else {
                // convert from (batch, channel) to (channel, batch)
                return tensor.transpose(0, 1);
            }
        }

        // TODO: Handle exponentials in a cheaper way? It's a symmetric tensor so we can save a lot of work
        // Computes the exponential of the 'in' tensor. Each higher-order tensor is placed in 'out'.
        // It is 'restricted' in the sense that it only computes the exponential of a tensor belonging to the lowest
        // level of the tensor algebra, not the exponential of an arbitrary element of the tensor algebra.
        void compute_restricted_exp(torch::Tensor in, std::vector<torch::Tensor>& out, const SigSpec& sigspec) {
            out[0].copy_(in);
            for (depth_type i = 0; i < sigspec.depth - 1; ++i) {
                torch::Tensor view_out = out[i + 1].view({in.size(0), out[i].size(0), sigspec.batch_size});
                torch::mul_out(view_out, out[i].unsqueeze(0), in.unsqueeze(1));
                out[i + 1] *= sigspec.reciprocals[i];
            }
        }

        // forced inline
        #define COMPUTE_MULTDIV_INNER(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, multiplier,  \
                                      sigspec) {                                                                 \
            for (depth_type j = 0, k = depth_to_calculate - 1; j < depth_to_calculate; ++j, --k) {               \
                /* loop invariant:                                                                               \
                   j + k = depth_to_calculate - 1 */                                                             \
                torch::Tensor view_out = tensor_at_depth_to_calculate.view({arg1[j].size(0),                     \
                                                                            arg2[k].size(0),                     \
                                                                            sigspec.batch_size});                \
                view_out.addcmul_(arg2[k].unsqueeze(0),  /* += (this tensor times */                             \
                                  arg1[j].unsqueeze(1),  /* this tensor times     */                             \
                                  multiplier);           /* this scalar)          */                             \
            }                                                                                                    \
        }

        // Computes the tensor product of two members of the tensor algebra. It's not completely generic, as it imposes
        // that the scalar value of both members of the tensor algebra must be 1. (As we don't store the scalar value,
        // as for signatures - the most common use of this function - it is always 1.)
        //
        // that is, it computes arg1 \otimes arg2
        //
        // if rightret==false then it returns the result in arg1, and arg2 is left unchanged
        // if rightret==true then it returns the result in arg2, and arg1 is left unchanged
        void compute_mult(std::vector<torch::Tensor>& arg1, std::vector<torch::Tensor>& arg2, const SigSpec& sigspec,
                          bool rightret) {
            for (depth_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = (rightret ? arg2 : arg1)[depth_to_calculate];

                COMPUTE_MULTDIV_INNER(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, 1, sigspec)

                tensor_at_depth_to_calculate += (rightret ? arg1 : arg2)[depth_to_calculate];
            }
        }

        // forced inline
        #define COMPUTE_MULTDIV_INNER_BACKWARD(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1, arg2, \
                                               depth_to_calculate, sigspec) {                                       \
            for (depth_type j = depth_to_calculate - 1, k = 0; j >= 0; --j, ++k) {                                  \
                /* loop invariant:                                                                                  \
                   j + k = depth_to_calculate - 1 */                                                                \
                torch::Tensor out_view = grad_tensor_at_depth_to_calculate.view({arg1[j].size(0),                   \
                                                                                 arg2[k].size(0),                   \
                                                                                 sigspec.batch_size});              \
                /* This is just a batch matrix-multiply with the batch dimension in last place                      \
                   It's not totally clear what the optimal way of writing this operation is, but this is at least   \
                   faster than transposing and using .baddbmm_ (not surprising, given the transposing) */           \
                grad_arg1[j] += (out_view * arg2[k].unsqueeze(0)).sum(/*dim=*/1);                                   \
                grad_arg2[k] += (out_view * arg1[j].unsqueeze(1)).sum(/*dim=*/0);                                   \
            }                                                                                                       \
        }

        // Retains information needed for the backwards pass.
        struct BackwardsInfo{
            BackwardsInfo(SigSpec&& sigspec, std::vector<torch::Tensor>&& out_vector, torch::Tensor out,
                          torch::Tensor path_increments, bool stream) :
                sigspec{sigspec},
                out_vector{out_vector},
                out{out},
                path_increments{path_increments},
                stream{stream}
            {};

            void set_logsignature_data(std::vector<torch::Tensor>&& signature_vector_,
                                       std::vector<torch::Tensor>&& logsignature_vector_,
                                       Mode mode_,
                                       int64_t logsig_channels_) {
                signature_vector = signature_vector_;
                logsignature_vector = logsignature_vector_;
                mode = mode_;
                logsig_channels = logsig_channels_;
            }

            SigSpec sigspec;
            std::vector<torch::Tensor> out_vector;
            std::vector<torch::Tensor> signature_vector;  // will be the same as out_vector when computing logsignatures
                                                          // with stream==true. But we provide a separate vector here
                                                          // for a consistent interface with the stream==false case as
                                                          // well.
            std::vector<torch::Tensor> logsignature_vector;
            torch::Tensor out;
            torch::Tensor path_increments;
            int64_t logsig_channels;
            Mode mode;
            bool stream;
        };

        constexpr auto backwards_info_capsule_name = "signatory.BackwardsInfoCapsule";

        // Frees the memory consumed retaining information for the backwards pass. The BackwardsInfo object is wrapped
        // into a PyCapsule.
        void BackwardsInfoCapsuleDestructor(PyObject* capsule) {
            delete static_cast<BackwardsInfo*>(PyCapsule_GetPointer(capsule, backwards_info_capsule_name));
        }

        // Unwraps a capsule to get at the BackwardsInfo object
        BackwardsInfo* get_backwards_info(py::object backwards_info_capsule) {
            return static_cast<detail::BackwardsInfo*>(
                    PyCapsule_GetPointer(backwards_info_capsule.ptr(), backwards_info_capsule_name));
        }

        // Checks the arguments for the backwards pass. Only grad_out is really checked to make sure it is as expected
        // The objects we get from the PyCapsule-wrapped BackwardsInfo object are just assumed to be correct.
        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec, bool stream,
                                int64_t num_channels = -1) {
            if (num_channels == -1) {
                num_channels = sigspec.output_channels;
            }

            if (stream) {
                if (grad_out.ndimension() != 3) {
                    throw std::invalid_argument("Gradient must be a 3-dimensional tensor, with dimensions "
                                                "corresponding to (batch, stream, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != sigspec.output_stream_size ||
                    grad_out.size(2) != num_channels) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }
            else {
                if (grad_out.ndimension() != 2) {
                    throw std::invalid_argument("Gradient must be a 2-dimensional tensor, with dimensions"
                                                "corresponding to (batch, channel) respectively.");
                }
                if (grad_out.size(0) != sigspec.batch_size ||
                    grad_out.size(1) != num_channels) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }
        }

        // forced inline
        #define SHOULD_INVERT(index) (((index) % 2) == 0)

        /************************************************/
        /* Stuff that's only used for signature_forward */
        /************************************************/

        // Checks the arguments for the forwards pass
        void checkargs(torch::Tensor path, depth_type depth, bool basepoint, torch::Tensor basepoint_value) {
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

        /*************************************************/
        /* Stuff that's only used for signature_backward */
        /*************************************************/

        // Computes the tensor 'divison' of two members of the tensor algebra. It's not completely generic, as it
        // imposes that the scalar value of both members of the tensor algebra must be 1. (As we don't store the scalar
        // value, as for signatures - the most common use of this function - it is always 1.)
        //
        // that is, it computes arg1 \otimes -arg2
        //
        // it returns the result in arg1 and leaves arg2 unchanged.
        void compute_div(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                         const SigSpec& sigspec) {
            for (depth_type depth_to_calculate = sigspec.depth - 1; depth_to_calculate >= 0; --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg1[depth_to_calculate];

                COMPUTE_MULTDIV_INNER(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate,
                                      SHOULD_INVERT(k) ? -1 : 1, sigspec)

                if (SHOULD_INVERT(depth_to_calculate)) {
                    tensor_at_depth_to_calculate -= arg2[depth_to_calculate];
                }
                else {
                    tensor_at_depth_to_calculate += arg2[depth_to_calculate];
                }
            }
        }

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
                                   const SigSpec& sigspec, bool add_not_copy) {
            for (depth_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth; ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg2[depth_to_calculate];

                if (add_not_copy) {
                    grad_arg1[depth_to_calculate] += grad_tensor_at_depth_to_calculate;
                }
                else {
                    grad_arg1[depth_to_calculate].copy_(grad_tensor_at_depth_to_calculate);
                }

                COMPUTE_MULTDIV_INNER_BACKWARD(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1, arg2,
                                               depth_to_calculate, sigspec)
            }
        }

        // Computes the backwards pass through the restricted exponential. 'in' should be the input to the forward pass
        // of the exponential, but 'out' should be the result of the forward pass of the exponential. (I.e. what it
        // holds after the function has been called - recall that the function operates with 'out' as an out-argument.)
        // Argument 'grad_out' should have the gradient on the output of the forward pass, and has in-place changes
        // occurring to it.
        // Argument 'grad_in' will have the gradients resulting from this operation placed into it, overwriting whatever
        // is current present.
        void compute_restricted_exp_backward(torch::Tensor grad_in, std::vector<torch::Tensor>& grad_out,
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

                for (depth_type i = sigspec.depth - 3; i >= 0; --i) {
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

        /********************************************************************/
        /* Stuff that's only used for log signatures (forward and backward) */
        /********************************************************************/

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
                                  const SigSpec& sigspec, torch::Scalar scalar_term_value,
                                  depth_type top_terms_to_skip) {
            for (depth_type depth_to_calculate = sigspec.depth - top_terms_to_skip - 1; depth_to_calculate >= 0;
                 --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg1[depth_to_calculate];

                // corresponding to the zero scalar assumed to be associated with arg2
                tensor_at_depth_to_calculate.zero_();

                COMPUTE_MULTDIV_INNER(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate, 1, sigspec)

                tensor_at_depth_to_calculate.add_(arg2[depth_to_calculate], scalar_term_value);
            }
        }

        // computes arg1 \otimes -arg2, but only partially, in the same manner as compute_mult_partial. The scalar term
        // of arg1 is taken to be scalar_term_value; the scalar term of arg2 is taken to be zero.
        void compute_div_partial(std::vector<torch::Tensor>& arg1, const std::vector<torch::Tensor>& arg2,
                                 const SigSpec& sigspec, torch::Scalar scalar_term_value,
                                 depth_type top_terms_to_skip) {
            for (depth_type depth_to_calculate = sigspec.depth - top_terms_to_skip - 1; depth_to_calculate >= 0;
                 --depth_to_calculate) {
                torch::Tensor tensor_at_depth_to_calculate = arg1[depth_to_calculate];

                // corresponding to the zero scalar assumed to be associated with arg2
                tensor_at_depth_to_calculate.zero_();

                COMPUTE_MULTDIV_INNER(tensor_at_depth_to_calculate, arg1, arg2, depth_to_calculate,
                                      SHOULD_INVERT(k) ? -1 : 1, sigspec)  // TODO: SHOULD_INVERT needs to used in a higher-order macro

                if (SHOULD_INVERT(depth_to_calculate)) {
                    tensor_at_depth_to_calculate.sub_(arg2[depth_to_calculate], scalar_term_value);
                }
                else {
                    tensor_at_depth_to_calculate.add_(arg2[depth_to_calculate], scalar_term_value);
                }
            }
        }

        // Backward pass through compute_mult_partial. Is somewhat simplified compared to the naive implementation.
        // grad_arg1 is the input gradient, and will be modified in-place.
        // grad_arg2 is the output gradient. (Note that this is the other way around to compute_mult_backward...)
        // arg1, arg2, sigspec, top_terms_to_skip should be as in the forward call to compute_mult_partial.
        void compute_mult_partial_backward(std::vector<torch::Tensor>& grad_arg1,
                                           std::vector<torch::Tensor>& grad_arg2,
                                           const std::vector<torch::Tensor>& arg1,
                                           const std::vector<torch::Tensor>& arg2,
                                           const SigSpec& sigspec,
                                           torch::Scalar scalar_value_term,
                                           depth_type top_terms_to_skip) {
            for (depth_type depth_to_calculate = 0; depth_to_calculate < sigspec.depth - top_terms_to_skip;
                 ++depth_to_calculate) {
                torch::Tensor grad_tensor_at_depth_to_calculate = grad_arg1[depth_to_calculate];

                grad_arg2[depth_to_calculate].add_(grad_tensor_at_depth_to_calculate, scalar_value_term);

                COMPUTE_MULTDIV_INNER_BACKWARD(grad_tensor_at_depth_to_calculate, grad_arg1, grad_arg2, arg1, arg2,
                                               depth_to_calculate, sigspec)

                grad_tensor_at_depth_to_calculate.zero_();
            }
        }

        // forced inline
        #define LOGSIG_INVERT(sigspec, depth_index) \
            ((SHOULD_INVERT(depth_index) ? -1 : 1) * sigspec.reciprocals[depth_index]).item()

        // forced inline
        #define LOGSIGNATURE_COMPUTATION(logsignature_vector, signature_vector, sigspec) {       \
            for (depth_type depth_index = sigspec.depth - 3; depth_index >= 0; --depth_index) {  \
                compute_mult_partial(logsignature_vector, signature_vector, sigspec,             \
                                     LOGSIG_INVERT(sigspec, depth_index), depth_index + 1);      \
            }                                                                                    \
            compute_mult_partial(logsignature_vector, signature_vector, sigspec, 1, 0);          \
        }

        // forced inline
        #define LOGSIGNATURE_COMPUTATION_BACKWARD(grad_logsig_vector, grad_sig_vector, logsignature_vector,    \
                                                  signature_vector, sigspec) {                                 \
            compute_div_partial(logsignature_vector, signature_vector, sigspec, 1, 0);                         \
            compute_mult_partial_backward(grad_logsig_vector, grad_sig_vector, logsignature_vector,            \
                                                  signature_vector, sigspec, 1, 0);                            \
            for (depth_type depth_index = 0; depth_index < sigspec.depth - 2; ++depth_index) {                 \
                torch::Scalar invert = LOGSIG_INVERT(sigspec, depth_index);                                    \
                compute_div_partial(logsignature_vector, signature_vector, sigspec, invert, depth_index + 1);  \
                compute_mult_partial_backward(grad_logsig_vector, grad_sig_vector, logsignature_vector,        \
                                              signature_vector, sigspec, invert, depth_index + 1);             \
            }                                                                                                  \
        }

        void emplace_word_to_index(const std::vector<int64_t>& word,
                                   std::vector<std::pair<depth_type, int64_t>>& out_indices,
                                   int64_t num_channels) {
            depth_type depth = word.size();
            int64_t index = 0;
            int64_t current_stride = 1;
            for (auto word_index = word.rbegin(); word_index != word.rend(); ++word_index) {
                index += *word_index * current_stride;
                current_stride *= num_channels;
            }
            out_indices.emplace_back(depth - 1, index);  // -1 because the signature-depth-one elements are at index
                                                         // 0 in the vector of tensors.
        }

        // Implements Duval's algorithm for generating Lyndon words
        // J.-P. Duval, Theor. Comput. Sci. 1988, doi:10.1016/0304-3975(88)90113-2.
        // credit to
        // https://gist.github.com/dvberkel/1950267
        // for inspiration
        void lyndon_word_generator(int64_t num_channels,
                                   depth_type max_depth,
                                   std::vector<std::pair<depth_type, int64_t>>& out_indices,
                                   Mode mode) {
            int64_t num_channels_minus_one = num_channels - 1;
            std::vector<int64_t> word;
            word.reserve(max_depth);
            word.push_back(-1);

            while (word.size() != 0) {
                word.back() += 1;
                emplace_word_to_index(word, out_indices, num_channels);
                int64_t pos = 0;
                while (static_cast<int64_t>(word.size()) < max_depth) {
                    word.push_back(word[pos]);
                    ++pos;
                }
                while (word.size() && word.back() == num_channels_minus_one) {
                    word.pop_back();
                }
            }

            if (mode == Mode::Lex) {
                std::stable_sort(out_indices.begin(),
                                 out_indices.end(),
                                 [] (std::pair<depth_type, int64_t> depth_index_pair1,
                                     std::pair<depth_type, int64_t> depth_index_pair2)
                                    {return depth_index_pair1.first < depth_index_pair2.first;});
            }
        }
    }  // namespace signatory::detail


    int64_t signature_channels(int64_t input_channels, int64_t depth) {
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
    signature_forward(torch::Tensor path, depth_type depth, bool stream, bool basepoint,
                      torch::Tensor basepoint_value) {
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
        detail::compute_restricted_exp(path_increments.narrow(/*dim=*/0, /*start=*/0, /*len=*/1).squeeze(0),
                                       stream ? scratch_vector : stream_vector, sigspec);

        for (int64_t stream_index = 1; stream_index < sigspec.output_stream_size; ++stream_index) {
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
            detail::compute_restricted_exp(path_increments.narrow(/*dim=*/0,
                                                                  /*start=*/stream_index,
                                                                  /*len=*/1).squeeze(0),
                                           scratch_vector,
                                           sigspec);
            // multiply on what we have so far in stream_vector onto scratch_vector, to calculate the signature for
            // the path up to this next time step.
            // if stream==true then return this value in scratch vector, so we don't overwrite our intermediate
            // signature stored in stream_vector.
            // if stream==false then just return this value in stream_vector
            detail::compute_mult(stream_vector, scratch_vector, sigspec, stream);
        }

        torch::Tensor out_with_transposes = detail::stream_transpose(out, stream);
        return {out_with_transposes,
                py::reinterpret_steal<py::object>(PyCapsule_New(new detail::BackwardsInfo{std::move(sigspec),
                                                                                          std::move(out_vector),
                                                                                          out,
                                                                                          path_increments,
                                                                                          stream},
                                                                detail::backwards_info_capsule_name,
                                                                detail::BackwardsInfoCapsuleDestructor))};
        }

    std::tuple<torch::Tensor, torch::Tensor>
    signature_backward(torch::Tensor grad_out, py::object backwards_info_capsule) {
        detail::BackwardsInfo* backwards_info = detail::get_backwards_info(backwards_info_capsule);

        // Unpacked backwards_info
        const detail::SigSpec& sigspec = backwards_info->sigspec;
        const std::vector<torch::Tensor>& out_vector = backwards_info->out_vector;
        torch::Tensor out = backwards_info->out;
        torch::Tensor path_increments = backwards_info->path_increments;
        bool stream = backwards_info->stream;

        // Check arguments
        detail::checkargs_backward(grad_out, sigspec, stream);

        // Transpose and clone. (Clone so we don't leak changes through grad_out.)
        grad_out = detail::stream_transpose_reverse(grad_out, stream);
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
            // overwrite the memory then, when recomputing our way back along the path with compute_div.)
            detail::slice_by_term(out.clone(), stream_vector, 0, sigspec);
            detail::slice_by_term(grad_out, grad_prev_stream_vector, 0, sigspec);
        }

        // grad_path_increments is what we want to compute throughout the for loop.
        torch::Tensor grad_path_increments = torch::empty({sigspec.output_stream_size,
                                                           sigspec.input_channels,
                                                           sigspec.batch_size},
                                                          sigspec.opts);
        for (int64_t stream_index = sigspec.output_stream_size - 1; stream_index > 0; --stream_index) {
            // Recompute the exponential of a path increment and put it in scratch_vector
            detail::compute_restricted_exp(path_increments.narrow(/*dim=*/0,
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
                detail::compute_div(stream_vector, scratch_vector, sigspec);

                // Set grad_next_stream_vector to grad_prev_stream_vector, and then we'll overwrite the contents of
                // grad_prev_stream_vector in a moment.
                grad_prev_stream_vector.swap(grad_next_stream_vector);
            }

            // Now actually do the computations
            detail::compute_mult_backward(grad_prev_stream_vector, grad_next_stream_vector, stream_vector,
                                          scratch_vector, sigspec, stream);
            detail::compute_restricted_exp_backward(grad_path_increments.narrow(/*dim=*/0,
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
        detail::compute_restricted_exp_backward(grad_path_increments.narrow(/*dim=*/0,
                                                                            /*start=*/0,
                                                                            /*len=*/1).squeeze(0),
                                                grad_prev_stream_vector,
                                                path_increments.narrow(/*dim=*/0,
                                                                       /*start=*/0,
                                                                       /*len=*/1).squeeze(0),
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

    std::tuple<torch::Tensor, py::object>
    logsignature_forward(torch::Tensor path, depth_type depth, bool stream, bool basepoint,
                         torch::Tensor basepoint_value, Mode mode) {
        if (depth == 1) {
            return signature_forward(path, depth, stream, basepoint, basepoint_value);
        }  // this isn't just a fast return path: we also can't index the reciprocals tensor if depth == 1, so we'd need
           // faffier code below - and it's already quite faffy enough

        // first call the regular signature
        torch::Tensor signature;
        py::object backwards_info_capsule;
        std::tie(signature, backwards_info_capsule) = signature_forward(path, depth, stream, basepoint,
                                                                        basepoint_value);

        // unpack sigspec
        detail::BackwardsInfo* backwards_info = detail::get_backwards_info(backwards_info_capsule);
        const detail::SigSpec& sigspec = backwards_info->sigspec;

        // undo the transposing we just did in signature_forward...
        signature = detail::stream_transpose_reverse(signature, stream);

        int64_t channel_dim = stream ? 1 : 0;

        // put the signature in memory
        std::vector<torch::Tensor> signature_vector;
        signature_vector.reserve(depth);
        detail::slice_by_term(signature, signature_vector, channel_dim, sigspec);

        // and allocate memory for the logsignature
        // TODO: only invert the lowest terms? The higher terms aren't used?
        torch::Tensor logsignature = signature * LOGSIG_INVERT(sigspec, depth - 2);
        std::vector<torch::Tensor> logsignature_vector;
        logsignature_vector.reserve(depth);
        detail::slice_by_term(logsignature, logsignature_vector, channel_dim, sigspec);

        if (stream) {
            // allocate vectors for the signature and logsignature by stream index
            std::vector<torch::Tensor> signature_stream_vector;
            std::vector<torch::Tensor> logsignature_stream_vector;
            signature_stream_vector.reserve(depth);
            logsignature_stream_vector.reserve(depth);
            for (int64_t stream_index = 0; stream_index < sigspec.output_stream_size; ++stream_index) {
                detail::slice_at_stream(signature_vector, signature_stream_vector, stream_index);
                detail::slice_at_stream(logsignature_vector, logsignature_stream_vector, stream_index);
                LOGSIGNATURE_COMPUTATION(logsignature_stream_vector, signature_stream_vector, sigspec)
            }
        }
        else {
            LOGSIGNATURE_COMPUTATION(logsignature_vector, signature_vector, sigspec)
        }

        if (mode != Mode::Expand) {
            // Extract terms corresponding to Lyndon words
            // This does mean that we just did a whole bunch of computation that isn't actually used in the output - we
            // don't really have good ways to compute logsignatures. Even the Baker-Campbell-Hausdoff formula is
            // expensive, and not obviously better than what we do.
            // It also means that we're holding on to a lot of memory until the backward pass.

            std::vector<std::pair<depth_type, int64_t>> out_indices;
            detail::lyndon_word_generator(sigspec.input_channels, sigspec.depth, out_indices, mode);

            torch::Tensor compressed_logsignature;
            if (stream) {
                compressed_logsignature = torch::empty({sigspec.output_stream_size,
                                                        static_cast<int64_t>(out_indices.size()),
                                                        sigspec.batch_size},
                                                       sigspec.opts);
            }
            else {
                compressed_logsignature = torch::empty({static_cast<int64_t>(out_indices.size()),
                                                        sigspec.batch_size},
                                                       sigspec.opts);
            }
            for (int64_t compressed_channel_index = 0;
                 compressed_channel_index < static_cast<int64_t>(out_indices.size());
                 ++compressed_channel_index) {
                std::pair<depth_type, int64_t> depth_index_pair = out_indices[compressed_channel_index];
                depth_type depth = depth_index_pair.first;
                int64_t index = depth_index_pair.second;
                compressed_logsignature.narrow(/*dim=*/channel_dim,
                                               /*start=*/compressed_channel_index,
                                               /*length=*/1).copy_(
                                                       logsignature_vector[depth].narrow(/*dim=*/channel_dim,
                                                                                         /*start=*/index,
                                                                                         /*length=*/1)
                                                                   );
            }

            logsignature = compressed_logsignature;
        }

        backwards_info->set_logsignature_data(std::move(signature_vector), std::move(logsignature_vector), mode,
                                              logsignature.size(channel_dim));

        logsignature = detail::stream_transpose(logsignature, stream);
        return {logsignature, backwards_info_capsule};
    }

    std::tuple<torch::Tensor, torch::Tensor>
    logsignature_backward(torch::Tensor grad_logsig, py::object backwards_info_capsule) {
        detail::BackwardsInfo* backwards_info = detail::get_backwards_info(backwards_info_capsule);
        const detail::SigSpec& sigspec = backwards_info->sigspec;
        if (sigspec.depth == 1) {
            return signature_backward(grad_logsig, backwards_info_capsule);
        }
        std::vector<torch::Tensor>& logsignature_vector = backwards_info->logsignature_vector;  // Not const so we can
                                                                                                // clone its contents
                                                                                                // and assign them back.
                                                                                                // Ironically this is
                                                                                                // so that we leave its
                                                                                                // current contents
                                                                                                // unchanged.
        const std::vector<torch::Tensor>& signature_vector = backwards_info->signature_vector;
        bool stream = backwards_info->stream;
        Mode mode = backwards_info->mode;
        int64_t logsig_channels = backwards_info->logsig_channels;

        detail::checkargs_backward(grad_logsig, sigspec, stream, logsig_channels);

        grad_logsig = detail::stream_transpose_reverse(grad_logsig, stream);
        if (!grad_logsig.is_floating_point()) {
            grad_logsig = grad_logsig.to(torch::kFloat32);
        }

        int64_t channel_dim = 0;

        if (mode == Mode::Expand) {
            // Clone so we don't leak changes through grad_logsig.
            grad_logsig = grad_logsig.clone();
        }
        else {
            torch::Tensor grad_logsig_expanded;
            std::vector<torch::Tensor> grad_logsig_expanded_vector;
            if (stream) {
                channel_dim = 1;
                grad_logsig_expanded = torch::zeros({sigspec.output_stream_size,
                                                     sigspec.output_channels,
                                                     sigspec.batch_size},
                                                    sigspec.opts);
            }
            else {
                grad_logsig_expanded = torch::zeros({sigspec.output_channels,
                                                     sigspec.batch_size},
                                                    sigspec.opts);
            }
            detail::slice_by_term(grad_logsig_expanded, grad_logsig_expanded_vector, channel_dim, sigspec);

            std::vector<std::pair<depth_type, int64_t>> out_indices;
            detail::lyndon_word_generator(sigspec.input_channels, sigspec.depth, out_indices, mode);

            for (int64_t compressed_channel_index = 0;
                 compressed_channel_index < static_cast<int64_t>(out_indices.size());
                 ++compressed_channel_index) {
                std::pair<depth_type, int64_t> depth_index_pair = out_indices[compressed_channel_index];
                depth_type depth = depth_index_pair.first;
                int64_t index = depth_index_pair.second;
                grad_logsig_expanded_vector[depth].narrow(/*dim=*/channel_dim,
                                                          /*start=*/index,
                                                          /*length=*/1)
                                                          .copy_(grad_logsig.narrow(/*dim=*/channel_dim,
                                                                                    /*start=*/compressed_channel_index,
                                                                                    /*length=*/1)
                                                                 );
            }

            grad_logsig = grad_logsig_expanded;
        }

        // Our old friend.
        // Memory management.
        torch::Tensor grad_sig = torch::zeros_like(grad_logsig);
        std::vector<torch::Tensor> grad_logsig_vector;
        std::vector<torch::Tensor> grad_sig_vector;
        grad_logsig_vector.reserve(sigspec.depth);
        grad_sig_vector.reserve(sigspec.depth);
        detail::slice_by_term(grad_logsig, grad_logsig_vector, channel_dim, sigspec);
        detail::slice_by_term(grad_sig, grad_sig_vector, channel_dim, sigspec);

        if (stream) {
            // allocate vectors for the signature and logsignature by stream index
            std::vector<torch::Tensor> grad_logsig_stream_vector;
            std::vector<torch::Tensor> grad_sig_stream_vector;
            std::vector<torch::Tensor> logsignature_stream_vector;
            std::vector<torch::Tensor> signature_stream_vector;
            grad_logsig_stream_vector.reserve(sigspec.depth);
            grad_sig_stream_vector.reserve(sigspec.depth);
            logsignature_stream_vector.reserve(sigspec.depth);
            signature_stream_vector.reserve(sigspec.depth);
            for (int64_t stream_index = 0; stream_index < sigspec.output_stream_size; ++stream_index) {
                detail::slice_at_stream(grad_logsig_vector, grad_logsig_stream_vector, stream_index);
                detail::slice_at_stream(grad_sig_vector, grad_sig_stream_vector, stream_index);
                detail::slice_at_stream(logsignature_vector, logsignature_stream_vector, stream_index);
                detail::slice_at_stream(signature_vector, signature_stream_vector, stream_index);
                if (mode == Mode::Expand) {
                    // if mode == Mode::Expand then logsignature_vector (and thus logsignature_stream_vector) is holding
                    // memory that is available externally, as the result of the forward pass. Since we're going to
                    // modify it in-place we need to clone it here.
                    for (auto &elem : logsignature_stream_vector) {
                        elem = elem.clone();
                    }
                }  // if mode != Mode::Expand then we are free to overwrite the memory used from the forward pass
                   // because it's not externally visible: different memory is used for the compressed logsignature.
                   // This is just one of those many details about memory efficiency that have made implementing this
                   // whole project... interesting.

                LOGSIGNATURE_COMPUTATION_BACKWARD(grad_logsig_stream_vector, grad_sig_stream_vector,
                                                  logsignature_stream_vector, signature_stream_vector, sigspec)
            }
        }
        else {
            if (mode == Mode::Expand) {
                for (auto& elem : logsignature_vector) {
                    elem = elem.clone();
                }
            }
            LOGSIGNATURE_COMPUTATION_BACKWARD(grad_logsig_vector, grad_sig_vector, logsignature_vector,
                                              signature_vector, sigspec)
        }
        grad_sig = detail::stream_transpose(grad_sig, stream);
        return signature_backward(grad_sig, backwards_info_capsule);
    }
}  // namespace signatory