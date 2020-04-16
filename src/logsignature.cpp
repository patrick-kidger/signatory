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
#include <memory>     // std::unique_ptr
#include <omp.h>
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::tie, std::tuple
#include <utility>     // std::pair
#include <vector>     // std::vector

#include "logsignature.hpp"
#include "lyndon.hpp"
#include "misc.hpp"
#include "pycapsule.hpp"
#include "signature.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    namespace logsignature {
        namespace detail {
            // This struct will be wrapped into a PyCapsule. Using it allows for computing certain aspects of the
            // logsignature transformation just once, so that repeated use of the logsignature transformation is more
            // efficient.
            struct LyndonInfo {
                LyndonInfo(std::unique_ptr<lyndon::LyndonWords> lyndon_words,
                           std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>&& transforms,
                std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>&& transforms_backward) :
                lyndon_words{std::move(lyndon_words)},
                transforms{transforms},
                transforms_backward{transforms_backward}
                {};

                // A list of Lyndon words
                std::unique_ptr<lyndon::LyndonWords> lyndon_words;

                // The transforms for going from Lyndon words to Lyndon basis
                // This is in terms of the 'compressed' index, i.e. in the free Lie algebra
                // They are grouped (the outermost vector) by anagram class
                std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms;

                // The transforms for going from Lyndon basis to Lyndon words
                // This is in terms of the tensor algebra index
                // They are grouped (the outermost vector) by anagram class
                std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms_backward;

                constexpr static auto capsule_name = "signatory.LyndonInfoCapsule";
            };

            // Compresses a representation of a member of the free Lie algebra.
            // In the tensor algebra it is represented by coefficients of all words. This just extracts the coefficients
            // of all the Lyndon words.
            // The list of all Lyndon words must have already been computed, and passed in as an argument.
            torch::Tensor compress(const lyndon::LyndonWords& lyndon_words, torch::Tensor input)
            {
                // TODO: avoid the need for this copy operation entirely by having all of the `tensor_algebra_index`s be
                //       a std::vector<int64_t> attribute of lyndon_words instead, and then just use torch::from_blob.
                torch::Tensor indices = torch::empty({lyndon_words.amount}, torch::dtype(torch::kInt64));
                auto index_accessor = indices.accessor<int64_t, 1>();
                for (s_size_type depth_index = 0; depth_index < lyndon_words.depth; ++depth_index){
                    for (auto& lyndon_word : lyndon_words[depth_index]) {
                        index_accessor[lyndon_word.compressed_index] = lyndon_word.tensor_algebra_index;
                    }
                }
                indices = indices.to(input.device());

                return torch::index_select(input, /*dim=*/channel_dim, /*index=*/indices);
            }

            // The backwards operation corresponding to compress.
            torch::Tensor compress_backward(torch::Tensor grad_compressed, const lyndon::LyndonWords& lyndon_words,
                                            torch::TensorOptions opts, bool stream, int64_t output_channel_size) {
                int64_t batch_size = grad_compressed.size(batch_dim);
                torch::Tensor grad_expanded;
                if (stream) {
                    int64_t output_stream_size = grad_compressed.size(stream_dim);
                    grad_expanded = torch::zeros({output_stream_size,
                                                  batch_size,
                                                  output_channel_size}, opts);
                }
                else {
                    grad_expanded = torch::zeros({batch_size,
                                                  output_channel_size}, opts);
                }

                // TODO: avoid the need for this copy operation entirely by having all of the `tensor_algebra_index`s be
                //       a std::vector<int64_t> attribute of lyndon_words instead, and then just use torch::from_blob.
                torch::Tensor indices = torch::empty({lyndon_words.amount}, torch::dtype(torch::kInt64));
                auto index_accessor = indices.accessor<int64_t, 1>();
                for (s_size_type depth_index = 0; depth_index < lyndon_words.depth; ++depth_index){
                    for (auto& lyndon_word : lyndon_words[depth_index]) {
                        index_accessor[lyndon_word.compressed_index] = lyndon_word.tensor_algebra_index;
                    }
                }
                indices = indices.to(grad_compressed.device());

                indices = indices.expand_as(grad_compressed);

                return grad_expanded.scatter_(channel_dim, indices, grad_compressed);
            }

            void logsignature_checkargs(torch::Tensor signature, int64_t input_channel_size, s_size_type depth,
                                        bool stream, bool scalar_term) {
                misc::checkargs_channels_depth(input_channel_size, depth);
                if (stream) {
                    if (signature.ndimension() != 3) {
                        throw std::invalid_argument("Argument 'signature' must be a 3-dimensional tensor, with "
                                                    "dimensions corresponding to (batch, stream, channel) "
                                                    "respectively.");
                    }
                    if (signature.size(stream_dim) == 0) {
                        throw std::invalid_argument("Argument 'signature' cannot have dimensions of size zero.");
                    }
                }
                else {
                    if (signature.ndimension() != 2) {
                        throw std::invalid_argument("Argument 'signature' must be a 2-dimensional tensor, with "
                                                    "dimensions corresponding to (batch, channel) respectively.");
                    }
                }
                if (signature.size(batch_dim) == 0 || signature.size(channel_dim) == 0) {
                    throw std::invalid_argument("Argument 'signature' cannot have dimensions of size zero.");
                }
                if (signature.size(channel_dim)
                    != signature_channels(input_channel_size, depth, scalar_term)) {
                    throw std::invalid_argument("Argument 'signature' has the wrong number of channels for the "
                                                "specified channels and depth.");
                }
                if (!signature.is_floating_point()) {
                    throw std::invalid_argument("Argument 'signature' must be of floating point type.");
                }
            }
        }  // namespace signatory::logsignature::detail
    }  // namespace signatory::logsignature

    py::object make_lyndon_info(int64_t channels, s_size_type depth, LogSignatureMode mode) {
        misc::checkargs_channels_depth(channels, depth);

        py::gil_scoped_release release;

        std::unique_ptr<lyndon::LyndonWords> lyndon_words;
        std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms;
        std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms_backward;

        // no make_unique in C++11
        if (mode == LogSignatureMode::Words) {
            lyndon_words.reset(new lyndon::LyndonWords(channels, depth, lyndon::LyndonWords::word_tag));
        }
        else if (mode == LogSignatureMode::Brackets) {
            lyndon_words.reset(new lyndon::LyndonWords(channels, depth, lyndon::LyndonWords::bracket_tag));
            lyndon_words->to_lyndon_basis(transforms, transforms_backward);
            lyndon_words->delete_extra();
        }

        return misc::wrap_capsule<logsignature::detail::LyndonInfo>(std::move(lyndon_words),
                                                                    std::move(transforms),
                                                                    std::move(transforms_backward));
    }

    std::tuple<torch::Tensor, py::object>
    signature_to_logsignature_forward(torch::Tensor signature, int64_t input_channel_size, s_size_type depth,
                                      bool stream, LogSignatureMode mode, py::object lyndon_info_capsule,
                                      bool scalar_term) {
        logsignature::detail::logsignature_checkargs(signature, input_channel_size, depth, stream, scalar_term);

        // must finish using Python objects before we release the GIL
        if (lyndon_info_capsule.is_none()) {
            lyndon_info_capsule = make_lyndon_info(input_channel_size, depth, mode);
        }
        logsignature::detail::LyndonInfo* lyndon_info =
                misc::unwrap_capsule<logsignature::detail::LyndonInfo>(lyndon_info_capsule);

        torch::Tensor logsignature;
        {  // release GIL
            py::gil_scoped_release release;

            if (scalar_term) {
                signature = signature.narrow(/*dim=*/channel_dim, /*start=*/1,
                                             /*length=*/signature.size(channel_dim) - 1);
            }

            // Don't need to track gradients when we have a custom backward
            signature = signature.detach();

            torch::TensorOptions opts = signature.options();
            torch::Tensor reciprocals = misc::make_reciprocals(depth, opts);
            int64_t output_stream_size = stream ? signature.size(stream_dim) : -1;

            // and allocate memory for the logsignature
            logsignature = torch::empty_like(signature);
            std::vector <torch::Tensor> signature_by_term;
            std::vector <torch::Tensor> logsignature_by_term;
            misc::slice_by_term(signature, signature_by_term, input_channel_size, depth);
            misc::slice_by_term(logsignature, logsignature_by_term, input_channel_size, depth);

            if (stream) {
                std::vector <torch::Tensor> signature_by_term_at_stream;

                // The if statement is for safety's sake... we haven't had issues with this one, but there have been
                // other issues we've run into with OpenMP+GPU on other for loops.
                // (even though presumably those threads are just scheduling work for the GPU to do... ?)
                #pragma omp parallel for default(none) \
                                     if(!signature.is_cuda()) \
                                     shared(output_stream_size, logsignature_by_term, signature_by_term, reciprocals)
                for (int64_t stream_index = 0;
                     stream_index < output_stream_size;
                     ++stream_index) {
                    std::vector <torch::Tensor> signature_by_term_at_stream;
                    std::vector <torch::Tensor> logsignature_by_term_at_stream;

                    misc::slice_at_stream(signature_by_term, signature_by_term_at_stream, stream_index);
                    misc::slice_at_stream(logsignature_by_term, logsignature_by_term_at_stream, stream_index);

                    ta_ops::log(logsignature_by_term_at_stream, signature_by_term_at_stream, reciprocals);
                }
            }
            else {
                ta_ops::log(logsignature_by_term, signature_by_term, reciprocals);
            }

            // Brackets and Words are the two possible compressed forms of the logsignature. So here we perform the
            // compression.
            if (mode == LogSignatureMode::Words) {
                logsignature = logsignature::detail::compress(*lyndon_info->lyndon_words, logsignature);
            }
            else if (mode == LogSignatureMode::Brackets) {
                logsignature = logsignature::detail::compress(*lyndon_info->lyndon_words, logsignature);
                // This is essentially solving a sparse linear system... and it's horrendously slow on a GPU.
                // There may well be ways of speeding this up beyond what's done here, but the brackets mode is
                // definitely the least favoured child out of the mode options we provide. (It's typically a strange
                // choice in machine learning anyway, when the words mode is available.)
                // iisignature does manage to provide this mode efficiently on the CPU by collecting together Lyndon
                // anagrams and then using pseudoinverses, which is an approach that might well work efficiently on the
                // GPU, so that is a possibility
                auto device = logsignature.device();
                logsignature = logsignature.cpu();
                // Then apply the transforms. We rely on the triangularity property of the Lyndon basis for this to work.
                #pragma omp parallel for default(none) \
                                     shared(lyndon_info, logsignature) schedule(dynamic, 1)
                for (s_size_type transform_class_index = 0;
                     transform_class_index < static_cast<s_size_type>(lyndon_info->transforms.size());
                     ++transform_class_index) {
                    // Note that it is very important that this inner loop operate serially!
                    for (const auto& transform : lyndon_info->transforms[transform_class_index]) {
                        int64_t source_index = std::get<0>(transform);
                        int64_t target_index = std::get<1>(transform);
                        int64_t coefficient = std::get<2>(transform);
                        torch::Tensor source = logsignature.narrow(/*dim=*/channel_dim,
                                                                   /*start=*/source_index,
                                                                   /*length=*/1);
                        torch::Tensor target = logsignature.narrow(/*dim=*/channel_dim,
                                                                   /*start=*/target_index,
                                                                   /*length=*/1);
                        target.sub_(source, coefficient);
                    }
                }
                logsignature = logsignature.to(device);
            }
        }  // finish released GIL

        return std::tuple<torch::Tensor, py::object> {logsignature, lyndon_info_capsule};
    }

    torch::Tensor signature_to_logsignature_backward(torch::Tensor grad_logsignature,
                                                     torch::Tensor signature,
                                                     int64_t input_channel_size,
                                                     s_size_type depth,
                                                     bool stream,
                                                     LogSignatureMode mode,
                                                     py::object lyndon_info_capsule,
                                                     bool scalar_term) {

        // Must do this before releasing the GIL.
        logsignature::detail::LyndonInfo* lyndon_info =
                misc::unwrap_capsule<logsignature::detail::LyndonInfo>(lyndon_info_capsule);

        py::gil_scoped_release release;

        if (scalar_term) {
            signature = signature.narrow(/*dim=*/channel_dim, /*start=*/1, /*length=*/signature.size(channel_dim) - 1);
        }

        grad_logsignature = grad_logsignature.detach();
        signature = signature.detach();

        torch::TensorOptions opts = signature.options();
        torch::Tensor reciprocals = misc::make_reciprocals(depth, opts);
        int64_t output_stream_size = stream ? signature.size(stream_dim) : -1;
        int64_t output_channel_size = signature.size(channel_dim);

        std::vector<torch::Tensor> signature_by_term;
        misc::slice_by_term(signature, signature_by_term, input_channel_size, depth);

        // Decompress the logsignature
        if (mode == LogSignatureMode::Expand) {
            grad_logsignature = grad_logsignature.clone();  // Clone so we don't leak changes through grad_logsignature.
        }
        else if (mode == LogSignatureMode::Words){
            grad_logsignature = logsignature::detail::compress_backward(grad_logsignature, *lyndon_info->lyndon_words,
                                                                        opts, stream,
                                                                        output_channel_size);
        }
        else {  // mode == LogSignatureMode::Brackets
            grad_logsignature = logsignature::detail::compress_backward(grad_logsignature, *lyndon_info->lyndon_words,
                                                                        opts, stream,
                                                                        output_channel_size);

            /* This is a deliberate asymmetry between the forwards and backwards: in the forwards pass we applied the
             * linear transformation after compression, but on the backwards we don't apply the transforms before
             * decompressing. Instead we apply a different (equivalent) transformation after decompressing. This is
             * because otherwise we would have to clone the grad_logsignature we were given, to be sure that the
             * transformations (which necessarily operate in-place) don't leak out. By doing it this way the memory that
             * we operate on is internal memory that we've claimed, not memory that we've been given in an input.
             */
            auto device = grad_logsignature.device();
            grad_logsignature = grad_logsignature.cpu();
            // This is essentially solving a sparse linear system... and it's horrendously slow on a GPU.
            #pragma omp parallel for default(none) \
                                     shared(lyndon_info, grad_logsignature) schedule(dynamic,1)
            for (s_size_type transform_class_index = 0;
                 transform_class_index < static_cast<s_size_type>(lyndon_info->transforms_backward.size());
                 ++transform_class_index) {
                for (auto tptr = lyndon_info->transforms_backward[transform_class_index].rbegin();
                     tptr != lyndon_info->transforms_backward[transform_class_index].rend();
                     ++tptr)  {
                    int64_t source_index = std::get<0>(*tptr);
                    int64_t target_index = std::get<1>(*tptr);
                    int64_t coefficient = std::get<2>(*tptr);
                    torch::Tensor grad_source = grad_logsignature.narrow(/*dim=*/channel_dim,
                                                                         /*start=*/source_index,
                                                                         /*length=*/1);
                    torch::Tensor grad_target = grad_logsignature.narrow(/*dim=*/channel_dim,
                                                                         /*start=*/target_index,
                                                                         /*length=*/1);
                    grad_source.sub_(grad_target, coefficient);
                }
            }
            grad_logsignature = grad_logsignature.to(device);
        }

        torch::Tensor grad_signature;
        torch::Tensor grad_signature_with_scalar;
        if (scalar_term) {
            if (stream) {
                grad_signature_with_scalar = torch::zeros({grad_logsignature.size(stream_dim),
                                                           grad_logsignature.size(batch_dim),
                                                           grad_logsignature.size(channel_dim) + 1},
                                                          opts);
            }
            else {
                grad_signature_with_scalar = torch::zeros({grad_logsignature.size(batch_dim),
                                                           grad_logsignature.size(channel_dim) + 1},
                                                          opts);
            }
            grad_signature = grad_signature_with_scalar.narrow(/*dim=*/channel_dim, /*start=*/1,
                                                               /*length=*/grad_logsignature.size(channel_dim));
        }
        else {
            grad_signature = torch::zeros_like(grad_logsignature);
            grad_signature_with_scalar = grad_signature;
        }

        std::vector<torch::Tensor> grad_logsignature_by_term;
        std::vector<torch::Tensor> grad_signature_by_term;
        misc::slice_by_term(grad_logsignature, grad_logsignature_by_term, input_channel_size, depth);
        misc::slice_by_term(grad_signature, grad_signature_by_term, input_channel_size, depth);

        if (stream) {
            // The if statement is because this sometimes hangs on the GPU... for some reason.
            #pragma omp parallel for default(none) \
                                     if(!grad_logsignature.is_cuda()) \
                                     shared(grad_logsignature_by_term, \
                                            grad_signature_by_term, \
                                            signature_by_term, \
                                            reciprocals, \
                                            output_stream_size)
            for (int64_t stream_index = 0; stream_index < output_stream_size; ++stream_index) {
                std::vector<torch::Tensor> grad_logsignature_by_term_at_stream;
                std::vector<torch::Tensor> grad_signature_by_term_at_stream;
                std::vector<torch::Tensor> signature_by_term_at_stream;

                misc::slice_at_stream(grad_logsignature_by_term,
                                      grad_logsignature_by_term_at_stream,
                                      stream_index);
                misc::slice_at_stream(grad_signature_by_term,
                                      grad_signature_by_term_at_stream,
                                      stream_index);
                misc::slice_at_stream(signature_by_term,
                                      signature_by_term_at_stream,
                                      stream_index);

                ta_ops::log_backward(grad_logsignature_by_term_at_stream, grad_signature_by_term_at_stream,
                                     signature_by_term_at_stream, reciprocals);
            }
        }
        else {
            ta_ops::log_backward(grad_logsignature_by_term, grad_signature_by_term, signature_by_term, reciprocals);
        }

        return grad_signature_with_scalar;
    }
}  // namespace signatory