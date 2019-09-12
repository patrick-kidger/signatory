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
 // Provides various miscellaneous things used throughout the project


#ifndef SIGNATORY_MISC_HPP
#define SIGNATORY_MISC_HPP

#include <torch/extension.h>
#include <cstdint>      // int64_t
#include <tuple>        // std::tuple
#include <type_traits>  // std::make_signed, std::make_unsigned
#include <vector>       // std::vector


namespace signatory {
    // Modes for the return value of logsignature
    // See signatory.logsignature for further documentation
    enum class LogSignatureMode { Expand, Brackets, Words };

    // signed-ness is important because we'll sometimes iterate downwards
    // it is very deliberately not called 'size_type' because otherwise when using it in e.g. the constructor for
    // something inheriting from std::vector, then 'size_type' will there refer to std::vector::size_type instead.
    using s_size_type = std::make_signed<std::vector<torch::Tensor>::size_type>::type;
    using u_size_type = std::make_unsigned<s_size_type>::type;

    // For clarity about which dimension we're slicing over.
    // Done negatively so that it works even when the stream dimension isn't present
    // (All tensors we consider are of shape either (stream, batch, channel) or (batch, channel).)
    // Note that these do not define the order of the dimensions. Other pieces of code (in particular torch::zeros,
    // torch::empty and torch::Tensor::view) will implicitly rely on the order of dimensions
    constexpr auto stream_dim = -3;
    constexpr auto batch_dim = -2;
    constexpr auto channel_dim = -1;

    // See signatory.signature_channels for documentation
    int64_t signature_channels(int64_t input_channels, int64_t depth);

    namespace misc {
        // Encapsulates the things necessary for certain computations. This will get passed around through
        // most such functions.
        struct MinimalSpec {
            MinimalSpec(int64_t input_channels, s_size_type depth);

            int64_t input_channels;
            s_size_type depth;
        };

        // Encapsulates all the things that aren't tensors for signature and logsignature computations. This will get
        // passed around through most such functions.
        struct SigSpec : MinimalSpec {
            SigSpec(torch::Tensor path, s_size_type depth, bool stream, bool basepoint, bool inverse);

            torch::TensorOptions opts;
            int64_t input_stream_size;
            int64_t batch_size;
            int64_t output_stream_size;
            int64_t output_channels;
            int64_t n_output_dims;
            torch::Tensor reciprocals;
            bool stream;
            bool basepoint;
            bool inverse;
        };

        // Argument 'in' is assumed to be a tensor with channel dimension of size minimalspec.input_channels.
        // It is sliced up along that dimension, and the resulting tensors placed into 'out'.
        // Each resulting tensor corresponds to one of the (tensor, not scalar) terms in the signature.
        inline void slice_by_term(torch::Tensor in, std::vector<torch::Tensor>& out, const MinimalSpec& minimalspec);

        // Argument 'in' is assumed to be a tensor for which its first dimension corresponds to the stream dimension.
        // Its slices along a particular index of that dimension are put in 'out'.
        inline void slice_at_stream(std::vector<torch::Tensor> in, std::vector<torch::Tensor>& out,
                                    int64_t stream_index);

        inline bool is_even(s_size_type index);

        // Retains information needed for the backwards pass.
        struct BackwardsInfo{
            BackwardsInfo(SigSpec&& sigspec_, const std::vector<torch::Tensor>& stream_vector_,
                          torch::Tensor signature_, torch::Tensor path_increments_, bool initial_);

            void set_logsignature_data(const std::vector<torch::Tensor>& signature_vector_,
                                       py::object lyndon_info_capsule_,
                                       LogSignatureMode mode_,
                                       int64_t logsignature_channels_);

            SigSpec sigspec;
            std::vector<torch::Tensor> signature_by_term;
            torch::Tensor signature;
            torch::Tensor path_increments;
            bool initial;

            py::object lyndon_info_capsule;
            LogSignatureMode mode;
            int64_t logsignature_channels;

            constexpr static auto capsule_name = "signatory.BackwardsInfoCapsule";
        };

        // Checks the arguments for a bunch of functions only depending on channels and depth.
        void checkargs_channels_depth(int64_t channels, s_size_type depth);

        // Checks the arguments for the forwards pass in the signature function (kept here for consistency with the
        // other checkarg functions).
        void checkargs(torch::Tensor path, s_size_type depth, bool basepoint, torch::Tensor basepoint_value,
                       bool initial, torch::Tensor initial_value);

        // Checks the arguments for the backwards pass in the signature and logsignature function. Only grad_out is
        // checked to make sure it is as expected. The objects we get from the PyCapsule-wrapped BackwardsInfo object
        // are assumed to be correct.
        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec, int64_t num_channels=-1);
    }  // namespace signatory::misc
}  // namespace signatory

#include "misc.inl"

#endif //SIGNATORY_MISC_HPP
