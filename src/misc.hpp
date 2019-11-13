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
    int64_t signature_channels(int64_t input_channel_size, int64_t depth);

    // See signatory.max_parallelism for documentation
    // But basically, this is the amount of extra parallelisation we allow ourselves for those operations which, as a
    // result of parallelising, use extra memory.
    // As a result this should be used to bound the parallelisation whenever this is the case. (And should _not_ be used
    // to bound all parallelisation!)
    void set_max_parallelism(int64_t value);
    int64_t get_max_parallelism();

    namespace misc {
        inline torch::TensorOptions make_opts(torch::Tensor tensor);

        inline torch::Tensor make_reciprocals(s_size_type depth, torch::TensorOptions opts);

        // Argument 'in' is assumed to be a tensor with channel dimension of size minimalspec.input_channels.
        // It is sliced up along that dimension, and the resulting tensors placed into 'out'.
        // Each resulting tensor corresponds to one of the (tensor, not scalar) terms in the signature.
        inline void slice_by_term(torch::Tensor in, std::vector<torch::Tensor>& out, int64_t input_channel_size,
                                  s_size_type depth);

        // Argument 'in' is assumed to be a tensor for which its first dimension corresponds to the stream dimension.
        // Its slices along a particular index of that dimension are put in 'out'.
        inline void slice_at_stream(const std::vector<torch::Tensor>& in, std::vector<torch::Tensor>& out,
                                    int64_t stream_index);

        // Checks the arguments for a bunch of functions only depending on channels and depth.
        void checkargs_channels_depth(int64_t channels, s_size_type depth);
    }  // namespace signatory::misc
}  // namespace signatory

#include "misc.inl"

#endif //SIGNATORY_MISC_HPP
