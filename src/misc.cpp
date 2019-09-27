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
#include <Python.h>     // PyCapsule
#include <cstdint>      // int64_t
#include <stdexcept>    // std::invalid_argument
#include <limits>       // std::numeric_limits
#include <tuple>        // std::tuple
#include <vector>       // std::vector

#include "misc.hpp"


namespace signatory {
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
            // In theory it'd probably be slightly quicker to calculate this via the geometric formula, but that
            // involves a division which gives inaccurate results for large numbers.
            int64_t output_channels = input_channels;
            int64_t mul_limit = std::numeric_limits<int64_t>::max() / input_channels;
            int64_t add_limit = std::numeric_limits<int64_t>::max() - input_channels;
            for (int64_t depth_index = 1; depth_index < depth; ++depth_index) {
                if (output_channels > mul_limit) {
                    throw std::invalid_argument("Integer overflow detected.");
                }
                output_channels *= input_channels;
                if (output_channels > add_limit) {
                    throw std::invalid_argument("Integer overflow detected.");
                }
                output_channels += input_channels;
            }
            return output_channels;
        }
    }

    namespace misc {
        MinimalSpec::MinimalSpec(int64_t input_channels, s_size_type depth) :
            input_channels{input_channels},
            depth{depth}
        {};

        SigSpec::SigSpec(torch::Tensor path, s_size_type depth, bool stream, bool basepoint, bool inverse) :
            MinimalSpec(path.size(channel_dim), depth),
            opts{torch::TensorOptions().dtype(path.dtype()).device(path.device())},
            input_stream_size{path.size(stream_dim)},
            batch_size{path.size(batch_dim)},
            output_stream_size{path.size(stream_dim) - (basepoint ? 0 : 1)},
            output_channels{signature_channels(path.size(channel_dim), depth)},
            reciprocals{torch::ones({depth - 1}, opts)},
            stream{stream},
            basepoint{basepoint},
            inverse{inverse}
        {
            if (depth > 1) {
                                                  // Cast to torch::Scalar is ambiguous
                reciprocals /= torch::linspace(2, static_cast<torch::Scalar>(static_cast<int64_t>(depth)),
                                               depth - 1, opts);
            }  // and reciprocals will be empty - of size 0 - if depth == 1.
        };

        BackwardsInfo::BackwardsInfo(SigSpec&& sigspec_, const std::vector<torch::Tensor>& signature_by_term_,
                                     torch::Tensor signature_, torch::Tensor path_increments_, bool initial_) :
            // Call to detach works around PyTorch bug 25340, which is a won't-fix. Basically, it makes sure that
            // backwards_info doesn't have references to any other tensors, and therefore in particular doesn't have
            // references to the tensor that is the output of the signature function: because this output tensor has a
            // reference to the Python-level 'ctx' variable, which in turn has a reference to the BackwardsInfo object,
            // and we get an uncollected cycle. (Some of the references are at the C++ level so this isn't picked up by
            // Python.)
            // Thus not doing this gives a massive memory leak.
            sigspec{sigspec_},
            signature{signature_.detach()},
            path_increments{path_increments_.detach()},
            initial{initial_}
            {
                signature_by_term.reserve(signature_by_term_.size());
                for (const auto& elem : signature_by_term_) {
                    signature_by_term.push_back(elem.detach());
                }
            };

        void BackwardsInfo::set_logsignature_data(const std::vector<torch::Tensor>& signature_by_term_,
                                                  py::object lyndon_info_capsule_,
                                                  LogSignatureMode mode_,
                                                  int64_t logsignature_channels_) {
            if (signature_by_term.size() == 0) {
                // We set signature_by_term if:
                // (a) signature, stream=True
                // (b) logsignature, stream=True
                // (c) logsignature, stream=False
                // In particular this function is called in cases (b) and (c). However (b) implies (a), so we don't need
                // to set it in this case; this is what we check here.
                signature_by_term.reserve(signature_by_term_.size());
                for (const auto& elem : signature_by_term_) {
                    signature_by_term.push_back(elem.detach());
                }
            }
            lyndon_info_capsule = lyndon_info_capsule_;
            mode = mode_;
            logsignature_channels = logsignature_channels_;
        }

        void checkargs_channels_depth(int64_t channels, s_size_type depth) {
            if (channels < 1) {
                throw std::invalid_argument("Argument 'channels' must be at least one.");
            }
            if (depth < 1) {
                throw std::invalid_argument("Argument 'depth' must be an integer greater than or equal to one.");
            }
        }

        void checkargs(torch::Tensor path, s_size_type depth, bool basepoint, torch::Tensor basepoint_value,
                       bool initial, torch::Tensor initial_value) {
            if (path.ndimension() == 2) {
                // Friendlier help message for a common mess-up.
                throw std::invalid_argument("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                                            "corresponding to (batch, stream, channel) respectively. If you just want "
                                            "the signature or logsignature of a single path then wrap it in a single "
                                            "batch dimension by replacing e.g. signature(path, depth) with "
                                            "signature(path.unsqueeze(0), depth).squeeze(0).");
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
            }
        }

        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec, int64_t num_channels) {
            if (num_channels == -1) {
                num_channels = sigspec.output_channels;
            }

            if (sigspec.stream) {
                if (grad_out.ndimension() != 3) {
                    throw std::invalid_argument("Gradient must be a 3-dimensional tensor, with dimensions "
                                                "corresponding to (batch, stream, channel) respectively.");
                }
                if (grad_out.size(batch_dim) != sigspec.batch_size ||
                    grad_out.size(stream_dim) != sigspec.output_stream_size ||
                    grad_out.size(channel_dim) != num_channels) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }
            else {
                if (grad_out.ndimension() != 2) {
                    throw std::invalid_argument("Gradient must be a 2-dimensional tensor, with dimensions"
                                                "corresponding to (batch, channel) respectively.");
                }
                if (grad_out.size(batch_dim) != sigspec.batch_size ||
                    grad_out.size(channel_dim) != num_channels) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }
        }
    }  // namespace signatory::misc
}  // namespace signatory