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
    int64_t signature_channels(int64_t input_channel_size, int64_t depth) {
        if (input_channel_size < 1) {
            throw std::invalid_argument("input_channels must be at least 1");
        }
        if (depth < 1) {
            throw std::invalid_argument("depth must be at least 1");
        }

        if (input_channel_size == 1) {
            return depth;
        }
        else {
            // In theory it'd probably be slightly quicker to calculate this via the geometric formula, but that
            // involves a division which gives inaccurate results for large numbers.
            int64_t output_channels = input_channel_size;
            int64_t mul_limit = std::numeric_limits<int64_t>::max() / input_channel_size;
            int64_t add_limit = std::numeric_limits<int64_t>::max() - input_channel_size;
            for (int64_t depth_index = 1; depth_index < depth; ++depth_index) {
                if (output_channels > mul_limit) {
                    throw std::invalid_argument("Integer overflow detected.");
                }
                output_channels *= input_channel_size;
                if (output_channels > add_limit) {
                    throw std::invalid_argument("Integer overflow detected.");
                }
                output_channels += input_channel_size;
            }
            return output_channels;
        }
    }

    namespace misc {
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
            torch::TensorOptions path_opts = make_opts(path);
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
                if (path_opts != make_opts(basepoint_value)) {
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
                if (path_opts != make_opts(initial_value)) {
                    throw std::invalid_argument("Argument 'initial' does not have the same dtype or device as 'path'.");
                }
            }
        }

        void checkargs_backward(torch::Tensor grad_out, bool stream, int64_t output_stream_size, int64_t batch_size,
                                int64_t channel_size, torch::TensorOptions opts) {
            if (stream) {
                if (grad_out.ndimension() != 3) {
                    throw std::invalid_argument("Gradient must be a 3-dimensional tensor, with dimensions "
                                                "corresponding to (batch, stream, channel) respectively.");
                }
                if (grad_out.size(batch_dim) != batch_size ||
                    grad_out.size(stream_dim) != output_stream_size ||
                    grad_out.size(channel_dim) != channel_size) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }
            else {
                if (grad_out.ndimension() != 2) {
                    throw std::invalid_argument("Gradient must be a 2-dimensional tensor, with dimensions"
                                                "corresponding to (batch, channel) respectively.");
                }
                if (grad_out.size(batch_dim) != batch_size ||
                    grad_out.size(channel_dim) != channel_size) {
                    throw std::invalid_argument("Gradient has the wrong size.");
                }
            }

            if (opts != make_opts(grad_out)) {
                throw std::invalid_argument("Argument 'grad_signature' does not have the correct dtype or device.");
            }
        }
    }  // namespace signatory::misc
}  // namespace signatory