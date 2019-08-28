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
#include <Python.h>   // PyCapsule
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <limits>     // std::numeric_limits
#include <tuple>      // std::tuple
#include <vector>     // std::vector

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
            // In theory it'd be slightly quicker to calculate this via the geometric formula, but that involves a
            // division which gives inaccurate results for large numbers.
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
        LyndonSpec::LyndonSpec(int64_t input_channels, s_size_type depth) :
            input_channels{input_channels},
            depth{depth}
        {};

        SigSpec::SigSpec(torch::Tensor path, s_size_type depth, bool stream, bool basepoint) :
            LyndonSpec(path.size(channel_dim), depth),
            opts{torch::TensorOptions().dtype(path.dtype()).device(path.device())},
            input_stream_size{path.size(stream_dim)},
            batch_size{path.size(batch_dim)},
            output_stream_size{path.size(stream_dim) - (basepoint ? 0 : 1)},
            output_channels{signature_channels(path.size(channel_dim), depth)},
            n_output_dims{stream ? 3 : 2},
            reciprocals{torch::ones({depth - 1}, opts)},
            stream{stream},
            basepoint{basepoint}
        {
            if (depth > 1) {
                reciprocals /= torch::linspace(2, depth, depth - 1, opts);
            }  // and reciprocals will be empty - of size 0 - if depth == 1.
        };

        BackwardsInfo::BackwardsInfo(SigSpec&& sigspec, std::vector<torch::Tensor>&& out_vector, torch::Tensor out,
                                     torch::Tensor path_increments) :
            sigspec{sigspec},
            out_vector{out_vector},
            out{out},
            path_increments{path_increments}
            {};

        void BackwardsInfo::set_logsignature_data(std::vector<torch::Tensor>&& signature_vector_,
                                                  py::object lyndon_info_capsule_,
                                                  LogSignatureMode mode_,
                                                  int64_t logsignature_channels_) {
            signature_vector = signature_vector_;
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

        void checkargs(torch::Tensor path, s_size_type depth, bool basepoint, torch::Tensor basepoint_value) {
            // This function is called before we even transpose anything (as we don't yet know that we can do a
            // transpose). As a result path should be of size (batch, stream, channel) at this point
            if (path.ndimension() != 3) {
                throw std::invalid_argument("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                                            "corresponding to (batch, stream, channel) respectively.");
            }
            if (path.size(0) == 0 || path.size(1) == 0 || path.size(2) == 0) {
                throw std::invalid_argument("Argument 'path' cannot have dimensions of size zero.");
            }
            if (!basepoint && path.size(1) == 1) {
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

        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec, int64_t num_channels) {
            // This function is called before we even transpose anything (as we don't yet know that we can do a
            // transpose). As a result grad_out should be of size (batch, stream, channel) at this point
            if (num_channels == -1) {
                num_channels = sigspec.output_channels;
            }

            if (sigspec.stream) {
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
    }  // namespace signatory::misc
}  // namespace signatory