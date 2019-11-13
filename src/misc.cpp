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
    namespace misc {
        namespace detail {
            // This represents the maximum amount of parallelisation that we allow Signatory, _not_ counting the normal
            // parallelisation over the batch dimension.
            // Essentially, it's how much extra memory we're willing to use to try and speed up calculations by doing
            // clever tricks.
            int64_t max_parallelism = 8;
        }  // namespace signatory::misc::detail

        void checkargs_channels_depth(int64_t channels, s_size_type depth) {
            if (channels < 1) {
                throw std::invalid_argument("Argument 'channels' must be at least one.");
            }
            if (depth < 1) {
                throw std::invalid_argument("Argument 'depth' must be an integer greater than or equal to one.");
            }
        }
    }  // namespace signatory::misc

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

    void set_max_parallelism(int64_t value) {
        if (value < 1 && value != -1) {
            throw std::invalid_argument("value must be either -1 or greater than or equal to 1.");
        }
        misc::detail::max_parallelism = value;
    }

    int64_t get_max_parallelism() {
        if (misc::detail::max_parallelism == -1) {
            return std::numeric_limits<decltype(misc::detail::max_parallelism)>::max();
        }
        else {
            return misc::detail::max_parallelism;
        }
    }
}  // namespace signatory