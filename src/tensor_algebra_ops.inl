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
#include <cmath>      // pow
#include <vector>     // std::vector

#include "misc.hpp"  // s_size_type


namespace signatory {
    namespace ta_ops {
        template <typename scalar_t, bool inverse>
        void mult_fused_restricted_exp_single_cpu(torch::TensorAccessor<scalar_t, 1> next_a,
                                                  std::vector<torch::TensorAccessor<scalar_t, 1>>& prev_a,
                                                  torch::TensorAccessor<scalar_t, 1> reciprocals_a) {
            // This whole thing is pretty much just a rewriting of mult_fused_restricted_exp in normal C++

            int64_t input_channel_size = next_a.size(0);
            s_size_type depth = prev_a.size();

            std::vector<std::vector<scalar_t>> next_divided;
            next_divided.resize(reciprocals_a.size(0));
            for (int64_t reciprocal_index = 0; reciprocal_index < reciprocals_a.size(0); ++reciprocal_index) {
                next_divided[reciprocal_index].resize(input_channel_size);
                for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                    next_divided[reciprocal_index][channel_index] = reciprocals_a[reciprocal_index] *
                                                                    next_a[channel_index];
                }
            }

            if (depth > 1) {
                std::vector<scalar_t> new_scratch;
                std::vector<scalar_t> old_scratch;
                // Figure out how large each vector is going to get by the end of the computation.
                if ((depth % 2) == 0) {
                    old_scratch.reserve(pow(input_channel_size, depth - 2));
                    new_scratch.reserve(old_scratch.size() * input_channel_size);
                }
                else {
                    new_scratch.reserve(pow(input_channel_size, depth - 2));
                    old_scratch.reserve(new_scratch.size() * input_channel_size);
                }

                for (s_size_type depth_index = depth - 1; depth_index >= 1; --depth_index) {
                    new_scratch.resize(input_channel_size);
                    for (int64_t scratch_index = 0; scratch_index < input_channel_size; ++scratch_index) {
                        new_scratch[scratch_index] = prev_a[0][scratch_index] + next_divided[depth_index - 1][scratch_index];
                    }

                    for (s_size_type j = 1, k = depth_index - 2; j < depth_index; ++j, --k) {
                        old_scratch.swap(new_scratch);
                        new_scratch.resize(old_scratch.size() * input_channel_size);
                        for (int64_t old_scratch_index = 0;
                             old_scratch_index < static_cast<int64_t>(old_scratch.size());
                             ++old_scratch_index) {
                            for (int64_t next_divided_index = 0; next_divided_index < input_channel_size; ++next_divided_index)
                            {
                                int64_t new_scratch_index;
                                if (inverse) {
                                    new_scratch_index = next_divided_index * old_scratch.size() + old_scratch_index;
                                }
                                else {
                                    new_scratch_index = old_scratch_index * input_channel_size + next_divided_index;
                                }
                                new_scratch[new_scratch_index] = prev_a[j][new_scratch_index] +
                                                                 old_scratch[old_scratch_index] *
                                                                 next_divided[k][next_divided_index];
                            }
                        }
                    }

                    for (int64_t new_scratch_index = 0;
                         new_scratch_index < static_cast<int64_t>(new_scratch.size());
                         ++new_scratch_index) {
                        for (int64_t next_index = 0; next_index < input_channel_size; ++next_index)
                        {
                            int64_t prev_a_index;
                            if (inverse) {
                                prev_a_index = next_index * new_scratch.size() + new_scratch_index;
                            }
                            else {
                                prev_a_index = new_scratch_index * input_channel_size + next_index;
                            }
                            prev_a[depth_index][prev_a_index] += new_scratch[new_scratch_index] * next_a[next_index];
                        }
                    }
                }
            }

            for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                prev_a[0][channel_index] += next_a[channel_index];
            }
        }
    }  // namespace signatory::ta_ops
}  // namespace signatory