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
#include <cstdint>      // int64_t
#include <vector>       // std::vector


namespace signatory {
    namespace misc {
        inline void slice_by_term(torch::Tensor in, std::vector<torch::Tensor>& out, int64_t dim,
                                  const SigSpec& sigspec) {
            int64_t current_memory_pos = 0;
            int64_t current_memory_length = sigspec.input_channels;
            out.clear();
            out.reserve(sigspec.depth);
            for (int64_t i = 0; i < sigspec.depth; ++i) {
                out.push_back(in.narrow(/*dim=*/dim,
                        /*start=*/current_memory_pos,
                        /*len=*/current_memory_length));
                current_memory_pos += current_memory_length;
                current_memory_length *= sigspec.input_channels;
            }
        }

        inline void slice_at_stream(std::vector<torch::Tensor> in, std::vector<torch::Tensor>& out,
                                    int64_t stream_index) {
            out.clear();
            out.reserve(in.size());
            for (auto elem : in) {
                out.push_back(elem.narrow(/*dim=*/0, /*start=*/stream_index, /*len=*/1).squeeze(0));
            }
        }

        inline torch::Tensor transpose(torch::Tensor tensor, const SigSpec& sigspec) {
            if (sigspec.stream) {
                // convert from (stream, channel, batch) to (batch, stream, channel)
                return tensor.transpose(1, 2).transpose(0, 1);
            } else {
                // convert from (channel, batch) to (batch, channel)
                return tensor.transpose(0, 1);
            }
        }

        inline torch::Tensor transpose_reverse(torch::Tensor tensor, const SigSpec& sigspec) {
            if (sigspec.stream) {
                // convert from (batch, stream, channel) to (stream, channel, batch)
                return tensor.transpose(0, 1).transpose(1, 2);
            } else {
                // convert from (batch, channel) to (channel, batch)
                return tensor.transpose(0, 1);
            }
        }

        inline bool is_even(s_size_type index) {
            return (((index) % 2) == 0);
        }
    }  // namespace signatory::misc
}  // namespace signatory