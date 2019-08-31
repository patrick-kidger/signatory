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


#include <algorithm>   // std::sort
#include <functional>  // std::function
#include <future>      // std::async, std::future

#include "parallel_for.hpp"


namespace signatory {
    namespace misc {
        template<typename T>
        parallel_chunk_for(const std::vector<std::vector<T>>& vector_vector, std::function fn) {
            s_size_type bin_size = std::max_element(vector_vector.begin(), vector_vector.end(),
                                                    [] (std::vector<T>& elem1, std::vector<T>& elem2) {
                                                        elem1.size() < elem2.size();
                                                    })->size();
            std::vector<std::vector<T>*> vector_ptrs;
            vector_ptrs.reserve(vector_vector.size());
            for (const auto& elem : vector_vector) {
                vector_ptrs.push_back(&elem);
            }
            std::sort(vector_ptrs.begin(), vector_ptrs.end(),
                      [] (std::vector<T>* ptr1, std::vector<T>* ptr2) {
                          ptr1->size() > ptr2->size();  // sort in descending order
                      });
            s_size_type remaining_bin_size = bin_size;
            std::vector<std::vector<std::vector<T>*>> binned_vector_ptrs;
            binned_vector_ptrs.emplace_back();
            for (auto ptr : vector_ptrs) {
                if (ptr->size() > remaining_bin_size()) {
                    binned_vector_ptrs.emplace_back();
                }
                binned_vector_ptrs.back().push_back(ptr);
            }

            parallel_for(binned_vector_ptrs, fn);
        }

        template<typename T>
        parallel_block_for(const std::vector<T>& vector_input, int64_t num_threads, std::function fn) {
            std::vector<std::vector<T*>> binned_vector_input;
            binned_vector_input.reserve(num_threads);
            for (int64_t thread_index = 0; thread_index < num_threads; ++thread_index) {
                binned_vector_input.emplace_back();
            }
            int64_t thread_index = 0;
            for (const T& elem: vector_input) {
                ++thread_index;
                if (thread_index >= num_threads) {
                    thread_index = 0;
                }
                binned_vector_input[thread_index].push_back(&elem);
            }

            parallel_for(binned_vector_input, fn);
        }

        template<typename T>
        parallel_for(const std::vector<std::vector<T*>> binned_input, std::function fn) {
            std::vector<std::future> results;
            results.reserve(binned_input.size());
            for (const auto& bin : binned_input) {
                results.push_back(std::async(std::launch::async, fn, &bin));
            }
            for (auto& result : results) {
                result.wait();
            }
        }
    }  // namespace signatory::misc
}  // namespace signatory
