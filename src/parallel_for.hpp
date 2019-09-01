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
// Here we define helpers for parallelising over for loops


#ifndef SIGNATORY_PARALLEL_FOR_HPP
#define SIGNATORY_PARALLEL_FOR_HPP

#include <algorithm>   // std::sort
#include <future>      // std::async, std::future
#include <iterator>    // std::advance, std::distance, std::iterator_traits
#include <thread>      // std::thread
#include <utility>     // std::pair


namespace signatory {
    namespace misc {
        // Runs fn on element of the iterator
        template<typename ForwardIt, typename F>
        // Where F = void(ForwardIt)
        void parallel_for(ForwardIt first, ForwardIt last, F fn, u_size_type size_hint=0) {
            std::vector<std::future<void>> results;
            if (size_hint == 0) {
                size_hint = std::distance(first, last);
            }
            results.reserve(size_hint);
            for (auto bin = first; bin != last; ++bin) {
                results.push_back(std::async(std::launch::async, fn, bin));
            }
            for (auto& result : results) {
                result.wait();
            }
        }

        // Splits up a range based on the size of its elements; they are grouped together into equally-sized
        // bins and each bin is run in its own thread.
        template<typename ForwardIt, typename F>
        // Must be such that there exists the function std::iterator_traits<ForwardIt>::reference::size()
        // Where F = void(typename std::vector<std::vector<ForwardIt>>::iterator)
        void parallel_chunk_for(ForwardIt first, ForwardIt last, F fn, u_size_type size_hint=0) {
            if (size_hint == 0) {
                size_hint = std::distance(first, last);
            }

            // Bins vectors using first fit decreasing algorithm
            s_size_type bin_size = std::max_element(first, last,
                                                    [] (typename std::iterator_traits<ForwardIt>::reference elem1,
                                                        typename std::iterator_traits<ForwardIt>::reference elem2) {
                                                        return elem1.size() < elem2.size();
                                                    })->size();
            std::vector<ForwardIt> vector_ptrs;
            vector_ptrs.reserve(size_hint);
            for (auto elem = first; elem != last; ++elem) {
                vector_ptrs.push_back(elem);
            }
            std::sort(vector_ptrs.begin(), vector_ptrs.end(),
                      [] (const ForwardIt& ptr1,
                          const ForwardIt& ptr2) {
                          return ptr1->size() > ptr2->size();  // sort in descending order
                      });
            std::vector<u_size_type> remaining_bin_sizes;
            std::vector<std::vector<ForwardIt>> binned_vector_ptrs;
            remaining_bin_sizes.push_back(bin_size);
            binned_vector_ptrs.emplace_back();
            for (auto ptr : vector_ptrs) {
                s_size_type selected_bin_index = -1;
                for (u_size_type bin_index = 0; bin_index < remaining_bin_sizes.size(); ++bin_index) {
                    if (ptr->size() < remaining_bin_sizes[bin_index]) {
                        selected_bin_index = bin_index;
                    }
                }
                if (selected_bin_index == -1) {
                    binned_vector_ptrs.emplace_back();
                    remaining_bin_sizes.push_back(bin_size);
                    selected_bin_index = remaining_bin_sizes.size() - 1;
                }
                remaining_bin_sizes[selected_bin_index] -= ptr->size();
                binned_vector_ptrs[selected_bin_index].push_back(ptr);
            }

            parallel_for(binned_vector_ptrs.begin(), binned_vector_ptrs.end(), fn, binned_vector_ptrs.size());
        }

        // Splits up vector_input into num_threads many bins, and each bin is run in its own thread.
        template<typename ForwardIt, typename F>
        // Where F = void(typename std::vector<std::pair<ForwardIt, ForwardIt>>::iterator)
        // i.e. it takes subranges to operate over
        void parallel_group_for(ForwardIt first, ForwardIt last, F fn, u_size_type size_hint=0,
                                u_size_type num_threads=0) {
            if (size_hint == 0) {
                // TODO: make more general
//                size_hint = std::distance(first, last);
                size_hint = last - first;
            }
            if (num_threads == 0) {
                num_threads = std::thread::hardware_concurrency();
                if (num_threads == 0) {
                    num_threads = 8;
                }
            }

            u_size_type batch_size = size_hint / num_threads;
            u_size_type remainder = size_hint % num_threads;
            ForwardIt prev_border = first;
            ForwardIt border = first;

            std::vector<std::pair<ForwardIt, ForwardIt>> binned_vector_input;
            binned_vector_input.reserve(num_threads);
            u_size_type remainder_counter = 0;
            for (u_size_type thread_index = 0; thread_index < num_threads; ++thread_index) {
                // TODO: make more general
//                std::advance(border, batch_size);
                border += batch_size;
                if (remainder_counter < remainder) {
                    ++border;
                    ++remainder_counter;
                }

                binned_vector_input.emplace_back(prev_border, border);
                prev_border = border;
            }

            parallel_for(binned_vector_input.begin(), binned_vector_input.end(), fn, binned_vector_input.size());
        }
    }  // namespace signatory::misc
}  // namespace signatory

#endif //SIGNATORY_PARALLEL_FOR_HPP
