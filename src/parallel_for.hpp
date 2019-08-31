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

#include <functional>  // std::function


namespace signatory {
    namespace misc {
        // Splits up vector_vector based on the sizes of its subvectors; they are group together into equally-sized bins
        // and each bin is run in its own thread.
        // fn should be of type void fn(std::vector<std::vector<T>*>*)
        // fn is given the address of a bin
        template<typename T>
        parallel_chunk_for(const std::vector<std::vector<T>>& vector_vector, std::function fn);

        // Splits up vector_input into num_threads many bins, and each bin is run in its own thread.
        // fn should be of type void fn(std::vector<T*>*)
        // fn is given the address of a bin
        template<typename T>
        parallel_block_for(const std::vector<T>& vector_input, int64_t num_threads, std::function fn);

        // Runs fn on each subvector of binned_input.
        // fn should be of type void fn(std::vector<T*>*)
        // fn is given the address of each bin.
        template<typename T>
        parallel_for(const std::vector<std::vector<T*>> binned_input, std::function fn);
    }  // namespace signatory::misc
}  // namespace signatory

#endif //SIGNATORY_PARALLEL_FOR_HPP
