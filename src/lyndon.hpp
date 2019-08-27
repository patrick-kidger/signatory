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
 // Computes certain utilities relating to Lyndon words


#ifndef SIGNATORY_LYNDON_HPP
#define SIGNATORY_LYNDON_HPP

#include <cstdint>    // int64_t


namespace signatory {
    // See signatory.lyndon_words for documentation
    std::vector<std::vector<int64_t>> lyndon_words(int64_t channels, int64_t depth);

    // See signatory.lyndon_brackets for documentation
    std::vector<py::object> lyndon_brackets(int64_t channels, int64_t depth);

    // See signatory.utility.lyndon_words_to_basis_transform for documentation
    std::vector<std::tuple<int64_t, int64_t, int64_t>> lyndon_words_to_basis_transform(int64_t channels, int64_t depth);
}  // namespace signatory

#endif //SIGNATORY_LYNDON_HPP
