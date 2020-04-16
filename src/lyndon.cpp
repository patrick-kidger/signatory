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


#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "misc.hpp"
#include "lyndon.hpp"

namespace signatory {
    namespace lyndon {
        namespace detail {
            template<typename T>
            std::vector<T> concat_vectors(const std::vector<T>& vector1, const std::vector<T>& vector2) {
                std::vector<T> concat;
                concat.reserve(vector1.size() + vector2.size());
                concat.insert(concat.end(), vector1.begin(), vector1.end());
                concat.insert(concat.end(), vector2.begin(), vector2.end());
                return concat;
            }

            // Caution! Only suitable for use on LyndonWords which have their extra information set.
            // (Which is why it's not set as operator< on LyndonWord itself.)
            // We could check whether it's been set in this function (extra==nullptr), but we don't, just like [] vs at.
            struct CompareWords {
                bool operator()(const LyndonWord& w1, const LyndonWord& w2) {
                    return w1.extra->word < w2.extra->word;
                }
                bool operator()(const LyndonWord* w1, const std::vector<int64_t> w2) {
                    return w1->extra->word < w2;
                }
                bool operator()(const std::vector<int64_t> w1, const LyndonWord* w2) {
                    return w1 < w2->extra->word;
                }
                bool operator()(const LyndonWord* w1, const LyndonWord* w2) {
                    return w1->extra->word < w2->extra->word;
                }
            };
            constexpr CompareWords compare_words {};
        }  // namespace signatory::lyndon::detail

        LyndonWords::LyndonWords(int64_t input_channel_size, s_size_type depth, WordTag) :
            input_channel_size{input_channel_size}, depth{depth}
        {
            this->reserve(depth);
            for (s_size_type depth_index = 0; depth_index < depth; ++depth_index) {
                this->emplace_back();
            }

            std::vector<int64_t> word;
            word.reserve(depth);
            word.push_back(-1);

            while (word.size()) {
                ++word.back();
                (*this)[word.size() - 1].emplace_back(word, false, input_channel_size);
                int64_t pos = 0;
                while (static_cast<s_size_type>(word.size()) < depth) {
                    word.push_back(word[pos]);
                    ++pos;
                }
                while (word.size() && word.back() == input_channel_size - 1) {
                    word.pop_back();
                }
            }
            finalise();
        }

        LyndonWords::LyndonWords(int64_t input_channel_size, s_size_type depth, BracketTag) :
            input_channel_size{input_channel_size}, depth{depth}
        {
            this->reserve(depth);
            for (s_size_type depth_index = 0; depth_index < depth; ++depth_index) {
                this->emplace_back();
            }

            (*this)[0].reserve(input_channel_size);
            for (int64_t channel_index = 0; channel_index < input_channel_size; ++channel_index) {
                (*this)[0].emplace_back(std::vector<int64_t> {channel_index}, true, input_channel_size);
            }

            for (s_size_type target_depth_index = 1; target_depth_index < depth; ++target_depth_index) {
                auto& target_depth_class = (*this)[target_depth_index];

                auto& depth_class1 = (*this)[0];
                auto& depth_class2 = (*this)[target_depth_index - 1];
                for (auto& elem : depth_class1) {
                    auto index_start = std::upper_bound(depth_class2.begin(), depth_class2.end(), elem,
                                                        detail::compare_words);
                    for (auto elemptr = index_start; elemptr != depth_class2.end(); ++elemptr) {
                        target_depth_class.emplace_back(&elem, &*elemptr, input_channel_size);
                    }
                }

                for (s_size_type depth_index1 = 1; depth_index1 < target_depth_index; ++depth_index1) {
                    s_size_type depth_index2 = target_depth_index - depth_index1 - 1;
                    auto& depth_class1 = (*this)[depth_index1];
                    auto& depth_class2 = (*this)[depth_index2];

                    for (auto& elem : depth_class1) {
                        auto index_start = std::upper_bound(depth_class2.begin(), depth_class2.end(), elem,
                                                            detail::compare_words);
                        auto index_end = std::upper_bound(index_start, depth_class2.end(), *elem.extra->second_child,
                                                          detail::compare_words);
                        for (auto elemptr = index_start; elemptr != index_end; ++elemptr) {
                            target_depth_class.emplace_back(&elem, &*elemptr, input_channel_size);
                        }
                    }
                }
                std::sort(target_depth_class.begin(), target_depth_class.end(), detail::compare_words);
            }
            finalise();
        }

        void LyndonWords::to_lyndon_basis(std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>& transforms,
                                          std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>& transforms_backward){

            std::vector<std::map<std::multiset<int64_t>, std::vector<LyndonWord*>>> lyndon_anagrams;
            //                   \--------------------/  \----------------------/
            //                Letters in a Lyndon word    All Lyndon words of a particular anagram class, ordered
            //                                                                lexicographically
            //          \--------------------------------------------------------/
            //                  All anagram classes of the same depth
            lyndon_anagrams.reserve(depth);
            for (s_size_type depth_index = 0; depth_index < depth; ++depth_index) {
                lyndon_anagrams.emplace_back();
            }

            std::vector<s_size_type> anagram_class_sizes;
            anagram_class_sizes.reserve(amount);
            // First go through and figure out the anagram classes
            for (s_size_type depth_index = 0; depth_index < depth; ++depth_index) {
                for (auto& lyndon_word : (*this)[depth_index]) {
                    auto& word = lyndon_word.extra->word;
                    auto& anagram_class = lyndon_anagrams[depth_index][std::multiset<int64_t> (word.begin(), word.end())];

                    anagram_class.push_back(&lyndon_word);
                    lyndon_word.extra->anagram_class = &anagram_class;

                    anagram_class_sizes.push_back(anagram_class.size());
                }
            }

            // Now go through and set where each Lyndon word appears in its anagram class. By a triangularity property
            // of Lyndon bases we can restrict our search space for anagrams.
            // Note that we couldn't do this in the above for loop because anagram_class was changing size (and thus
            // reallocating memory), so anagram_class.end() ends up becoming invalid.
            s_size_type counter = 0;
            for (auto& depth_class : *this) {
                for (auto& lyndon_word : depth_class) {
                    lyndon_word.extra->anagram_limit = lyndon_word.extra->anagram_class->begin() +
                                                       anagram_class_sizes[counter];
                    ++counter;
                }
            }

            // Make every length-one Lyndon word have itself as its own expansion (with coefficient 1)
            for (auto& lyndon_word : (*this)[0]) {
                lyndon_word.extra->expansion[lyndon_word.extra->word] = 1;
            }

            // Now unpack each bracket to find the coefficients we're interested in. This takes quite a lot of work.

            // not exact: we don't know precisely how many nontrivial anagram classes there are
            transforms.reserve(lyndon_anagrams.size());
            transforms_backward.reserve(lyndon_anagrams.size());

            transforms.emplace_back();
            transforms_backward.emplace_back();

            for (const auto& depth_class : lyndon_anagrams) {  // important to iterate by increasing depth
                for (const auto& key_value : depth_class) {
                    const std::multiset<int64_t>& letters = key_value.first;
                    if (letters.size() == 1) {
                        // The lowest level can't be decomposed into two subwords
                        continue;
                    }
                    if (transforms.back().size() != 0) {
                        transforms.emplace_back();
                        transforms_backward.emplace_back();
                    }
                    auto& transforms_back = transforms.back();
                    auto& transforms_backward_back = transforms_backward.back();
                    for (const auto& lyndon_word : key_value.second) {
                        // Record the coefficients of each word in the expansion
                        std::map<std::vector<int64_t>, int64_t> bracket_expansion;

                        const auto& first_bracket_expansion = lyndon_word->extra->first_child->extra->expansion;
                        const auto& second_bracket_expansion = lyndon_word->extra->second_child->extra->expansion;

                        // Iterate over every word in the expansion of the first element of the bracket
                        for (const auto& first_word_coeff : first_bracket_expansion) {
                            const std::vector<int64_t>& first_word = first_word_coeff.first;
                            int64_t first_coeff = first_word_coeff.second;

                            // And over every word in the expansion of the second element of the bracket
                            for (const auto& second_word_coeff : second_bracket_expansion) {
                                const std::vector<int64_t>& second_word = second_word_coeff.first;
                                int64_t second_coeff = second_word_coeff.second;

                                // And put them together to get every word in the expansion of the bracket
                                std::vector<int64_t> first_then_second = detail::concat_vectors(first_word, second_word);
                                std::vector<int64_t> second_then_first = detail::concat_vectors(second_word, first_word);


                                int64_t product = first_coeff * second_coeff;

                                // At the final depth we only need to
                                // record the coefficients of Lyndon words. At lower depths we need to record the
                                // coefficients of non-Lyndon words in case some concatenation on to them becomes a Lyndon
                                // word at higher depths.
                                if (static_cast<s_size_type>(letters.size()) < depth ||
                                    lyndon_word->is_lyndon_anagram(first_then_second)) {
                                    bracket_expansion[first_then_second] += product;
                                }
                                if (static_cast<s_size_type>(letters.size()) < depth ||
                                    lyndon_word->is_lyndon_anagram(second_then_first)) {
                                    bracket_expansion[second_then_first] -= product;
                                }
                            }
                        }

                        // Record the transformations we're interested in
                        auto end = lyndon_word->extra->anagram_class->end();
                        for (const auto& word_coeff : bracket_expansion) {
                            const std::vector<int64_t>& word = word_coeff.first;
                            int64_t coeff = word_coeff.second;

                            // Filter out non-Lyndon words. (If letters.size() == depth then we've essentially
                            // already done this above so the if statement should always be true, so we check that
                            // preferentially as it's probably faster to check. Probably - I know I know I should time it
                            // but it's not that big a deal either way...)
                            auto ptr_to_word = std::lower_bound(lyndon_word->extra->anagram_limit, end, word,
                                                                detail::compare_words);
                            if (ptr_to_word != end) {
                                if (static_cast<s_size_type>(letters.size()) == depth ||
                                    (*ptr_to_word)->extra->word == word) {
                                    transforms_back.emplace_back(lyndon_word->compressed_index,
                                                                 (*ptr_to_word)->compressed_index,
                                                                 coeff);
                                    transforms_backward_back.emplace_back(lyndon_word->tensor_algebra_index,
                                                                          (*ptr_to_word)->tensor_algebra_index,
                                                                          coeff);
                                }
                            }
                        }

                        // At the final depth then we don't need to record what we've found
                        if (static_cast<s_size_type>(letters.size()) < depth) {
                            lyndon_word->extra->expansion = std::move(bracket_expansion);
                        }
                    }
                }
            }
        }

        void LyndonWords::delete_extra() {
            for (auto& depth_class : (*this)) {
                for (auto& lyndon_word : depth_class) {
                    lyndon_word.extra.reset();
                }
            }
        }

        void LyndonWords::finalise() {
            // Used to set indices for a collection of Lyndon words. In some sense this behaviour really belongs in the
            // constructors of each individual LyndonWord (as without this function call they aren't really completely
            // initialised) but it's a boatload more efficient to do this after all the Lyndon words are generated,
            // rather than applying this to them one-by-one.
            int64_t tensor_algebra_offset = 0;
            int64_t num_words = input_channel_size;
            s_size_type compressed_offset = 0;
            for (auto& depth_class : (*this)) {
                for (s_size_type compressed_index = 0;
                     compressed_index < static_cast<s_size_type>(depth_class.size());
                     ++compressed_index) {
                    auto& lyndon_word = depth_class[compressed_index];
                    lyndon_word.tensor_algebra_index += tensor_algebra_offset;
                    lyndon_word.compressed_index = compressed_offset + compressed_index;
                }
                tensor_algebra_offset += num_words;
                num_words *= input_channel_size;
                compressed_offset += depth_class.size();
            }

            // Figure out the total amount of Lyndon words
            if (input_channel_size == 1) {
                // In this case there only exists a singe Lyndon word '0', at (*this)[0].back(). There are no
                // higher-depth words: (*this)[1], (*this)[2], ... etc. are all size-0 vectors.
                amount = 1;
            }
            else {
                amount = this->back().back().compressed_index + 1;
            }
        }

        LyndonWord::ExtraLyndonInformation::ExtraLyndonInformation(const std::vector<int64_t>& word_,
                                                                   LyndonWord* first_child_,
                                                                   LyndonWord* second_child_) :
                word{word_},
                first_child{first_child_},
                second_child{second_child_}
        {};

        LyndonWord::LyndonWord(const std::vector<int64_t>& word, bool extra, int64_t input_channel_size)
        {
            init(word, extra, nullptr, nullptr, input_channel_size);
        };

        LyndonWord::LyndonWord(LyndonWord* first_child, LyndonWord* second_child, int64_t input_channel_size)
        {
            std::vector<int64_t> word = detail::concat_vectors(first_child->extra->word, second_child->extra->word);
            init(word, true, first_child, second_child, input_channel_size);
        };

        // Checks if the given 'word' is:
        // (a) later in the lexicographic order than 'this'
        // (b) also a Lyndon word itself
        // (c) an anagram of 'this'
        bool LyndonWord::is_lyndon_anagram (const std::vector<int64_t>& word) const {
            return std::binary_search(extra->anagram_limit,  extra->anagram_class->end(), word, detail::compare_words);
        }

        // Actually performs the initialisation
        void LyndonWord::init(const std::vector<int64_t>& word, bool extra_, LyndonWord* first_child,
                              LyndonWord* second_child, int64_t input_channel_size) {
            int64_t current_stride = 1;
            for (auto word_index = word.rbegin(); word_index != word.rend(); ++word_index) {
                tensor_algebra_index += *word_index * current_stride;
                current_stride *= input_channel_size;
            }
            // We still need to add on to tensor_algebra_index the offset corresponding to number of all smaller
            // words.
            // We also need to set compressed_index, but we don't know that until we've generated all Lyndon words.
            // Thus both of these are handled by the set_indices function, called after all Lyndon words have been
            // generated.

            if (extra_) {
                // no make_unique in C++11
                extra.reset(new LyndonWord::ExtraLyndonInformation(word, first_child, second_child));
            }
        }
    }

    std::vector<std::vector<int64_t>> lyndon_words(int64_t channels, int64_t depth) {
        misc::checkargs_channels_depth(channels, depth);

        py::gil_scoped_release release;

        lyndon::LyndonWords lyndon_words(channels, depth, lyndon::LyndonWords::bracket_tag);

        std::vector<std::vector<int64_t>> lyndon_words_as_words;
        lyndon_words_as_words.reserve(lyndon_words.amount);

        for (const auto& depth_class : lyndon_words) {
            for (const auto& lyndon_word : depth_class) {
                lyndon_words_as_words.push_back(lyndon_word.extra->word);
            }
        }

        return lyndon_words_as_words;
    }

    std::vector<py::object> lyndon_brackets(int64_t channels, int64_t depth) {
        misc::checkargs_channels_depth(channels, depth);

        lyndon::LyndonWords lyndon_words(channels, depth, lyndon::LyndonWords::bracket_tag);

        std::vector<py::object> lyndon_words_as_brackets;
        lyndon_words_as_brackets.reserve(lyndon_words.amount);

        for (const auto& depth_class : lyndon_words) {
            for (const auto& lyndon_word : depth_class) {
                if (lyndon_word.extra->first_child == nullptr) {
                    lyndon_words_as_brackets.emplace_back(py::cast(lyndon_word.extra->word.back()));
                }
                else {
                    // Using the property that compressed_index corresponds to the order in which we iterate over them
                    const py::object& first_child = lyndon_words_as_brackets[lyndon_word.extra->first_child
                                                                             ->compressed_index];
                    const py::object& second_child = lyndon_words_as_brackets[lyndon_word.extra->second_child
                                                                              ->compressed_index];
                    // Why a list, you might ask? After all, it has to be a pair of just two elements, so a tuple is
                    // a better fit.
                    // And I completely agree.
                    // Except that lists use square [] brackets and tuples use round () brackets, and the commutators
                    // that these object represent are traditionally written with square [] brackets, so this looks more
                    // immediately understandable to any mathematician looking at this.
                    // Possibly one of the odder reasons anyone has ever had for how they chose to represent their data.
                    py::list lyndon_bracket;
                    lyndon_bracket.append(first_child);
                    lyndon_bracket.append(second_child);
                    lyndon_words_as_brackets.push_back(std::move(lyndon_bracket));
                }
            }
        }
        return lyndon_words_as_brackets;
    }

    std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> lyndon_words_to_basis_transform(int64_t channels,
                                                                                                    int64_t depth)
    {
        misc::checkargs_channels_depth(channels, depth);

        py::gil_scoped_release release;

        lyndon::LyndonWords lyndon_words(channels, depth, lyndon::LyndonWords::bracket_tag);
        std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms;
        std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> transforms_backward;
        lyndon_words.to_lyndon_basis(transforms, transforms_backward);
        return transforms;
    }
}  // namespace signatory