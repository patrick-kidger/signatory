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
    namespace lyndon {
        struct LyndonWord;

        /* Represents all possible Lyndon words up to a certain order, for a certain alphabet.
         *
         *                       All Lyndon words of all depths, ordered by depth
         *                           /----------------------------------\
         *                    All Lyndon words of a particular depth, ordered lexicographically
         *                                       /---------------------\                          */
        struct LyndonWords : private std::vector<std::vector<LyndonWord>> {
            constexpr static struct WordTag {} word_tag {};
            constexpr static struct BracketTag {} bracket_tag {};

            /* Implements Duval's algorithm for generating Lyndon words
             * J.-P. Duval, Theor. Comput. Sci. 1988, doi:10.1016/0304-3975(88)90113-2.
             * Each LyndonWord we produce does _not_ have any ExtraLyndonInformation set.
             */
            LyndonWords(int64_t input_channel_size, s_size_type depth, WordTag);

            /* Generates Lyndon words with their standard bracketing. No reference for this algorithm I'm afraid,
             * I made it up myself.
             * Each LyndonWord we produce _does_ have ExtraLyndonInformation set. After it has been used for whatever,
             * consider calling LyndonWords::delete_extra(), to reclaim the memory corresponding to
             * ExtraLyndonInformation.
             */
            LyndonWords(int64_t input_channel_size, s_size_type depth, BracketTag);

            using std::vector<std::vector<LyndonWord>>::operator[];
            using std::vector<std::vector<LyndonWord>>::begin;
            using std::vector<std::vector<LyndonWord>>::end;

            /* Computes the transforms that need to be applied to the coefficients of the Lyndon words to produce the
             * coefficients of the Lyndon basis.
             * The transforms are returned in the transforms argument.
             */
            void to_lyndon_basis(std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>& transforms,
                                 std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>>& transforms_backward);

            /* Deletes the ExtraLyndonInformation associated with each word, if it is present. This is to reclaim memory
             * when we know we don't need it any more.
             */
            void delete_extra();

            // Should be treated as read-only attributes.
            int64_t amount;
            int64_t input_channel_size;
            s_size_type depth;
        private:
            void finalise();
        };

        /* Represents a single Lyndon word. It is primarily represented by a pair of indices, corresponding to how
         * far into the list of all words and the list of all Lyndon words it is. (In both cases ordered by depth
         * and then by lexicographic order.)
         */
        struct LyndonWord {
            /* Stores extra information about Lyndon words for when we want to perform calculations beyond simply
             * extracting coefficients: these are set when using the bracket-based constructor for LyndonWords.
             */
            struct ExtraLyndonInformation {
                ExtraLyndonInformation(const std::vector<int64_t>& word_,
                                       LyndonWord* first_child_,
                                       LyndonWord* second_child_);

                // Information set at creation time in LyndonWords(..., LyndonWords::bracket_tag)
                std::vector<int64_t> word;  // The word itself
                LyndonWord* first_child;    // The first part of its standard bracketing
                LyndonWord* second_child;   // The second part of its standard bracketing

                friend struct LyndonWords;
                friend struct LyndonWord;
            private:
                // Information set once all Lyndon words are known. At present it is set and used exclusively in
                // LyndonWords::to_lyndon_basis.
                std::vector<LyndonWord*>* anagram_class;
                std::vector<LyndonWord*>::iterator anagram_limit;
                std::map<std::vector<int64_t>, int64_t> expansion;
            };

            // Constructor for LyndonWords(..., LyndonWords::word_tag) (with extra==false) and
            // constructor for LyndonWords(..., LyndonWords::bracket_tag) for the depth == 1 words (with extra==true).
            LyndonWord(const std::vector<int64_t>& word, bool extra, int64_t input_channel_size);

            // Constructor for LyndonWords(..., LyndonWords::bracket_tag) for the depth > 1 words.
            LyndonWord(LyndonWord* first_child, LyndonWord* second_child, int64_t input_channel_size);

            /* The index of this element in the sequence of all Lyndon words i.e. given some input_channel_size, depth,
             * and tag:
             *
             * LyndonWords lyndon_words(input_channel_size, depth, tag);
             * s_size_type counter = 0
             * for (auto& depth_class : lyndon_words) {
             *     for (auto& lyndon_word : depth_class) {
             *         lyndon_word.compressed_index == counter;
             *     }
             * }
             */
            s_size_type compressed_index;

            // The index of this element in the sequence of all words (not necessarily Lyndon).
            int64_t tensor_algebra_index {0};

            std::unique_ptr<ExtraLyndonInformation> extra {nullptr};

            friend struct LyndonWords;
        private:
            bool is_lyndon_anagram (const std::vector<int64_t>& word) const;
            void init(const std::vector<int64_t>& word, bool extra_, LyndonWord* first_child, LyndonWord* second_child,
                      int64_t input_channel_size);
        };
    }  // namespace signatory::lyndon

    // See signatory.lyndon_words for documentation
    std::vector<std::vector<int64_t>> lyndon_words(int64_t channels, int64_t depth);

    // See signatory.lyndon_brackets for documentation
    std::vector<py::object> lyndon_brackets(int64_t channels, int64_t depth);

    // See signatory.utility.lyndon_words_to_basis_transform for documentation
    std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> lyndon_words_to_basis_transform(int64_t channels,
                                                                                                    int64_t depth);
}  // namespace signatory

#endif //SIGNATORY_LYNDON_HPP
