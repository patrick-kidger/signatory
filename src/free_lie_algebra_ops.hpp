#ifndef SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP
#define SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <memory>     // std::unique_ptr
#include <tuple>      // std::tuple
#include <vector>     // std::vector

#include "misc.hpp"


namespace signatory { namespace fla_ops {
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

        /* Implements Duval's algorithm for generating Lyndon words (without their standard bracketing)
         * J.-P. Duval, Theor. Comput. Sci. 1988, doi:10.1016/0304-3975(88)90113-2.
         */
        LyndonWords(const misc::LyndonSpec& lyndonspec, WordTag);

        /* Generates Lyndon words with their standard bracketing. No reference for this algorithm I'm afraid,
         * I made it up myself.
         */
        LyndonWords(const misc::LyndonSpec& lyndonspec, BracketTag);

        using base = std::vector<std::vector<LyndonWord>>;
        using base::operator[];
        using base::begin;
        using base::end;

        /* Computes the transforms that need to be applied to the coefficients of the Lyndon words to produce the
         * coefficients of the Lyndon basis.
         * The transforms are returned in the transforms argument.
         */
        void to_lyndon_basis(std::vector<std::tuple<int64_t, int64_t, int64_t>>& transforms);

        int64_t amount;
    private:
        void finalise();

        misc::LyndonSpec lyndonspec;
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
            std::vector<int64_t> word;
            LyndonWord* first_child;
            LyndonWord* second_child;

            friend class LyndonWords;
            friend class LyndonWord;
        private:
            // Information set once all Lyndon words are known.
            std::vector<LyndonWord*>* anagram_class;
            std::vector<LyndonWord*>::iterator anagram_limit;
            std::map<std::vector<int64_t>, int64_t> expansion;
        };

        LyndonWord(const std::vector<int64_t>& word, bool extra, const misc::LyndonSpec& lyndonspec);
        LyndonWord(LyndonWord* first_child, LyndonWord* second_child, const misc::LyndonSpec& lyndonspec);

        /* The index of this element in the sequence of all Lyndon words i.e. given some lyndonspec and some tag:
         *
         * LyndonWords lyndon_words(lyndonspec, tag);
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

        friend class LyndonWords;
    private:
        bool is_lyndon_anagram (const std::vector<int64_t>& word) const;
        void init(const std::vector<int64_t>& word, bool extra_, LyndonWord* first_child,
                  LyndonWord* second_child, const misc::LyndonSpec& lyndonspec);
    };

    // Compresses a representation of a member of the free Lie algebra.
    // In the tensor algebra it is represented by coefficients of all words. This just extracts the coefficients of
    // all the Lyndon words.
    torch::Tensor compress(const LyndonWords& lyndon_words, torch::Tensor input,
                           const misc::SigSpec& sigspec);

    torch::Tensor compress_backward(torch::Tensor grad_logsignature, const misc::SigSpec& sigspec);
}  /* namespace signatory::fla_ops */ }  // namespace signatory

#endif //SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP
