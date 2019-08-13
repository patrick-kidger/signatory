#ifndef SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP
#define SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <memory>     // std::unique_ptr
#include <tuple>      // std::tuple
#include <vector>     // std::vector

#include "misc.hpp"


namespace signatory {
    namespace fla_ops {
        // Represents a single Lyndon word. It is primarily represented by a pair of indices, corresponding to how far
        // into the list of all words and the list of all Lyndon words it is. (In both cases ordered by depth and then
        // by lexicographic order.)
        struct LyndonWord {
            struct ExtraLyndonInformation;
            struct ExtraLyndonInformationDeleter { void operator()(ExtraLyndonInformation* p); };

            LyndonWord(const std::vector<int64_t>& word, bool extra, const misc::SigSpec& sigspec);
            LyndonWord(LyndonWord* first_child, LyndonWord* second_child, const misc::SigSpec& sigspec);

            void init(const std::vector<int64_t>& word, bool extra_, LyndonWord* first_child,
                      LyndonWord* second_child, const misc::SigSpec& sigspec);

            size_type compressed_index;
            int64_t tensor_algebra_index {0};
            std::unique_ptr<ExtraLyndonInformation, ExtraLyndonInformationDeleter> extra {nullptr};
        };

        // Implements Duval's algorithm for generating Lyndon words
        // J.-P. Duval, Theor. Comput. Sci. 1988, doi:10.1016/0304-3975(88)90113-2.
        void lyndon_word_generator(std::vector<std::vector<LyndonWord>>& lyndon_words, const misc::SigSpec& sigspec);
            /*                                             \--------/
             *                                         A single Lyndon word
             *
             *                                 \---------------------/
             *            All Lyndon words of a particular depth, ordered lexicographically
             *
             *                     \----------------------------------/
             *               All Lyndon words of all depths, ordered by depth
             *
             * Duval's algorithm produces words of the same depth in lexicographic order, but words of different depths
             * are muddled together. So in order to recover the full lexicographic order we put them into a bin
             * corresponding to the depth of each generated word.
            */

        // Generates Lyndon words with their standard bracketing. No reference for this algorithm I'm afraid, I made it
        // up myself.
        void lyndon_bracket_generator(std::vector<std::vector<LyndonWord>>& lyndon_words, const misc::SigSpec& sigspec);

        // Computes the transforms that need to be applied to the coefficients of the Lyndon words to produce the
        // coefficients of the Lyndon basis.
        // The transforms are returned in the transforms argument.
        void lyndon_words_to_lyndon_basis(std::vector<std::vector<LyndonWord>>& lyndon_words,
                                          std::vector<std::tuple<int64_t, int64_t, int64_t>>& transforms,
                                          const misc::SigSpec& sigspec);

        // Compresses a representation of a member of the free Lie algebra.
        // In the tensor algebra it is represented by coefficients of all words. This just extracts the coefficients of
        // all the Lyndon words.
        torch::Tensor compress(const std::vector<std::vector<LyndonWord>>& lyndon_words, torch::Tensor input,
                               const misc::SigSpec& sigspec);

        torch::Tensor compress_backward(torch::Tensor grad_logsignature, const misc::SigSpec& sigspec);
    }  // namespace signatory::fla_ops
}  // namespace signatory

#endif //SIGNATORY_FREE_LIE_ALGEBRA_OPS_HPP
