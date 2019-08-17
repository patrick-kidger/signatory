#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "free_lie_algebra_ops.hpp"
#include "misc.hpp"
#include "utilities.hpp"

namespace signatory {
    int64_t signature_channels(int64_t input_channels, int64_t depth) {
        if (input_channels < 1) {
            throw std::invalid_argument("input_channels must be at least 1");
        }
        if (depth < 1) {
            throw std::invalid_argument("depth must be at least 1");
        }

        if (input_channels == 1) {
            return depth;
        }
        else {
            return input_channels * ((pow(input_channels, depth) - 1) / (input_channels - 1));
        }
    }

    std::vector<std::vector<int64_t>> lyndon_words(int64_t channels, int64_t depth) {
        fla_ops::LyndonWords lyndon_words(misc::LyndonSpec {channels, depth}, fla_ops::LyndonWords::bracket_tag);

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
        fla_ops::LyndonWords lyndon_words(misc::LyndonSpec {channels, depth}, fla_ops::LyndonWords::bracket_tag);

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

    std::vector<std::tuple<int64_t, int64_t, int64_t>> lyndon_words_to_basis_transform(int64_t channels, int64_t depth)
    {
        fla_ops::LyndonWords lyndon_words(misc::LyndonSpec {channels, depth}, fla_ops::LyndonWords::bracket_tag);
        std::vector<std::tuple<int64_t, int64_t, int64_t>> transforms;
        std::vector<std::tuple<int64_t, int64_t, int64_t>> transforms_backward;
        lyndon_words.to_lyndon_basis(transforms, transforms_backward);
        return transforms;
    }
}  // namespace signatory