#ifndef SIGNATORY_UTILITIES_HPP
#define SIGNATORY_UTILITIES_HPP

#include <cstdint>    // int64_t


namespace signatory {
    int64_t signature_channels(int64_t input_channels, int64_t depth);

    std::vector<std::vector<int64_t>> lyndon_words(int64_t channels, int64_t depth);

    std::vector<py::object> lyndon_brackets(int64_t channels, int64_t depth);

    std::vector<std::tuple<int64_t, int64_t, int64_t>> lyndon_words_to_basis_transform(int64_t channels, int64_t depth);
}  // namespace signatory

#endif //SIGNATORY_UTILITIES_HPP
