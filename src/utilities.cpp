#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument

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
}  // namespace signatory