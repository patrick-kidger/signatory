#include "src/extension.hpp"
#include <iostream>

#include "test.hpp"

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}