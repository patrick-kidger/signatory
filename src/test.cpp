#include "extension.hpp"
#include <iostream>

#include "test.hpp"

Tensor d_sigmoid(Tensor z) {
    auto s = sigmoid(z);
    return (1 - s) * s;
}