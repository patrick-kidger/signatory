#include <torch/extension.h>
#include <iostream>

int main() {
    std::cout << "Hello, Worldl" << std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    return 0;
}