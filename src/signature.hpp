#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP


namespace signatory{
    namespace detail{
    }
    int signature_channels(int input_channels, int depth);
    std::vector<torch::Tensor> signature(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten,
            bool batch_first);
}

#endif //TORCHTEST_SRC_SIGNATURE_HPP
