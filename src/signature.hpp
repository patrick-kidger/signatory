#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP


namespace signatory{
    int signature_channels(int input_channels, int depth);

    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> signature(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten);

    // TODO: remove
    torch::Tensor signature_backward(std::vector<torch::Tensor> grad_out_vector, std::vector<torch::Tensor> out_vector,
            /*torch::Tensor path_increments, */torch::Tensor path, int depth,
                                     bool basepoint, bool stream, bool flatten);
}

#endif //TORCHTEST_SRC_SIGNATURE_HPP
