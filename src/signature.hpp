#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP

#include <cstdint>    // int64_t


namespace signatory{
    struct SigSpec {
        // Encapsulates all the things that aren't tensors
        SigSpec(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten);
        int64_t input_channels;
        int64_t input_stream_size;
        int64_t batch_size;
        int64_t output_channels;
        int64_t output_stream_size;
        int depth;
        int n_output_dims;
        bool basepoint;
        bool stream;
        bool flatten;
        torch::TensorOptions opts;
    };

    int64_t signature_channels(int64_t input_channels, int depth);

    std::tuple<std::vector<torch::Tensor>,
               std::vector<torch::Tensor>,
               torch::Tensor,
               SigSpec>
    signature_forward(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten);

    torch::Tensor signature_backward(std::vector<torch::Tensor> grad_out_vector,
                                     std::vector<torch::Tensor> out_vector, torch::Tensor path_increments,
                                     SigSpec sigspec, int depth, bool basepoint, bool stream, bool flatten);
}

#endif //TORCHTEST_SRC_SIGNATURE_HPP
