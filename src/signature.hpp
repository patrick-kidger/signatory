#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP


namespace signatory{
    struct SigSpec {
        // Encapsulates all the things that aren't tensors
        SigSpec(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten);
        int input_channels;
        int input_stream_size;
        int batch_size;
        int output_channels;
        int output_stream_size;
        int depth;
        int n_output_dims;
        bool basepoint;
        bool stream;
        bool flatten;
        torch::TensorOptions opts;
    };

    int signature_channels(int input_channels, int depth);

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
