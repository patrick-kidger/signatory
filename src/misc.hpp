#ifndef SIGNATORY_MISC_HPP
#define SIGNATORY_MISC_HPP

#include <torch/extension.h>
#include <cstdint>      // int64_t
#include <stdexcept>    // std::invalid_argument
#include <tuple>        // std::tuple
#include <type_traits>  // std::make_signed, std::make_unsigned
#include <vector>       // std::vector


namespace signatory {
    // Modes for the return value of logsignature
    enum class LogSignatureMode { Expand, Brackets, Words };

    // signed-ness is important because we'll sometimes iterate downwards
    using size_type = std::make_signed<std::vector<torch::Tensor>::size_type>::type;
    using u_size_type = std::make_unsigned<size_type>::type;

    namespace misc {
        // Encapsulates all the things that aren't tensors
        struct SigSpec {
            SigSpec(torch::Tensor path, size_type depth, bool stream, bool basepoint);

            torch::TensorOptions opts;
            int64_t input_stream_size;
            int64_t input_channels;
            int64_t batch_size;
            int64_t output_stream_size;
            int64_t output_channels;
            int64_t output_channel_dim{-2};  // always -2 but provided here for clarity
            int64_t n_output_dims;
            size_type depth;
            torch::Tensor reciprocals;
            bool stream;
            bool basepoint;
        };

        // Argument 'in' is assumed to be a tensor for which one dimension has size equal to sigspec.output_channels
        // It is sliced up along that dimension, specified by 'dim', and the resulting tensors placed into 'out'.
        // Each resulting tensor corresponds to one of the (tensor, not scalar) terms in the signature.
        void slice_by_term(torch::Tensor in, std::vector <torch::Tensor> &out, int64_t dim, const SigSpec &sigspec);

        // Argument 'in' is assumed to be a tensor for which its first dimension corresponds to the stream dimension.
        // Its slices along a particular index of that dimension are put in 'out'.
        void slice_at_stream(std::vector <torch::Tensor> in, std::vector <torch::Tensor> &out, int64_t stream_index);

        // Convert from internally-used axis ordering to externally-visible axis ordering
        torch::Tensor transpose(torch::Tensor tensor, const SigSpec &sigspec);

        // Convert from externally-visible axis ordering to internally-used axis ordering
        torch::Tensor transpose_reverse(torch::Tensor tensor, const SigSpec &sigspec);

        bool is_even(size_type index);

        // Retains information needed for the backwards pass.
        struct BackwardsInfo{
            BackwardsInfo(SigSpec&& sigspec, std::vector<torch::Tensor>&& out_vector, torch::Tensor out,
                          torch::Tensor path_increments);

            void set_logsignature_data(std::vector<torch::Tensor>&& signature_vector_,
                                       std::vector<std::tuple<int64_t, int64_t, int64_t>>&& transforms_,
                                       LogSignatureMode mode_,
                                       int64_t logsignature_channels_);

            SigSpec sigspec;
            std::vector<torch::Tensor> out_vector;
            torch::Tensor out;
            torch::Tensor path_increments;

            std::vector<torch::Tensor> signature_vector;  // will be the same as out_vector when computing logsignatures
                                                          // with stream==true. But we provide a separate vector here
                                                          // for a consistent interface with the stream==false case as
                                                          // well.
            std::vector<std::tuple<int64_t, int64_t, int64_t>> transforms;
            LogSignatureMode mode;
            int64_t logsignature_channels;
        };

        // Makes a BackwardsInfo object and wraps it into a PyCapsule and wraps that into a py::object
        py::object make_backwards_info(std::vector<torch::Tensor>& out_vector, torch::Tensor out,
                                       torch::Tensor path_increments, const SigSpec& sigspec);

        // Unwraps a py::object to unwrap a PyCapsule to get a BackwardsInfo object
        BackwardsInfo* get_backwards_info(py::object backwards_info_capsule);

        // Checks the arguments for the forwards pass
        void checkargs(torch::Tensor path, size_type depth, bool basepoint, torch::Tensor basepoint_value);

        // Checks the arguments for the backwards pass. Only grad_out is checked to make sure it is as expected.
        // The objects we get from the PyCapsule-wrapped BackwardsInfo object are assumed to be correct.
        void checkargs_backward(torch::Tensor grad_out, const SigSpec& sigspec, int64_t num_channels=-1);
    }  // namespace signatory::misc
}  // namespace signatory

#endif //SIGNATORY_MISC_HPP
