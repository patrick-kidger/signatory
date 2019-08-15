#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <tuple>      // std::tie, std::tuple
#include <vector>     // std::vector

#include "free_lie_algebra_ops.hpp"
#include "logsignature.hpp"
#include "misc.hpp"
#include "signature.hpp"
#include "tensor_algebra_ops.hpp"


namespace signatory {
    std::tuple<torch::Tensor, py::object>
    logsignature_forward(torch::Tensor path, s_size_type depth, bool stream, bool basepoint,
                         torch::Tensor basepoint_value, LogSignatureMode mode) {
        if (depth == 1) {
            return signature_forward(path, depth, stream, basepoint, basepoint_value);
        }  // this isn't just a fast return path: we also can't index the reciprocals tensor if depth == 1, so we'd need
        // faffier code below - and it's already quite faffy enough

        // first call the regular signature
        torch::Tensor signature;
        py::object backwards_info_capsule;
        std::tie(signature, backwards_info_capsule) = signature_forward(path, depth, stream, basepoint,
                                                                        basepoint_value);

        // unpack sigspec
        misc::BackwardsInfo* backwards_info = misc::get_backwards_info(backwards_info_capsule);
        const misc::SigSpec& sigspec = backwards_info->sigspec;

        // undo the transposing we just did in signature_forward...
        signature = misc::transpose_reverse(signature, sigspec);

        // organise the memory into a vector
        std::vector<torch::Tensor> signature_vector;
        misc::slice_by_term(signature, signature_vector, sigspec.output_channel_dim, sigspec);

        // and allocate memory for the logsignature
        // TODO: only invert the lowest terms? The higher terms aren't used?
        torch::Tensor logsignature = signature * ta_ops::log_coefficient_at_depth(depth - 2, sigspec);
        std::vector<torch::Tensor> logsignature_vector;
        misc::slice_by_term(logsignature, logsignature_vector, sigspec.output_channel_dim, sigspec);

        if (stream) {
            // allocate vectors for the signature and logsignature by stream index
            std::vector<torch::Tensor> signature_stream_vector;
            std::vector<torch::Tensor> logsignature_stream_vector;
            for (int64_t stream_index = 0; stream_index < sigspec.output_stream_size; ++stream_index) {
                misc::slice_at_stream(signature_vector, signature_stream_vector, stream_index);
                misc::slice_at_stream(logsignature_vector, logsignature_stream_vector, stream_index);
                ta_ops::compute_log(logsignature_stream_vector, signature_stream_vector, sigspec);
            }
        }
        else {
            ta_ops::compute_log(logsignature_vector, signature_vector, sigspec);
        }

        // Brackets and Words are the two possible compressed forms of the logsignature. So here we perform the
        // compression.
        std::vector<std::tuple<int64_t, int64_t, int64_t>> transforms;
        if (mode == LogSignatureMode::Words) {
            fla_ops::LyndonWords lyndon_words(sigspec, fla_ops::LyndonWords::word_tag);
            logsignature = fla_ops::compress(lyndon_words, logsignature, sigspec);
        }
        else if (mode == LogSignatureMode::Brackets){
            fla_ops::LyndonWords lyndon_words(sigspec, fla_ops::LyndonWords::bracket_tag);
            logsignature = fla_ops::compress(lyndon_words, logsignature, sigspec);

            // First find all the transforms
            lyndon_words.to_lyndon_basis(transforms);
            // Then apply the transforms. We rely on the triangularity property of the Lyndon basis for this to work.
            for (const auto& transform : transforms) {
                int64_t source_index = std::get<0>(transform);
                int64_t target_index = std::get<1>(transform);
                int64_t coefficient = std::get<2>(transform);
                torch::Tensor source = logsignature.narrow(/*dim=*/sigspec.output_channel_dim,
                        /*start=*/source_index,
                        /*length=*/1);
                torch::Tensor target = logsignature.narrow(/*dim=*/sigspec.output_channel_dim,
                        /*start=*/target_index,
                        /*length=*/1);
                target.sub_(source, coefficient);
            }
        }

        // I'm not an experienced enough C++ programmer to know if these moves are actually helpful here.
        // backwards_info points to a heap object and signature_vector et al are stack objects. Does that mean that a
        // copy needs to be performed anyway, to move from stack to heap?
        // All the things being moved are vectors, which hold the nontrivial part of their memory on the heap anyway, so
        // this probably shouldn't be a performance issue, as the already-on-the-heap elements of the vector certainly
        // won't need to be copied.
        backwards_info->set_logsignature_data(std::move(signature_vector),
                                              std::move(transforms),
                                              mode,
                                              logsignature.size(sigspec.output_channel_dim));

        logsignature = misc::transpose(logsignature, sigspec);
        return {logsignature, backwards_info_capsule};
    }

    std::tuple<torch::Tensor, torch::Tensor>
    logsignature_backward(torch::Tensor grad_logsignature, py::object backwards_info_capsule) {
        // Unpack sigspec
        misc::BackwardsInfo* backwards_info = misc::get_backwards_info(backwards_info_capsule);
        const misc::SigSpec& sigspec = backwards_info->sigspec;
        if (sigspec.depth == 1) {
            return signature_backward(grad_logsignature, backwards_info_capsule);
        }

        // Unpack everything else from backwards_info
        torch::Tensor signature = backwards_info->out;
        const std::vector<torch::Tensor>& signature_vector = backwards_info->signature_vector;
        const std::vector<std::tuple<int64_t, int64_t, int64_t>>& transforms = backwards_info->transforms;
        LogSignatureMode mode = backwards_info->mode;
        int64_t logsignature_channels = backwards_info->logsignature_channels;

        misc::checkargs_backward(grad_logsignature, sigspec, logsignature_channels);

        grad_logsignature = misc::transpose_reverse(grad_logsignature, sigspec);
        if (!grad_logsignature.is_floating_point()) {
            grad_logsignature = grad_logsignature.to(torch::kFloat32);
        }

        // Decompress the logsignature
        if (mode == LogSignatureMode::Expand) {
            grad_logsignature = grad_logsignature.clone();  // Clone so we don't leak changes through grad_logsignature.
        }
        else if (mode == LogSignatureMode::Words){
            // Don't need to clone grad_logsignature as it gets put into new memory when decompressing

            grad_logsignature = fla_ops::compress_backward(grad_logsignature, sigspec);
        }
        else {  // mode == LogSignatureMode::Brackets
            // TODO: have transforms record the tensor algebra indices as well so we can do this after decompressing
            //       and not need to clone here
            grad_logsignature = grad_logsignature.clone();  // Clone so we don't leak changes through grad_logsignature.

            for (auto tptr = transforms.rbegin(); tptr != transforms.rend(); ++tptr) {
                int64_t source_index = std::get<0>(*tptr);
                int64_t target_index = std::get<1>(*tptr);
                int64_t coefficient = std::get<2>(*tptr);
                torch::Tensor grad_source = grad_logsignature.narrow(/*dim=*/sigspec.output_channel_dim,
                        /*start=*/source_index,
                        /*length=*/1);
                torch::Tensor grad_target = grad_logsignature.narrow(/*dim=*/sigspec.output_channel_dim,
                        /*start=*/target_index,
                        /*length=*/1);
                grad_source.sub_(grad_target, coefficient);
            }

            grad_logsignature = fla_ops::compress_backward(grad_logsignature, sigspec);
        }

        // Our old friend.
        // Memory management.
        torch::Tensor grad_signature = torch::zeros_like(grad_logsignature);
        torch::Tensor scratch = torch::empty({sigspec.output_channels, sigspec.batch_size}, sigspec.opts);
        std::vector<torch::Tensor> grad_logsignature_vector;
        std::vector<torch::Tensor> grad_signature_vector;
        std::vector<torch::Tensor> scratch_vector;
        misc::slice_by_term(grad_logsignature, grad_logsignature_vector, sigspec.output_channel_dim, sigspec);
        misc::slice_by_term(grad_signature, grad_signature_vector, sigspec.output_channel_dim, sigspec);
        misc::slice_by_term(scratch, scratch_vector, sigspec.output_channel_dim, sigspec);

        if (sigspec.stream) {
            // allocate vectors for the signature and logsignature by stream index
            std::vector<torch::Tensor> grad_logsignature_stream_vector;
            std::vector<torch::Tensor> grad_signature_stream_vector;
            std::vector<torch::Tensor> signature_stream_vector;
            for (int64_t stream_index = 0; stream_index < sigspec.output_stream_size; ++stream_index) {
                misc::slice_at_stream(grad_logsignature_vector, grad_logsignature_stream_vector, stream_index);
                misc::slice_at_stream(grad_signature_vector, grad_signature_stream_vector, stream_index);
                misc::slice_at_stream(signature_vector, signature_stream_vector, stream_index);
                torch::Tensor signature_at_stream = signature.narrow(/*dim=*/0,
                        /*start=*/stream_index,
                        /*len=*/1).squeeze(0);

                ta_ops::compute_log_backward(grad_logsignature_stream_vector, grad_signature_stream_vector,
                                             scratch_vector, signature_stream_vector, scratch,
                                             signature_at_stream, sigspec);
            }
        }
        else {
            ta_ops::compute_log_backward(grad_logsignature_vector, grad_signature_vector, scratch_vector,
                                         signature_vector, scratch, signature, sigspec);
        }

        grad_signature.add_(grad_logsignature, ta_ops::log_coefficient_at_depth(sigspec.depth - 2, sigspec));

        grad_signature = misc::transpose(grad_signature, sigspec);
        return signature_backward(grad_signature, backwards_info_capsule, false);
    }
}  // namespace signatory