#ifndef TORCHTEST_SRC_SIGNATURE_HPP
#define TORCHTEST_SRC_SIGNATURE_HPP

namespace signatory{
    namespace detail{
        torch::Tensor compute_increments(torch::Tensor path, bool basepoint);
        void compute_first_term(torch::Tensor path, torch::Tensor nth_term, bool basepoint);
        void compute_nth_term(torch::Tensor n_minus_one_th_term, torch::Tensor path_increments,
                                       torch::Tensor nth_term);
    }
    std::vector<torch::Tensor> signature(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten);
}

#endif //TORCHTEST_SRC_SIGNATURE_HPP
