#include "extension.hpp"
#include <cmath>      // pow
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "signature.hpp"

// TODO: test rearranging dimensions
// TODO: benchmark against iisignature

namespace signatory {
    namespace detail {
        torch::Tensor compute_increments(torch::Tensor path, bool basepoint) {
            if (basepoint) {
                torch::Tensor path_increments = path.clone();
                path_increments.narrow(/*dim=*/2, /*start=*/1, /*length=*/path.size(2) - 1) -=
                           path.narrow(/*dim=*/2, /*start=*/0, /*length=*/path.size(2) - 1);
                return path_increments;
            }
            else {
                return path.narrow(/*dim=*/2, /*start=*/1, /*length=*/path.size(2) - 1) -
                       path.narrow(/*dim=*/2, /*start=*/0, /*length=*/path.size(2) - 1);
            }
        }

        void compute_first_term(torch::Tensor path, torch::Tensor nth_term, bool basepoint) {
            if (basepoint) {
                nth_term.copy_(path);
            }
            else {
                nth_term.copy_(path.narrow(/*dim=*/2, /*start=*/1, /*length=*/path.size(2) - 1));
                nth_term -= path.narrow(/*dim=*/2, /*start=*/0, /*length=*/1).expand({path.size(0), path.size(1), path.size(2) - 1});
            }
        }

        void compute_nth_term(torch::Tensor n_minus_one_th_term, torch::Tensor path_increments,
                                       torch::Tensor nth_term) {
            // We want to take a tensor of shape (batch, d^(n-1), stream) and a tensor of shape (batch, d, stream), and
            // perform every possible multiplication along the middle dimension, to get a tensor of shape
            // (batch, d^n, stream). This is equivalent to taking an outer product between the two middle dimensions and
            // then flattening the result.
            nth_term = torch::einsum("bjs,bks->bjks", {n_minus_one_th_term, path_increments}).flatten(1, 2);
            // sum down stream axis
            torch::cumsum_out(/*out=*/nth_term, /*in=*/nth_term, /*dim=*/2);
        }
    }  // namespace signatory::detail

    // It'd be nice to return std::variant<torch::Tensor, std::vector<torch::Tensor>> here instead, based on whether
    // 'flatten' is true or false respectively, rather than wrapping it into a vector like we actually do.
    // Sadly we're targeting C++11 so instead it is the caller's responsibility to unwrap the single-element vector if
    // they call this function with flatten=true.
    std::vector<torch::Tensor> signature(torch::Tensor path, int depth, bool basepoint=false, bool stream=false,
                                         bool flatten=true) {
        if (depth < 1) {
            throw new std::invalid_argument("Depth must be an integer greater than or equal to one.");
        }
        path.accessor<float, 3>();  // check dtype and shape

        torch::Tensor path_increments = detail::compute_increments(path, basepoint);

        torch::Tensor out;
        torch::Tensor nth_term;
        int feature_size = path.size(1) * ((pow(path.size(1), depth) - 1) / (path.size(1) - 1));
        if (!stream || !flatten) {
            // In this branch we create new tensors to store the results in.
            // (If !stream then we'll have to make another copy later to make sure we discard what is then irrelevant
            // information, but either way for now, we'll just assign this memory...)
            nth_term = torch::empty({path.size(0), path.size(1), path.size(2) - (basepoint ? 0 : 1)},
                                    torch::TensorOptions().dtype(path.dtype()).device(path.device()));
        }
        else {
            // ...however in the special case of stream && flatten, then we can save ourselves a boatload of copying by
            // using the same memory for both computation and output, so we special-case this.
            out = torch::empty({path.size(0), feature_size, path.size(2) - (basepoint ? 0 : 1)},
                               torch::TensorOptions().dtype(path.dtype()).device(path.device()));

            nth_term = out.narrow(/*dim=*/1, /*start=*/0, /*length=*/path.size(1));
        }

        detail::compute_first_term(path, nth_term, basepoint);

        std::vector<torch::Tensor> out_vector;
        if (!stream || !flatten) {
            out_vector.reserve(depth);
            out_vector.push_back(nth_term);
        } // don't bother in the stream && flatten special case; we're just going to return the 'out' memory anyway.

        int current_memory_pos = path.size(1);
        int current_length = current_memory_pos * path.size(1);
        torch::Tensor n_minus_one_th_term;
        for (int n = 2; n <= depth; ++n) {
            n_minus_one_th_term = nth_term;

            if (!stream || !flatten) {
                nth_term = torch::empty({path.size(0), current_length, path.size(2) - (basepoint ? 0 : 1)},
                                        torch::TensorOptions().dtype(path.dtype()).device(path.device()));
            }
            else{
                nth_term = out.narrow(/*dim=*/1, /*start=*/current_memory_pos, /*length=*/current_length);
                current_memory_pos += current_length;
            }
            current_length *= path.size(1);

            detail::compute_nth_term(n_minus_one_th_term, path_increments, nth_term);

            if (!stream || !flatten) {
                out_vector.push_back(nth_term);
            }
        }

        if (stream) {
            if (flatten) {
                return std::vector<torch::Tensor> {out};
            }
            else {
                return out_vector;
            }
        }
        else {
            // If we're not returning a stream then we want to make a copy of the data before we return it. This is
            // because in this case, we only want to return the set of elements which is at the very end of the stream
            // dimension - but if we don't make a copy then we're still using the same underlying storage that was used
            // for the whole thing, stream dimension included, and we can't free up that storage until we're no longer
            // using any of it. So this takes a little extra time but saves a lot of memory.

            if (flatten) {
                torch::Tensor out_flattened = torch::empty({path.size(0), feature_size},
                                                           torch::TensorOptions().dtype(path.dtype()).device(path.device()));
                current_memory_pos = 0;
                current_length = path.size(1);
                torch::Tensor trimmed_tensor_element;
                for (auto tensor_element : out_vector) {
                    trimmed_tensor_element = tensor_element.narrow(/*dim=*/2, /*start=*/tensor_element.size(2) - 1, /*length=*/1).squeeze(2);
                    out_flattened.narrow(/*dim=*/1, /*start=*/current_memory_pos, /*length=*/current_length).copy_(trimmed_tensor_element);
                    current_memory_pos += current_length;
                    current_length *= path.size(1);
                }
                return std::vector<torch::Tensor> {out_flattened};
            }
            else{
                torch::Tensor trimmed_tensor_element;
                for (auto& tensor_element : out_vector) {
                    trimmed_tensor_element = tensor_element.narrow(/*dim=*/2, /*start=*/tensor_element.size(2) - 1, /*length=*/1).squeeze(2);
                    tensor_element = trimmed_tensor_element.clone();
                }
                return out_vector;
            }
        }
    }
}  // namespace signatory