#include <torch/extension.h>
#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <sstream>    // std::stringstream
#include <stdexcept>  // std::invalid_argument
#include <vector>     // std::vector

#include "signature.hpp"

// TODO: test reordering axes...
// TODO: write Chen method + test against iisignature for small batch sizes?
// TODO: tests
// TODO: numpy, tensorflow
// TODO: CUDA
// TODO: backprop!


namespace signatory {
    namespace detail {
        void checkargs(torch::Tensor path, int depth, bool batch_first) {
            if (path.ndimension() != 3) {
                std::stringstream error_msg;
                error_msg << "path must be a 3-dimensional tensor, corresponding to (";
                if (batch_first) {
                    error_msg << "batch, stream";
                }
                else {
                    error_msg << "stream, batch";
                }
                error_msg << ", channels) respectively.";
                throw std::invalid_argument(error_msg.str());
            }
            if (depth < 1) {
                throw std::invalid_argument("depth must be an integer greater than or equal to one.");
            }
        }


        torch::Tensor format_input(torch::Tensor path, bool batch_first) {
            if (path.type().scalarType() != torch::kFloat64) {
                // we don't want to convert kFloat64 down to kFloat32.
                // ideally we'd want to future-proof against kFloat128 etc. as well, but I'm not sure how to do that,
                // except perhaps to check the type isn't one of the (long) list of current non-float types. Would be
                // nice to have a float "Concept" to check against...
                path = path.to(torch::kFloat32);
            }
            if (batch_first) {
                // we don't put batches first for speed reasons (contiguous memory yadda yadda): batch-first is about 8%
                // slower on a simple test I tried.
                // still, most things put batch first, so we offer this option and just prepare to pay that price. Of
                // course using this option gives strides that aren't monotonically decreasing, but fixing that by
                // cloning is slower still: about 16% overall.
                path = path.transpose(0, 1);
                // not path.transpose_(0, 1) because that will modify the tensor we were given as input!
            }
            return path;
        }


        // In the special case of stream && flatten, we can save ourselves a boatload of copying by using the same
        // memory for both computation and output, so we special-case this (and thus handle every other case as well,
        // wrapped into this class.) This is both a common use-case and the most expensive use-case, which is why we
        // bother to handle the memory management ourselves.
        class OutMemory {
        public:
            OutMemory(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten, bool batch_first) :
            path{path}, depth{depth}, basepoint{basepoint}, stream{stream}, flatten{flatten}, batch_first{batch_first},
            counter{0}, current_mem_pos{0}, current_length{path.size(2)},
            output_channels{signature_channels(path.size(2), depth)}
            {
                if (stream && flatten) {
                    out = torch::empty({path.size(0) - (basepoint ? 0 : 1),
                                        path.size(1),
                                        output_channels},
                                       torch::dtype(path.dtype()).device(path.device()));
                }
                else {
                    out_vector.reserve(depth);
                }
            }

            torch::Tensor next() {
                if (counter >= depth) {
                    // shouldn't ever get here!
                    throw std::range_error("Eek!");
                }
                torch::Tensor nth_term;
                if (stream && flatten) {  // TODO: this is actually slower! To fix...
                    nth_term = out.narrow(/*dim=*/2, /*start=*/current_mem_pos, /*length=*/current_length);
                    current_mem_pos += current_length;
                }
                else {
                    nth_term = torch::empty({path.size(0) - (basepoint ? 0 : 1),
                                             path.size(1),
                                             current_length},
                                            torch::dtype(path.dtype()).device(path.device()));
                    out_vector.push_back(nth_term);
                }
                ++counter;
                current_length *= path.size(2);
                return nth_term;
            }

            // not technically to do with memory management... but it uses all the same things so we bundle it here.
            void format_output() {
                if (stream) {
                    if (flatten) {
                        out_vector.push_back(out);
                    }
                }
                else {
                    // If we're not returning a stream then we want to make a copy of the data before we return it. This
                    // is because in this case, we only want to return the set of elements which is at the very end of
                    // the stream dimension - but if we don't make a copy then we're still using the same underlying
                    // storage that was used for the whole thing, stream dimension included, and we can't free up that
                    // storage until we're no longer using any of it. So this takes a little extra time but saves a lot
                    // of memory.

                    if (flatten) {
                        torch::Tensor out_flattened = torch::empty({path.size(1), output_channels},
                                                                   torch::dtype(path.dtype()).device(path.device()));
                        int current_mem_pos = 0;         // deliberate variable shadowing
                        int current_length = path.size(2);  //
                        torch::Tensor trimmed_tensor_element;
                        for (auto tensor_element : out_vector) {
                            trimmed_tensor_element = tensor_element.narrow(/*dim=*/0,
                                                                           /*start=*/tensor_element.size(0) - 1,
                                                                           /*length=*/1).squeeze(0);
                            out_flattened.narrow(/*dim=*/1,
                                                 /*start=*/current_mem_pos,
                                                 /*length=*/current_length).copy_(trimmed_tensor_element);
                            current_mem_pos += current_length;
                            current_length *= path.size(2);
                        }
                        (std::vector<torch::Tensor> {out_flattened}).swap(out_vector);
                    }
                    else{
                        torch::Tensor trimmed_tensor_element;
                        for (auto& tensor_element : out_vector) {
                            trimmed_tensor_element = tensor_element.narrow(/*dim=*/0,
                                                                           /*start=*/tensor_element.size(2) - 1,
                                                                           /*length=*/1).squeeze(0);
                            tensor_element = trimmed_tensor_element.clone();
                        }
                    }
                }

                if (stream && batch_first) {
                    // if(!stream) then our output has axes (batch, feature), which is fine.
                    // if(stream) then our output has axes (stream, batch, feature). if(batch_first) as well then we
                    // want to switch them round to (batch, stream, feature).
                    for (auto tensor_element : out_vector) {
                        tensor_element.transpose_(0, 1);
                    }
                }
            }

            const std::vector<torch::Tensor> get_out_vector() const {
                return out_vector;
            }

        private:
            torch::Tensor path;
            int depth;
            bool basepoint;
            bool stream;
            bool flatten;
            bool batch_first;

            int counter;
            int64_t current_mem_pos;
            int64_t current_length;
            int output_channels;
            torch::Tensor out;
            std::vector<torch::Tensor> out_vector;
        };


        void compute_first_term(torch::Tensor out, torch::Tensor path, bool basepoint) {
            if (basepoint) {
                out.copy_(path);
            }
            else {
                out.copy_(path.narrow(/*dim=*/0, /*start=*/1, /*length=*/path.size(0) - 1));
                out -= path.narrow(/*dim=*/0, /*start=*/0, /*length=*/1);  // broadcasting
            }
        }


        torch::Tensor compute_increments(torch::Tensor path, bool basepoint) {
            if (basepoint) {
                torch::Tensor path_increments = path.clone();
                path_increments.narrow(/*dim=*/0, /*start=*/1, /*length=*/path.size(0) - 1) -=
                        path.narrow(/*dim=*/0, /*start=*/0, /*length=*/path.size(0) - 1);
                return path_increments;
            }
            else {
                return path.narrow(/*dim=*/0, /*start=*/1, /*length=*/path.size(0) - 1) -
                       path.narrow(/*dim=*/0, /*start=*/0, /*length=*/path.size(0) - 1);
            }
        }


        void compute_nth_term(torch::Tensor out, torch::Tensor prev_term, torch::Tensor path_increments) {
            torch::Tensor out_view = out.view({out.size(0), out.size(1), prev_term.size(2), path_increments.size(2)});
            torch::mul_out(out_view, path_increments.unsqueeze(2), prev_term.unsqueeze(3));
            // So torch::cumsum_out is excruciatingly slow.
            // I can't find anyone else running into that problem on the internet, but at least in this use case it's
            // ~20 times slower than my own code. (Which is admittedly specialised to the particular case of an
            // in-place cumsum along a particular dimension.)
            for (int i = 1; i < out.size(0); ++i) {
                out.narrow(/*dim=*/0, /*start=*/i, /*length=*/1) +=
                        out.narrow(/*dim=*/0, /*start=*/i - 1, /*length=*/1);
            }
        }
    }  // namespace signatory::detail


    int signature_channels(int input_channels, int depth) {
        if (input_channels < 1) {
            throw std::invalid_argument("input_channels must be at least 1");
        }
        if (depth < 1) {
            throw std::invalid_argument("depth must be at least 1");
        }

        if (input_channels == 1) {
            return depth;
        }
        else {
            return input_channels * ((pow(input_channels, depth) - 1) / (input_channels - 1));
        }
    }


    // It'd be nice to return std::variant<torch::Tensor, std::vector<torch::Tensor>> here instead, based on whether
    // 'flatten' is true or false respectively, rather than wrapping it into a vector like we actually do.
    // Sadly we're targeting C++11 so instead it is the caller's responsibility to unwrap the single-element vector if
    // they call this function with flatten=true.
    std::vector<torch::Tensor> signature(torch::Tensor path, int depth, bool basepoint=false, bool stream=false,
                                         bool flatten=true, bool batch_first=false) {
        detail::checkargs(path, depth, batch_first);
        path = detail::format_input(path, batch_first);

        detail::OutMemory out_memory {path, depth, basepoint, stream, flatten, batch_first};
        torch::Tensor nth_term = out_memory.next();

        detail::compute_first_term(nth_term, path, basepoint);

        torch::Tensor prev_term;
        torch::Tensor path_increments = detail::compute_increments(path, basepoint);
        for (int n = 2; n <= depth; ++n) {
            prev_term = nth_term;
            nth_term = out_memory.next();
            detail::compute_nth_term(nth_term, prev_term, path_increments);
        }

        out_memory.format_output();
        return out_memory.get_out_vector();
    }


    //std::vector<torch::Tensor> signature_backward(std::vector<torch::Tensor> grad_out)
}  // namespace signatory