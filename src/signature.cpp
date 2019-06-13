#include <torch/extension.h>
#include <cmath>      // pow
#include <cstdint>    // int64_t
#include <stdexcept>  // std::invalid_argument
#include <string>     // std::string
#include <tuple>      // std::tie, std::tuple
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "signature.hpp"

// TODO: write Chen method + test against iisignature for small batch sizes?
// TODO: more tests
// TODO: numpy, tensorflow
// TODO: CUDA
// TODO: update README
// TODO: handle warnings. (int64_t etc.)
// TODO: support torchscript? https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html


namespace signatory {
    namespace detail {
        std::vector<torch::Tensor> slice_into_terms(torch::Tensor out, const SigSpec& sigspec) {
            int current_mem_pos{0};
            int current_length{sigspec.input_channels};
            std::vector<torch::Tensor> out_vector;
            out_vector.reserve(sigspec.depth);
            for (int i = 0; i < sigspec.depth; ++i) {
                out_vector.push_back(out.narrow(/*dim=*/0, /*start=*/current_mem_pos, /*len=*/current_length));
                current_mem_pos += current_length;
                current_length *= sigspec.input_channels;
            }
            return out_vector;
        }


        void checkargs(torch::Tensor path, int depth) {
            if (path.ndimension() != 3) {
                throw std::invalid_argument("path must be a 3-dimensional tensor, corresponding to (batch, channel, "
                                            "stream) respectively.");
            }
            if (depth < 1) {
                throw std::invalid_argument("depth must be an integer greater than or equal to one.");
            }
        }


        std::pair<std::vector<torch::Tensor>, torch::Tensor> get_out_memory(const SigSpec& sigspec) {
            if (sigspec.stream && sigspec.flatten) {
                torch::Tensor out = torch::empty({sigspec.output_channels,
                                                  sigspec.output_stream_size,
                                                  sigspec.batch_size},
                                                 sigspec.opts);
                return {slice_into_terms(out, sigspec), out};
            }
            else {
                int current_length{sigspec.input_channels};
                std::vector<torch::Tensor> out_vector;
                out_vector.reserve(sigspec.depth);
                for (int i = 0; i < sigspec.depth; ++i) {
                    out_vector.push_back(torch::empty({current_length,
                                                       sigspec.output_stream_size,
                                                       sigspec.batch_size},
                                                      sigspec.opts));
                    current_length *= sigspec.input_channels;
                }
                return {out_vector, torch::Tensor {}};
            }
        }


        void compute_first_term(torch::Tensor nth_term, torch::Tensor path, const SigSpec& sigspec) {
            if (sigspec.basepoint) {
                nth_term.copy_(path);
            }
            else {
                nth_term.copy_(path.narrow(/*dim=*/1, /*start=*/1, /*len=*/sigspec.output_stream_size));
                nth_term -= path.narrow(/*dim=*/1, /*start=*/0, /*len=*/1);  // broadcasting
            }
        }


        torch::Tensor compute_increments(torch::Tensor path, const SigSpec& sigspec) {
            int num_increments = sigspec.input_stream_size - 1;  // == path.size(1) - 1
            if (sigspec.basepoint) {
                torch::Tensor path_increments = path.clone();
                path_increments.narrow(/*dim=*/1, /*start=*/1, /*len=*/num_increments) -=
                        path.narrow(/*dim=*/1, /*start=*/0, /*len=*/num_increments);
                return path_increments;
            }
            else {
                return path.narrow(/*dim=*/1, /*start=*/1, /*len=*/num_increments) -
                       path.narrow(/*dim=*/1, /*start=*/0, /*len=*/num_increments);
            }
        }


        void compute_nth_term(torch::Tensor nth_term, torch::Tensor prev_term, torch::Tensor path_increments,
                              const SigSpec& sigspec) {
            torch::Tensor nth_term_view = nth_term.view({prev_term.size(0), sigspec.input_channels,
                                                         sigspec.output_stream_size, sigspec.batch_size});
            torch::mul_out(nth_term_view, path_increments.unsqueeze(0), prev_term.unsqueeze(1));
            // So torch::cumsum_out is excruciatingly slow.
            // I can't find anyone else running into that problem on the internet, but at least in this use case it's
            // ~20 times slower than my own code. (Which is admittedly specialised to the particular case of an
            // in-place cumsum along a particular dimension.)
            for (int i = 1; i < sigspec.output_stream_size; ++i) {
                nth_term.narrow(/*dim=*/1, /*start=*/i, /*len=*/1) +=
                        nth_term.narrow(/*dim=*/1, /*start=*/i - 1, /*len=*/1);
            }
        }


        std::vector<torch::Tensor> format_output(std::vector<torch::Tensor>& out_vector, torch::Tensor out,
                                                 const SigSpec& sigspec) {
            std::vector<torch::Tensor> formatted_out_vector;
            if (sigspec.stream) {
                if (sigspec.flatten) {
                    formatted_out_vector.push_back(out);
                }
                else {
                    formatted_out_vector = out_vector;
                }
            }
            else {
                // If we're not returning a stream then we want to make a copy of the data before we return it. This
                // is because in this case, we only want to return the set of elements which is at the very end of
                // the stream dimension - but if we don't make a copy then we're still using the same underlying
                // storage that was used for the whole thing, stream dimension included, and we can't free up that
                // storage until we're no longer using any of it. So this takes a little extra time but saves a lot
                // of memory.

                if (sigspec.flatten) {
                    torch::Tensor out_flattened = torch::empty({sigspec.output_channels,
                                                                sigspec.batch_size},
                                                               sigspec.opts);
                    int current_mem_pos = 0;
                    int current_length = sigspec.input_channels;
                    torch::Tensor trimmed_tensor_element;
                    for (auto tensor_element : out_vector) {
                        trimmed_tensor_element = tensor_element.narrow(/*dim=*/1,
                                                                       /*start=*/tensor_element.size(1) - 1,
                                                                       /*len=*/1).squeeze(1);
                        out_flattened.narrow(/*dim=*/0,
                                             /*start=*/current_mem_pos,
                                             /*len=*/current_length).copy_(trimmed_tensor_element);
                        current_mem_pos += current_length;
                        current_length *= sigspec.input_channels;
                    }
                    formatted_out_vector.push_back(out_flattened);
                }
                else{
                    for (auto tensor_element : out_vector) {
                        formatted_out_vector.push_back(tensor_element.narrow(/*dim=*/1,
                                                                             /*start=*/tensor_element.size(1) - 1,
                                                                             /*len=*/1).squeeze(1).clone());
                    }
                }
            }
            for (auto& tensor_element : formatted_out_vector) {
                if (sigspec.stream) {
                    // switch from (channel, stream, batch) used internally to (batch, channel, stream), which is its
                    // interface.
                    tensor_element = tensor_element.transpose(1, 2).transpose(0, 1);
                    // must not be in-place else the tensors in out_vector also get modified.
                }
                else {
                    // switch from (channel, batch) used internally to (batch, channel), which is its interface.
                    tensor_element = tensor_element.transpose(0, 1);
                }
            }
            return formatted_out_vector;
        }


        void checkargs_backward(const std::vector<torch::Tensor>& grad_out_vector, const SigSpec& sigspec) {
            std::string err = "Misconfigured call to backward. Error code: ";
            if (sigspec.depth < 1) {
                throw std::invalid_argument(err + "0");
            }
            if (sigspec.flatten) {
                if (grad_out_vector.size() != 1) {
                    throw std::invalid_argument(err + "1");
                }
                torch::Tensor tensor_element = grad_out_vector[0];
                if (tensor_element.ndimension() != sigspec.n_output_dims) {
                    throw std::invalid_argument(err + "2");
                }
                // Inside this function is one of the few places where the axes convention is (batch, channel, stream),
                // because we haven't transposed dimensions yet.
                if (tensor_element.size(0) != sigspec.batch_size) {
                    throw std::invalid_argument(err + "3");
                }
                if (tensor_element.size(1) != sigspec.output_channels) {
                    throw std::invalid_argument(err + "4");
                }
                if (sigspec.stream && tensor_element.size(2) != sigspec.output_stream_size) {
                    throw std::invalid_argument(err + "5");
                }
            }
            else {
                if (grad_out_vector.size() != sigspec.depth) {
                    throw std::invalid_argument(err + "6");
                }
                int total_input_channels = 0;
                int input_channels = 1;
                for (int i = 0; i < sigspec.depth; ++i) {
                    torch::Tensor tensor_element = grad_out_vector[i];
                    input_channels *= sigspec.input_channels;
                    total_input_channels += input_channels;
                    if (tensor_element.ndimension() != sigspec.n_output_dims) {
                        throw std::invalid_argument(err + "7");
                    }
                    if (tensor_element.size(0) != sigspec.batch_size) {
                        throw std::invalid_argument(err + "8");
                    }
                    if (tensor_element.size(1) != input_channels) {
                        throw std::invalid_argument(err + "9");
                    }
                    if (sigspec.stream && tensor_element.size(2) != sigspec.output_stream_size) {
                        throw std::invalid_argument(err + "10");
                    }
                }
                if (total_input_channels != sigspec.output_channels) {
                    throw std::invalid_argument(err + "11");
                }
            }

        }

        // TODO: think about memory
        void compute_first_term_backward(torch::Tensor grad_nth_term, torch::Tensor grad_path, const SigSpec& sigspec) {
            if (sigspec.basepoint) {
                grad_path += grad_nth_term;
            }
            else {
                torch::Tensor first_part {grad_path.narrow(/*dim=*/1, /*start=*/0, /*len=*/1)};
                for (int i = 0; i < grad_nth_term.size(1); ++i) {
                    first_part -= grad_nth_term.narrow(/*dim=*/1, /*start=*/i, /*len=*/1);
                }
                grad_path.narrow(/*dim=*/1, /*start=*/1, /*len=*/sigspec.input_stream_size - 1) += grad_nth_term;
            }
        }


        torch::Tensor compute_increments_backward(torch::Tensor grad_path_increments, const SigSpec& sigspec) {
            int num_increments = sigspec.input_stream_size - 1;
            if (sigspec.basepoint) {
                torch::Tensor grad_path = grad_path_increments.clone();
                grad_path.narrow(/*dim=*/1, /*start=*/0, /*len=*/num_increments)
                    -= grad_path_increments.narrow(/*dim=*/1, /*start=*/1, /*len=*/num_increments);
                return grad_path;
            }
            else {
                torch::Tensor grad_path = torch::zeros({sigspec.input_channels,
                                                        sigspec.input_stream_size,
                                                        sigspec.batch_size},
                                                       sigspec.opts);
                grad_path.narrow(/*dim=*/1, /*start=*/1, /*len=*/num_increments) += grad_path_increments;
                grad_path.narrow(/*dim=*/1, /*start=*/0, /*len=*/num_increments) -= grad_path_increments;
                return grad_path;
            }
        }


        void compute_nth_term_backward(torch::Tensor grad_nth_term,
                                       torch::Tensor grad_prev_term,
                                       torch::Tensor grad_path_increments,
                                       torch::Tensor prev_term,
                                       torch::Tensor path_increments,
                                       const SigSpec& sigspec) {
            for (int i = sigspec.output_stream_size - 1; i >= 1; --i) {
                grad_nth_term.narrow(/*dim=*/1, /*start=*/i - 1, /*len=*/1) +=
                        grad_nth_term.narrow(/*dim=*/1, /*start=*/i, /*len=*/1);
            }
            torch::Tensor grad_nth_term_view = grad_nth_term.view({prev_term.size(0), sigspec.input_channels,
                                                                   sigspec.output_stream_size, sigspec.batch_size});
            grad_path_increments += (grad_nth_term_view * prev_term.unsqueeze(1)).sum(/*dim=*/0);  // broadcasting
            grad_prev_term += (grad_nth_term_view * path_increments.unsqueeze(0)).sum(/*dim=*/1);  // broadcasting
        }


        void format_output_backward(std::vector<torch::Tensor>& grad_out_vector, const SigSpec& sigspec) {
            for (auto& tensor_element : grad_out_vector) {
                if (sigspec.stream) {
                    // switch to (channel, stream, batch) used internally from (batch, channel, stream), which is its
                    // interface.
                    tensor_element = tensor_element.transpose(0, 1).transpose(1, 2);
                } else {
                    // switch to (channel, batch) used internally from (batch, channel), which is its interface.
                    tensor_element = tensor_element.transpose(0, 1);
                }
            }
            if (sigspec.flatten) {
                slice_into_terms(grad_out_vector[0], sigspec).swap(grad_out_vector);
            }
            if (!sigspec.stream) {
                // TODO: do something more space efficient each time instead.
                std::vector<torch::Tensor> grad_out_vector_replacement;
                int input_channels = 1;
                for (int i = 0; i < sigspec.depth; ++i) {
                    input_channels *= sigspec.input_channels;
                    torch::Tensor tensor_element = torch::zeros({input_channels,
                                                                 sigspec.output_stream_size,
                                                                 sigspec.batch_size},
                                                                sigspec.opts);
                    tensor_element.narrow(/*dim=*/1,
                            /*start=*/tensor_element.size(1) - 1,
                            /*len=*/1).squeeze(1).copy_(grad_out_vector[i]);
                    grad_out_vector_replacement.push_back(tensor_element);
                }
                grad_out_vector_replacement.swap(grad_out_vector);
            }
        }
    }  // namespace signatory::detail


    SigSpec::SigSpec(torch::Tensor path, int depth, bool basepoint, bool stream, bool flatten) :
                     input_channels{path.size(0)},
                     input_stream_size{path.size(1)},
                     batch_size{path.size(2)},
                     output_channels{signature_channels(path.size(0), depth)},
                     output_stream_size{path.size(1) - (basepoint ? 0 : 1)},
                     depth{depth},
                     n_output_dims{stream ? 3 : 2},
                     basepoint{basepoint},
                     stream{stream},
                     flatten{flatten}
                     { opts = torch::TensorOptions().dtype(path.dtype()).device(path.device()); };


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


    std::tuple<std::vector<torch::Tensor>,  // it'd be nice to make this one
                                            // std::variant<torch::Tensor, std::vector<torch::Tensor>> but we're
                                            // targetting C++11. So it's the caller's responsibility to unwrap the
                                            // single-element vector if flatten==True.
               std::vector<torch::Tensor>,
               torch::Tensor,
               SigSpec>
    signature_forward(torch::Tensor path, int depth, bool basepoint=false, bool stream=false, bool flatten=true) {
        detail::checkargs(path, depth);
        if (!path.is_floating_point()) {
            path = path.to(torch::kFloat32);
        }
        // convert from (batch, channel, stream) to (channel, stream, batch), which is the representation we use
        // internally for speed (fewer cache misses).
        // having 'path' have non-monotonically-decreasing strides doesn't slow things down very much, as 'path' is only
        // really used to compute 'path_increments' below, and the extra speed from a more efficient internal
        // representation more than compensates
        path = path.transpose(0, 1).transpose(1, 2);

        std::vector<torch::Tensor> out_vector;
        torch::Tensor out;
        SigSpec sigspec{path, depth, basepoint, stream, flatten};
        std::tie(out_vector, out) = detail::get_out_memory(sigspec);

        detail::compute_first_term(out_vector[0], path, sigspec);

        torch::Tensor path_increments = detail::compute_increments(path, sigspec);
        for (int n = 1; n < depth; ++n) {
            detail::compute_nth_term(out_vector[n], out_vector[n - 1], path_increments, sigspec);
        }

        std::vector<torch::Tensor> formatted_out_vector = detail::format_output(out_vector, out, sigspec);
        return {formatted_out_vector, out_vector, path_increments, sigspec};
    }


    // implemented manually for speed (in particular, autograd doesn't like the in-place operations we have to do in the
    // forward pass for efficiency's sake.)
    torch::Tensor signature_backward(std::vector<torch::Tensor> grad_out_vector,
                                     std::vector<torch::Tensor> out_vector, torch::Tensor path_increments,
                                     SigSpec sigspec, int depth, bool basepoint, bool stream, bool flatten) {
        for (auto& tensor_element : grad_out_vector) {
            tensor_element = tensor_element.clone();
        }

        detail::checkargs_backward(grad_out_vector, sigspec);
        detail::format_output_backward(grad_out_vector, sigspec);

        torch::Tensor grad_path_increments = torch::zeros_like(path_increments, sigspec.opts);
        for (int n = depth - 1; n >= 1; --n) {
            detail::compute_nth_term_backward(grad_out_vector[n],
                                              grad_out_vector[n - 1],
                                              grad_path_increments,
                                              out_vector[n - 1],
                                              path_increments,
                                              sigspec);
        }

        torch::Tensor grad_path = detail::compute_increments_backward(grad_path_increments, sigspec);

        detail::compute_first_term_backward(grad_out_vector[0], grad_path, sigspec);

        grad_path = grad_path.transpose(1, 2).transpose(0, 1);  // convert back to the normal axis ordering
        return grad_path;
    }
};  // namespace signatory