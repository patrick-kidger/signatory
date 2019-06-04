#ifndef TORCHTEST_SRC_EXTENSION_HPP
#define TORCHTEST_SRC_EXTENSION_HPP

// must define TYPEFLAG before including this file, to specify what framework we're building for.
#if TYPEFLAG == torch
    #include <torch/extension.h>
    using Tensor = torch::Tensor;
    using sigmoid = torch::sigmoid;
#elif TYPEFLAG == numpy
    #include <pybind11/pybind11.h>
#else
    #error Unrecognised typeflag
#endif

#endif //TORCHTEST_SRC_EXTENSION_HPP
