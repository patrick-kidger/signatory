#ifndef SIGNATORY_PYCAPSULE_HPP
#define SIGNATORY_PYCAPSULE_HPP

#include <torch/extension.h>


namespace signatory { namespace misc {
    template <typename T, typename ...Args>
    inline py::object wrap_capsule(Args&&... args);

    template <typename T>
    inline T* unwrap_capsule(py::object capsule);
}  /* namespace signatory::misc */ }  // namespace signatory

#include "pycapsule.inl"

#endif //SIGNATORY_PYCAPSULE_HPP
