/* Copyright 2019 Patrick Kidger. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */
// Provides for wrapping things into PyCapsules
 

#ifndef SIGNATORY_PYCAPSULE_HPP
#define SIGNATORY_PYCAPSULE_HPP

#include <torch/extension.h>


namespace signatory { namespace misc {
    // Makes an instance of a struct of type T and wraps it into a PyCapsule.
    template <typename T, typename ...Args>
    inline py::object wrap_capsule(Args&&... args);

    // Unwraps a capsule to give a struct of type T
    template <typename T>
    inline T* unwrap_capsule(py::object capsule);
}  /* namespace signatory::misc */ }  // namespace signatory

#include "pycapsule.inl"

#endif //SIGNATORY_PYCAPSULE_HPP
