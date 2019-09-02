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


namespace signatory { namespace misc {
    namespace detail {
        template <typename T>
        void CapsuleDestructor(PyObject* capsule) {
            delete static_cast<T*>(PyCapsule_GetPointer(capsule, T::capsule_name));
        }
    }  // namespace signatory::misc::detail

    template <typename T, typename ...Args>
    inline py::object wrap_capsule(Args&&... args) {
        return py::reinterpret_steal<py::object>(PyCapsule_New(new T{std::forward<Args>(args)...},
                                                               T::capsule_name,
                                                               detail::CapsuleDestructor<T>));
    }

    template <typename T>
    inline T* unwrap_capsule(py::object capsule) {
        return static_cast<T*>(PyCapsule_GetPointer(capsule.ptr(), T::capsule_name));
    }
}  /* namespace signatory::misc */ }  // namespace signatory