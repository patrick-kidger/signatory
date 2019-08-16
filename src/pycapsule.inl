namespace signatory { namespace misc {
    namespace detail {
        template <typename T>
        void CapsuleDestructor(PyObject* capsule, const char* name) {
            delete static_cast<T*>(PyCapsule_GetPointer(capsule, name));
        }
    }  // namespace signatory::misc::detail

    template <typename T, typename ...Ts>
    inline py::object wrap_capsule(Ts... args) {
        return py::reinterpret_steal<py::object>(PyCapsule_New(new T{args...}, T::capsule_name, detail::CapsuleDestructor<T>));
    }

    template <typename T>
    inline T* unwrap_capsule(py::object capsule) {
        return static_cast<T*>(PyCapsule_GetPointer(capsule.ptr(), T::capsule_name));
    }
}  /* namespace signatory::misc */ }  // namespace signatory