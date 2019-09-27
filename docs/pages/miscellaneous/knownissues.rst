.. _miscellaneous-knownissues:

Known Issues
############

* Exceptions messages aren't very helpful on a Mac.

This isn't an issue directly to do with Signatory. We use pybind11 to translate C++ exceptions to Python exceptions, and some part of this process breaks down when on a Mac.

If you're trying to debug your code then the best (somewhat unhelpful) advice is to try running the problematic code on either Windows or Linux to check what the error message is.