.. currentmodule:: miscellaneous-faq

FAQ
---
* What's the difference between Signatory and iisignature_?

The essential difference (and the reason for Signatory's existence) is that iisignature is limited to the CPU, whilst Signatory is for both CPU and GPU. This allows Signatory to run *much* faster. (See the next question.) Other than that, iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides a few operations that the other doesn't.

* What's the difference in speed between Signatory and iisignature_?

Depends on your CPU and GPU, really. But to throw some numbers out there: on the CPU, Signatory tends to be about twice as fast. With the GPU, it's roughly 65 times as fast.

* I get an ``ImportError`` when I try to install Signatory.

You probably haven't installed PyTorch. Do that, then run ``pip`` to install Signatory.

* How do I backpropagate through the signature transform?

Just call ``.backward()`` like you normally would in PyTorch!

.. _iisignature: https://github.com/bottler/iisignature