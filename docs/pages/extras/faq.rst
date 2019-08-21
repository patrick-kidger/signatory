.. currentmodule:: extras-faq

FAQ
---
* What's the difference between Signatory and iisignature_?

The essential difference (and indeed the reason for Signatory's existence) is that iisignature is CPU-only, whilst Signatory is for both CPU and GPU, to provide the speed necessary for machine learning. (In particular this removes the need to copy data back and forth between the CPU and the GPU.) iisignature is NumPy-based, whilst Signatory is for PyTorch. There are also a few differences in the provided functionality; each package provides a few operations that the other doesn't.

* I'm only using the CPU. Does it matter whether I use Signatory or iisignature_?

Not particularly!

.. _iisignature: https://github.com/bottler/iisignature