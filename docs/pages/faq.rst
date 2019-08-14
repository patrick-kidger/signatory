.. currentmodule:: faq

FAQ
---
* What's the difference between Signatory and iisignature_?

The essential difference is that iisignature is NumPy-based and CPU-only. Meanwhile Signatory is for PyTorch and may also run on the GPU. Empirically Signatory is also about twice as fast (on the CPU) as iisignature on reasonably sized batches. (Signatory is optimised for batched operations, using batches to be `cache-friendly <https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code>`__.)

.. _iisignature: https://github.com/bottler/iisignature