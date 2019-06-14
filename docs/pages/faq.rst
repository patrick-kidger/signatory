.. currentmodule:: faq

FAQ
---
* What's the difference between :ref:`signatory-label` and iisignature_?

The main difference is that iisignature_ is NumPy-based and CPU-only. Meanwhile :ref:`signatory-label` is for PyTorch and may also run on the GPU, as it is targeted towards machine learning applications. The two packages also use different embeddings from streams of data into path space -- see the next question.

* Why is the signature of a path different when I use the iisignature_ package?

The signature transform is defined on paths; given a stream of data we must decide how to embed it into a path. iisignature_ uses a piecewise linear embedding, whilst :ref:`signatory-label` uses a piecewise constant embedding. From a data science point of view, both are equally arbitrary -- so as long as you pick one and stick with it then it shouldn't matter.

This embedding was selected for :ref:`signatory-label` because signature calculations for this embedding may be done much more rapidly, with provably fewer scalar multiplications. Empirically it runs 3-4 times faster on the CPU than iisignature_ on reasonably sized batches. (:ref:`signatory-label` is optimised for batched operations, using batches to be `cache-friendly <https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code>`__.)

.. _iisignature: https://github.com/bottler/iisignature