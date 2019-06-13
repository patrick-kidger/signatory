FAQ
---
*What's the difference between `signatory` and [`iisignature`](https://github.com/bottler/iisignature)?*

The main difference is that `iisignature` is NumPy-based and CPU-only; I believe it was written primarily with mathematical research in mind. Meanwhile `signatory` is for PyTorch and may also run on the GPU; it is targeted towards machine learning applications.

The two packages also use different embeddings from streams of data into path space; see the next question.

*I get different results when I use the [`iisignature`](https://github.com/bottler/iisignature) package?*

The signature transform is defined on paths; given a stream of data we must decide how to embed it into a path. `iisignature` uses a piecewise linear embedding, whilst `signatory` uses a piecewise constant embedding. From a data science point of view, both are equally arbitrary, so as long as you pick one and stick with it shouldn't matter.

This embedding was selected for `signatory` because signature calculations for this embedding may be done much more rapidly (with provably fewer scalar multiplications); empirically it runs 3-4 times faster on the CPU than `iisignature` on reasonably sized batches. (`signatory` is optimised for batched operations, using batches to be [cache-friendly](https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code).)