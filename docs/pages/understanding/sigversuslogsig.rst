.. currentmodule:: understanding-sigversuslogsig

Signatures vs. Logsignatures
----------------------------
Signatures can get quite large. This is in fact the whole point of them! They provide a way to linearise all possible functions of their input. In contrast logsignatures tend to be reasonably modestly sized.

If you know that you want to try and capture particularly high order interactions between your input channels then you may prefer to use logsignatures over signatures, which will capture this information in a more information-dense way. This comes with a price though. First of all, the logsignature is a little slower to compute than the signature. (Have a look at the difference between :func:`signatory.logsignature` and :func:`signatory.LogSignature` for a way to amortize this extra cost.) Secondly, the universal nonlinearity property does not hold for the logsignature: by condensing the information, this property has been lost.

Note that as the logsignature is computed by going via the signature, it is not more memory-efficient to compute the logsignature than the signature.

In summary, the signature is usually preferred over the logsignature for machine learning applications.