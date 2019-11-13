.. _understanding-sigversuslogsig:

Signatures vs. Logsignatures
############################
Signatures can get quite large. This is in fact the whole point of them! They provide a way to linearise all possible functions of their input. In contrast logsignatures tend to be reasonably modestly sized.

If you know that you want to try and capture particularly high order interactions between your input channels then you may prefer to use logsignatures over signatures, as this will capture this the same information, but in a more information-dense way. This comes with a price though, as the logsignature is somewhat slower to compute than the signature.

Note that as the logsignature is computed by going via the signature, it is not more memory-efficient to compute the logsignature than the signature.