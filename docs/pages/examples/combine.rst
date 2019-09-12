.. currentmodule:: examples-combine

.. _examples-combine:

Combining signatures
####################
Suppose we have two paths, and want to combine their signatures. That is, we know the signatures of the two paths, and would like to know the signature of the two paths concatenated together. This can be done with the :func:`signatory.signature_combine` function.

.. code-block:: python

    import torch
    import signatory

    depth = 3
    input_channels = 5
    path1 = torch.rand(1, 10, input_channels)
    path2 = torch.rand(1, 5, input_channels)

    ### OPTION 1 using signature_combine
    sig_path1 = signatory.signature(path1, depth)
    sig_path2 = signatory.signature(path2, depth,
                                    basepoint=path1[:, -1, :])
    sig_combined = signatory.signature_combine(sig_path1, sig_path2,
                                               input_channels, depth)

    ### OPTION 2 without using signature_combine
    path_combined = torch.cat([path1, path2], dim=1)
    sig_combined = signatory.signature(path_combined, depth)

    ### Both options will produce the same value for sig_combined

.. danger::

    Note in particular that the end of :attr:`path1` is used as the :attr:`basepoint` when calculating :attr:`sig_path2` in Option 1. If this is not done then the code will run but the result will be meaningless! It is important that :attr:`path2` starts from the same place that :attr:`path1` finishes. Otherwise there will be a jump between the end of :attr:`path1` and the start of :attr:`path2` which the signature will not see.

    If it is known that :attr:`path1[:, -1, :] == path2[:, 0, :]`, so that in fact :attr:`path1` does finish where :attr:`path2` starts, then only in this case can the use of :attr:`basepoint` safely be skipped. (And if :attr:`basepoint` is set then it will not change the result.)

However Option 1 requires knowing :attr:`sig_path1` when computing the next signature. However if it is desired to know :attr:`sig_path2` independently, then Option 1 using :func:`signatory.signature_combine` is the preferable option.

With Option 2 it is clearest what is being computed. However this is also going to be much slower: the assumption is that the signature of :attr:`path1` is already known, but Option 2 does not use this information at all, and will instead perform a lot of unnecessary computation. Furthermore its calculation requires holding all of :attr:`path1` in memory, instead of just :attr:`path1[:, -1, :]`. This option should be avoided.

Note how with Options 1, then once :attr:`sig_path1` has been computed, then the only thing that must now be held in memory is :attr:`sig_path1` and :attr:`path1[:, -1, :]`. The amount of memory required is independent of the length of :attr:`path1`. Thus if :attr:`path` is very long, or can grow to arbitrary length as time goes by, then the use of this option (over Option 2) is crucial.

.. tip::

    Combining signatures in this way is the most sensible way to do things only if the signature of :attr:`path2` is actually desirable information on its own. If the only aim is to use the signature of :attr:`path1` when computing the signature of the combined path, then this can be done most easily by

    .. code-block::

        sig_path1 = signatory.signature(path1, depth)
        sig_combined = signatory.signature(path2, depth,
                                           basepoint=path1[:, -1, :],
                                           initial=sig_path1)

    For further examples of this nature, see :ref:`examples-online`.