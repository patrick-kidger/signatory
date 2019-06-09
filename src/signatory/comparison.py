import signatory
import iisignature
import timeit
import torch


def speed(batch_size=1000, number=5, depth=3, stream=True):
    me_x = torch.rand(1000, batch_size, 10)
    him_x = torch.rand(batch_size, 1000, 10)

    def me():
        signatory.signature(me_x, depth, stream=stream)

    def him():
        for him_ix in him_x:
            iisignature.sig(him_ix, depth, 2 if stream else 0)

    me = timeit.timeit(me, number=number)
    him = timeit.timeit(him, number=number)
    print("Me: " + str(me))
    print("Him: " + str(him))
    print("Me/Him: " + str(me / him))
