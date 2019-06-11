import signatory
import iisignature
import timeit
import torch


def speed(batch_size=1000, number=10, depth=3, stream=True):
    signatory_x = torch.rand(batch_size, 10, 100)
    iisignature_x = torch.rand(batch_size, 100, 10)

    def signatory_fn():
        signatory.signature(signatory_x, depth, stream=stream)

    def iisignature_fn():
        for him_ix in iisignature_x:
            iisignature.sig(him_ix, depth, 2 if stream else 0)

    signatory_time = timeit.timeit(signatory_fn, number=number)
    iisignature_time = timeit.timeit(iisignature_fn, number=number)
    print("signatory:   " + str(signatory_time))
    print("iisignature: " + str(iisignature_time))
    print("ratio:       " + str(signatory_time / iisignature_time))
