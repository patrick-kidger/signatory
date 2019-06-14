import signatory
import torch
import unittest


class TestArguments(unittest.TestCase):
    def test_arguments(self):
        for basepoint in (True, False):
            for stream in (True, False):
                for flatten in (True, False):
                    for _ in range(5):
                        size = torch.randint(low=1, high=10, size=(3,))
                        depth = int(torch.randint(low=1, high=4, size=(1,)))
                        try:
                            signatory.signature(size, depth, basepoint, stream, flatten)
                        except Exception:
                            print("Failed with basepoint={basepoint}, stream={stream}, flatten={flatten}, size={size}, "
                                  "depth={depth}".format(basepoint=basepoint, stream=stream, flatten=flatten, size=size,
                                                         depth=depth))
                            # Technically we'll get an error rather than a failure if it doesn't work, but this way we
                            # also get an informative traceback.
                            raise
