import os
import sys
import unittest


failfast = '-f' in sys.argv
unittest.record_test_times = not ('--notimes' in sys.argv)

loc = os.path.dirname(__file__)
loader = unittest.defaultTestLoader
suite = loader.discover(loc)

runner = unittest.TextTestRunner(failfast=failfast)
try:
    result = runner.run(suite)
finally:  # in case of KeyboardInterrupt on a long test
    if hasattr(unittest, 'test_times'):
        print('Time taken for each test:')
        for r in unittest.test_times:
            print(r)
