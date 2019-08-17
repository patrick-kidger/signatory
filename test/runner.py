import os
import sys
import unittest

loc = os.path.dirname(__file__)
loader = unittest.defaultTestLoader
suite = loader.discover(loc)

failfast = '-f' in sys.argv
runner = unittest.TextTestRunner(failfast=failfast)
result = runner.run(suite)
