import os
import unittest

loc = os.path.dirname(__file__)
loader = unittest.defaultTestLoader
suite = loader.discover(loc)

runner = unittest.TextTestRunner()
result = runner.run(suite)
