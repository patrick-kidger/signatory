import os
with open(os.path.join(os.path.dirname(__file__), '..', 'VERSION'), 'r') as f:
    __version__ = f.read()
del os
del f
