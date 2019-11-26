"""Each module here corresponds to a particular operation that we would like to test the speed or memory usage of.

Each module should specify two functions: 'setup' and 'run'.

'setup' should specify initial setup that should not be benchmarked.
'run' is the actual operation that will be benchmarked, and will be run multiple times. (In particular make sure that it
can be run multiple times!)

An argparse.Namespace(size=size, depth=depth) object will be passed to 'setup', then passed to 'run'. This object is
used to allow these two functions to communicate with one another.
"""
