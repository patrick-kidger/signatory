"""Each module here corresponds to a particular operation that we would like to test the speed or memory usgae of.

Each module should specify three functions: 'setup', 'mem_include' and 'run'.

'setup' should specify initial setup that should not be benchmarked.
'mem_include' is setup that should be included in memory usage but not in time usage.
'run' is the actual operation that will be benchmarked, and will be run multiple times. (In particular make sure that it
can be run multiple times!)

An argparse.Namespace(size=size, depth=depth) object will be passed to 'setup', then the same object passed to
'mem_include', then the the same object passed to 'run'. This object is used to allow these three functions to
communicate with one another.
"""
