"""Each module here corresponds to a particular operation that we would like to test the speed or memory usage of.

Each module should specify three functions: 'setup' and 'run' and 'teardown'.

'setup' should specify initial setup that should not be benchmarked.
'run' is the actual operation that will be benchmarked, and will be run multiple times. (In particular make sure that it
can be run multiple times!)
'teardown' will be run after the benchmark operation.

An argparse.Namespace(size=size, depth=depth) object will be passed to 'setup', then passed to 'run', then to
'teardown'. This object is used to allow these functions to communicate with one another.
"""
