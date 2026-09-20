"""Instance suites for the NN-equivalence benchmarks.

Each module exposes ``load_suite(suite_options) -> InstanceSuite`` and is selected
by name via the runners' ``--suite`` flag (e.g. ``--suite pruning_mnist``).
"""
