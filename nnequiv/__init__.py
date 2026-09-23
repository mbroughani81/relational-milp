"""Clean core for the NN-equivalence verifier-comparison harness.

Phase 1 of the rewrite: this package holds the dependency-free domain model
(:mod:`nnequiv.core`) and the shared bound-propagation layer
(:mod:`nnequiv.bounds`). The legacy ``nn_equivalence`` and ``benchmarks``
packages now re-export from here, so existing imports keep working while the
migration proceeds.
"""
