"""Verifier plugin layer.

One :class:`~nnequiv.verifiers.base.Verifier` seam and a registry of backends.
Importing this package registers the three built-in verifiers (``milp``,
``crown``, ``diff``); ``create(name, options)`` builds a configured one and
``available()`` lists them. No heavy modules (torch, pyomo, the C tools) are
imported until a verifier's ``verify_suite`` actually runs.
"""

from nnequiv.verifiers.base import Verifier, VerifierOptions, parse_bool
from nnequiv.verifiers.registry import available, create, register
from nnequiv.verifiers.crown.verifier import CrownVerifier
from nnequiv.verifiers.diff.verifier import DiffVerifier
from nnequiv.verifiers.milp.verifier import MilpVerifier

__all__ = [
    "Verifier",
    "VerifierOptions",
    "parse_bool",
    "available",
    "create",
    "register",
    "MilpVerifier",
    "CrownVerifier",
    "DiffVerifier",
]
