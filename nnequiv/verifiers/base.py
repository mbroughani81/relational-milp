"""The ``Verifier`` seam: one contract every equivalence verifier implements.

A verifier is ``(Instance) -> InstanceResult``. The contract is batch-first
(``verify_suite``) because the real backends share per-run setup — CROWN caches
torch models across a run, the diff tool caches serialized ``.nnet`` files — and
``verify`` is the one-instance convenience on top. ``supports`` encodes, in code,
which equivalence properties a backend can actually express (only the MILP
verifier handles ``linf`` / ``top1``; CROWN and the diff tools do ``logit_class``
only), replacing prose caveats scattered across the runners.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping

from nnequiv.core import EquivalenceProperty, Instance, InstanceResult

VerifierOptions = Mapping[str, str]


def parse_bool(value: str) -> bool:
    """Parse a string option value into a bool (for ``--verifier-opt k=v`` bags)."""
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean option value, got {value!r}")


class Verifier(ABC):
    #: Registry key / CSV label for this verifier.
    name: ClassVar[str]
    #: Equivalence properties this backend can actually express.
    supported_properties: ClassVar[frozenset[EquivalenceProperty]]

    def supports(self, property_kind: EquivalenceProperty) -> bool:
        return property_kind in self.supported_properties

    @abstractmethod
    def verify_suite(self, instances: list[Instance]) -> list[InstanceResult]:
        """Verify a batch of instances, returning one result per instance in order."""

    def verify(self, instance: Instance) -> InstanceResult:
        return self.verify_suite([instance])[0]

    @classmethod
    @abstractmethod
    def from_options(cls, options: VerifierOptions) -> "Verifier":
        """Build a configured verifier from a string ``KEY=VALUE`` option bag."""
