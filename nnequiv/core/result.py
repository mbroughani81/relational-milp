"""The outcome of verifying one instance, plus per-phase solve statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

InstanceStatus = Literal["sat", "unsat", "timeout", "unknown"]


@dataclass(frozen=True)
class SolveStats:
    name: str
    timings: list[tuple[str, float]] = field(default_factory=list)
    details: list[tuple[str, str | int | float]] = field(default_factory=list)

    @property
    def measured_total_sec(self) -> float:
        return sum(runtime_sec for _, runtime_sec in self.timings)


@dataclass(frozen=True)
class InstanceResult:
    instance_id: str
    suite_name: str
    status: InstanceStatus
    runtime_sec: float
    epsilon: float
    expected_status: InstanceStatus | None
    stats: list[SolveStats] = field(default_factory=list)

    @property
    def matched_expected(self) -> bool | None:
        if self.expected_status is None:
            return None
        return self.status == self.expected_status
