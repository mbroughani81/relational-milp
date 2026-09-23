"""Per-instance progress lines for the unified runner.

The ``[index/total] ... status=<status>`` shape is kept compatible with the
fleet worker's progress parser, and now also carries the verifier and property.
"""

from __future__ import annotations

import sys
from typing import TextIO

from nnequiv.core import Instance, InstanceResult


def print_progress(
    index: int,
    total: int,
    verifier: str,
    instance: Instance,
    result: InstanceResult,
    *,
    stream: TextIO | None = None,
) -> None:
    if result.expected_status is None:
        expected = "-"
    else:
        expected = (
            f"{result.expected_status}:"
            f"{'yes' if result.matched_expected else 'no'}"
        )
    print(
        f"[{index}/{total}] {result.instance_id} "
        f"[{verifier}/{instance.property_kind}]: "
        f"status={result.status} expected={expected} "
        f"runtime_sec={result.runtime_sec:.3f} epsilon={result.epsilon:.17g}",
        file=stream if stream is not None else sys.stdout,
        flush=True,
    )
