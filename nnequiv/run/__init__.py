"""Orchestration: select instances, run a verifier, report results."""

from nnequiv.run.runner import run_suite
from nnequiv.run.selection import extract_selection, select_instances

__all__ = ["run_suite", "extract_selection", "select_instances"]
