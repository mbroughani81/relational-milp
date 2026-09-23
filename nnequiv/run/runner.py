"""The one orchestration loop: select instances, run a verifier, return results.

Replaces the near-identical ``main()`` bodies of ``run_pyomo`` / ``run_crown`` /
``run_diffverifier``: load the suite, pull ``limit`` / ``ids`` out of the option
bag and apply them, build the configured verifier, check it can express every
instance's property, then hand the whole batch to ``verify_suite``.
"""

from __future__ import annotations

from nnequiv.core import Instance, InstanceResult
from nnequiv.run.selection import extract_selection, select_instances
from nnequiv.suites import SuiteOptions, load_suite
from nnequiv.verifiers import VerifierOptions, create


def run_suite(
    verifier_name: str,
    suite_name: str,
    suite_options: SuiteOptions | None = None,
    verifier_options: VerifierOptions | None = None,
) -> tuple[list[Instance], list[InstanceResult]]:
    remaining_options, limit, ids = extract_selection(dict(suite_options or {}))
    suite = load_suite(suite_name, remaining_options)
    instances = select_instances(suite.instances, limit, ids)

    verifier = create(verifier_name, verifier_options or {})
    unsupported = {
        instance.property_kind
        for instance in instances
        if not verifier.supports(instance.property_kind)
    }
    if unsupported:
        raise ValueError(
            f"verifier {verifier_name!r} cannot express property "
            f"{sorted(unsupported)}; it supports "
            f"{sorted(verifier.supported_properties)}"
        )

    results = verifier.verify_suite(instances)
    return instances, results
