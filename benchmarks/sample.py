from __future__ import annotations

from benchmarks.common import Instance, InstanceSuite, InputRegion, SuiteOptions
from nn_equivalence.nn_loader import load_nn_pair_1, load_nn_pair_2


def load_suite(suite_options: SuiteOptions | None = None) -> InstanceSuite:
    del suite_options
    region = InputRegion(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
    )
    different_nn1, different_nn2 = load_nn_pair_1()
    identical_nn1, identical_nn2 = load_nn_pair_2()

    return InstanceSuite(
        name="sample",
        instances=[
            Instance(
                instance_id="sample_different_eps_001",
                suite_name="sample",
                nn1=different_nn1,
                nn2=different_nn2,
                input_region=region,
                epsilon=0.01,
                expected_status="sat",
                metadata={"pair_type": "slightly_different"},
            ),
            Instance(
                instance_id="sample_different_eps_006",
                suite_name="sample",
                nn1=different_nn1,
                nn2=different_nn2,
                input_region=region,
                epsilon=0.6,
                expected_status="unsat",
                metadata={"pair_type": "slightly_different"},
            ),
            Instance(
                instance_id="sample_identical_eps_001",
                suite_name="sample",
                nn1=identical_nn1,
                nn2=identical_nn2,
                input_region=region,
                epsilon=0.001,
                expected_status="unsat",
                metadata={"pair_type": "identical"},
            ),
        ],
    )
