"""The equivalence property an instance asserts between its two networks."""

from __future__ import annotations

from typing import Literal

# Which equivalence property an instance asserts between nn1 and nn2:
#   logit_class  |nn1(x)[i] - nn2(x)[i]| <= epsilon for the single output i
#                given by ``Instance.output_index``. The repo's original
#                property, kept as the default so existing suites are unchanged.
#   linf         max_i |nn1(x)[i] - nn2(x)[i]| <= epsilon, over every output.
#                Definition 1 (epsilon-equivalence) of Teuber et al. 2021.
#   top1         argmax nn1(x) == argmax nn2(x). Definition 2 of the same paper.
#                Has no epsilon; ``Instance.epsilon`` is a tie margin (see
#                ``encoder_pyomo.encode_instance_top1``).
EquivalenceProperty = Literal["logit_class", "linf", "top1"]
EQUIVALENCE_PROPERTIES: tuple[EquivalenceProperty, ...] = (
    "logit_class",
    "linf",
    "top1",
)
