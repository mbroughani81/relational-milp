"""Pre-activation bound propagation for the equivalence harness.

``interval`` is pure Python (safe to import anywhere, including the encoder);
``crown`` pulls in torch / auto_LiRPA and is imported explicitly by callers that
need certified alpha-beta-CROWN tightening. ``network_bounds`` dispatches
between them. Importing this package does NOT import torch.
"""

from nnequiv.bounds.api import BoundMode, network_bounds
from nnequiv.bounds.interval import (
    affine_bounds,
    compute_interval_bounds,
    relu_bounds,
    tighten_bounds,
)

__all__ = [
    "network_bounds",
    "BoundMode",
    "affine_bounds",
    "relu_bounds",
    "tighten_bounds",
    "compute_interval_bounds",
]
