"""Name -> verifier-factory registry.

Each verifier module registers a factory that turns a string option bag into a
configured :class:`~nnequiv.verifiers.base.Verifier`. Importing
``nnequiv.verifiers`` triggers registration of the built-in verifiers.
"""

from __future__ import annotations

from typing import Callable

from nnequiv.verifiers.base import Verifier, VerifierOptions

VerifierFactory = Callable[[VerifierOptions], Verifier]

_REGISTRY: dict[str, VerifierFactory] = {}


def register(name: str, factory: VerifierFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"verifier {name!r} is already registered")
    _REGISTRY[name] = factory


def create(name: str, options: VerifierOptions | None = None) -> Verifier:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown verifier {name!r}; available: {available()}"
        ) from None
    return factory(options or {})


def available() -> list[str]:
    return sorted(_REGISTRY)
