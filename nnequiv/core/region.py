"""Input regions: hyperrectangles and half-spaces, plus containment tests."""

from __future__ import annotations

from dataclasses import dataclass

from nnequiv.core.types import Bounds


class AbstractPolytope:
    pass


@dataclass(frozen=True)
class HalfSpace(AbstractPolytope):
    a: list[float]
    b: float

    def validate_dimension(self, dimension: int) -> None:
        if len(self.a) != dimension:
            raise ValueError("halfspace dimension does not match input region")


@dataclass(frozen=True)
class Hyperrectangle(AbstractPolytope):
    low: list[float]
    high: list[float]

    def bounds(self) -> Bounds:
        if len(self.low) != len(self.high):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        region_bounds = list(zip(self.low, self.high))
        for lower, upper in region_bounds:
            if lower > upper:
                raise ValueError("input lower bound exceeds upper bound")
        return region_bounds

    @staticmethod
    def overapproximate(set_: AbstractPolytope) -> Hyperrectangle:
        if isinstance(set_, Hyperrectangle):
            return set_
        raise TypeError(f"unsupported polytope type: {type(set_).__name__}")


def constraints_list(set_: AbstractPolytope | HalfSpace) -> tuple[HalfSpace, ...]:
    if isinstance(set_, HalfSpace):
        return (set_,)
    if isinstance(set_, Hyperrectangle):
        return ()
    raise TypeError(f"unsupported polytope type: {type(set_).__name__}")


def dim(set_: AbstractPolytope) -> int:
    if isinstance(set_, HalfSpace):
        return len(set_.a)
    if isinstance(set_, Hyperrectangle):
        return len(set_.low)
    raise TypeError(f"unsupported polytope type: {type(set_).__name__}")


def contains(
    set_: AbstractPolytope,
    values: list[float],
    tolerance: float = 0.0,
) -> bool:
    if isinstance(set_, HalfSpace):
        value = sum(
            coefficient * input_value
            for coefficient, input_value in zip(set_.a, values)
        )
        return value <= set_.b + tolerance

    if len(values) != dim(set_):
        return False
    if isinstance(set_, Hyperrectangle):
        region_bounds = set_.bounds()
        bounds_satisfied = all(
            lower - tolerance <= value <= upper + tolerance
            for value, (lower, upper) in zip(values, region_bounds)
        )
        if not bounds_satisfied:
            return False
    return all(
        contains(constraint, values, tolerance)
        for constraint in constraints_list(set_)
    )
