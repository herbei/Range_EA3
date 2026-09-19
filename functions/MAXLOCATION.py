"""Sample the location of the maximum of a Brownian bridge.

The sampler is conditional on the bridge ending at r and attaining maximum m.
The returned location will eventually lie strictly between 0 and 1.
"""

import math
import random


def _maxlocation_check_scalar(value, name, positive=False, integer=False):
    """Check that value is finite and satisfies any requested restrictions."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if integer and (value < 1 or value != math.floor(value)):
        raise ValueError(f"{name} must be a positive integer")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _maxlocation_check_parameters(r, m):
    """Check that m can be the maximum of a bridge from 0 to r."""
    _maxlocation_check_scalar(r, "r")
    _maxlocation_check_scalar(m, "m")

    lower_support = max(r, 0)

    if m <= lower_support:
        raise ValueError("m must be strictly larger than max(r, 0)")


def _maxlocation_check_max_iterations(max_iterations):
    """Check for a positive integer iteration limit or positive infinity."""
    is_number = isinstance(max_iterations, (int, float))
    is_number = is_number and not isinstance(max_iterations, bool)

    if not is_number or math.isnan(max_iterations):
        raise ValueError("max_iterations must be a positive integer or infinity")

    if math.isinf(max_iterations) and max_iterations > 0:
        return

    if (
        not math.isfinite(max_iterations)
        or max_iterations < 1
        or max_iterations != math.floor(max_iterations)
    ):
        raise ValueError("max_iterations must be a positive integer or infinity")


def _maxlocation_exceeded(iteration, max_iterations):
    """Return whether a finite iteration limit has been reached."""
    return math.isfinite(max_iterations) and iteration >= max_iterations


def _maxlocation_one(r, m, max_iterations):
    """Draw one maximum location using one of three rejection cases."""
    sqrt2 = math.sqrt(2)

    if m >= sqrt2:
        iteration = 0

        while True:
            iteration = iteration + 1
            U = random.random()
            N = random.gauss(0, 1)
            Y = 1 + (m - r) ** 2 / N**2

            if math.log(U) - m**2 / 2 <= math.log(Y) - Y * m**2 / 2:
                return 1 / Y

            if _maxlocation_exceeded(iteration, max_iterations):
                raise RuntimeError(
                    "MAXLOCATION exceeded max_iterations in Case I"
                )

    if m - r >= sqrt2:
        iteration = 0

        while True:
            iteration = iteration + 1
            U = random.random()
            N = random.gauss(0, 1)
            Y = 1 + m**2 / N**2

            if math.log(U) - (m - r) ** 2 / 2 <= math.log(Y) - Y * (m - r) ** 2 / 2:
                return 1 - 1 / Y

            if _maxlocation_exceeded(iteration, max_iterations):
                raise RuntimeError(
                    "MAXLOCATION exceeded max_iterations in Case II"
                )

    iteration = 0

    while True:
        iteration = iteration + 1
        U = random.random()
        X = random.betavariate(0.5, 0.5)

        if X <= 0 or X >= 1:
            if _maxlocation_exceeded(iteration, max_iterations):
                raise RuntimeError(
                    "MAXLOCATION exceeded max_iterations in Case III"
                )

            continue

        lhs_log = (
            math.log(U)
            + math.log(4)
            - 0.5 * math.log(X * (1 - X))
            - 2
            - 2 * math.log(m)
            - 2 * math.log(m - r)
        )
        rhs_log = (
            -m**2 / (2 * X)
            - (m - r) ** 2 / (2 * (1 - X))
            - 1.5 * math.log(X * (1 - X))
        )

        if not math.isfinite(lhs_log) or not math.isfinite(rhs_log):
            if _maxlocation_exceeded(iteration, max_iterations):
                raise RuntimeError(
                    "MAXLOCATION exceeded max_iterations in Case III"
                )

            continue

        if lhs_log <= rhs_log:
            return X

        if _maxlocation_exceeded(iteration, max_iterations):
            raise RuntimeError(
                "MAXLOCATION exceeded max_iterations in Case III"
            )


def MAXLOCATION(r, m, n=1, max_iterations=math.inf):
    """Draw n locations of the maximum of a Brownian bridge."""
    _maxlocation_check_parameters(r, m)
    _maxlocation_check_scalar(n, "n", positive=True, integer=True)
    _maxlocation_check_max_iterations(max_iterations)

    n = int(n)
    locations = []

    for i in range(n):
        location = _maxlocation_one(r, m, max_iterations)
        locations.append(location)

    return locations
