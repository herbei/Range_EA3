"""Devroye's exact algorithm for a Brownian-meander maximum.

The meander is observed on [0, 1] and has a fixed endpoint r.

Reference:
    Devroye, L. (2010). On exact simulation algorithms for some
    distributions related to Brownian motion and Brownian meanders.
    In Recent Developments in Applied Probability and Statistics,
    pages 1-35. Physica-Verlag HD, Heidelberg.
"""

import math
import random


_MAXMEANDER_CONSTANTS = {
    "cutoff": 3 / 2,
    "xi": 6.8 * math.exp(-9),
    "zeta": 2.2 * math.exp(-9),
    "mu": 16 * math.exp(-2 * math.pi**2 / 3),
    "nu": 16 * math.exp(-9),
    "tau": 4 * math.exp(-9),
}


def _maxmeander_check_scalar(
    value,
    name,
    nonnegative=False,
    positive=False,
    integer=False,
):
    """Check that value satisfies the requested restrictions."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if integer and (value < 1 or value != math.floor(value)):
        raise ValueError(f"{name} must be a positive integer")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")

    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _maxmeander_f(k, r, x):
    """Calculate the F_k term used in the MAXMEANDER series."""
    exponent = -(k**2) * math.pi**2 / (2 * x**2)

    if r == 0:
        raise ValueError(
            "F_k is singular at r = 0; use the r -> 0 product limit"
        )

    return (
        math.sqrt(2 * math.pi)
        * (1 / x**2)
        * (1 / r)
        * math.exp(r**2 / 2)
        * math.pi
        * k
        * math.exp(exponent)
    )


def _maxmeander_f_times_r(k, r, x):
    """Calculate r times F_k without dividing by r."""
    exponent = -(k**2) * math.pi**2 / (2 * x**2)

    return (
        math.sqrt(2 * math.pi)
        * (1 / x**2)
        * math.exp(r**2 / 2)
        * math.pi
        * k
        * math.exp(exponent)
    )


def _maxmeander_psi(k, r, x):
    """Calculate the psi_k term used in the small-x series."""
    exponent = -(k**2) * math.pi**2 / (2 * x**2)

    if r == 0:
        return (
            math.sqrt(2 * math.pi)
            * math.pi**2
            * k**2
            * (k**2 * math.pi**2 - 3 * x**2)
            * math.exp(exponent)
            / x**6
        )

    f_value = _maxmeander_f(k, r, x)
    angle = math.pi * k * r / x

    first_term = (
        (k**2 * math.pi**2 - 2 * x**2)
        / x**3
        * math.sin(angle)
    )
    second_term = (
        math.pi
        * k
        * r
        / x**2
        * math.cos(angle)
    )

    return f_value * (first_term - second_term)


def _maxmeander_psi_tail_bound(k, r, x):
    """Bound the uncomputed tail of the psi_k series."""
    constants = _MAXMEANDER_CONSTANTS

    return (
        _maxmeander_f_times_r(k, r, x)
        * (k**3 * math.pi**3 / x**4)
        / (1 - constants["mu"])
    )


def _maxmeander_fk(k, r, x):
    """Calculate the f_k term used in the large-x series."""
    exponent = -2 * k**2 * x**2

    if r == 0:
        return (
            8
            * k**2
            * x
            * (4 * k**2 * x**2 - 3)
            * math.exp(exponent)
        )

    positive_image = (
        1 - (r + 2 * k * x) ** 2
    ) * math.exp(exponent - 2 * k * x * r)

    negative_image = (
        1 - (r - 2 * k * x) ** 2
    ) * math.exp(exponent + 2 * k * x * r)

    return (2 * k / r) * (positive_image - negative_image)


def _maxmeander_large_one(r, max_terms):
    """Draw one maximum when the endpoint r is at least 3/2."""
    constants = _MAXMEANDER_CONSTANTS
    c = 5 * r / (10 * r**2 - 8)

    while True:
        E = random.expovariate(1.0)
        X = r + c * E
        V = random.random()
        Y = 10 * r * V * math.exp(-E) / (1 - constants["xi"])

        k = 2
        S = _maxmeander_fk(1, r, X)
        decision = "undecided"

        while decision == "undecided":
            if k > max_terms:
                raise RuntimeError(
                    "MAXMEANDER exceeded max_terms in the r >= 3/2 branch"
                )

            exponent = -2 * k**2 * X**2 + 2 * k * X * r
            lower_tail = (
                4 * k * (1 + 4 * k * X * r) / r
            ) * math.exp(exponent) / (1 - constants["zeta"])
            upper_tail = (
                2 * k * (r + 4 * k**2 * X**2 / r)
            ) * math.exp(exponent) / (1 - constants["xi"])

            if Y <= S - lower_tail:
                decision = "accept"
            elif Y >= S + upper_tail:
                decision = "reject"
            else:
                S = S + _maxmeander_fk(k, r, X)
                k = k + 1

        if decision == "accept":
            return X


def _maxmeander_small_one(r, max_terms):
    """Draw one maximum when the endpoint r is below 3/2."""
    constants = _MAXMEANDER_CONSTANTS
    cutoff = constants["cutoff"]
    p = 3 * math.exp(9 / 8) / (1 - constants["mu"])

    # The paper's printed upper-tail constant in the x >= 3/2 subcase is
    # too small at r = 0 when used with the derivative-form f_k. Doubling
    # this dominating mass is conservative and preserves exactness.
    q = (
        123
        * math.exp(3 * r - 9 / 2)
        / ((1 - constants["nu"]) * (4 - 2 * r))
    )

    while True:
        U = random.random()
        V = random.random()

        if U <= p / (p + q):
            N = random.gauss(0, 1)
            E1 = random.expovariate(1.0)
            E2 = random.expovariate(1.0)
            X = math.pi / math.sqrt(N**2 + 2 * E1 + 2 * E2)

            g = (
                math.sqrt(2 * math.pi)
                * math.exp(9 / 8)
                * math.pi**4
                * math.exp(-math.pi**2 / (2 * X**2))
                / ((1 - constants["mu"]) * X**6)
            )
            Y = V * g

            if X < r or X >= cutoff:
                continue

            k = 2
            S = _maxmeander_psi(1, r, X)
            decision = "undecided"

            while decision == "undecided":
                if k > max_terms:
                    raise RuntimeError(
                        "MAXMEANDER exceeded max_terms in the x < 3/2 branch"
                    )

                tail_bound = _maxmeander_psi_tail_bound(k, r, X)

                if Y <= S - tail_bound:
                    decision = "accept"
                elif Y >= S + tail_bound:
                    decision = "reject"
                else:
                    S = S + _maxmeander_psi(k, r, X)
                    k = k + 1
        else:
            rate = 4 - 2 * r
            E = random.expovariate(1.0)
            X = cutoff + E / rate
            Y = V * q * rate * math.exp(-E)

            k = 2
            S = _maxmeander_fk(1, r, X)
            decision = "undecided"

            while decision == "undecided":
                if k > max_terms:
                    raise RuntimeError(
                        "MAXMEANDER exceeded max_terms in the x >= 3/2 branch"
                    )

                exponent = 2 * k * X * r - 2 * k**2 * X**2
                lower_tail = (
                    8 * k**2 * X * math.exp(exponent)
                    / (1 - constants["tau"])
                )
                upper_tail = (
                    2 * 164 * k**4 * X**3 * math.exp(exponent)
                    / (9 * (1 - constants["nu"]))
                )

                if Y <= S - lower_tail:
                    decision = "accept"
                elif Y >= S + upper_tail:
                    decision = "reject"
                else:
                    S = S + _maxmeander_fk(k, r, X)
                    k = k + 1

        if decision == "accept":
            return X


def MAXMEANDER(r, n=1, max_terms=100000):
    """Draw n Brownian-meander maxima conditional on endpoint r."""
    _maxmeander_check_scalar(r, "r", nonnegative=True)
    _maxmeander_check_scalar(n, "n", positive=True, integer=True)
    _maxmeander_check_scalar(
        max_terms,
        "max_terms",
        positive=True,
        integer=True,
    )

    if r >= _MAXMEANDER_CONSTANTS["cutoff"]:
        sampler = _maxmeander_large_one
    else:
        sampler = _maxmeander_small_one

    n = int(n)
    max_terms = int(max_terms)
    maxima = []

    for i in range(n):
        maximum = sampler(r, max_terms)
        maxima.append(maximum)

    return maxima


def dMAXMEANDER(x, r, terms=120):
    """Evaluate the Brownian-meander maximum density at x."""
    _maxmeander_check_scalar(r, "r", nonnegative=True)
    _maxmeander_check_scalar(
        terms,
        "terms",
        positive=True,
        integer=True,
    )

    terms = int(terms)

    x_is_number = (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
    )

    if x_is_number:
        x_values = [x]
    else:
        try:
            x_values = list(x)
        except TypeError as error:
            raise ValueError(
                "x must be a number or an iterable of numbers"
            ) from error

    for value in x_values:
        is_number = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

        if not is_number:
            raise ValueError("x must contain only numbers")

    densities = [0.0] * len(x_values)
    cutoff = _MAXMEANDER_CONSTANTS["cutoff"]

    for i in range(len(x_values)):
        xx = x_values[i]

        if not math.isfinite(xx) or xx < r or xx <= 0:
            continue

        if r <= cutoff and xx < cutoff:
            term_function = _maxmeander_psi
        else:
            term_function = _maxmeander_fk

        density = 0.0

        for k in range(1, terms + 1):
            density = density + term_function(k, r, xx)

        densities[i] = max(0.0, density)

    return densities
