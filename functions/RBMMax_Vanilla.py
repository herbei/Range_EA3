"""Vanilla exact sampler for a restricted Brownian-meander maximum.

RBMMax_Vanilla implements Algorithm 5 from Article01.pdf.  It samples the
maximum with MAXMEANDER and uses the Bessel exit-time functions in
BESSELEXIT.py to sample the location of that maximum.
"""

import math
import random

from .BESSELEXIT import BESSEL_F_T, BESSEL_G_accept
from .MAXMEANDER import MAXMEANDER


def _rbmmax_vanilla_check_scalar(
    value,
    name,
    positive=False,
    integer=False,
):
    """Check that value is one finite number with the requested properties."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")

    if integer and (value < 1 or value != math.floor(value)):
        raise ValueError(f"{name} must be a positive integer")


def _rbmmax_vanilla_one(r, T, max_refinements):
    """Draw one restricted-meander maximum and its location."""
    square_root_T = math.sqrt(T)

    maximum_unit = MAXMEANDER(
        r / square_root_T,
        n=1,
    )[0]

    maximum = square_root_T * maximum_unit

    while True:
        postmaximum_duration = BESSEL_F_T(
            maximum,
            r,
            T,
            n=1,
            max_refinements=max_refinements,
        )[0]

        tau = T - postmaximum_duration
        threshold = random.random() * (4.0 / maximum**2)

        if BESSEL_G_accept(
            tau,
            maximum,
            threshold,
            max_refinements=max_refinements,
        ):
            return {
                "M": maximum,
                "tau": tau,
            }


def RBMMax_Vanilla(r, T, n=1, max_refinements=100000):
    """Draw restricted-meander maxima and their locations."""
    _rbmmax_vanilla_check_scalar(r, "r", positive=True)
    _rbmmax_vanilla_check_scalar(T, "T", positive=True)
    _rbmmax_vanilla_check_scalar(
        n,
        "n",
        positive=True,
        integer=True,
    )
    _rbmmax_vanilla_check_scalar(
        max_refinements,
        "max_refinements",
        integer=True,
    )

    n = int(n)
    max_refinements = int(max_refinements)

    if n == 1:
        return _rbmmax_vanilla_one(
            r,
            T,
            max_refinements,
        )

    draws = []

    for _ in range(n):
        draw = _rbmmax_vanilla_one(
            r,
            T,
            max_refinements,
        )
        draws.append(draw)

    return draws
