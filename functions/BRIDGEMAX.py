"""Devroye's exact algorithm for a Brownian-bridge maximum and its location.

Reference:
    Devroye, L. (2010). On exact simulation algorithms for some
    distributions related to Brownian motion and Brownian meanders.
    In Recent Developments in Applied Probability and Statistics,
    pages 1-35. Physica-Verlag HD, Heidelberg.
"""

import math
import random

from .MAXLOCATION import MAXLOCATION


def _bridgemax_check_scalar(value, name, positive=False):
    """Check that value is finite and, when requested, positive."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def BRIDGEMAX(x, z, T):
    """Draw the maximum and its location for a bridge from x to z."""
    _bridgemax_check_scalar(x, "x")
    _bridgemax_check_scalar(z, "z")
    _bridgemax_check_scalar(T, "T", positive=True)

    endpoint = (z - x) / math.sqrt(T)

    if endpoint == 0:
        location_unit = random.random()
        gamma_draw = random.gammavariate(1.5, 1.0)
        maximum_unit = math.sqrt(
            2 * location_unit * (1 - location_unit) * gamma_draw
        )
    else:
        exponential_draw = random.expovariate(1.0)
        maximum_unit = 0.5 * (
            endpoint
            + math.sqrt(endpoint**2 + 2 * exponential_draw)
        )
        location_unit = MAXLOCATION(
            r=endpoint,
            m=maximum_unit,
        )[0]

    M = x + math.sqrt(T) * maximum_unit
    tau = T * location_unit

    return {
        "M": M,
        "tau": tau,
    }
