"""Sample Brownian bridges conditional on staying in a fixed interval.

ICB samples a bridge from x to z over [0, Delta] at the times in calT,
conditional on the entire bridge staying in [l, u]. Values are sampled
sequentially, each conditional on the preceding value and the final endpoint.

The result is a list of dictionaries with keys draw, time, value, and
attempts, ordered by draw number and then by time.

The file also provides numerical transition-density and survival-probability
helpers, and a sampler for an unconstrained Brownian bridge.

Reference:
    Herbei, R. and Somnath, K. (2026). Interval-Constrained Brownian Paths:
    Exact Interpolation and Extrapolation. Manuscript draft, Section 3.2.
"""

import math
import random

if __package__:
    from .ICB_gh_samplers import ICB_interpolate_step
else:
    from ICB_gh_samplers import ICB_interpolate_step


def _icb_check_scalar(value, name, positive=False):
    """Check that value is one finite number, and optionally positive."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _icb_check_integer(value, name, positive=False):
    """Check for a finite integer-valued number without converting it."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be an integer scalar")

    if value != math.floor(value):
        raise ValueError(f"{name} must be an integer scalar")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _icb_clamp_probability(value):
    """Restrict a numerical probability to [0, 1]; reject undefined NaN."""
    if math.isnan(value):
        raise ValueError("Probability must not be NaN")

    if not math.isfinite(value):
        if value > 0:
            return 1.0
        return 0.0

    return min(1.0, max(0.0, value))


def _icb_nonnegative(value):
    """Replace negative or nonfinite numerical kernel values with zero."""
    if not math.isfinite(value):
        return 0.0

    return max(0.0, value)


def _icb_endpoint_kind(value, l, u):
    """Classify an endpoint exactly, preserving every interior input."""
    _icb_check_scalar(value, "Endpoint")

    if value < l or value > u:
        raise ValueError("Endpoints must lie in [l, u]")

    if value == l:
        return "lower"

    if value == u:
        return "upper"

    return "interior"


def _icb_shift_endpoint(value, l, u):
    """Translate an endpoint to [0, u-l] without losing its classification."""
    kind = _icb_endpoint_kind(value, l, u)
    if kind == "lower":
        return 0.0
    if kind == "upper":
        return u - l

    shifted = value - l
    if not 0.0 < shifted < u - l:
        raise FloatingPointError(
            "Translating an interior endpoint to [0, u-l] rounded it "
            "onto a boundary"
        )
    return shifted


def _icb_absorbed_density_image(t, a, b, l, u, terms):
    """Approximate the killed transition density by the manuscript's image sum.

    Here a and b are endpoints, not the interval width used in the paper.
    Requires t > 0, l < u, and a nonnegative integer number of terms.
    """
    width = u - l
    x = a - l
    y = b - l
    normal_constant = 1.0 / math.sqrt(2.0 * math.pi * t)
    total = 0.0

    for k in range(-int(terms), int(terms) + 1):
        offset = 2.0 * k * width
        positive = math.exp(-(y - x + offset)**2 / (2.0 * t))
        negative = math.exp(-(y + x + offset)**2 / (2.0 * t))
        total = total + normal_constant * (positive - negative)

    return total


def _icb_absorbed_density_spectral(t, a, b, l, u, terms):
    """Approximate the killed transition density by the manuscript's sine sum."""
    width = u - l
    x = a - l
    y = b - l
    total = 0.0

    for n in range(1, int(terms) + 1):
        start_sine = math.sin(n * math.pi * x / width)
        end_sine = math.sin(n * math.pi * y / width)
        decay = math.exp(-n**2 * math.pi**2 * t / (2.0 * width**2))
        total = total + start_sine * end_sine * decay

    return (2.0 / width) * total


def _icb_absorbed_density(t, a, b, l, u, terms):
    """Choose a finite image or spectral sum for the killed transition density."""
    if a <= l or a >= u or b <= l or b >= u:
        return 0.0

    width = u - l
    scaled_time = t / width**2

    if scaled_time <= 0.15:
        density = _icb_absorbed_density_image(t, a, b, l, u, terms)
        if math.isfinite(density) and density >= 0.0:
            return density

    density = _icb_absorbed_density_spectral(t, a, b, l, u, terms)
    if math.isfinite(density) and density >= 0.0:
        return density

    return max(0.0, _icb_absorbed_density_image(t, a, b, l, u, terms))


def _icb_boundary_density_image_lower(t, y, l, u, terms):
    """Approximate the inward derivative of the killed kernel at l.

    This has the shape of the manuscript's function h, but includes the factor
    2 / (t * sqrt(2*pi*t)). Keep this derivative normalization in kernels.
    """
    width = u - l
    x = y - l
    if x <= 0.0 or x >= width:
        return 0.0

    normal_constant = 1.0 / math.sqrt(2.0 * math.pi * t)
    total = 0.0

    for k in range(-int(terms), int(terms) + 1):
        shifted = x + 2.0 * k * width
        normal_density = normal_constant * math.exp(-shifted**2 / (2.0 * t))
        total = total + 2.0 * shifted * normal_density / t

    return _icb_nonnegative(total)


def _icb_boundary_density_spectral(t, y, boundary, l, u, terms):
    """Approximate the inward boundary derivative using the sine series."""
    width = u - l
    x = y - l
    if x <= 0.0 or x >= width:
        return 0.0

    total = 0.0

    for n in range(1, int(terms) + 1):
        sign = 1
        if boundary == "upper":
            sign = (-1)**(n + 1)

        sine = math.sin(n * math.pi * x / width)
        decay = math.exp(-n**2 * math.pi**2 * t / (2.0 * width**2))
        total = total + sign * n * sine * decay

    return _icb_nonnegative((2.0 * math.pi / width**2) * total)


def _icb_boundary_density(t, y, boundary, l, u, terms):
    """Choose the image or spectral boundary kernel; reflect at u if needed."""
    if t <= 0.0 or y <= l or y >= u:
        return 0.0

    width = u - l
    scaled_time = t / width**2

    if scaled_time <= 0.15:
        if boundary == "lower":
            return _icb_boundary_density_image_lower(t, y, l, u, terms)

        return _icb_boundary_density_image_lower(t, l + u - y, l, u, terms)

    return _icb_boundary_density_spectral(t, y, boundary, l, u, terms)


def _icb_boundary_boundary_kernel(t, from_kind, to_kind, l, u, terms):
    """Approximate the two inward boundary derivatives of the killed kernel.

    Opposite boundaries give alternating signs; matching boundaries do not.
    This fixed spectral sum can suffer cancellation at very small times.
    """
    if t <= 0.0:
        return float(from_kind == to_kind)

    width = u - l
    total = 0.0

    for n in range(1, int(terms) + 1):
        sign = 1
        if from_kind != to_kind:
            sign = (-1)**(n + 1)

        decay = math.exp(-n**2 * math.pi**2 * t / (2.0 * width**2))
        total = total + sign * n**2 * decay

    return _icb_nonnegative((2.0 * math.pi**2 / width**3) * total)


def _icb_transition_kernel(t, from_value, from_kind, to_value, to_kind, l, u, terms):
    """Select the numerical transition kernel for the two endpoint kinds."""
    if from_kind == "interior" and to_kind == "interior":
        return _icb_absorbed_density(t, from_value, to_value, l, u, terms)

    if from_kind != "interior" and to_kind == "interior":
        return _icb_boundary_density(t, to_value, from_kind, l, u, terms)

    if from_kind == "interior" and to_kind != "interior":
        return _icb_boundary_density(t, from_value, to_kind, l, u, terms)

    return _icb_boundary_boundary_kernel(t, from_kind, to_kind, l, u, terms)


def _icb_bridge_survival_image(t, a, b, l, u, terms):
    """Approximate bridge survival by an image sum divided by the free density."""
    width = u - l
    x = a - l
    y = b - l
    base = (y - x)**2
    total = 0.0

    for k in range(-int(terms), int(terms) + 1):
        offset = 2.0 * k * width
        positive = math.exp(-((y - x + offset)**2 - base) / (2.0 * t))
        negative = math.exp(-((y + x + offset)**2 - base) / (2.0 * t))
        total = total + (positive - negative)

    return _icb_clamp_probability(total)


def _icb_bridge_survival(t, a, b, l, u, terms):
    """Approximate the survival probability for an interior-to-interior bridge."""
    if t <= 0.0:
        return float(l < a < u and l < b < u)

    if a <= l or a >= u or b <= l or b >= u:
        return 0.0

    width = u - l
    scaled_time = t / width**2

    if scaled_time <= 0.15:
        return _icb_bridge_survival_image(t, a, b, l, u, terms)

    free_density = math.exp(-(b - a)**2 / (2.0 * t)) / math.sqrt(2.0 * math.pi * t)
    if free_density == 0.0 or not math.isfinite(free_density):
        return _icb_bridge_survival_image(t, a, b, l, u, terms)

    absorbed_density = _icb_absorbed_density(t, a, b, l, u, terms)
    return _icb_clamp_probability(absorbed_density / free_density)


def _icb_unconstrained_bridge(Delta, x, z, calT):
    """Sample an ordinary Brownian bridge at sorted, strictly interior times.

    This comparison helper is not used by the constrained ICB sampler.
    """
    values = []
    current_time = 0.0
    current_value = x

    for next_time in calT:
        elapsed = next_time - current_time
        remaining = Delta - current_time

        mean = current_value + elapsed * (z - current_value) / remaining
        variance = elapsed * (Delta - next_time) / remaining
        next_value = random.gauss(mean, math.sqrt(variance))
        values.append(next_value)

        current_time = next_time
        current_value = next_value

    return values


def _icb_acceptance_log_probability(times, values, l, u, terms):
    """Sum log survival probabilities for consecutive bridge segments.

    Requires corresponding time/value sequences, with increasing times.
    Uses fixed numerical series; it is not an exact ICB acceptance test.
    """
    log_probability = 0.0

    for i in range(len(times) - 1):
        survival = _icb_bridge_survival(
            t=times[i + 1] - times[i],
            a=values[i],
            b=values[i + 1],
            l=l,
            u=u,
            terms=terms,
        )

        if survival <= 0.0:
            return -math.inf

        log_probability = log_probability + math.log(survival)

    return log_probability


def _icb_exact_bridge(Delta, x, z, l, u, calT, include_endpoints):
    """Build one constrained bridge at sorted, strictly interior times.

    Follow the sequential construction in Section 3.2 of the manuscript.
    Each sampled value becomes the starting point for the remaining bridge;
    the final endpoint z and the final absolute time Delta stay fixed.
    """
    path = []
    if include_endpoints:
        path.append({"time": 0.0, "value": x, "attempts": 1})

    width = u - l
    current_time = 0.0
    if calT:
        current_value_shifted = _icb_shift_endpoint(x, l, u)
        final_value_shifted = _icb_shift_endpoint(z, l, u)

    for next_time in calT:
        elapsed = next_time - current_time
        remaining_horizon = Delta - current_time

        # Subtract l to use the one-step sampler on [0, width].
        next_value_shifted = ICB_interpolate_step(
            t=elapsed,
            T=remaining_horizon,
            x=current_value_shifted,
            z=final_value_shifted,
            a=width,
        )

        # Add l back to report the value in the original interval.
        next_value = l + next_value_shifted
        if not l < next_value < u:
            raise FloatingPointError(
                "Translating an interior bridge value back to [l, u] "
                "lost its interior position"
            )
        path.append({
            "time": next_time,
            "value": next_value,
            "attempts": 1,
        })

        current_time = next_time
        current_value_shifted = next_value_shifted

    if include_endpoints:
        path.append({"time": Delta, "value": z, "attempts": 1})

    return path


def _icb_one(Delta, x, z, l, u, calT, include_endpoints):
    """Draw one constrained bridge by sequential conditional sampling."""
    return _icb_exact_bridge(Delta, x, z, l, u, calT, include_endpoints)


def ICB(
    Delta, x, z, l, u, calT, n=1, terms=200,
    max_attempts=math.inf, include_endpoints=True, rel_tol=1e-8,
):
    """Sample interval-constrained Brownian bridges at the times in calT.

    Delta is the total duration, x and z are the endpoints, and [l, u] is
    the allowed interval. calT must be a list or tuple of distinct, finite
    times strictly between 0 and Delta; a sorted copy is used internally.
    n is the number of independent bridge draws.

    Return a list of rows with keys draw, time, value, and attempts. Draw
    numbers start at 1. Include the two endpoints unless include_endpoints
    is False. An empty calT gives endpoints only, or an empty list if the
    endpoints are excluded. attempts is always 1; it does not
    count internal rejection proposals.

    terms and rel_tol are validated for API compatibility, but neither
    controls the exact sampler. max_attempts must be positive infinity.
    Same-boundary interpolation is not implemented by the one-step sampler.
    Endpoints must lie in [l, u]; only exact equality identifies a boundary.
    Raises FloatingPointError if translating or reflecting an interior
    endpoint or sampled value loses its interior position.
    """
    _icb_check_scalar(Delta, "Delta", positive=True)
    _icb_check_scalar(x, "x")
    _icb_check_scalar(z, "z")
    _icb_check_scalar(l, "l")
    _icb_check_scalar(u, "u")
    _icb_check_integer(n, "n", positive=True)
    _icb_check_integer(terms, "terms", positive=True)
    _icb_check_scalar(rel_tol, "rel_tol", positive=True)

    if l >= u:
        raise ValueError("l must be smaller than u")
    if not math.isfinite(u - l):
        raise ValueError("u - l must be finite")

    _icb_endpoint_kind(x, l, u)
    _icb_endpoint_kind(z, l, u)

    if not isinstance(calT, (list, tuple)):
        raise ValueError("calT must be a list or tuple of finite numbers")
    for next_time in calT:
        _icb_check_scalar(next_time, "calT entries")
        if next_time <= 0.0 or next_time >= Delta:
            raise ValueError("calT values must lie strictly inside (0, Delta)")

    calT = sorted(calT)
    for i in range(1, len(calT)):
        if calT[i] == calT[i - 1]:
            raise ValueError("calT values must be distinct")

    if not isinstance(include_endpoints, bool):
        raise ValueError("include_endpoints must be True or False")

    is_number = isinstance(max_attempts, (int, float)) and not isinstance(max_attempts, bool)
    if not is_number or math.isnan(max_attempts) or max_attempts <= 0.0:
        raise ValueError("max_attempts must be positive infinity")
    if math.isfinite(max_attempts):
        raise ValueError("finite max_attempts is not supported by the exact ICB sampler")

    n = int(n)
    draws = []

    for draw_number in range(1, n + 1):
        path = _icb_one(Delta, x, z, l, u, calT, include_endpoints)

        for row in path:
            draws.append({
                "draw": draw_number,
                "time": row["time"],
                "value": row["value"],
                "attempts": row["attempts"],
            })

    return draws
