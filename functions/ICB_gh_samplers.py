"""Exact proposal samplers and interval-constrained Brownian interpolation.

ICB_sample_g and ICB_sample_h return one endpoint under survival.
ICB_interpolate_step returns one bridge value for interior or boundary endpoints.
Same-boundary bridges (0 to 0, or a to a) remain unsupported.

Some numerical helpers accept scaled=True. This cancels the common factor
exp(-alpha) from a series or bound, avoiding underflow in comparisons.
Their default, scaled=False, retains the original mathematical quantity.
Scaling is not normalization to a probability density.

Density comparisons use the image bounds in Propositions 6--7 for
alpha < pi, and the spectral bounds in Propositions 4--5 for alpha >= pi.
This choice is independent of the proposal-envelope cutoffs.

References:
    Herbei, R. and Somnath, K. (2026). Interval-Constrained Brownian Paths:
    Exact Interpolation and Extrapolation. Manuscript draft.
    Devroye, L. (2010). On Exact Simulation Algorithms for Some Distributions
    Related to Brownian Motion and Brownian Meanders. In Recent Developments
    in Applied Probability and Statistics, pp. 1-35.
    DOI: 10.1007/978-3-7908-2598-5_1.
"""

import math
import random


# Crossover I1(t, a) = I2(t, a), calculated by Figure_002.py.
# Keep full precision here; the draft displays six decimal places.
ICB_ALPHA0 = 1.0861647656269326
ICB_BETA0 = math.pi**2 / (4.0 * ICB_ALPHA0)

# Section 3.1: comparison sequences, independent of the proposal cutoff above.
ICB_BOUND_ALPHA0 = math.pi


def _icb_gh_alpha(t, a):
    """Return the manuscript's dimensionless time alpha_t = pi^2*t/(2*a^2)."""
    return math.pi**2 * t / (2.0 * a**2)


def _icb_gh_beta(t, a):
    """Return the manuscript's dimensionless inverse time beta_t = a^2/(2*t)."""
    return a**2 / (2.0 * t)


def _icb_gh_Kh(t, a):
    """Return K^h, the coefficient of the manuscript's spectral series for h."""
    return math.sqrt(2.0) * math.pi**1.5 * t**1.5 / a**2


def _icb_gh_n0(alpha, beta, rho=0.5):
    """ Return the value N0"""
    T1 = (0.5) * (math.log(4/rho)/alpha - 1.0)
    T2 = (0.5) * (math.log(1/rho)/beta - 1.0)
    N = max(T1, T2)

    N0 = max(math.ceil(N)+1, 2) + 1
    return int(N0)


def _icb_gh_image_n0(t, a):
    """Image starting index for h in Proposition 7 (g starts at zero)."""
    return max(0, math.ceil((math.sqrt(t) / a - 1.0) / 2.0))


def _icb_gh_bound_start(t, a, kind):
    """First admissible density index for the representation selected at pi."""
    alpha = _icb_gh_alpha(t, a)
    if alpha < ICB_BOUND_ALPHA0:
        return 0 if kind == "g" else _icb_gh_image_n0(t, a)
    return _icb_gh_n0(alpha, _icb_gh_beta(t, a)) + 1


def _icb_gh_endpoint_kind(value, a):
    """Classify an endpoint exactly, preserving every interior input."""
    if not math.isfinite(value) or value < 0.0 or value > a:
        raise ValueError("ICB_interpolate_step endpoints must lie in [0, a]")

    if value == 0.0:
        return "lower"

    if value == a:
        return "upper"

    return "interior"


def _icb_gh_reflect_interior(value, a):
    """Reflect a known interior value without rounding it onto a boundary."""
    reflected = a - value
    if not 0.0 < reflected < a:
        raise FloatingPointError(
            "Reflecting an interior endpoint or bridge value in [0, a] "
            "lost its interior position"
        )
    return reflected


def _icb_gh_g0(y, x):
    """Evaluate the f_0 proposal envelope at one y, with x > 0.

    This envelope is used in Devroye's (2010) interior-start procedure,
    as described in Section 3.1 of the manuscript.
    """
    exponential = math.exp(-(y - x)**2 / 2.0)

    term1 = 0.0
    term2 = 2.0 * x**2 * exponential

    if y >= x:
        term1 = 2.0 * x * (y - x) * exponential

    return term1 + term2


def _icb_gh_f_n(y, x, a, n):
    """Return one term of the manuscript's image series for g, with time t = 1.

    Requires x > 0. The index n may be negative, zero, or positive.
    Factor out the larger exponential and use expm1 for their difference,
    preserving small image terms when x is close to the lower boundary.
    """
    if y < 0.0 or y > a:
        return 0.0

    shifted = y + 2 * n * a
    distance = abs(shifted)
    exponential = math.exp(-(distance - x)**2 / 2.0)
    difference = exponential * (-math.expm1(-2.0 * x * distance))

    return math.copysign(difference, shifted) / math.sqrt(2.0 * math.pi)


def _icb_gh_simulate_f0(x, a):
    """Draw from a density proportional to f_0 at unit time.

    Requires 0 < x <= a / 2.
    Uses the proposal constructions of Devroye (2010).
    """
    if not (x > 0.0 and a > 0.0 and 2.0 * x <= a):
        raise ValueError("conditions for f_0 are not met")

    if a <= 1.0:
        while True:
            U = random.random()
            V = random.random()
            candidate = a * math.sqrt(U)

            exponential = math.exp(-(candidate - x)**2 / 2.0)
            difference = exponential * (-math.expm1(-2.0 * x * candidate))
            threshold = V * 2.0 * x * candidate

            if threshold < difference:
                return candidate

    if x >= 1.0:
        while True:
            V = random.random()
            candidate = x + random.gauss(0.0, 1.0)

            if 0.0 <= candidate <= a:
                if V >= math.exp(-2.0 * x * candidate):
                    return candidate

    mixture_mass = 2.0 * x + x**2 * math.sqrt(8.0 * math.pi)

    while True:
        U = random.random()
        V = random.random()

        if U * mixture_mass <= 2.0 * x:
            exponential_draw = random.expovariate(1.0)
            candidate = x + math.sqrt(2.0 * exponential_draw)
        else:
            candidate = x + random.gauss(0.0, 1.0)

        if candidate < 0.0 or candidate > a:
            continue

        threshold = V * _icb_gh_g0(candidate, x)
        exponential = math.exp(-(candidate - x)**2 / 2.0)
        difference = exponential * (-math.expm1(-2.0 * x * candidate))

        if threshold <= difference:
            return candidate


def ICB_g_series(y, t, x, a, Nmax=500, scaled=False):
    """Evaluate the first Nmax spectral terms of g at one y.

    This is the partial sum S_N^g in Section 3.1 of the manuscript.
    Nmax is a nonnegative integer.
    With scaled=True, cancel the common factor exp(-alpha).
    Stop if the exponential rounds to zero: all later factors are also zero.
    """
    if y < 0.0 or y > a:
        return 0.0

    alpha = _icb_gh_alpha(t, a)
    total = 0.0
    shift = 0
    if scaled:
        shift = 1

    for n in range(1, Nmax + 1):
        exponential = math.exp(-alpha * (n**2 - shift))
        if exponential == 0.0:
            break
        sin_y = math.sin(math.pi * n * y / a)
        sin_x = math.sin(math.pi * n * x / a)

        total = total + sin_y * sin_x * exponential

    return (2.0 / a) * total


def _icb_gh_check_image_index(N):
    """Require an integer image radius, including the central term at N=0."""
    if not isinstance(N, int) or N < 0:
        raise ValueError("The image-series index must be a nonnegative integer")


def ICB_g_image_series(y, t, x, a, Nmax=0, scaled=False):
    """Return the image partial sum of g over -Nmax,...,Nmax (Proposition 6).

    Pair opposite images using expm1 to preserve small boundary distances.
    Reflection includes the finite-sum correction, so this is the stated
    partial sum even when both arguments are reflected, rather than just
    the same infinite limit. Symmetry in x,y also holds for the finite sum.
    With scaled=True, multiply by exp(alpha) inside each exponential.
    """
    _icb_gh_check_image_index(Nmax)
    if y < 0.0 or y > a:
        return 0.0
    shift = _icb_gh_alpha(t, a) if scaled else 0.0
    normalizer = math.sqrt(2.0 * math.pi) * math.sqrt(t)
    if x > a / 2.0 and y > a / 2.0:
        reflected = ICB_g_image_series(a - y, t, a - x, a, Nmax, scaled)
        distance = 2.0 * Nmax * a + (a - x) + (a - y)
        gap = 2.0 * (x - (a - y))
        correction = math.exp(-distance**2 / (2.0 * t) + shift)
        correction *= -math.expm1(-gap * (2.0 * distance + gap) / (2.0 * t))
        return reflected + correction / normalizer

    if x > y:
        x, y = y, x
    if y > a / 2.0:
        # Pair images around the upper boundary. Each complete block
        # factors both x and a-y, preserving opposite-boundary values.
        delta = a - y
        distance = 2.0 * Nmax * a + y
        last = math.exp(-(distance - x)**2 / (2.0 * t) + shift)
        last *= -math.expm1(-2.0 * x * distance / t)
        terms = [last]
        for n in range(Nmax):
            center = (2.0 * n + 1.0) * a
            x_exponent = -2.0 * x * (center - delta) / t
            delta_exponent = -2.0 * delta * (center - x) / t
            difference = math.fsum([
                (-math.expm1(x_exponent)) * (-math.expm1(delta_exponent)),
                math.exp(x_exponent + delta_exponent) * math.expm1(-4.0 * x * delta / t),
            ])
            terms.append(math.exp(-(2.0 * n * a + y - x)**2 / (2.0 * t) + shift) * difference)
        return math.fsum(terms) / normalizer

    central = math.exp(-(y - x)**2 / (2.0 * t) + shift)
    central *= -math.expm1(-2.0 * x * y / t)
    terms = [central]
    for n in range(1, Nmax + 1):
        distance = 2.0 * (n - 1) * a + (a - x) + (a - y)
        x_exponent = -2.0 * x * (distance + x) / t
        y_exponent = -2.0 * y * (distance + y) / t
        # f_{-n}+f_n, factored in both x and y.
        difference = math.fsum([
            (-math.expm1(x_exponent)) * math.expm1(y_exponent),
            math.exp(x_exponent + y_exponent) * (-math.expm1(-4.0 * x * y / t)),
        ])
        terms.append(math.exp(-distance**2 / (2.0 * t) + shift) * difference)
    return math.fsum(terms) / normalizer


def ICB_g_image_error(y, t, x, a, N, scaled=False):
    """Return the geometric image-tail radius in Proposition 6, N>=0."""
    _icb_gh_check_image_index(N)
    distance = 2.0 * N * a + (a - x) + (a - y)
    denominator = -math.expm1(-2.0 * a * (distance + a) / t)
    shift = _icb_gh_alpha(t, a) if scaled else 0.0
    numerator = math.exp(-distance**2 / (2.0 * t) + shift)
    return numerator / (math.sqrt(2.0 * math.pi) * math.sqrt(t) * denominator)


def ICB_lower_g_bound(y, t, x, a, N, scaled=False):
    """Lower g bound: Proposition 6 if alpha<pi, Proposition 4 otherwise.

    Start at _icb_gh_bound_start(t, a, 'g'). Image bounds may reflect both
    spatial arguments when both exceed a/2, using
    g(y;t,x,a)=g(a-y;t,a-x,a), for numerical stability.
    With scaled=True, multiply both the sum and radius by exp(alpha).
    """
    alpha = _icb_gh_alpha(t, a)
    if alpha < ICB_BOUND_ALPHA0:
        if x > a / 2.0 and y > a / 2.0:
            x, y = a - x, a - y
        partial_sum = ICB_g_image_series(y, t, x, a, Nmax=N, scaled=scaled)
        return partial_sum - ICB_g_image_error(y, t, x, a, N, scaled=scaled)
    partial_sum = ICB_g_series(y, t, x, a, Nmax=N, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1

    coefficient = 2.0 * x * math.pi**2 * y / a**3
    next_index = N + 1
    tail_bound = 2.0 * next_index**2 * math.exp(-alpha * (next_index**2 - shift))

    return partial_sum - coefficient * tail_bound


def ICB_upper_g_bound(y, t, x, a, N, scaled=False):
    """Upper g bound using the same representation as ICB_lower_g_bound."""
    alpha = _icb_gh_alpha(t, a)
    if alpha < ICB_BOUND_ALPHA0:
        if x > a / 2.0 and y > a / 2.0:
            x, y = a - x, a - y
        partial_sum = ICB_g_image_series(y, t, x, a, Nmax=N, scaled=scaled)
        return partial_sum + ICB_g_image_error(y, t, x, a, N, scaled=scaled)
    partial_sum = ICB_g_series(y, t, x, a, Nmax=N, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1

    coefficient = 2.0 * x * math.pi**2 * y / a**3
    next_index = N + 1
    tail_bound = 2.0 * next_index**2 * math.exp(-alpha * (next_index**2 - shift))

    return partial_sum + coefficient * tail_bound


def _icb_gh_compare_g(threshold, y, t, x, a, scaled=False):
    """Decide threshold<g by refining the selected representation one index."""
    N = _icb_gh_bound_start(t, a, "g")
    while True:
        lower = ICB_lower_g_bound(y, t, x, a, N, scaled=scaled)
        upper = ICB_upper_g_bound(y, t, x, a, N, scaled=scaled)
        if threshold < lower:
            return True
        if threshold >= upper:
            return False
        N = N + 1


def _icb_gh_sample_g0(x, a):
    """Draw from a density proportional to g at unit time.

    Requires 0 < x <= a / 2.
    Section 3.1 of the manuscript refers to Devroye's (2010) algorithm.
    Keep its proposals for a >= 2 and a < 2, using the manuscript's
    image/spectral bounds to resolve their acceptance comparisons.
    In the a < 2 branch, cancel the common factor exp(-alpha) from
    the envelope, series, and tail bound to avoid numerical underflow.
    """
    if not (x > 0.0 and a > 0.0 and 2.0 * x <= a):
        raise ValueError("conditions for f_0 are not met")

    if a >= 2.0:
        while True:
            candidate = _icb_gh_simulate_f0(x, a)
            if candidate <= 0.0 or candidate >= a:
                continue
            V = random.random()

            S = _icb_gh_f_n(candidate, x, a, 0)
            if not math.isfinite(S) or S <= 0.0:
                # Retrying here would silently discard positive-density
                # proposals and bias the sample if the density underflows.
                raise FloatingPointError(
                    "Image-series proposal density is not positive and finite "
                    "at an interior candidate"
                )
            Y = V * S
            if _icb_gh_compare_g(Y, candidate, 1.0, x, a):
                return candidate

    rho_term = 4.0 * math.exp(-3.0 * math.pi**2 / 8.0)
    while True:
        candidate = a * math.sqrt(random.random())
        V = random.random()

        # The common factor exp(-alpha) is cancelled throughout this branch.
        envelope = 2.0 * math.pi**2 * x * candidate / (a**3 * (1.0 - rho_term))
        Y = V * envelope
        if _icb_gh_compare_g(Y, candidate, 1.0, x, a, scaled=True):
            return candidate


def ICB_sample_g(t, x, a):
    """Sample a Brownian endpoint at time t conditional on staying in [0, a].

    Requires t > 0, a > 0, and 0 < x < a. Returns one sampled value.
    Uses the scaling and reflection identities of Devroye (2010), as
    described in Section 3.1 of the manuscript.
    Raises FloatingPointError if an interior image-series proposal density
    cannot be represented as a positive finite float.
    """
    if not math.isfinite(t) or not math.isfinite(x) or not math.isfinite(a):
        raise ValueError("ICB_sample_g requires finite t, x, and a")
    if t <= 0.0 or x <= 0.0 or x >= a or a <= 0.0:
        raise ValueError("ICB_sample_g requires t > 0 and x in (0, a)")

    scale = math.sqrt(t)
    scaled_a = a / scale

    if x <= a / 2.0:
        candidate = _icb_gh_sample_g0(x / scale, scaled_a)
        return scale * candidate

    candidate = _icb_gh_sample_g0((a - x) / scale, scaled_a)
    return a - scale * candidate


def _icb_gh_series1(alpha, Nmax=500, scaled=False):
    """Return the manuscript's partial sum S_N^(1)(alpha).

    Requires alpha > 0 and a nonnegative integer Nmax.
    Nmax selects the number of terms; it does not set an error tolerance.
    With scaled=True, cancel the common factor exp(-alpha).
    Stop if the exponential rounds to zero: all later factors are also zero.
    """
    total = 0.0
    shift = 0
    if scaled:
        shift = 1

    for n in range(1, Nmax + 1):
        exponential = math.exp(-alpha * (n**2 - shift))
        if exponential == 0.0:
            break
        term = n**2 * exponential
        total = total + term

    return total


def _icb_gh_series2(beta, Nmax=500):
    """Return the manuscript's partial sum S_N^(2)(beta).

    Requires beta > 0 and a nonnegative integer Nmax.
    Nmax selects the number of terms; it does not set an error tolerance.
    Stop if a term rounds to zero: every later term is also zero.
    """
    total = 0.0

    for n in range(1, Nmax + 1):
        term = math.exp(-beta * n**2)
        if term == 0.0:
            break
        total = total + term

    return total


def _icb_gh_q_R(y, t, a):
    """Evaluate the truncated Rayleigh proposal density at one y.

    Requires t > 0 and a > 0. The density is zero outside [0, a].
    This is f_R with normalizing constant D_1 in the manuscript.
    """
    if y < 0.0 or y > a:
        return 0.0

    beta = _icb_gh_beta(t, a)
    normalizer = t * (-math.expm1(-beta))
    numerator = y * math.exp(-y**2 / (2.0 * t))

    return numerator / normalizer


def _icb_gh_q_root_U(y, a):
    """Evaluate the density of a * sqrt(U), with U uniform on [0, 1].

    Requires a > 0. Evaluate at one y; return zero outside [0, a].
    This is a component of the mixture proposal in Algorithm 1.
    """
    if y < 0.0 or y > a:
        return 0.0

    normalizer = a**2 / 2.0
    return y / normalizer


def _icb_gh_q_triangle(y, a):
    """Evaluate the symmetric triangular proposal density at one y.

    Requires a > 0. The density is zero outside [0, a].
    This is the proposal density used by Algorithm 2.
    """
    if y < 0.0 or y > a:
        return 0.0

    normalizer = a**2 / 4.0
    numerator = min(y, a - y)

    return numerator / normalizer


def ICB_h_series(y, t, a, Nmax=500, scaled=False):
    """Evaluate the first Nmax spectral terms of h at one y.

    This is the partial sum S_N^h in Section 3.1 of the manuscript.
    Requires t > 0, a > 0, and a nonnegative integer Nmax.
    Nmax sets the cutoff, not an error tolerance.
    With scaled=True, cancel the common factor exp(-alpha).
    Stop if the exponential rounds to zero: all later factors are also zero.
    """
    if y < 0.0 or y > a:
        return 0.0

    Kh = _icb_gh_Kh(t, a)
    alpha = _icb_gh_alpha(t, a)
    total = 0.0
    shift = 0
    if scaled:
        shift = 1

    for n in range(1, Nmax + 1):
        exponential = math.exp(-alpha * (n**2 - shift))
        if exponential == 0.0:
            break
        sin_y = math.sin(math.pi * n * y / a)
        term = n * sin_y * exponential
        total = total + term

    return Kh * total


def _icb_gh_image_q(value, t, shift):
    """One nonnegative image term, value*exp(-value**2/(2*t)+shift)."""
    return value * math.exp(-value**2 / (2.0 * t) + shift)


def _icb_gh_image_q_difference(value, gap, t, shift):
    """Return q(value)-q(value+gap), preserving a small nonnegative gap."""
    if value == 0.0:
        return -_icb_gh_image_q(gap, t, shift)
    log_ratio = math.log1p(gap / value) - gap * (2.0 * value + gap) / (2.0 * t)
    if log_ratio <= 0.0:
        return _icb_gh_image_q(value, t, shift) * (-math.expm1(log_ratio))
    return _icb_gh_image_q(value + gap, t, shift) * math.expm1(-log_ratio)


def ICB_h_image_series(y, t, a, Nmax=0, scaled=False):
    """Return the image partial sum of h over -Nmax,...,Nmax (Proposition 7).

    Regroup the same finite sum near a so paired terms retain a-y.
    Nmax may be any nonnegative integer; the tail bound has its own cutoff.
    With scaled=True, multiply each term by exp(alpha) in its exponent.
    """
    _icb_gh_check_image_index(Nmax)
    if y < 0.0 or y > a:
        return 0.0
    shift = _icb_gh_alpha(t, a) if scaled else 0.0
    if y <= a / 2.0:
        terms = [_icb_gh_image_q(y, t, shift)]
        for n in range(1, Nmax + 1):
            terms.append(-_icb_gh_image_q_difference(2.0 * n * a - y, 2.0 * y, t, shift))
    else:
        terms = [_icb_gh_image_q(2.0 * Nmax * a + y, t, shift)]
        for n in range(Nmax):
            terms.append(_icb_gh_image_q_difference(2.0 * n * a + y, 2.0 * (a - y), t, shift))
    return math.fsum(terms)


def ICB_h_image_error(y, t, a, N, scaled=False):
    """Return the one-sided image-tail radius in Proposition 7.

    Requires N>=max(0,ceil((sqrt(t)/a-1)/2)). Below this index the
    monotonicity needed for the tail bound is not guaranteed.
    """
    _icb_gh_check_image_index(N)
    if N < _icb_gh_image_n0(t, a):
        raise ValueError("The h image bound requires N >= the Proposition 7 cutoff")
    distance = 2.0 * N * a + a + (a - y)
    shift = _icb_gh_alpha(t, a) if scaled else 0.0
    return _icb_gh_image_q(distance, t, shift)


def ICB_lower_h_bound(y, t, a, N, scaled=False):
    """Lower h bound: Proposition 7 if alpha<pi, Proposition 5 otherwise.

    Start at _icb_gh_bound_start(t, a, 'h'). The image lower bound is
    S_N-error_N; the upper bound uses the sharper S_N from Proposition 7.
    With scaled=True, multiply both the sum and radius by exp(alpha).
    """
    alpha = _icb_gh_alpha(t, a)
    if alpha < ICB_BOUND_ALPHA0:
        error = ICB_h_image_error(y, t, a, N, scaled=scaled)
        if a / 2.0 < y <= a:
            # This is S_N-error_N, regrouped before subtraction to retain
            # small positive values when y is close to a.
            shift = alpha if scaled else 0.0
            terms = []
            for n in range(N + 1):
                terms.append(_icb_gh_image_q_difference(2.0 * n * a + y, 2.0 * (a - y), t, shift))
            return math.fsum(terms)
        return ICB_h_image_series(y, t, a, Nmax=N, scaled=scaled) - error
    Kh = _icb_gh_Kh(t, a)
    partial_sum = ICB_h_series(y, t, a, Nmax=N, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1

    coefficient = Kh * math.pi * y / a
    next_index = N + 1
    tail_bound = 2.0 * next_index**2 * math.exp(-alpha * (next_index**2 - shift))

    return partial_sum - coefficient * tail_bound


def ICB_upper_h_bound(y, t, a, N, scaled=False):
    """Upper h bound: the sharper S_N in images, S_N+radius in spectral."""
    alpha = _icb_gh_alpha(t, a)
    if alpha < ICB_BOUND_ALPHA0:
        _icb_gh_check_image_index(N)
        if N < _icb_gh_image_n0(t, a):
            raise ValueError("The h image bound requires N >= the Proposition 7 cutoff")
        return ICB_h_image_series(y, t, a, Nmax=N, scaled=scaled)
    Kh = _icb_gh_Kh(t, a)
    partial_sum = ICB_h_series(y, t, a, Nmax=N, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1

    coefficient = Kh * math.pi * y / a
    next_index = N + 1
    tail_bound = 2.0 * next_index**2 * math.exp(-alpha * (next_index**2 - shift))

    return partial_sum + coefficient * tail_bound


def _icb_gh_compare_h(threshold, y, t, a, scaled=False):
    """Decide threshold<h by refining the selected representation one index."""
    N = _icb_gh_bound_start(t, a, "h")
    while True:
        lower = ICB_lower_h_bound(y, t, a, N, scaled=scaled)
        upper = ICB_upper_h_bound(y, t, a, N, scaled=scaled)
        if threshold < lower:
            return True
        if threshold >= upper:
            return False
        N = N + 1


def _icb_gh_lower_D3_bound(t, a, N, scaled=False):
    """Return a lower bound for D_3, the triangular-envelope constant.

    See the approximating sequences in Section 3.1 of the manuscript.
    Requires t > 0, a > 0, and integer N > N0.
    This uses N + 1 terms of S^(1).
    With scaled=True, cancel the common factor exp(-alpha).
    """
    alpha = _icb_gh_alpha(t, a)
    coefficient = alpha**1.5 * a**2 / math.sqrt(math.pi)
    partial_sum = _icb_gh_series1(alpha, Nmax=N + 1, scaled=scaled)

    return coefficient * partial_sum


def _icb_gh_upper_D3_bound(t, a, N, scaled=False):
    """Return the upper bound for the triangular-envelope constant D_3.

    See the approximating sequences in Section 3.1 of the manuscript.
    Requires t > 0, a > 0, and integer N > _icb_gh_n0(alpha, beta),
    with its default rho = 0.5.
    With scaled=True, cancel the common factor exp(-alpha).
    """
    alpha = _icb_gh_alpha(t, a)
    coefficient = alpha**1.5 * a**2 / math.sqrt(math.pi)
    partial_sum = _icb_gh_series1(alpha, Nmax=N + 1, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1

    next_index = N + 1
    tail_bound = 2.0 * next_index**2 * math.exp(-alpha * (next_index**2 - shift))
    return coefficient * (partial_sum + tail_bound)


def ICB_sample_h(t, a):
    """Sample one value with density proportional to h(y; t, a).

    Implements Algorithms 1 and 2 in Section 3.1 of the manuscript.
    This is the lower-boundary starting case. Requires finite t > 0, a > 0.
    Selects proposals at ICB_ALPHA0 and density representations at alpha=pi.
    Refine one image pair or spectral term at a time. The triangular branch
    cancels exp(-alpha) from both sides of its comparison.
    """
    if not math.isfinite(t) or not math.isfinite(a) or t <= 0.0 or a <= 0.0:
        raise ValueError("ICB_sample_h requires finite t > 0 and a > 0")

    alpha = _icb_gh_alpha(t, a)
    beta = _icb_gh_beta(t, a)
    N0 = _icb_gh_n0(alpha, beta)

    if alpha <= ICB_ALPHA0:
        rayleigh_mass = -math.expm1(-beta)
        D1 = t * rayleigh_mass
        series = _icb_gh_series2(beta, Nmax=N0 + 1)
        extra_term = math.exp(-beta * (N0 + 1)**2)
        D2 = (a**2 / 2.0) * (series + extra_term)
        p = D1 / (D1 + D2)

        while True:
            if random.random() < p:
                U = random.random()
                Y = math.sqrt(-2.0 * t * math.log1p(-U * rayleigh_mass))
            else:
                Y = a * math.sqrt(random.random())

            # A computer's uniform generator can return exactly zero.
            if Y <= 0.0 or Y >= a:
                continue

            denominator = D1 * _icb_gh_q_R(Y, t, a) + D2 * _icb_gh_q_root_U(Y, a)
            U = random.random()
            threshold = U * denominator
            if _icb_gh_compare_h(threshold, Y, t, a):
                return Y

    while True:
        U = random.random()
        if U < 0.5:
            Y = (a / math.sqrt(2.0)) * math.sqrt(U)
        else:
            Y = a - (a / math.sqrt(2.0)) * math.sqrt(1.0 - U)

        if Y <= 0.0 or Y >= a:
            continue

        U = random.random()
        proposal_density = _icb_gh_q_triangle(Y, a)
        N = _icb_gh_bound_start(t, a, "h")
        # D_3 remains a spectral series, even when h uses image bounds.
        envelope_N = N0 + 1

        while True:
            lower = ICB_lower_h_bound(Y, t, a, N, scaled=True)
            upper = ICB_upper_h_bound(Y, t, a, N, scaled=True)
            lower_D3 = _icb_gh_lower_D3_bound(t, a, envelope_N, scaled=True)
            upper_D3 = _icb_gh_upper_D3_bound(t, a, envelope_N, scaled=True)

            # Bound both sides because D_3 also contains an infinite series.
            if U * upper_D3 * proposal_density < lower:
                return Y
            if U * lower_D3 * proposal_density >= upper:
                break
            N = N + 1
            envelope_N = envelope_N + 1


def ICB_C1_g(t, x, a, scaled=False):
    """Return the first constant g upper bound from Proposition 3.

    Requires t > 0, 0 < x < a, and a / sqrt(t) < 2.
    With scaled=True, cancel the common factor exp(-alpha).
    """
    delta_term = 4.0 * math.exp(-3.0 * math.pi**2 / 8.0)
    coefficient = 2.0 * math.pi**2 * x / (a**2 * (1.0 - delta_term))
    if scaled:
        return coefficient
    alpha = _icb_gh_alpha(t, a)
    return coefficient * math.exp(-alpha)


def ICB_C2_g(t, x, a, scaled=False):
    """Return the second constant g upper bound from Proposition 3.

    Requires t > 0 and 0 < x < a. The sampler uses this when a/sqrt(t) >= 2.
    With scaled=True, cancel exp(-alpha); intended for that low-alpha regime.
    """
    exponent = (x + a)**2 / (2.0 * t)
    bound = (-math.expm1(-exponent)) / math.sqrt(2.0 * math.pi * t)
    if scaled:
        bound = bound * math.exp(_icb_gh_alpha(t, a))
    return bound


def ICB_D1_h(t, a, scaled=False):
    """Return the first constant h upper bound used for bridge interpolation.

    Requires t > 0 and a > 0. With scaled=True, cancel exp(-alpha);
    this option is intended for the low-alpha regime.
    """
    scale = math.sqrt(t)
    if a < scale:
        maximum = a * math.exp(-a**2 / (2.0 * t))
    else:
        maximum = scale * math.exp(-0.5)

    alpha = _icb_gh_alpha(t, a)
    beta = _icb_gh_beta(t, a)
    N0 = _icb_gh_n0(alpha, beta)
    series = _icb_gh_series2(beta, Nmax=N0 + 1)
    extra_term = math.exp(-beta * (N0 + 1)**2)
    bound = maximum + a * (series + extra_term)
    if scaled:
        bound = bound * math.exp(alpha)
    return bound


def ICB_D2_h(t, a, scaled=False):
    """Return the second constant h upper bound used for bridge interpolation.

    Requires t > 0 and a > 0. With scaled=True, cancel exp(-alpha).
    The bound includes the N0+1 partial sum and one additional tail term.
    """
    alpha = _icb_gh_alpha(t, a)
    beta = _icb_gh_beta(t, a)
    N0 = _icb_gh_n0(alpha, beta)
    coefficient = 4.0 * alpha**1.5 / math.sqrt(math.pi)
    series = _icb_gh_series1(alpha, Nmax=N0 + 1, scaled=scaled)
    shift = 0
    if scaled:
        shift = 1
    last_index = N0 + 1
    extra_term = last_index**2 * math.exp(-alpha * (last_index**2 - shift))
    return coefficient * (series + extra_term) * (a / 2.0)


def _icb_gh_sample_g_h_product(g_time, h_time, x, a, reflect_h=False):
    """Propose from g and accept against the existing constant bound for h.

    The target is proportional to g(y; g_time, x, a) times h(y; h_time, a),
    or h(a-y; h_time, a) when reflect_h is True. Inputs are validated by the
    bridge routines. Keep both durations explicitly to avoid subtracting
    twice. Only the acceptance argument is reflected; return the proposal.
    """
    alpha = _icb_gh_alpha(h_time, a)
    scaled = alpha >= ICB_ALPHA0
    if scaled:
        D = ICB_D2_h(h_time, a, scaled=True)
    else:
        D = ICB_D1_h(h_time, a)

    while True:
        Y = ICB_sample_g(g_time, x, a)
        if Y <= 0.0 or Y >= a:
            continue
        h_argument = _icb_gh_reflect_interior(Y, a) if reflect_h else Y
        threshold = random.random() * D
        if _icb_gh_compare_h(threshold, h_argument, h_time, a, scaled=scaled):
            return Y


def ICB_interpolate_BB_I(t, T, x, z, a):
    """Sample a constrained bridge from interior x to interior z at time t.

    Algorithm 3 in Section 3.2 of the manuscript. Requires 0 < t < T,
    a > 0, and 0 < x, z < a. All inputs must be finite.
    Propose from the endpoint with the shorter duration by reversing time
    when t > T-t. The target product is unchanged; the acceptance factor
    uses the longer duration. At equal durations retain the original side.
    """
    if not math.isfinite(t) or not math.isfinite(T) or not math.isfinite(a):
        raise ValueError("ICB_interpolate_BB_I requires finite times and a")
    if not (0.0 < t < T and a > 0.0 and 0.0 < x < a and 0.0 < z < a):
        raise ValueError("ICB_interpolate_BB_I requires 0 < t < T and 0 < x, z < a")

    remaining_time = T - t
    if t > remaining_time:
        t, remaining_time = remaining_time, t
        x, z = z, x
    if a / math.sqrt(remaining_time) < 2.0:
        C = ICB_C1_g(remaining_time, z, a, scaled=True)
    else:
        C = ICB_C2_g(remaining_time, z, a, scaled=True)

    while True:
        Y = ICB_sample_g(t, x, a)
        if Y <= 0.0 or Y >= a:
            continue
        threshold = random.random() * C
        if _icb_gh_compare_g(threshold, Y, remaining_time, z, a, scaled=True):
            return Y


def ICB_interpolate_BB_II(t, T, z, a):
    """Sample a constrained bridge from 0 to interior z at time t.

    Algorithm 4 in Section 3.2 of the manuscript. Requires finite inputs,
    0 < t < T, a > 0, and 0 < z < a.
    Use the h proposal when t <= T-t. Otherwise propose from g at the
    final endpoint and accept against h at duration t using its existing
    constant bound. Choose the side before drawing any random numbers.
    """
    if not math.isfinite(t) or not math.isfinite(T) or not math.isfinite(a):
        raise ValueError("ICB_interpolate_BB_II requires finite times and a")
    if not (0.0 < t < T and a > 0.0 and 0.0 < z < a):
        raise ValueError("ICB_interpolate_BB_II requires 0 < t < T and 0 < z < a")

    remaining_time = T - t
    if t > remaining_time:
        return _icb_gh_sample_g_h_product(remaining_time, t, z, a)
    if a / math.sqrt(remaining_time) < 2.0:
        C = ICB_C1_g(remaining_time, z, a, scaled=True)
    else:
        C = ICB_C2_g(remaining_time, z, a, scaled=True)

    while True:
        Y = ICB_sample_h(t, a)
        if Y <= 0.0 or Y >= a:
            continue
        threshold = random.random() * C
        if _icb_gh_compare_g(threshold, Y, remaining_time, z, a, scaled=True):
            return Y


def ICB_interpolate_BB_III(t, T, x, a):
    """Sample a constrained bridge from interior x to a at time t.

    Algorithm 5 in Section 3.2 of the manuscript. Requires finite inputs,
    0 < t < T, a > 0, and 0 < x < a.
    Use the reflected h proposal when T-t <= t. Otherwise propose from g
    at the initial endpoint and accept against h(a-y; T-t, a) using its
    existing constant bound. Return y without reflecting the accepted draw.
    """
    if not math.isfinite(t) or not math.isfinite(T) or not math.isfinite(a):
        raise ValueError("ICB_interpolate_BB_III requires finite times and a")
    if not (0.0 < t < T and a > 0.0 and 0.0 < x < a):
        raise ValueError("ICB_interpolate_BB_III requires 0 < t < T and 0 < x < a")

    remaining_time = T - t
    if t < remaining_time:
        return _icb_gh_sample_g_h_product(t, remaining_time, x, a, reflect_h=True)

    if a / math.sqrt(t) < 2.0:
        C = ICB_C1_g(t, x, a, scaled=True)
    else:
        C = ICB_C2_g(t, x, a, scaled=True)

    while True:
        Y = _icb_gh_reflect_interior(ICB_sample_h(remaining_time, a), a)
        threshold = random.random() * C
        if _icb_gh_compare_g(threshold, Y, t, x, a, scaled=True):
            return Y


def ICB_interpolate_BB_IV(t, T, a):
    """Sample a constrained bridge from 0 to a at time t.

    Algorithm 6 in Section 3.2 of the manuscript. Requires finite inputs,
    0 < t < T and a > 0. Uses the common crossover ICB_ALPHA0,
    displayed as alpha0 = 1.086165 in the draft.

    Near the final boundary, reverse time and reflect the bridge:
    X_t has the same law as a - X_(T-t) for a constrained 0-to-a bridge.
    We therefore sample at the earlier of t and T-t, and reflect the result
    if needed. This exact symmetry avoids a very short remaining-time test.
    The orientation is chosen before drawing any random numbers.
    """
    if not math.isfinite(t) or not math.isfinite(T) or not math.isfinite(a):
        raise ValueError("ICB_interpolate_BB_IV requires finite times and a")
    if not (0.0 < t < T and a > 0.0):
        raise ValueError("ICB_interpolate_BB_IV requires 0 < t < T and a > 0")

    remaining_time = T - t
    reflect_result = False
    if t > T / 2.0:
        # Save both durations rather than subtracting twice at working precision.
        original_time = t
        t = remaining_time
        remaining_time = original_time
        reflect_result = True
    alpha = _icb_gh_alpha(remaining_time, a)
    scaled = False
    if alpha < ICB_ALPHA0:
        D = ICB_D1_h(remaining_time, a)
    else:
        scaled = True
        D = ICB_D2_h(remaining_time, a, scaled=True)

    while True:
        Y = ICB_sample_h(t, a)
        if Y <= 0.0 or Y >= a:
            continue
        threshold = random.random() * D
        if _icb_gh_compare_h(threshold, a - Y, remaining_time, a, scaled=scaled):
            if reflect_result:
                return _icb_gh_reflect_interior(Y, a)
            return Y


def ICB_interpolate_step(t, T, x, z, a):
    """Sample one interval-constrained bridge value, selecting the endpoint case.

    Requires finite inputs, 0 < t < T, a > 0, and endpoints in [0, a].
    Only exact equality identifies a boundary; endpoints are not snapped.
    Raises FloatingPointError if reflection loses an interior position.
    See Algorithms 3-6 in Section 3.2 of the manuscript.
    Same-boundary cases remain unsupported.
    """
    if not math.isfinite(t) or not math.isfinite(T) or not math.isfinite(a):
        raise ValueError("ICB_interpolate_step requires finite times and a")
    if not (0.0 < t < T and a > 0.0):
        raise ValueError("ICB_interpolate_step requires 0 < t < T and a > 0")

    start_kind = _icb_gh_endpoint_kind(x, a)
    final_kind = _icb_gh_endpoint_kind(z, a)

    if start_kind == "interior" and final_kind == "interior":
        return ICB_interpolate_BB_I(t, T, x, z, a)
    if start_kind == "lower" and final_kind == "interior":
        return ICB_interpolate_BB_II(t, T, z, a)
    if start_kind == "lower" and final_kind == "upper":
        return ICB_interpolate_BB_IV(t, T, a)
    if start_kind == "interior" and final_kind == "upper":
        return ICB_interpolate_BB_III(t, T, x, a)
    if start_kind == "interior" and final_kind == "lower":
        reflected_x = _icb_gh_reflect_interior(x, a)
        return _icb_gh_reflect_interior(
            ICB_interpolate_BB_III(t, T, reflected_x, a), a
        )
    if start_kind == "upper" and final_kind == "lower":
        return _icb_gh_reflect_interior(ICB_interpolate_BB_IV(t, T, a), a)
    if start_kind == "upper" and final_kind == "interior":
        reflected_z = _icb_gh_reflect_interior(z, a)
        return _icb_gh_reflect_interior(
            ICB_interpolate_BB_II(t, T, reflected_z, a), a
        )

    raise NotImplementedError("Same-boundary bridge interpolation is not implemented yet")
