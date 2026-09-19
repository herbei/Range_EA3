"""Optimized exact samplers for three-dimensional Bessel exit times.

BESSEL_G samples the entrance-boundary hitting time of level x.
BESSEL_F samples the hitting time of level x when starting at r > 0.
BESSEL_F_T samples from BESSEL_F conditional on occurring before T.

Reference:
    Herbei, R. (2026). Exact and Efficient Simulation of Exit Times for
    Three-Dimensional Bessel Processes.
"""

import math
import random
import sys


_BESSELEXIT_KAPPA_1_F = 1.262
_BESSELEXIT_KAPPA_2_F = 1.536
_BESSELEXIT_G_SWITCH = 0.403723
_BESSELEXIT_G_IMAGE_UPPER = 0.9033402881
_BESSELEXIT_U_E = 2.0 * math.log(2.0) / math.pi**2


def _besselexit_check_finite_scalar(value, name):
    """Check that value is one finite number."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")

    return value


def _besselexit_check_positive_scalar(value, name):
    """Check that value is one finite, positive number."""
    _besselexit_check_finite_scalar(value, name)

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _besselexit_check_nonnegative_integer(value, name="n"):
    """Check that value is one non-negative integer."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if (
        not is_number
        or not math.isfinite(value)
        or value < 0
        or value != math.floor(value)
        or value > 2_147_483_647
    ):
        raise ValueError(f"{name} must be a non-negative integer scalar")

    return int(value)


def _besselexit_check_f_parameters(x, r):
    """Check the parameters for the positive-start exit-time density f."""
    _besselexit_check_positive_scalar(x, "x")
    _besselexit_check_positive_scalar(r, "r")

    if r >= x:
        raise ValueError("r must satisfy 0 < r < x")

    return True


def _besselexit_lambda_x(x):
    """Return the spectral rate pi squared divided by 2 x squared."""
    _besselexit_check_positive_scalar(x, "x")

    return math.pi**2 / (2.0 * x**2)


def _besselexit_levy_density(t, a):
    """Return the Levy density at time t with distance parameter a."""
    is_number = isinstance(t, (int, float)) and not isinstance(t, bool)

    if not is_number:
        raise ValueError("t must be numeric")

    _besselexit_check_positive_scalar(a, "a")

    if math.isnan(t):
        return math.nan

    if not math.isfinite(t) or t <= 0:
        return 0.0

    return (
        a
        / (math.sqrt(2.0 * math.pi) * t ** (3.0 / 2.0))
        * math.exp(-(a**2) / (2.0 * t))
    )


def _besselexit_levy_log_cdf(s, a):
    """Return the logarithm of the Levy CDF at s."""
    is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not is_number:
        raise ValueError("s must be numeric")

    _besselexit_check_positive_scalar(a, "a")

    if math.isnan(s):
        return math.nan

    if s == math.inf:
        return 0.0

    if not math.isfinite(s) or s <= 0:
        return -math.inf

    cdf = math.erfc(a / math.sqrt(2.0 * s))

    if cdf == 0.0:
        return -math.inf

    return math.log(cdf)


def _besselexit_levy_cdf(s, a):
    """Return the Levy CDF at s."""
    log_cdf = _besselexit_levy_log_cdf(s, a)

    return math.exp(log_cdf)


def _besselexit_levy_cdf_diff(s, a_lower, a_upper):
    """Return F_a_lower(s) minus F_a_upper(s) stably."""
    is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not is_number:
        raise ValueError("s must be numeric")

    _besselexit_check_positive_scalar(a_lower, "a_lower")
    _besselexit_check_positive_scalar(a_upper, "a_upper")

    if a_lower >= a_upper:
        raise ValueError("a_lower must be smaller than a_upper")

    log_lower = _besselexit_levy_log_cdf(s, a_lower)
    log_upper = _besselexit_levy_log_cdf(s, a_upper)

    if math.isnan(log_lower) or math.isnan(log_upper):
        return math.nan

    if not math.isfinite(log_lower) or not math.isfinite(log_upper):
        return 0.0

    log_ratio = min(log_upper - log_lower, 0.0)

    return math.exp(log_lower) * (-math.expm1(log_ratio))


def _besselexit_levy_density_pair_diff(t, a_lower, r):
    """Return ell_a_lower(t) minus ell_(a_lower + 2r)(t) stably."""
    is_number = isinstance(t, (int, float)) and not isinstance(t, bool)

    if not is_number:
        raise ValueError("t must be numeric")

    _besselexit_check_positive_scalar(a_lower, "a_lower")
    _besselexit_check_positive_scalar(r, "r")

    if math.isnan(t):
        return math.nan

    if not math.isfinite(t) or t <= 0:
        return 0.0

    levy_lower = _besselexit_levy_density(t, a_lower)
    log_ratio = (
        math.log1p(2.0 * r / a_lower)
        - 2.0 * r * (a_lower + r) / t
    )

    return -levy_lower * math.expm1(log_ratio)


def _besselexit_exp_density_rate(t, rate):
    """Return the exponential density at t for the given rate."""
    is_number = isinstance(t, (int, float)) and not isinstance(t, bool)

    if not is_number:
        raise ValueError("t must be numeric")

    _besselexit_check_positive_scalar(rate, "rate")

    if math.isnan(t):
        return math.nan

    if not math.isfinite(t) or t < 0:
        return 0.0

    return rate * math.exp(-rate * t)


def _besselexit_rinv_gamma_truncated_upper(n, shape, beta, upper):
    """Draw inverse-gamma values conditional on being below upper."""
    n = _besselexit_check_nonnegative_integer(n)
    _besselexit_check_positive_scalar(shape, "shape")
    _besselexit_check_positive_scalar(beta, "beta")
    _besselexit_check_positive_scalar(upper, "upper")

    if n == 0:
        return []

    gamma_lower_bound = beta / upper

    if not math.isfinite(gamma_lower_bound):
        raise ValueError(
            "upper is too small for the truncated inverse-gamma sampler"
        )

    draws = []

    while len(draws) < n:
        # Python uses a scale parameter here.  Scale 1 equals R's rate 1.
        gamma_draw = random.gammavariate(shape, 1.0)

        # beta / gamma_draw is below upper exactly when this is true.
        if gamma_draw >= gamma_lower_bound:
            inverse_gamma_draw = beta / gamma_draw
            draws.append(inverse_gamma_draw)

    return draws


def _besselexit_log_gamma_f(s, r, x):
    """Return log Gamma(s; r, x) for the paired-image correction."""
    s_is_number = isinstance(s, (int, float)) and not isinstance(s, bool)
    r_is_number = isinstance(r, (int, float)) and not isinstance(r, bool)
    x_is_number = isinstance(x, (int, float)) and not isinstance(x, bool)

    if not s_is_number:
        raise ValueError("s must be numeric")

    if not r_is_number or not math.isfinite(r) or r < 0:
        raise ValueError("r must be a finite non-negative scalar")

    if not x_is_number or not math.isfinite(x) or x <= r:
        raise ValueError("x must be a finite scalar with x > r")

    d_minus = x - r

    if not math.isfinite(s) or s <= 0 or s >= d_minus**2:
        raise ValueError(
            "s must satisfy 0 < s < (x - r)^2"
        )

    return (
        math.log((3.0 * x - r) ** 2 - s)
        - math.log(d_minus**2 - s)
        - 2.0 * x * (2.0 * x - r) / s
    )


def _besselexit_one_minus_gamma_f(s, r, x):
    """Return one minus Gamma(s; r, x) stably."""
    log_gamma = _besselexit_log_gamma_f(s, r, x)

    return -math.expm1(log_gamma)


def _besselexit_c_i_g(s, x):
    """Return the image-envelope constant C_I for density g."""
    _besselexit_check_positive_scalar(x, "x")

    s_is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not s_is_number or not math.isfinite(s) or s <= 0 or s >= x**2:
        raise ValueError("s must satisfy 0 < s < x^2")

    one_minus_gamma = _besselexit_one_minus_gamma_f(s, 0.0, x)

    if one_minus_gamma <= 0:
        raise ValueError("C_I_g requires Gamma(s, 0, x) < 1")

    return 1.0 / one_minus_gamma


def _besselexit_d_i_g(s, x):
    """Return the mass D_I of the image proposal for density g."""
    _besselexit_check_positive_scalar(x, "x")

    s_is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not s_is_number or not math.isfinite(s) or s <= 0 or s >= x**2:
        raise ValueError("s must satisfy 0 < s < x^2")

    return (
        2.0
        * x
        * math.sqrt(2.0 / (math.pi * s))
        * math.exp(-(x**2) / (2.0 * s))
    )


def _besselexit_d_e_g(s, x):
    """Return the mass D_E of the exponential proposal for density g."""
    _besselexit_check_positive_scalar(x, "x")

    s_is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not s_is_number or not math.isfinite(s) or s < 0:
        raise ValueError("s must be a finite non-negative number")

    rate = _besselexit_lambda_x(x)

    return math.exp(-rate * s)


def _besselexit_b_g_i_term(t, x, k):
    """Return term k of the image series for density g."""
    a = (2.0 * k + 1.0) ** 2 * x**2
    scaled = a / t

    return (
        2.0
        * x
        / (math.sqrt(2.0 * math.pi) * t ** (3.0 / 2.0))
        * (scaled - 1.0)
        * math.exp(-scaled / 2.0)
    )


def _besselexit_l_n_g_i(t, x, n=0):
    """Return the image-series partial sum L_N for density g."""
    n = _besselexit_check_nonnegative_integer(n, "N")
    value = 0.0

    for k in range(n + 1):
        term = _besselexit_b_g_i_term(t, x, k)
        value = value + term

    return value


def _besselexit_u_n_g_i(t, x, n=0):
    """Return the image-series upper bound U_N for density g."""
    n = _besselexit_check_nonnegative_integer(n, "N")
    _besselexit_check_positive_scalar(x, "x")

    t_is_number = isinstance(t, (int, float)) and not isinstance(t, bool)

    if not t_is_number or not math.isfinite(t) or t <= 0 or t >= x**2:
        return math.nan

    one_minus_gamma = _besselexit_one_minus_gamma_f(t, 0.0, x)

    if one_minus_gamma <= 0:
        return math.nan

    lower_bound = _besselexit_l_n_g_i(t, x, n)
    next_term = _besselexit_b_g_i_term(t, x, n + 1)

    return lower_bound + next_term / one_minus_gamma


def _besselexit_g_e_partial_sum(t, x, n_max):
    """Return the first n_max terms of the spectral series for density g."""
    n_max = _besselexit_check_nonnegative_integer(n_max, "n_max")
    rate = _besselexit_lambda_x(x)
    value = 0.0

    for n in range(1, n_max + 1):
        sign = (-1.0) ** (n + 1)
        term = (
            sign
            * (math.pi**2 / x**2)
            * n**2
            * math.exp(-rate * t * n**2)
        )
        value = value + term

    return value


def _besselexit_l_n_g_e(t, x, n=0):
    """Return the spectral-series lower bound L_N for density g."""
    n = _besselexit_check_nonnegative_integer(n, "N")
    number_of_terms = 2 * n + 2

    return _besselexit_g_e_partial_sum(t, x, number_of_terms)


def _besselexit_u_n_g_e(t, x, n=0):
    """Return the spectral-series upper bound U_N for density g."""
    n = _besselexit_check_nonnegative_integer(n, "N")
    number_of_terms = 2 * n + 1

    return _besselexit_g_e_partial_sum(t, x, number_of_terms)


def BESSEL_G_accept(
    t,
    x,
    threshold,
    representation="auto",
    max_refinements=100000,
):
    """Decide whether threshold lies below the density g(t; x)."""
    _besselexit_check_positive_scalar(t, "t")
    _besselexit_check_positive_scalar(x, "x")
    _besselexit_check_finite_scalar(threshold, "threshold")
    max_refinements = _besselexit_check_nonnegative_integer(
        max_refinements,
        "max_refinements",
    )

    allowed_representations = ("auto", "image", "spectral")

    if representation not in allowed_representations:
        raise ValueError(
            "representation must be 'auto', 'image', or 'spectral'"
        )

    if threshold <= 0:
        return True

    if representation == "auto":
        if t < _BESSELEXIT_G_IMAGE_UPPER * x**2:
            representation = "image"
        else:
            representation = "spectral"

    n = 0
    refinements = 0

    while True:
        if representation == "image":
            lower = _besselexit_l_n_g_i(t, x, n)
            upper = _besselexit_u_n_g_i(t, x, n)
        else:
            lower = _besselexit_l_n_g_e(t, x, n)
            upper = _besselexit_u_n_g_e(t, x, n)

        if math.isfinite(lower) and threshold <= lower:
            return True

        if math.isfinite(upper) and threshold > upper:
            return False

        n = n + 1
        refinements = refinements + 1

        if refinements > max_refinements:
            raise RuntimeError("g retrospective comparison did not resolve")


def _besselexit_simulate_q_i_g(x, s, n=1):
    """Draw from the short-time image proposal for density g."""
    _besselexit_check_positive_scalar(x, "x")
    _besselexit_check_positive_scalar(s, "s")
    n = _besselexit_check_nonnegative_integer(n)

    if s >= x**2:
        raise ValueError("s must satisfy 0 < s < x^2")

    if n == 0:
        return []

    draws = []
    x_squared = x**2

    while len(draws) < n:
        gamma_draw = random.gammavariate(3.0 / 2.0, 1.0)

        if gamma_draw == 0.0:
            continue

        candidate = x_squared / (2.0 * gamma_draw)

        if candidate <= s:
            acceptance_probability = 1.0 - candidate / x_squared

            if random.random() <= acceptance_probability:
                draws.append(candidate)

    return draws


def BESSEL_G(x, n=1, max_refinements=100000):
    """Draw entrance-boundary Bessel exit times using Algorithm 6."""
    _besselexit_check_positive_scalar(x, "x")
    n = _besselexit_check_nonnegative_integer(n)
    max_refinements = _besselexit_check_nonnegative_integer(
        max_refinements,
        "max_refinements",
    )

    if n == 0:
        return []

    # Build the image-plus-exponential proposal envelope.
    switch_time = x**2 * _BESSELEXIT_G_SWITCH
    rate = _besselexit_lambda_x(x)
    c_i = _besselexit_c_i_g(switch_time, x)
    c_e = 2.0
    d_i = _besselexit_d_i_g(switch_time, x)
    d_e = _besselexit_d_e_g(switch_time, x)
    total_mass = c_i * d_i + c_e * d_e
    probability_image = c_i * d_i / total_mass

    draws = []

    while len(draws) < n:
        # Draw from one of the two proposal components.
        if random.random() <= probability_image:
            candidate = _besselexit_simulate_q_i_g(
                x,
                switch_time,
                n=1,
            )[0]
            envelope_height = (
                c_i * _besselexit_b_g_i_term(candidate, x, 0)
            )
            representation = "image"
        else:
            exponential_draw = random.expovariate(1.0)
            candidate = switch_time + exponential_draw / rate
            envelope_height = (
                c_e * _besselexit_exp_density_rate(candidate, rate)
            )
            representation = "spectral"

        # Keep the candidate only when the point under the envelope
        # also lies below the target density g.
        threshold = random.random() * envelope_height

        if BESSEL_G_accept(
            candidate,
            x,
            threshold,
            representation=representation,
            max_refinements=max_refinements,
        ):
            draws.append(candidate)

    return draws


def _besselexit_c_e_f(r, x):
    """Return the spectral-envelope constant C_E for density f."""
    _besselexit_check_f_parameters(x, r)
    rho = r / x

    first_bound = 2.0 * _BESSELEXIT_KAPPA_1_F / (math.pi * rho)
    second_bound = (
        2.0
        * _BESSELEXIT_KAPPA_2_F
        * min(1.0, (1.0 - rho) / rho)
    )

    return min(first_bound, second_bound)


def _besselexit_d_i_f(s, r, x):
    """Return the mass D_I of the one-image proposal for density f."""
    _besselexit_check_f_parameters(x, r)
    distance = x - r

    return _besselexit_levy_cdf(s, distance)


def _besselexit_d_p_f(s, r, x):
    """Return the mass D_P of the paired-image proposal for density f."""
    _besselexit_check_f_parameters(x, r)
    lower_distance = x - r
    upper_distance = x + r

    return _besselexit_levy_cdf_diff(
        s,
        lower_distance,
        upper_distance,
    )


def _besselexit_d_e_f(s, x):
    """Return the mass D_E of the exponential proposal for density f."""
    _besselexit_check_positive_scalar(x, "x")

    s_is_number = isinstance(s, (int, float)) and not isinstance(s, bool)

    if not s_is_number:
        raise ValueError("s must be numeric")

    rate = _besselexit_lambda_x(x)

    return math.exp(-rate * s)


def _besselexit_m_i_f_u(u, rho):
    """Return the one-image envelope mass at dimensionless switch u."""
    _besselexit_check_positive_scalar(rho, "rho")

    if rho >= 1:
        raise ValueError("rho must satisfy 0 < rho < 1")

    u_is_number = isinstance(u, (int, float)) and not isinstance(u, bool)
    upper = (1.0 + rho) ** 2

    if (
        not u_is_number
        or not math.isfinite(u)
        or u < _BESSELEXIT_U_E
        or u > upper
    ):
        return math.inf

    c_e = _besselexit_c_e_f(rho, 1.0)
    image_mass = _besselexit_levy_cdf(u, 1.0 - rho) / rho
    exponential_mass = c_e * math.exp(-(math.pi**2) * u / 2.0)

    return image_mass + exponential_mass


def _besselexit_m_p_f_u(u, rho):
    """Return the paired-image envelope mass at dimensionless switch u."""
    _besselexit_check_positive_scalar(rho, "rho")

    if rho >= 1:
        raise ValueError("rho must satisfy 0 < rho < 1")

    u_is_number = isinstance(u, (int, float)) and not isinstance(u, bool)
    upper = (1.0 - rho) ** 2

    if (
        not u_is_number
        or not math.isfinite(u)
        or u < _BESSELEXIT_U_E
        or u >= upper
    ):
        return math.inf

    log_gamma = _besselexit_log_gamma_f(u, rho, 1.0)

    if not math.isfinite(log_gamma) or log_gamma >= 0:
        return math.inf

    one_minus_gamma = -math.expm1(log_gamma)
    d_pair = _besselexit_levy_cdf_diff(
        u,
        1.0 - rho,
        1.0 + rho,
    )
    c_e = _besselexit_c_e_f(rho, 1.0)
    paired_mass = d_pair / (rho * one_minus_gamma)
    exponential_mass = c_e * math.exp(-(math.pi**2) * u / 2.0)

    return paired_mass + exponential_mass


def _besselexit_u_p_interval_f(rho):
    """Return the admissible interval for the paired-image switch u."""
    _besselexit_check_positive_scalar(rho, "rho")

    if rho >= 1:
        raise ValueError("rho must satisfy 0 < rho < 1")

    lower = _BESSELEXIT_U_E
    upper_domain = (1.0 - rho) ** 2

    if lower >= upper_domain:
        return None

    lower_value = _besselexit_log_gamma_f(lower, rho, 1.0)

    if lower_value >= 0:
        return None

    tolerance = math.sqrt(sys.float_info.epsilon)
    upper_probe = upper_domain - max(
        tolerance * upper_domain,
        sys.float_info.epsilon,
    )

    if upper_probe <= lower:
        return None

    upper_value = _besselexit_log_gamma_f(upper_probe, rho, 1.0)

    if not math.isfinite(upper_value) or upper_value <= 0:
        return None

    left = lower
    right = upper_probe

    while right - left > tolerance:
        midpoint = (left + right) / 2.0
        midpoint_value = _besselexit_log_gamma_f(midpoint, rho, 1.0)

        if midpoint_value < 0:
            left = midpoint
        else:
            right = midpoint

    root = (left + right) / 2.0

    if not math.isfinite(root) or root <= lower:
        return None

    return [lower, root]


def _besselexit_optimize_with_endpoints(fn, lower, upper):
    """Minimize fn on a closed interval using golden-section search."""
    _besselexit_check_finite_scalar(lower, "lower")
    _besselexit_check_finite_scalar(upper, "upper")

    if lower > upper:
        raise ValueError("invalid optimization interval")

    if not callable(fn):
        raise ValueError("fn must be a function")

    left = lower
    right = upper
    tolerance = math.sqrt(sys.float_info.epsilon)

    if left < right:
        golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
        point_1 = right - golden_fraction * (right - left)
        point_2 = left + golden_fraction * (right - left)
        value_1 = fn(point_1)
        value_2 = fn(point_2)

        while right - left > tolerance:
            if value_1 <= value_2:
                right = point_2
                point_2 = point_1
                value_2 = value_1
                point_1 = right - golden_fraction * (right - left)
                value_1 = fn(point_1)
            else:
                left = point_1
                point_1 = point_2
                value_1 = value_2
                point_2 = left + golden_fraction * (right - left)
                value_2 = fn(point_2)

        interior_minimum = (left + right) / 2.0
    else:
        interior_minimum = left

    candidates = [lower, upper, interior_minimum]
    best_u = None
    best_mass = math.inf

    for candidate in candidates:
        mass = fn(candidate)

        if math.isfinite(mass) and mass < best_mass:
            best_u = candidate
            best_mass = mass

    if best_u is None:
        raise RuntimeError("optimization failed to find a finite envelope mass")

    return {"u": best_u, "mass": best_mass}


def _besselexit_select_envelope_f(x, r):
    """Select the lower-mass envelope for the positive-start density f."""
    _besselexit_check_f_parameters(x, r)
    rho = r / x

    def one_image_mass(u):
        return _besselexit_m_i_f_u(u, rho)

    def paired_image_mass(u):
        return _besselexit_m_p_f_u(u, rho)

    one_image = _besselexit_optimize_with_endpoints(
        one_image_mass,
        _BESSELEXIT_U_E,
        (1.0 + rho) ** 2,
    )

    paired_interval = _besselexit_u_p_interval_f(rho)
    paired_image = None

    if paired_interval is not None:
        paired_image = _besselexit_optimize_with_endpoints(
            paired_image_mass,
            paired_interval[0],
            paired_interval[1],
        )

    if paired_image is None or one_image["mass"] <= paired_image["mass"]:
        method = "I"
        u_star = one_image["u"]
        mass = one_image["mass"]
    else:
        method = "P"
        u_star = paired_image["u"]
        mass = paired_image["mass"]

    if paired_image is None:
        paired_mass = math.nan
        paired_u = math.nan
    else:
        paired_mass = paired_image["mass"]
        paired_u = paired_image["u"]

    return {
        "method": method,
        "u": u_star,
        "s": x**2 * u_star,
        "mass": mass,
        "rho": rho,
        "mass_I": one_image["mass"],
        "mass_P": paired_mass,
        "u_I": one_image["u"],
        "u_P": paired_u,
        "CE": _besselexit_c_e_f(r, x),
    }


def _besselexit_f_image_bounds(t, x, r, n=0):
    """Return lower and upper bounds from the image series for density f."""
    _besselexit_check_f_parameters(x, r)
    n = _besselexit_check_nonnegative_integer(n, "N")

    d_minus = x - r
    d_plus = x + r
    pair_sum = 0.0

    for k in range(n + 1):
        lower_distance = d_minus + 2.0 * k * x
        pair = _besselexit_levy_density_pair_diff(
            t,
            lower_distance,
            r,
        )
        pair_sum = pair_sum + pair

    lower = (x / r) * pair_sum
    remainder_distance = d_plus + 2.0 * n * x
    remainder_bound = _besselexit_levy_density(
        t,
        remainder_distance,
    )
    upper = lower + (x / r) * remainder_bound

    return {"lower": lower, "upper": upper}


def _besselexit_f_eigen_partial_sum(t, x, r, n=0):
    """Return the first N terms of the eigenfunction series for density f."""
    _besselexit_check_f_parameters(x, r)
    n = _besselexit_check_nonnegative_integer(n, "N")

    if n == 0:
        return 0.0

    rate = _besselexit_lambda_x(x)
    value = 0.0

    for index in range(1, n + 1):
        weight = (
            (-1.0) ** (index + 1)
            * index
            * math.sin(index * math.pi * r / x)
        )
        exponential = math.exp(-rate * t * index**2)
        value = value + weight * exponential

    return (math.pi / (x * r)) * value


def _besselexit_n0_f_e(a):
    """Return the first index at which both spectral tail ratios are below 1."""
    _besselexit_check_positive_scalar(a, "a")
    n = 0

    while True:
        index_ratio = (n + 2.0) / (n + 1.0)
        exponential = math.exp(-a * (2.0 * n + 3.0))
        q_1 = index_ratio * exponential
        q_2 = index_ratio**2 * exponential

        if q_1 < 1.0 and q_2 < 1.0:
            return n

        n = n + 1

        if n > 2_147_483_646:
            raise RuntimeError("failed to find a finite spectral starting index")


def _besselexit_f_eigen_remainder_bound(a, x, r, n):
    """Return a bound for the uncomputed tail of the f eigenfunction series."""
    index_ratio = (n + 2.0) / (n + 1.0)
    step_decay = math.exp(-a * (2.0 * n + 3.0))
    q_1 = index_ratio * step_decay
    q_2 = index_ratio**2 * step_decay

    first_omitted_index = n + 1.0
    first_omitted_decay = math.exp(-a * first_omitted_index**2)

    bound_1 = (
        math.pi
        * first_omitted_index
        * first_omitted_decay
        / (x * r * (1.0 - q_1))
    )
    bound_2 = (
        math.pi**2
        * first_omitted_index**2
        * first_omitted_decay
        / (x**2 * (1.0 - q_2))
    )

    return min(bound_1, bound_2)


def _besselexit_f_eigen_bounds(t, x, r, n=None):
    """Return rigorous lower and upper eigenfunction bounds for density f."""
    _besselexit_check_positive_scalar(t, "t")
    _besselexit_check_f_parameters(x, r)

    a = _besselexit_lambda_x(x) * t
    n_0 = _besselexit_n0_f_e(a)

    if n is None:
        n = n_0
    else:
        n = max(_besselexit_check_nonnegative_integer(n, "N"), n_0)

    lower = -math.inf
    upper = math.inf

    for k in range(n_0, n + 1):
        partial_sum = _besselexit_f_eigen_partial_sum(t, x, r, k)
        remainder = _besselexit_f_eigen_remainder_bound(a, x, r, k)
        lower = max(lower, partial_sum - remainder)
        upper = min(upper, partial_sum + remainder)

    return {
        "lower": lower,
        "upper": upper,
        "N0": n_0,
        "N": n,
    }


def BESSEL_F_accept(
    t,
    x,
    r,
    threshold,
    representation="auto",
    switch=None,
    max_refinements=100000,
):
    """Decide whether threshold lies below the density f(t; r, x)."""
    _besselexit_check_positive_scalar(t, "t")
    _besselexit_check_f_parameters(x, r)
    _besselexit_check_finite_scalar(threshold, "threshold")
    max_refinements = _besselexit_check_nonnegative_integer(
        max_refinements,
        "max_refinements",
    )

    allowed_representations = ("auto", "image", "spectral")

    if representation not in allowed_representations:
        raise ValueError(
            "representation must be 'auto', 'image', or 'spectral'"
        )

    if threshold <= 0:
        return True

    if representation == "auto":
        if switch is None:
            switch = _besselexit_select_envelope_f(x, r)["s"]
        else:
            _besselexit_check_positive_scalar(switch, "switch")

        if t <= switch:
            representation = "image"
        else:
            representation = "spectral"

    if representation == "spectral":
        n = _besselexit_n0_f_e(_besselexit_lambda_x(x) * t)
    else:
        n = 0

    refinements = 0

    while True:
        if representation == "image":
            bounds = _besselexit_f_image_bounds(t, x, r, n)
        else:
            bounds = _besselexit_f_eigen_bounds(t, x, r, n)

        if threshold <= bounds["lower"]:
            return True

        if threshold > bounds["upper"]:
            return False

        n = n + 1
        refinements = refinements + 1

        if refinements > max_refinements:
            raise RuntimeError("f retrospective comparison did not resolve")


def _besselexit_simulate_q_i_f(x, r, s, n=1):
    """Draw from the one-image proposal for density f."""
    _besselexit_check_f_parameters(x, r)
    _besselexit_check_positive_scalar(s, "s")
    n = _besselexit_check_nonnegative_integer(n)

    if n == 0:
        return []

    distance = x - r
    beta = distance**2 / 2.0

    return _besselexit_rinv_gamma_truncated_upper(
        n,
        shape=1.0 / 2.0,
        beta=beta,
        upper=s,
    )


def _besselexit_simulate_q_p_f_one(x, r, s):
    """Draw one value from the paired-image proposal for density f."""
    d_minus = x - r
    log_distance_ratio = math.log1p(2.0 * r / d_minus)

    while True:
        uniform = random.random()
        distance = d_minus * math.exp(log_distance_ratio * uniform)
        gamma_draw = random.gammavariate(3.0 / 2.0, 1.0)

        if gamma_draw == 0.0:
            continue

        candidate = distance**2 / (2.0 * gamma_draw)

        if candidate <= s:
            acceptance_probability = 1.0 - candidate / distance**2

            if random.random() <= acceptance_probability:
                return candidate


def _besselexit_simulate_q_p_f(x, r, s, n=1):
    """Draw values from the paired-image proposal for density f."""
    _besselexit_check_f_parameters(x, r)
    _besselexit_check_positive_scalar(s, "s")
    n = _besselexit_check_nonnegative_integer(n)

    d_minus = x - r

    if s >= d_minus**2:
        raise ValueError(
            "s must satisfy 0 < s < (x - r)^2 "
            "for the paired-image sampler"
        )

    draws = []

    for draw_number in range(n):
        candidate = _besselexit_simulate_q_p_f_one(x, r, s)
        draws.append(candidate)

    return draws


def BESSEL_F(x, r, n=1, max_refinements=100000):
    """Draw positive-start Bessel exit times using Algorithm 4."""
    _besselexit_check_f_parameters(x, r)
    n = _besselexit_check_nonnegative_integer(n)
    max_refinements = _besselexit_check_nonnegative_integer(
        max_refinements,
        "max_refinements",
    )

    if n == 0:
        return []

    # Construct the optimized short-time-plus-exponential envelope.
    envelope = _besselexit_select_envelope_f(x, r)
    method = envelope["method"]
    switch_time = envelope["s"]
    d_minus = x - r
    rate = _besselexit_lambda_x(x)
    c_e = envelope["CE"]
    d_e = _besselexit_d_e_f(switch_time, x)

    if method == "I":
        c_star = x / r
        d_star = _besselexit_d_i_f(switch_time, r, x)
    else:
        one_minus_gamma = _besselexit_one_minus_gamma_f(
            switch_time,
            r,
            x,
        )
        c_star = x / (r * one_minus_gamma)
        d_star = _besselexit_d_p_f(switch_time, r, x)

    total_mass = c_star * d_star + c_e * d_e
    probability_short_time = c_star * d_star / total_mass

    if (
        not math.isfinite(total_mass)
        or total_mass <= 0
        or not math.isfinite(probability_short_time)
        or probability_short_time < 0
        or probability_short_time > 1
    ):
        raise RuntimeError("failed to construct a valid f proposal envelope")

    draws = []

    while len(draws) < n:
        # Draw from the short-time or exponential component.
        if random.random() <= probability_short_time:
            if method == "I":
                candidate = _besselexit_simulate_q_i_f(
                    x,
                    r,
                    switch_time,
                    n=1,
                )[0]
                envelope_height = (
                    c_star * _besselexit_levy_density(candidate, d_minus)
                )
            else:
                candidate = _besselexit_simulate_q_p_f(
                    x,
                    r,
                    switch_time,
                    n=1,
                )[0]
                envelope_height = (
                    c_star
                    * _besselexit_levy_density_pair_diff(
                        candidate,
                        d_minus,
                        r,
                    )
                )

            representation = "image"
        else:
            exponential_draw = random.expovariate(1.0)
            candidate = switch_time + exponential_draw / rate
            envelope_height = (
                c_e * _besselexit_exp_density_rate(candidate, rate)
            )
            representation = "spectral"

        threshold = random.random() * envelope_height

        if BESSEL_F_accept(
            candidate,
            x,
            r,
            threshold,
            representation=representation,
            max_refinements=max_refinements,
        ):
            draws.append(candidate)

    return draws


def BESSEL_F_T(x, r, T, n=1, max_refinements=100000):
    """Draw positive-start Bessel exit times conditional on being below T."""
    _besselexit_check_f_parameters(x, r)
    _besselexit_check_positive_scalar(T, "T")
    n = _besselexit_check_nonnegative_integer(n)

    if n == 0:
        return []

    draws = []

    while len(draws) < n:
        candidate = BESSEL_F(
            x,
            r,
            n=1,
            max_refinements=max_refinements,
        )[0]

        if candidate < T:
            draws.append(candidate)

    return draws


def BESSEL_EXIT(x, r=0.0, n=1, max_refinements=100000):
    """Draw exit times for a Bessel process from the interval [0, x]."""
    _besselexit_check_positive_scalar(x, "x")
    _besselexit_check_finite_scalar(r, "r")

    if r < 0.0 or r >= x:
        raise ValueError("r must satisfy 0 <= r < x")

    if r == 0.0:
        return BESSEL_G(
            x,
            n=n,
            max_refinements=max_refinements,
        )

    return BESSEL_F(
        x,
        r,
        n=n,
        max_refinements=max_refinements,
    )
