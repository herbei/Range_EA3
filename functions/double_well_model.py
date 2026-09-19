"""Double-well Langevin functions for the vanilla exact sampling algorithm.

The diffusion is dX_t = gamma * (X_t - X_t**3) dt + dW_t, with gamma > 0.
Endpoint and stationary sampling use exact Gaussian rejection, without
numerical integration. Probability and density diagnostics use deterministic
adaptive Simpson integration; these diagnostics are numerical, not exact.

Reference: Paper_H200.pdf, Section 4.2, Equations (36)-(50).
"""

import math
import random


def _double_well_check_scalar(value, name, positive=False):
    """Require one finite number, excluding Boolean values."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite scalar")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _double_well_check_count(n):
    """Accept a positive integer-valued count, including values such as 2.0."""
    _double_well_check_scalar(n, "n", positive=True)
    if n != math.floor(n):
        raise ValueError("n must be a positive integer")
    return int(n)


def _double_well_finite(value, name):
    """Fail explicitly if an intermediate result is not representable."""
    if not math.isfinite(value):
        raise ArithmeticError(f"{name} is not finite at working precision")
    return value


def _double_well_positive_exp(log_value, name):
    """Exponentiate a strictly positive result without hiding underflow."""
    _double_well_finite(log_value, name + " logarithm")
    try:
        value = math.exp(log_value)
    except OverflowError as error:
        raise ArithmeticError(f"{name} overflows; use its logarithm when available") from error
    if value == 0.0:
        raise ArithmeticError(f"{name} underflows; use its logarithm when available")
    return value


def _double_well_log_uniform():
    """Draw a log-uniform variate; retry a computer uniform equal to zero."""
    uniform = random.random()
    while uniform == 0.0:
        uniform = random.random()
    return math.log(uniform)


def _double_well_normal_expectation(weight, landmarks):
    """Numerically integrate a weight in [0, 1] against the N(0, 1) law.

    Use [-10, 10], 64 initial panels, and adaptive Simpson refinement.
    Additional landmarks expose the double-well peaks before refinement.
    The Gaussian mass omitted outside the interval is bounded by erfc.
    Simpson's error estimate plus that tail must be at most 1e-9 times
    the result. The error estimate is not a rigorous interval-arithmetic
    certificate. This diagnostic is intended for the paper's gamma=1 and
    moderate states/times, not arbitrary sharply concentrated parameters.
    Unresolved or extremely small integrals raise rather than returning
    a misleading probability. No sampler calls this function.
    """
    relative_tolerance = 1e-9
    cutoff = 10.0
    tail_bound = math.erfc(cutoff / math.sqrt(2.0))
    normalizer = math.sqrt(2.0 * math.pi)
    evaluations = 0

    def integrand(u):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 100000:
            raise ArithmeticError("Double-well diagnostic quadrature exhausted its evaluation limit")
        value = weight(u)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ArithmeticError("Double-well diagnostic weight must lie in [0, 1]")
        return math.exp(-u * u / 2.0) * value / normalizer

    def refine(left, right, f_left, f_middle, f_right, whole, tolerance, depth):
        middle = (left + right) / 2.0
        quarter_left = (left + middle) / 2.0
        quarter_right = (middle + right) / 2.0
        if quarter_left == left or quarter_right == right:
            raise ArithmeticError("Double-well quadrature reached floating-point resolution")
        f_quarter_left = integrand(quarter_left)
        f_quarter_right = integrand(quarter_right)
        first = (middle - left) * (f_left + 4 * f_quarter_left + f_middle) / 6.0
        second = (right - middle) * (f_middle + 4 * f_quarter_right + f_right) / 6.0
        difference = first + second - whole
        error = abs(difference) / 15.0
        if error <= tolerance:
            result = first + second + difference / 15.0
            if result < 0 or not math.isfinite(result):
                raise ArithmeticError("Invalid double-well quadrature result")
            return result, error
        if depth == 0:
            raise ArithmeticError("Double-well diagnostic quadrature did not converge")
        first, first_error = refine(
            left, middle, f_left, f_quarter_left, f_middle, first, tolerance / 2, depth - 1,
        )
        second, second_error = refine(
            middle, right, f_middle, f_quarter_right, f_right, second, tolerance / 2, depth - 1,
        )
        return first + second, first_error + second_error

    points = []
    for index in range(65):
        points.append(-cutoff + 2 * cutoff * index / 64)
    for point in landmarks:
        if math.isfinite(point) and -cutoff < point < cutoff:
            points.append(point)
    points = sorted(set(points))
    panels = []
    estimate = 0.0
    for index in range(len(points) - 1):
        left, right = points[index], points[index + 1]
        f_left = integrand(left)
        f_middle = integrand((left + right) / 2)
        f_right = integrand(right)
        whole = (right - left) * (f_left + 4 * f_middle + f_right) / 6
        panels.append((left, right, f_left, f_middle, f_right, whole))
        estimate += whole
    if estimate <= 0 or not math.isfinite(estimate):
        raise ArithmeticError("Double-well diagnostic integral is too small or unresolved")

    for attempt in range(3):
        tolerance = relative_tolerance * estimate / 8
        total = 0.0
        total_error = tail_bound
        for panel in panels:
            result, error = refine(*panel, tolerance / len(panels), 24)
            total += result
            total_error += error
        if total <= 0 or total > 1.0 or not math.isfinite(total):
            raise ArithmeticError("Invalid double-well diagnostic probability")
        if total_error <= relative_tolerance * total:
            return total
        if tail_bound > relative_tolerance * total:
            raise ArithmeticError("Double-well diagnostic probability is too small for its tail guarantee")
        estimate = total
    raise ArithmeticError("Double-well diagnostic quadrature did not achieve relative accuracy")


def double_well_model(gamma):
    """Return the drift, potential, exact samplers, and numerical diagnostics.

    The model is defined for gamma > 0. Very extreme finite parameters may
    be unrepresentable and then raise ArithmeticError. Gaussian rejection
    can be inefficient for very small/large gamma or distant starting
    states. There is no proposal-count cutoff that would bias the law.
    """
    _double_well_check_scalar(gamma, "gamma", positive=True)
    gamma = float(gamma)
    root = _double_well_finite(math.sqrt(1.0 + 9.0 / gamma), "stationary-point root")
    d = (3.0 / gamma) / (root + 1.0)
    y_plus = 1.0 + d
    # This equals (2-root)/3, without cancellation when gamma is near 3.
    y_minus = (gamma - 3.0) / (gamma * (2.0 + root))
    if not math.isfinite(y_plus) or y_plus <= 1.0:
        raise ArithmeticError("gamma is too extreme to resolve the potential's stationary points")
    sqrt_y_plus = math.sqrt(y_plus)
    kL = _double_well_finite(
        gamma / 2.0 * (gamma * y_plus * d * d - 2.0 - 3.0 * d), "kL",
    )
    endpoint_scale = math.sqrt(gamma) / 2.0
    stationary_scale = math.sqrt(gamma / 2.0)
    stationary_center = _double_well_finite(1.0 + 0.5 / gamma, "stationary envelope center")

    def alpha(u):
        """Return gamma * u * (1 - u**2)."""
        _double_well_check_scalar(u, "u")
        return _double_well_finite(gamma * u * (1.0 - u * u), "drift")

    def alpha_prime(u):
        """Return the derivative of the drift."""
        _double_well_check_scalar(u, "u")
        return _double_well_finite(gamma * (1.0 - 3.0 * u * u), "drift derivative")

    def A(u):
        """Return gamma*u**2/2 - gamma*u**4/4."""
        _double_well_check_scalar(u, "u")
        square = _double_well_finite(u * u, "squared state")
        return _double_well_finite(gamma * square * (2.0 - square) / 4.0, "A")

    def Psi(u):
        """Return (alpha(u)**2 + alpha_prime(u))/2."""
        drift = alpha(u)
        return _double_well_finite((drift * drift + alpha_prime(u)) / 2.0, "Psi")

    def phi(u):
        """Return Psi(u)-kL using its nonnegative factorization."""
        _double_well_check_scalar(u, "u")
        square = _double_well_finite(u * u, "squared state")
        factor = gamma * (square - y_plus) * math.sqrt((square + 2.0 * d) / 2.0)
        result = _double_well_finite(factor * factor, "phi")
        if result == 0.0 and factor != 0.0:
            raise ArithmeticError("phi underflows at working precision")
        return result

    def stationary_points():
        """Return every stationary point of phi, not just its minima."""
        points = [0.0, -sqrt_y_plus, sqrt_y_plus]
        if gamma > 3.0:
            small_root = math.sqrt(y_minus)
            points.extend([-small_root, small_root])
        return sorted(points)

    def phi_candidates(lower, upper):
        """Return endpoints and every stationary point inside [lower, upper]."""
        _double_well_check_scalar(lower, "lower")
        _double_well_check_scalar(upper, "upper")
        if lower > upper:
            raise ValueError("phi_candidates requires lower <= upper")
        candidates = [lower, upper]
        for point in stationary_points():
            if lower <= point <= upper:
                candidates.append(point)
        return sorted(set(candidates))

    def phi_bound(lower, upper):
        """Return the exact polynomial maximum on the closed interval."""
        values = []
        for point in phi_candidates(lower, upper):
            values.append(phi(point))
        return max(values)

    def log_endpoint_weight(z):
        _double_well_check_scalar(z, "z")
        factor = _double_well_finite(endpoint_scale * (z * z - 1.0), "endpoint penalty factor")
        return _double_well_finite(-factor * factor, "log endpoint acceptance")

    def sample_endpoint(x, T, n=1):
        """Return n Gaussian-rejection endpoint draws as a list, even for n=1."""
        _double_well_check_scalar(x, "x")
        _double_well_check_scalar(T, "T", positive=True)
        n = _double_well_check_count(n)
        out = []
        while len(out) < n:
            z = random.gauss(x, math.sqrt(T))
            log_probability = log_endpoint_weight(z)
            if _double_well_log_uniform() <= log_probability:
                out.append(z)
        return out

    def sample_stationary(n=1):
        """Return exact stationary draws by N(0,1) rejection, without quadrature.

        Completing the square in log(target/proposal) gives the acceptance
        exponent -gamma/2 * (z**2 - 1 - 1/(2*gamma))**2, always nonpositive.
        """
        n = _double_well_check_count(n)
        out = []
        while len(out) < n:
            z = random.gauss(0.0, 1.0)
            _double_well_check_scalar(z, "stationary candidate")
            factor = _double_well_finite(
                stationary_scale * (z * z - stationary_center), "stationary penalty factor",
            )
            log_probability = _double_well_finite(-factor * factor, "log stationary acceptance")
            if _double_well_log_uniform() <= log_probability:
                out.append(z)
        return out

    def endpoint_acceptance_probability(x, T):
        """Numerically integrate the Gaussian endpoint rejection probability."""
        _double_well_check_scalar(x, "x")
        _double_well_check_scalar(T, "T", positive=True)
        scale = math.sqrt(T)

        def weight(u):
            z = _double_well_finite(x + scale * u, "quadrature state")
            # Underflow of negligible integrand values is allowed; the total
            # integral must still pass the explicit relative-error/tail check.
            return math.exp(log_endpoint_weight(z))

        landmarks = [(z - x) / scale for z in [-1.0, 0.0, 1.0]]
        return _double_well_normal_expectation(weight, landmarks)

    def log_endpoint_acceptance_probability(x, T):
        """Return the logarithm of the numerically integrated endpoint mass."""
        return math.log(endpoint_acceptance_probability(x, T))

    def log_endpoint_normalizing_constant(x, T):
        """Return log integral exp(A(z)-(z-x)**2/(2T)) dz numerically."""
        log_mass = log_endpoint_acceptance_probability(x, T)
        return _double_well_finite(
            0.5 * (math.log(2.0 * math.pi) + math.log(T)) + gamma / 4.0 + log_mass,
            "log endpoint normalizing constant",
        )

    def endpoint_normalizing_constant(x, T):
        return _double_well_positive_exp(
            log_endpoint_normalizing_constant(x, T), "endpoint normalizing constant",
        )

    def endpoint_density(z, x, T):
        """Numerically normalize the endpoint density; fail on underflow."""
        _double_well_check_scalar(z, "z")
        log_mass = log_endpoint_acceptance_probability(x, T)
        standardized = (z - x) / math.sqrt(T)
        log_density = (-standardized * standardized / 2.0
                       - 0.5 * (math.log(2.0 * math.pi) + math.log(T))
                       + log_endpoint_weight(z) - log_mass)
        return _double_well_positive_exp(log_density, "endpoint density")

    def log_acceptance_probability(x, T):
        """Return log theoretical outer-loop acceptance using numerical mass."""
        log_mass = log_endpoint_acceptance_probability(x, T)
        # A(x)-gamma/4 equals this completed-square penalty; avoid subtracting
        # nearly equal exponentials or taking a ratio of underflowed values.
        value = _double_well_finite(log_endpoint_weight(x) + kL * T - log_mass,
                                    "log outer acceptance probability")
        if value > 0.0:
            raise ArithmeticError("Numerical outer acceptance exceeds one")
        return value

    def acceptance_probability(x, T):
        """Return the theoretical outer acceptance (a numerical diagnostic)."""
        return _double_well_positive_exp(log_acceptance_probability(x, T), "outer acceptance")

    def stationary_density_unnormalized(u):
        """Return exp(2*A(u)); this is not a normalized density."""
        return _double_well_positive_exp(2.0 * A(u), "unnormalized stationary density")

    return {
        "name": "double-well Langevin", "gamma": gamma,
        "y_plus": y_plus, "y_minus": y_minus, "kL": kL,
        "alpha": alpha, "alpha_prime": alpha_prime, "A": A, "Psi": Psi,
        "phi": phi, "stationary_points": stationary_points,
        "phi_candidates": phi_candidates, "phi_bound": phi_bound,
        "sample_endpoint": sample_endpoint, "sample_stationary": sample_stationary,
        "endpoint_acceptance_probability": endpoint_acceptance_probability,
        "log_endpoint_acceptance_probability": log_endpoint_acceptance_probability,
        "endpoint_normalizing_constant": endpoint_normalizing_constant,
        "log_endpoint_normalizing_constant": log_endpoint_normalizing_constant,
        "endpoint_density": endpoint_density,
        "acceptance_probability": acceptance_probability,
        "log_acceptance_probability": log_acceptance_probability,
        "stationary_density_unnormalized": stationary_density_unnormalized,
    }
