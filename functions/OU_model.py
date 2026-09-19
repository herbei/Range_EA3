"""Ornstein-Uhlenbeck model functions for the exact sampling algorithm.

The model is

    dX_t = (theta1 - theta2 * X_t) dt + dW_t,

where theta2 must be positive.
"""

import math
import random


def _ou_check_scalar(value, name, positive=False):
    """Check that value is one finite number, and optionally positive."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not is_number or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")

    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def OU_model(theta1, theta2):
    """Create the Ornstein-Uhlenbeck model functions."""
    _ou_check_scalar(theta1, "theta1")
    _ou_check_scalar(theta2, "theta2", positive=True)

    def alpha(u):
        """Return the drift at state u."""
        return theta1 - theta2 * u

    def A(u):
        """Return an antiderivative of the drift."""
        return theta1 * u - theta2 * u**2 / 2

    def phi(u):
        """Return the nonnegative potential at state u."""
        return 0.5 * (theta1 - theta2 * u) ** 2

    def phi_bound(lower, upper):
        """Return the maximum potential on the interval [lower, upper]."""
        _ou_check_scalar(lower, "lower")
        _ou_check_scalar(upper, "upper")

        if lower > upper:
            raise ValueError("phi_bound requires finite lower <= upper")

        return max(phi(lower), phi(upper))

    def endpoint_mean(x, T):
        """Return the mean of the proposal endpoint distribution."""
        _ou_check_scalar(x, "x")
        _ou_check_scalar(T, "T", positive=True)

        return (x + theta1 * T) / (1 + theta2 * T)

    def endpoint_var(T):
        """Return the variance of the proposal endpoint distribution."""
        _ou_check_scalar(T, "T", positive=True)

        return T / (1 + theta2 * T)

    def sample_endpoint(x, T, n=1):
        """Draw n values from the proposal endpoint distribution."""
        _ou_check_scalar(x, "x")
        _ou_check_scalar(T, "T", positive=True)

        is_integer = isinstance(n, int) and not isinstance(n, bool)
        if not is_integer or n <= 0:
            raise ValueError("n must be a positive integer")

        mean = endpoint_mean(x, T)
        standard_deviation = math.sqrt(endpoint_var(T))

        return [
            random.gauss(mean, standard_deviation)
            for _ in range(n)
        ]

    def endpoint_density(z, x, T):
        """Return the proposal endpoint density at z."""
        _ou_check_scalar(z, "z")
        _ou_check_scalar(x, "x")
        _ou_check_scalar(T, "T", positive=True)

        mean = endpoint_mean(x, T)
        variance = endpoint_var(T)
        squared_distance = (z - mean) ** 2

        return math.exp(-squared_distance / (2 * variance)) / math.sqrt(2 * math.pi * variance)

    def transition_mean(x, t):
        """Return the mean of X_t given that X_0 = x."""
        _ou_check_scalar(x, "x")
        _ou_check_scalar(t, "t")

        long_run_mean = theta1 / theta2

        return long_run_mean + (x - long_run_mean) * math.exp(-theta2 * t)

    def transition_var(t):
        """Return the variance of X_t given its value at time zero."""
        _ou_check_scalar(t, "t")

        if t == 0:
            return 0

        if t < 0:
            raise ValueError("t must be non-negative")

        return (1 - math.exp(-2 * theta2 * t)) / (2 * theta2)

    def transition_density(y, x, t):
        """Return the transition density at state y."""
        _ou_check_scalar(y, "y")
        _ou_check_scalar(x, "x")

        variance = transition_var(t)

        if variance == 0:
            if y == x:
                return math.inf

            return 0

        mean = transition_mean(x, t)
        squared_distance = (y - mean) ** 2

        return math.exp(-squared_distance / (2 * variance)) / math.sqrt(
            2 * math.pi * variance
        )

    def sample_transition(x, delta, n=1):
        """Draw n values from the OU transition distribution."""
        is_integer = isinstance(n, int) and not isinstance(n, bool)
        if not is_integer or n <= 0:
            raise ValueError("n must be a positive integer")

        mean = transition_mean(x, delta)
        variance = transition_var(delta)
        standard_deviation = math.sqrt(variance)

        return [random.gauss(mean, standard_deviation) for _ in range(n)]

    def bridge_mean(t, T, x, z):
        """Return the conditional mean at time t for an OU bridge."""
        _ou_check_scalar(t, "t")
        _ou_check_scalar(T, "T", positive=True)
        _ou_check_scalar(x, "x")
        _ou_check_scalar(z, "z")

        if t < 0 or t > T:
            raise ValueError("t must lie in [0, T]")

        if t == 0:
            return x

        if t == T:
            return z

        q_t = transition_var(t)
        q_T = transition_var(T)
        c_tT = math.exp(-theta2 * (T - t)) * q_t

        return transition_mean(x, t) + c_tT * (z - transition_mean(x, T)) / q_T

    def bridge_var(t, T):
        """Return the conditional variance at time t for an OU bridge."""
        _ou_check_scalar(t, "t")
        _ou_check_scalar(T, "T", positive=True)

        if t < 0 or t > T:
            raise ValueError("t must lie in [0, T]")

        if t == 0 or t == T:
            return 0

        q_t = transition_var(t)
        q_T = transition_var(T)
        c_tT = math.exp(-theta2 * (T - t)) * q_t

        return q_t - c_tT**2 / q_T

    def bridge_density(y, t, T, x, z):
        """Return the OU bridge density at state y and time t."""
        _ou_check_scalar(y, "y")
        _ou_check_scalar(x, "x")
        _ou_check_scalar(z, "z")

        variance = bridge_var(t, T)

        if variance == 0:
            target = x if t == 0 else z

            if y == target:
                return math.inf

            return 0

        mean = bridge_mean(t, T, x, z)
        squared_distance = (y - mean) ** 2

        return math.exp(-squared_distance / (2 * variance)) / math.sqrt(
            2 * math.pi * variance
        )

    def acceptance_probability(x, T):
        """Return the theoretical acceptance probability for Algorithm 2."""
        _ou_check_scalar(x, "x")
        _ou_check_scalar(T, "T", positive=True)

        exponent = (
            -theta2 * T / 2
            - T * (theta1 - theta2 * x) ** 2 / (2 * (1 + theta2 * T))
        )

        return math.sqrt(1 + theta2 * T) * math.exp(exponent)

    def bridge_acceptance_probability(x, z, T):
        """Return the theoretical acceptance probability for an OU bridge."""
        _ou_check_scalar(x, "x")
        _ou_check_scalar(z, "z")
        _ou_check_scalar(T, "T", positive=True)

        p_ou = transition_density(z, x, T)
        squared_distance = (z - x) ** 2
        p_bm = math.exp(-squared_distance / (2 * T)) / math.sqrt(2 * math.pi * T)
        kL = -theta2 / 2

        return (p_ou / p_bm) * math.exp(A(x) - A(z) + kL * T)

    return {
        "name": "Ornstein-Uhlenbeck",
        "theta1": theta1,
        "theta2": theta2,
        "kL": -theta2 / 2,
        "alpha": alpha,
        "A": A,
        "phi": phi,
        "phi_bound": phi_bound,
        "sample_endpoint": sample_endpoint,
        "endpoint_mean": endpoint_mean,
        "endpoint_var": endpoint_var,
        "endpoint_density": endpoint_density,
        "transition_mean": transition_mean,
        "transition_var": transition_var,
        "transition_density": transition_density,
        "sample_transition": sample_transition,
        "bridge_mean": bridge_mean,
        "bridge_var": bridge_var,
        "bridge_density": bridge_density,
        "acceptance_probability": acceptance_probability,
        "bridge_acceptance_probability": bridge_acceptance_probability,
    }
