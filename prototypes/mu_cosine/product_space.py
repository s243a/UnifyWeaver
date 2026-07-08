#!/usr/bin/env python3
"""Finite product-space transforms for Product-Kalman / correlated-PoE prototypes.

These helpers implement the clipping convention from `DESIGN_product_kalman_poe.md`.
They are intentionally small and numpy-only: they expose stable log/logit coordinates,
lower/upper product proxies, and link Jacobians without committing to a Kalman model.

Example:

    mu = [0.25, 0.8]
    weights = normalized_geometric_weights(len(mu))
    lower, upper, width = product_interval(mu, weights)
    J = product_interval_jacobian(mu, weights)

The product proxies are AND/noisy-OR style diagnostics, not calibrated confidence
intervals. Under correlated experts the true posterior can fall outside the proxy
interval unless a downstream model calibrates the dependence structure.
"""

import numpy as np


DEFAULT_EPS = 1e-6

__all__ = [
    "DEFAULT_EPS",
    "clip_mu",
    "link_jacobian",
    "log_mu",
    "logit_mu",
    "normalized_geometric_weights",
    "product_interval",
    "product_interval_jacobian",
    "product_lower",
    "product_lower_jacobian",
    "product_lower_log",
    "product_upper",
    "product_upper_complement_log",
    "product_upper_jacobian",
]


def clip_mu(mu, eps=DEFAULT_EPS):
    """Clip membership values into the finite open interval used by log/logit links."""
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    return np.clip(np.asarray(mu, dtype=float), eps, 1.0 - eps)


def log_mu(mu, eps=DEFAULT_EPS):
    """Finite log-membership evidence, `log(clip(mu))`."""
    return np.log(clip_mu(mu, eps))


def logit_mu(mu, eps=DEFAULT_EPS):
    """Finite logit-membership evidence, `log(clip(mu)/(1-clip(mu)))`."""
    m = clip_mu(mu, eps)
    return np.log(m) - np.log1p(-m)


def link_jacobian(mu, link="logit", eps=DEFAULT_EPS):
    """Derivative d link(mu) / d mu at clipped `mu`.

    Supported links:
    - `log`:   d log(mu) / d mu = 1 / mu
    - `logit`: d logit(mu) / d mu = 1 / (mu * (1 - mu))
    """
    m = clip_mu(mu, eps)
    if link == "log":
        return 1.0 / m
    if link == "logit":
        return 1.0 / (m * (1.0 - m))
    raise ValueError(f"unknown link {link!r}; expected 'log' or 'logit'")


def _source_vector(mu):
    m = np.asarray(mu, dtype=float)
    if m.ndim != 1:
        raise ValueError("product proxies expect a 1-D source vector; batch matrices are not supported")
    if m.size == 0:
        raise ValueError("product proxies require at least one source")
    return m


def _weights_like(mu, weights):
    m = _source_vector(mu)
    if weights is None:
        return np.ones_like(m, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != m.shape:
        raise ValueError(f"weights shape {w.shape} must match mu shape {m.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    return w


def _clipped_source_and_weights(mu, weights, eps):
    m = clip_mu(_source_vector(mu), eps)
    return m, _weights_like(m, weights)


def _product_lower_log_from_clipped(m, weights):
    return float(np.sum(weights * np.log(m)))


def _product_upper_complement_log_from_clipped(m, weights):
    return float(np.sum(weights * np.log1p(-m)))


def product_lower_log(mu, weights=None, eps=DEFAULT_EPS):
    """Log of the AND-style lower-support proxy, stable for high-dimensional products."""
    m, w = _clipped_source_and_weights(mu, weights, eps)
    return _product_lower_log_from_clipped(m, w)


def product_upper_complement_log(mu, weights=None, eps=DEFAULT_EPS):
    """Log complement of the noisy-OR upper proxy, `log(prod_i (1-mu_i)^w_i)`."""
    m, w = _clipped_source_and_weights(mu, weights, eps)
    return _product_upper_complement_log_from_clipped(m, w)


def product_lower(mu, weights=None, eps=DEFAULT_EPS):
    """AND-style lower-support proxy, `prod_i clip(mu_i)^w_i`.

    The regular-space value can underflow for many small inputs; use
    `product_lower_log` when the log-evidence value is the meaningful quantity.
    """
    return float(np.exp(product_lower_log(mu, weights, eps)))


def product_upper(mu, weights=None, eps=DEFAULT_EPS):
    """Noisy-OR upper-support proxy, `1 - prod_i (1 - clip(mu_i))^w_i`."""
    return float(1.0 - np.exp(product_upper_complement_log(mu, weights, eps)))


def product_interval(mu, weights=None, eps=DEFAULT_EPS):
    """Return `(lower, upper, width)` product proxies for one source vector."""
    lo = product_lower(mu, weights, eps)
    hi = product_upper(mu, weights, eps)
    return lo, hi, hi - lo


def product_lower_jacobian(mu, weights=None, eps=DEFAULT_EPS):
    """Derivative of the lower product proxy with respect to clipped `mu`."""
    m, w = _clipped_source_and_weights(mu, weights, eps)
    lower = np.exp(_product_lower_log_from_clipped(m, w))
    return lower * w / m


def product_upper_jacobian(mu, weights=None, eps=DEFAULT_EPS):
    """Derivative of the upper noisy-OR proxy with respect to clipped `mu`."""
    m, w = _clipped_source_and_weights(mu, weights, eps)
    complement = np.exp(_product_upper_complement_log_from_clipped(m, w))
    return complement * w / (1.0 - m)


def product_interval_jacobian(mu, weights=None, eps=DEFAULT_EPS):
    """Return Jacobian rows for `(lower, upper, width)` in clipped product space.

    The width row is `d(upper-lower)/dmu`; individual entries may be negative,
    because raising one source can narrow the lower/upper proxy interval.
    """
    lo_j = product_lower_jacobian(mu, weights, eps)
    hi_j = product_upper_jacobian(mu, weights, eps)
    return np.vstack([lo_j, hi_j, hi_j - lo_j])


def normalized_geometric_weights(n):
    """Equal product exponents normalized to sum to one for geometric-mean proxies."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(int(n), 1.0 / float(n))
