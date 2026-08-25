"""Tolerance structures from the Kernel Contracts paper, section 3.5.

Pure numpy. Runs anywhere, including CPU-only, with no GPU dependency.
This module implements tau: the function that decides whether a candidate
output y is acceptable against a reference y_star. It does not touch
silicon -- that's harness/run.py's job.
"""
import numpy as np


def backward_error(x, f, y, delta_grid=None, norm=np.inf):
    """tau_back(x, y*, y) per paper section 3.5.

    Finds the smallest perturbation delta to the input x such that the
    exact operator f, applied to x + delta, reproduces y exactly (within
    numerical search resolution). Returns ||delta|| / candidate scale --
    the caller divides by eta separately to get the acceptance test.

    f must be an exact (or high-precision reference) implementation of the
    mathematical operator, e.g. numpy in float64.

    This is a numerical search, not a closed form, except for operators
    where the backward-error problem has an analytic solution (linear
    operators like matmul admit one; nonlinear operators like softmax
    generally don't and need the search or a bound from the literature).
    For v0 we implement the closed-form case only (see matmul_backward_error
    below) and raise NotImplementedError otherwise -- do not silently
    fall back to forward error, which is a different tolerance structure
    with different guarantees.
    """
    raise NotImplementedError(
        "Generic backward-error search is not implemented in v0. "
        "Use an operator-specific closed form (see matmul_backward_error) "
        "or add one -- do not substitute forward error silently."
    )


def matmul_backward_error(A, B, y, y_ref_high_precision):
    """Closed-form backward error for GEMM: the minimal Frobenius-norm
    perturbation to (A, B) such that (A+dA)(B+dB) = y exactly is bounded,
    to first order, by ||y - y_ref|| / (||A|| ||B||) -- the standard
    numerical-analysis bound (Higham, Accuracy and Stability of Numerical
    Algorithms, ch. 3). y_ref_high_precision should be computed in float64
    with the same reduction order as the candidate where possible.
    """
    resid = np.linalg.norm(y - y_ref_high_precision, ord="fro")
    scale = np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro")
    if scale == 0:
        return 0.0 if resid == 0 else np.inf
    return resid / scale


def pointwise_relative(y_star, y, eps=1e-12):
    """tau_point with the relative metric (paper section 3.5)."""
    denom = np.maximum(np.abs(y_star), eps)
    return float(np.max(np.abs(y_star - y) / denom))


def pointwise_absolute(y_star, y):
    return float(np.max(np.abs(y_star - y)))


def invariant_row_sum(y, target=1.0, axis=-1):
    """tau_inv for the attention/softmax row-sum invariant
    (paper section 3.11, clause phi_4): rows of y should sum to `target`.
    Returns 0.0 if satisfied, else the max violation magnitude -- caller
    compares against eta, not against a hardcoded threshold.
    """
    sums = np.sum(y, axis=axis)
    return float(np.max(np.abs(sums - target)))


def derive_eta_from_unit_roundoff(dtype_accum, condition_number_correction=1.0):
    """Derive a backward-error tolerance eta from first principles, per
    paper section 3.11's fix to the v1 circularity: eta must come from the
    declared accumulator's unit roundoff, never from measuring the
    candidate under test.

    Unit roundoff values per IEEE 754-2019.
    """
    unit_roundoff = {
        "fp8_e4m3": 2 ** -3,   # ~0.0625, 3 mantissa bits
        "fp8_e5m2": 2 ** -2,   # ~0.25, 2 mantissa bits
        "fp16": 2 ** -11,
        "bf16": 2 ** -8,
        "fp32": 2 ** -24,
        "fp64": 2 ** -53,
    }
    if dtype_accum not in unit_roundoff:
        raise ValueError(f"unknown dtype_accum {dtype_accum!r}; add it to unit_roundoff before deriving eta")
    return unit_roundoff[dtype_accum] * condition_number_correction


def satisfies(divergence, eta):
    """The acceptance test: tau(x, y*, y) <= 1, restated as divergence <= eta.
    Returns a plain Python bool (not numpy.bool_) so callers can json.dump
    the result without a custom encoder. nan divergence correctly returns
    False -- a NaN output is not a passing candidate.
    """
    return bool(divergence <= eta)
