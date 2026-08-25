"""Three-state calibration (paper section 5.1): every contract must admit
a reference-conforming candidate and an injected-bad candidate that a
naive test suite would still pass. This module runs both candidates
against a tolerance function and records whether the calibration is
sound -- i.e. whether the contract would actually catch the bad candidate.

A calibration where the bad candidate passes is not a failed test run.
It is evidence the contract itself is under-specified (paper section 3.4,
case 3) -- report it as such, don't discard it.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import tolerance as tol


@dataclass
class CalibrationResult:
    reference_conforming_result: str  # "pass" | "fail"
    reference_conforming_divergence: float
    injected_bad_result: str
    injected_bad_divergence: float
    injection_method: str
    sound: bool  # True iff conforming passes AND bad fails


def run_calibration(
    x,
    reference_fn: Callable,
    conforming_fn: Callable,
    bad_fn: Callable,
    tolerance_fn: Callable,
    eta: float,
    injection_method: str,
) -> CalibrationResult:
    """
    x: input(s) drawn from the calibration distribution mu.
    reference_fn: high-precision reference (e.g. operators.matmul_reference).
    conforming_fn: candidate expected to satisfy the contract.
    bad_fn: candidate expected to violate the contract.
    tolerance_fn: callable(y_star, y) -> divergence (float).
    eta: tolerance bound.
    """
    y_star = reference_fn(*x) if isinstance(x, tuple) else reference_fn(x)

    y_conforming = conforming_fn(*x) if isinstance(x, tuple) else conforming_fn(x)
    d_conforming = tolerance_fn(y_star, y_conforming)
    conforming_result = "pass" if tol.satisfies(d_conforming, eta) else "fail"

    y_bad = bad_fn(*x) if isinstance(x, tuple) else bad_fn(x)
    d_bad = tolerance_fn(y_star, y_bad)
    bad_result = "pass" if tol.satisfies(d_bad, eta) else "fail"

    sound = (conforming_result == "pass") and (bad_result == "fail")

    return CalibrationResult(
        reference_conforming_result=conforming_result,
        reference_conforming_divergence=d_conforming,
        injected_bad_result=bad_result,
        injected_bad_divergence=d_bad,
        injection_method=injection_method,
        sound=sound,
    )


def check_soundness_or_raise(result: CalibrationResult, contract_class: str):
    """A contract whose bad candidate passes is not corpus-ready. Fail
    loudly rather than silently emitting an entry with sound=False --
    the schema records the calibration result either way, but run.py
    should not present an unsound calibration as a finished measurement
    without the operator explicitly overriding this check.
    """
    if not result.sound:
        raise ValueError(
            f"{contract_class}: calibration is NOT sound "
            f"(conforming={result.reference_conforming_result}, "
            f"bad={result.injected_bad_result}). "
            f"The contract's tolerance (eta) or clause set is under-specified "
            f"for this injection method ({result.injection_method}). "
            f"Fix the contract before recording this as a measured entry, "
            f"or record it explicitly as verdict=under_specified."
        )
