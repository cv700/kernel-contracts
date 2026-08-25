"""Reference and candidate implementations for the two operators implemented
in v0: GEMM (C-PRC-01) and softmax (C-PRC-02 / row-sum invariant).

Two implemented, three not (attention, reduction-order, OOB indexing) --
see README "Status" section. Do not add stub functions for the other three
that look complete; track them as open work instead.

Everything here runs on CPU via numpy. That's enough to validate the
tolerance math and the calibration logic (this file), and to produce
'illustrative' corpus entries. It is NOT enough to produce 'measured'
entries -- those require running the candidate on real silicon (an actual
NVIDIA or AMD GPU via torch), which harness/run.py's --device flag is for.
The reference computation (numpy float64) is silicon-independent by
construction and is the same regardless of which GPU produced the
candidate.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Reference implementations (float64, silicon-independent)
# ---------------------------------------------------------------------------

def matmul_reference(A, B):
    return (A.astype(np.float64) @ B.astype(np.float64))


def softmax_reference(x, axis=-1):
    x64 = x.astype(np.float64)
    m = np.max(x64, axis=axis, keepdims=True)
    e = np.exp(x64 - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# CPU-simulated candidates, for local testing of the harness without a GPU.
#
# These simulate what a precision-constrained kernel would do by rounding
# through numpy/ml_dtypes-equivalent casts. They are NOT a substitute for
# running the operator on real NVIDIA/AMD/Apple silicon -- silicon-specific
# behavior (accumulator width, softmax stabilization choices, warp-level
# reduction order) is exactly what this project measures, and a CPU
# simulation cannot produce it. Use these only to validate the schema and
# tolerance code paths; mark any entry produced this way as
# status: "illustrative", never "measured".
# ---------------------------------------------------------------------------

def matmul_candidate_downcast_accumulator(A, B, accum_dtype=np.float16):
    """Simulates a kernel that silently accumulates at a lower precision
    than declared -- the C-PRC-01 violation pattern from the Ascend case
    study (paper section 6.1). Declares fp32 accumulation, actually
    accumulates in accum_dtype.
    """
    A16 = A.astype(accum_dtype)
    B16 = B.astype(accum_dtype)
    return (A16 @ B16).astype(np.float32)


def matmul_candidate_conforming(A, B):
    """Accumulates at the declared precision (fp32). Should pass any
    reasonable C-PRC-01 contract -- this is the reference-conforming state
    of the three-state calibration.
    """
    return (A.astype(np.float32) @ B.astype(np.float32))


def softmax_candidate_no_stabilization(x, axis=-1):
    """Naive softmax without max-subtraction. Overflows on large-magnitude
    inputs -- the C-PRC-02 cancellation/overflow pattern. This is the
    injected-bad candidate for softmax contracts. The overflow to inf/nan
    is the point of this function, not a bug in it -- errstate suppresses
    the warning so it doesn't read as an unhandled harness error.
    """
    x32 = x.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x32)
        return e / np.sum(e, axis=axis, keepdims=True)


def softmax_candidate_stabilized(x, axis=-1):
    """Max-subtraction stabilized softmax at fp32. Reference-conforming
    candidate.
    """
    x32 = x.astype(np.float32)
    m = np.max(x32, axis=axis, keepdims=True)
    e = np.exp(x32 - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Not yet implemented -- see README "Status". Each needs a reference impl,
# a conforming candidate, and an injected-bad candidate before it can
# produce corpus entries.
# ---------------------------------------------------------------------------
#   - attention (row-sum invariant + accumulator preservation, section 3.11)
#   - reduction_order (C-ORD-01, path-dependence diameter under reordering)
#   - gather / index_select (C-EXC-02, OOB policy: RAISE|CLAMP|ZERO|UNDEFINED)
