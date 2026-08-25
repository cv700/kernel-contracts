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
# GPU-dispatched candidates (torch, device="cuda" -- covers both real CUDA
# and ROCm builds of torch, see silicon.py's docstring on that quirk).
# These are what actually make a corpus entry silicon-specific instead of
# a numpy simulation of one. Requires torch with a CUDA or ROCm build
# installed; import is deferred into the function bodies so this module
# still imports cleanly (and the CPU-simulated candidates above still run)
# on a machine with no torch at all, e.g. this one.
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for GPU-dispatched candidates. "
            "Install a CUDA or ROCm build (pip install torch --index-url ...; "
            "see README provisioning notes) -- not needed for the CPU-"
            "simulated candidates above."
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch is installed but no CUDA/ROCm device is visible. "
            "Check you're on a GPU instance and the driver matches the "
            "torch build (nvidia-smi / rocm-smi should show the device)."
        )
    return torch


def matmul_candidate_torch(A, B, accum_dtype_name, device="cuda"):
    """Runs GEMM on real GPU silicon via torch, accumulating at
    accum_dtype_name. This is the actual candidate a corpus entry's
    silicon-specific behavior comes from -- unlike
    matmul_candidate_downcast_accumulator above, this doesn't force a
    particular outcome; it reports whatever the vendor's kernel actually
    does, which is the entire point of measuring rather than simulating.
    """
    torch = _require_torch()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if accum_dtype_name not in dtype_map:
        raise ValueError(f"unsupported accum_dtype_name {accum_dtype_name!r}; add it to dtype_map if needed")
    dt = dtype_map[accum_dtype_name]
    A_t = torch.from_numpy(A).to(device=device, dtype=dt)
    B_t = torch.from_numpy(B).to(device=device, dtype=dt)
    y_t = A_t @ B_t
    return y_t.to(dtype=torch.float32).cpu().numpy()


def softmax_candidate_torch(x, device="cuda", stabilized=True):
    """Runs softmax on real GPU silicon via torch. stabilized=False
    disables torch's internal stabilization where possible by computing
    exp/sum manually instead of calling torch.softmax (which is always
    stabilized) -- this is what actually exercises the C-PRC-02 overflow
    pattern on real hardware rather than assuming numpy's behavior
    generalizes.
    """
    torch = _require_torch()
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    if stabilized:
        y_t = torch.softmax(x_t, dim=-1)
    else:
        e = torch.exp(x_t)
        y_t = e / torch.sum(e, dim=-1, keepdim=True)
    return y_t.cpu().numpy()


# ---------------------------------------------------------------------------
# Not yet implemented -- see README "Status". Each needs a reference impl,
# a conforming candidate, and an injected-bad candidate before it can
# produce corpus entries.
# ---------------------------------------------------------------------------
#   - attention (row-sum invariant + accumulator preservation, section 3.11)
#   - reduction_order (C-ORD-01, path-dependence diameter under reordering)
#   - gather / index_select (C-EXC-02, OOB policy: RAISE|CLAMP|ZERO|UNDEFINED)
