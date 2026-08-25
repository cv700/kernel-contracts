#!/usr/bin/env python3
"""CLI entrypoint. Two-step workflow, because real corpus entries need two
different physical machines (one per silicon), so one process can't
produce a full entry in one shot.

Step 1 -- on each machine, once per silicon:
    python3 -m harness.run measure --operator matmul --dtype-accum fp32 \\
        --candidate conforming --seed 0 --out reading_nvidia_conforming.json
    python3 -m harness.run measure --operator matmul --dtype-accum fp32 \\
        --candidate bad --seed 0 --out reading_nvidia_bad.json
    (repeat on the second machine for readings_amd_*.json)

Step 2 -- on any machine, combine two conforming+bad reading pairs into a
schema-valid corpus entry:
    python3 -m harness.run pair \\
        --reading-a-conforming reading_nvidia_conforming.json \\
        --reading-a-bad reading_nvidia_bad.json \\
        --reading-b-conforming reading_amd_conforming.json \\
        --reading-b-bad reading_amd_bad.json \\
        --contract-class C-PRC-01 --primitive path_dependence \\
        --entry-id KC-0001 --status measured \\
        --out corpus/v0.1/KC-0001.json

Running `measure` on this machine (no GPU) produces status=illustrative
readings via the CPU-simulated candidates in operators.py. Do not pass
--status measured for anything produced this way -- see operators.py's
module docstring for why.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

from . import operators as ops
from . import tolerance as tol
from . import calibration as cal
from . import silicon as sil

HARNESS_VERSION = "0.1.0"

OPERATOR_TABLE = {
    # tolerance_fn takes (x, y_star, y) so operators whose backward-error
    # bound depends on the input (matmul) and operators that only need the
    # candidate output (softmax's invariant check) share one interface.
    "matmul": dict(
        reference=ops.matmul_reference,
        conforming=ops.matmul_candidate_conforming,
        bad=ops.matmul_candidate_downcast_accumulator,
        injection_method="forced downcast to fp16 accumulator (declared fp32)",
        # Backward error, not pointwise: pointwise-relative blows up with
        # matrix conditioning (paper section 3.5) and would reject a
        # correct fp32 accumulation for no reason other than problem size.
        # condition_number_correction=n follows the standard GEMM backward-
        # error bound (Higham, ch. 3): error grows roughly linearly in the
        # inner dimension.
        tolerance_fn=lambda x, y_star, y: tol.matmul_backward_error(x[0], x[1], y, y_star),
        gen_input=lambda rng, n: (rng.standard_normal((n, n)).astype(np.float32),
                                   rng.standard_normal((n, n)).astype(np.float32)),
        condition_number_correction=lambda n: n,
        # GPU path: conforming accumulates at dtype_accum as declared; bad
        # forces fp16 accumulation regardless of what was declared. Real
        # divergence comes from whatever the vendor's kernel actually does
        # at that precision on that silicon -- not from a numpy simulation.
        gpu_conforming=lambda x, dtype_accum: ops.matmul_candidate_torch(x[0], x[1], dtype_accum),
        gpu_bad=lambda x, dtype_accum: ops.matmul_candidate_torch(x[0], x[1], "fp16"),
    ),
    "softmax": dict(
        reference=ops.softmax_reference,
        conforming=ops.softmax_candidate_stabilized,
        bad=ops.softmax_candidate_no_stabilization,
        injection_method="no max-subtraction stabilization (overflows on wide dynamic range)",
        tolerance_fn=lambda x, y_star, y: tol.invariant_row_sum(y),
        # Range must exceed float32's exp() overflow point (~88.7) for the
        # unstabilized candidate to actually fail -- narrower ranges don't
        # exercise the failure mode C-PRC-02 is about and would make the
        # calibration falsely look sound by accident.
        gen_input=lambda rng, n: rng.uniform(-110, 110, size=(8, n)).astype(np.float32),
        # n-term summation accumulates roundoff; using n as the correction
        # is the same conservative (worst-case, not average-case) choice
        # made for matmul above, for the same reason.
        condition_number_correction=lambda n: n,
        gpu_conforming=lambda x, dtype_accum: ops.softmax_candidate_torch(x[0], stabilized=True),
        gpu_bad=lambda x, dtype_accum: ops.softmax_candidate_torch(x[0], stabilized=False),
    ),
}


def cmd_measure(args):
    if args.operator not in OPERATOR_TABLE:
        sys.exit(f"unknown operator {args.operator!r}; choices: {list(OPERATOR_TABLE)}")
    spec = OPERATOR_TABLE[args.operator]
    rng = np.random.default_rng(args.seed)
    x = spec["gen_input"](rng, args.n)

    correction = spec["condition_number_correction"](args.n)
    eta = tol.derive_eta_from_unit_roundoff(args.dtype_accum, correction)

    x_arg = x if isinstance(x, tuple) else (x,)
    y_star = spec["reference"](x) if not isinstance(x, tuple) else spec["reference"](*x)

    if args.device == "cpu":
        candidate_fn = spec["conforming"] if args.candidate == "conforming" else spec["bad"]
        y = candidate_fn(x) if not isinstance(x, tuple) else candidate_fn(*x)
    else:
        if "gpu_conforming" not in spec:
            sys.exit(f"{args.operator}: no GPU-dispatched candidate wired up yet; "
                      f"add one to OPERATOR_TABLE before running with --device cuda")
        gpu_fn = spec["gpu_conforming"] if args.candidate == "conforming" else spec["gpu_bad"]
        y = gpu_fn(x_arg, args.dtype_accum)

    divergence = spec["tolerance_fn"](x_arg, y_star, y)

    detected = sil.detect_local()
    if args.device == "cuda" and "CPU" in detected.runtime:
        sys.exit(f"--device cuda was requested but silicon detection found no GPU "
                  f"({detected.runtime!r}). Refusing to write a reading that would "
                  f"mislabel a CPU run as GPU silicon.")

    reading = {
        "operator": args.operator,
        "candidate": args.candidate,
        "device": args.device,
        "dtype_accum": args.dtype_accum,
        "eta": eta,
        "divergence": divergence,
        "satisfies": tol.satisfies(divergence, eta),
        "injection_method": spec["injection_method"] if args.candidate == "bad" else None,
        "n_samples": args.n,
        "seed": args.seed,
        "silicon": detected.to_dict(),
        "harness_version": HARNESS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out, "w") as f:
        json.dump(reading, f, indent=2)
    print(f"wrote {args.out}: divergence={divergence:.6g} eta={eta:.6g} satisfies={reading['satisfies']}")


def cmd_pair(args):
    with open(args.reading_a_conforming) as f:
        a_conf = json.load(f)
    with open(args.reading_a_bad) as f:
        a_bad = json.load(f)
    with open(args.reading_b_conforming) as f:
        b_conf = json.load(f)
    with open(args.reading_b_bad) as f:
        b_bad = json.load(f)

    for pair, label in [((a_conf, a_bad), "a"), ((b_conf, b_bad), "b")]:
        conf, bad = pair
        if conf["operator"] != bad["operator"]:
            sys.exit(f"readings for silicon {label}: operator mismatch")
        if not conf["satisfies"]:
            sys.exit(f"readings for silicon {label}: conforming candidate does not satisfy its own contract; fix eta or the candidate before pairing")
        if bad["satisfies"]:
            print(f"WARNING: silicon {label}'s injected-bad candidate PASSED (divergence={bad['divergence']:.6g} <= eta={bad['eta']:.6g}). "
                  f"Calibration is unsound for this contract. Recording verdict=under_specified.", file=sys.stderr)

    if args.status == "measured":
        if a_conf.get("device") != "cuda" or b_conf.get("device") != "cuda":
            sys.exit("status=measured requires both readings to come from --device cuda runs. "
                      "CPU-simulated readings can only be paired as status=illustrative.")
        if a_conf["silicon"]["arch"] == b_conf["silicon"]["arch"]:
            sys.exit(f"status=measured requires two DIFFERENT silicon arches; both readings "
                      f"report {a_conf['silicon']['arch']!r}. Same GPU twice is not a cross-silicon pair.")

    calibration_sound = (a_conf["satisfies"] and not a_bad["satisfies"] and
                          b_conf["satisfies"] and not b_bad["satisfies"])
    verdict = "within_tolerance" if calibration_sound else "under_specified"
    # If both conforming candidates pass but this is a REAL cross-silicon
    # measured pair, the interesting verdict is whether a's and b's actual
    # (non-injected) outputs diverge from each other beyond eta -- that
    # comparison needs the raw candidate outputs, not just conforming/bad
    # against the reference. v0 records the calibration-soundness verdict;
    # add direct a-vs-b output comparison in v0.2 once real paired GPU runs
    # exist to design it against.

    entry = {
        "entry_id": args.entry_id,
        "status": args.status,
        "operator": {
            "name": a_conf["operator"],
            "shape": args.shape,
            "dtype_in": args.dtype_in,
            "dtype_accum": a_conf["dtype_accum"],
        },
        "silicon_pair": {"a": a_conf["silicon"], "b": b_conf["silicon"]},
        "contract_class": args.contract_class,
        "primitive": args.primitive,
        "tolerance": {
            "structure": args.tolerance_structure,
            "eta": a_conf["eta"],
            "derivation": f"derive_eta_from_unit_roundoff({a_conf['dtype_accum']!r}), see harness/tolerance.py",
        },
        "measurement": {
            "divergence": max(a_bad["divergence"], b_bad["divergence"]),
            "method": "max(injected-bad divergence on silicon a, on silicon b), each vs float64 reference",
            "n_samples": a_conf["n_samples"],
        },
        "calibration": {
            "reference_conforming": {"result": "pass" if (a_conf["satisfies"] and b_conf["satisfies"]) else "fail"},
            "injected_bad": {
                "result": "fail" if (not a_bad["satisfies"] and not b_bad["satisfies"]) else "pass",
                "injection_method": a_bad["injection_method"] or b_bad["injection_method"] or "n/a",
            },
        },
        "verdict": verdict,
        "reproduction": {
            "command": " ".join(sys.argv),
            "seed": a_conf["seed"],
            "harness_version": HARNESS_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    try:
        import jsonschema
        with open(args.schema) as f:
            schema = json.load(f)
        jsonschema.validate(entry, schema)
        print("schema: valid")
    except ImportError:
        print("schema: skipped (pip install jsonschema to validate)", file=sys.stderr)
    except Exception as e:
        sys.exit(f"schema: INVALID -- {e}")

    with open(args.out, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"wrote {args.out}: verdict={verdict}")


def cmd_selftest(args):
    """Runs the full three-state calibration locally, both candidates in
    one process, via calibration.run_calibration. Useful for validating
    the harness and a new operator's tolerance wiring before ever touching
    a rented GPU -- this is what caught the matmul tolerance_fn bug during
    development (see git history). Does not produce a corpus entry.
    """
    failures = []
    for name, spec in OPERATOR_TABLE.items():
        rng = np.random.default_rng(args.seed)
        x = spec["gen_input"](rng, args.n)
        x_arg = x if isinstance(x, tuple) else (x,)
        correction = spec["condition_number_correction"](args.n)
        eta = tol.derive_eta_from_unit_roundoff(args.dtype_accum, correction)

        result = cal.run_calibration(
            x=x_arg,
            reference_fn=spec["reference"],
            conforming_fn=spec["conforming"],
            bad_fn=spec["bad"],
            tolerance_fn=lambda y_star, y, _x=x_arg, _fn=spec["tolerance_fn"]: _fn(_x, y_star, y),
            eta=eta,
            injection_method=spec["injection_method"],
        )
        status = "SOUND" if result.sound else "UNSOUND"
        print(f"{name:10s} eta={eta:.4g}  conforming: {result.reference_conforming_result} "
              f"(d={result.reference_conforming_divergence:.4g})  "
              f"bad: {result.injected_bad_result} (d={result.injected_bad_divergence:.4g})  [{status}]")
        if not result.sound:
            failures.append(name)

    if failures:
        sys.exit(f"\nUNSOUND calibration for: {', '.join(failures)}. "
                  f"Fix the tolerance derivation or the candidates before using these operators.")
    print("\nall operators: calibration sound.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="run reference+candidate on this machine, write a reading")
    m.add_argument("--operator", required=True, choices=list(OPERATOR_TABLE))
    m.add_argument("--dtype-accum", required=True)
    m.add_argument("--candidate", required=True, choices=["conforming", "bad"])
    m.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="'cpu' uses the numpy simulation (fine for illustrative entries). "
                         "'cuda' dispatches to real GPU silicon via torch (covers CUDA and "
                         "ROCm builds) and is required for status=measured entries.")
    m.add_argument("--n", type=int, default=256, help="problem size (matrix dim / vector length)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_measure)

    pr = sub.add_parser("pair", help="combine two silicons' conforming+bad readings into a corpus entry")
    pr.add_argument("--reading-a-conforming", required=True)
    pr.add_argument("--reading-a-bad", required=True)
    pr.add_argument("--reading-b-conforming", required=True)
    pr.add_argument("--reading-b-bad", required=True)
    pr.add_argument("--contract-class", required=True)
    pr.add_argument("--primitive", required=True, choices=["path_dependence", "domain_violation", "resource_contention"])
    pr.add_argument("--tolerance-structure", default="pointwise_relative")
    pr.add_argument("--shape", required=True)
    pr.add_argument("--dtype-in", required=True)
    pr.add_argument("--entry-id", required=True)
    pr.add_argument("--status", required=True, choices=["measured", "illustrative"])
    pr.add_argument("--schema", default="schema/entry.schema.json")
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_pair)

    st = sub.add_parser("selftest", help="run all operators' three-state calibration locally, no GPU needed")
    st.add_argument("--dtype-accum", default="fp32")
    st.add_argument("--n", type=int, default=128)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
