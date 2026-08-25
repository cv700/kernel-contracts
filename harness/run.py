#!/usr/bin/env python3
"""CLI entrypoint. Three commands, in order of use:

    ./kc selftest                          # no GPU needed, validates the harness
    ./kc measure matmul --device cuda --out nvidia.json      # once per box
    ./kc measure matmul --device cuda --out amd.json         # once per box
    ./kc combine nvidia.json amd.json      # writes corpus/v0.1/KC-0001.json

`measure` runs both the conforming and the injected-bad candidate in one
call and writes one file. `combine` takes just the two files -- contract
class, primitive, tolerance structure, and dtype default all come from
OPERATOR_TABLE, entry id auto-increments, status is inferred from whether
both readings are real GPU runs on different silicon. Override any of it
with flags if you need to; you shouldn't need to for the common case.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from . import operators as ops
from . import tolerance as tol
from . import calibration as cal
from . import silicon as sil

HARNESS_VERSION = "0.2.0"
REPO_ROOT = Path(__file__).resolve().parent.parent

OPERATOR_TABLE = {
    # tolerance_fn takes (x, y_star, y) so operators whose backward-error
    # bound depends on the input (matmul) and operators that only need the
    # candidate output (softmax's invariant check) share one interface.
    #
    # contract_class / primitive / tolerance_structure / dtype_in are the
    # schema metadata that used to be CLI flags on `pair`. They're a
    # property of the operator, not something a caller should have to
    # know or re-type correctly every time.
    "matmul": dict(
        contract_class="C-PRC-01",
        primitive="path_dependence",
        tolerance_structure="backward_error",
        default_dtype_in="fp32",
        default_dtype_accum="fp32",
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
        shape=lambda n: f"({n},{n})x({n},{n})",
        # GPU path: conforming accumulates at dtype_accum as declared; bad
        # forces fp16 accumulation regardless of what was declared. Real
        # divergence comes from whatever the vendor's kernel actually does
        # at that precision on that silicon -- not from a numpy simulation.
        gpu_conforming=lambda x, dtype_accum: ops.matmul_candidate_torch(x[0], x[1], dtype_accum),
        gpu_bad=lambda x, dtype_accum: ops.matmul_candidate_torch(x[0], x[1], "fp16"),
    ),
    "softmax": dict(
        contract_class="C-PRC-02",
        primitive="path_dependence",
        tolerance_structure="invariant",
        default_dtype_in="fp32",
        default_dtype_accum="fp32",
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
        shape=lambda n: f"N=8,d={n}",
        gpu_conforming=lambda x, dtype_accum: ops.softmax_candidate_torch(x[0], stabilized=True),
        gpu_bad=lambda x, dtype_accum: ops.softmax_candidate_torch(x[0], stabilized=False),
    ),
}


def _run_one_candidate(spec, x, x_arg, y_star, device, dtype_accum, which):
    if device == "cpu":
        fn = spec["conforming"] if which == "conforming" else spec["bad"]
        return fn(x) if not isinstance(x, tuple) else fn(*x)
    if "gpu_conforming" not in spec:
        sys.exit(f"no GPU-dispatched candidate wired up for this operator yet; "
                  f"add one to OPERATOR_TABLE before running with --device cuda")
    gpu_fn = spec["gpu_conforming"] if which == "conforming" else spec["gpu_bad"]
    return gpu_fn(x_arg, dtype_accum)


def cmd_measure(args):
    if args.operator not in OPERATOR_TABLE:
        sys.exit(f"unknown operator {args.operator!r}; choices: {list(OPERATOR_TABLE)}")
    spec = OPERATOR_TABLE[args.operator]
    dtype_accum = args.dtype_accum or spec["default_dtype_accum"]

    rng = np.random.default_rng(args.seed)
    x = spec["gen_input"](rng, args.n)
    x_arg = x if isinstance(x, tuple) else (x,)
    y_star = spec["reference"](x) if not isinstance(x, tuple) else spec["reference"](*x)

    detected = sil.detect_local()
    if args.device == "cuda" and "CPU" in detected.runtime:
        sys.exit(f"--device cuda was requested but silicon detection found no GPU "
                  f"({detected.runtime!r}). Refusing to write a reading that would "
                  f"mislabel a CPU run as GPU silicon.")

    correction = spec["condition_number_correction"](args.n)
    eta = tol.derive_eta_from_unit_roundoff(dtype_accum, correction)

    results = {}
    for which in ("conforming", "bad"):
        y = _run_one_candidate(spec, x, x_arg, y_star, args.device, dtype_accum, which)
        divergence = spec["tolerance_fn"](x_arg, y_star, y)
        results[which] = {"divergence": divergence, "satisfies": tol.satisfies(divergence, eta)}

    sound = results["conforming"]["satisfies"] and not results["bad"]["satisfies"]
    status_word = "OK" if sound else "UNSOUND"
    conforming_word = "accepted, correctly" if results["conforming"]["satisfies"] else "REJECTED -- should have been accepted"
    bad_word = "rejected, correctly" if not results["bad"]["satisfies"] else "ACCEPTED -- contract is too loose"
    print(f"{args.operator} on {detected.arch} ({args.device}): "
          f"conforming d={results['conforming']['divergence']:.4g} ({conforming_word}); "
          f"bad d={results['bad']['divergence']:.4g} ({bad_word})  [{status_word}]")
    if not sound:
        print("WARNING: calibration not sound on this silicon -- writing the reading anyway, "
              "but `combine` will refuse to call it status=measured.", file=sys.stderr)

    reading = {
        "operator": args.operator,
        "device": args.device,
        "dtype_accum": dtype_accum,
        "dtype_in": spec["default_dtype_in"],
        "n": args.n,
        "eta": eta,
        "conforming": results["conforming"],
        "bad": {**results["bad"], "injection_method": spec["injection_method"]},
        "sound": sound,
        "seed": args.seed,
        "silicon": detected.to_dict(),
        "harness_version": HARNESS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = args.out or f"{args.operator}_{detected.arch.replace(' ', '_')}.json"
    with open(out, "w") as f:
        json.dump(reading, f, indent=2)
    print(f"wrote {out}")


def _next_entry_id():
    existing = list((REPO_ROOT / "corpus" / "v0.1").glob("KC-*.json")) + \
               list((REPO_ROOT / "corpus" / "examples").glob("KC-*.json"))
    nums = [int(m.group(1)) for f in existing if (m := re.match(r"KC-(\d+)", f.stem))]
    return f"KC-{(max(nums) + 1) if nums else 1:04d}"


def cmd_combine(args):
    with open(args.reading_a) as f:
        a = json.load(f)
    with open(args.reading_b) as f:
        b = json.load(f)

    if a["operator"] != b["operator"]:
        sys.exit(f"reading files are for different operators ({a['operator']!r} vs {b['operator']!r})")
    operator = a["operator"]
    spec = OPERATOR_TABLE[operator]

    for reading, label in [(a, args.reading_a), (b, args.reading_b)]:
        if not reading["conforming"]["satisfies"]:
            sys.exit(f"{label}: conforming candidate does not satisfy its own contract; "
                      f"fix the tolerance or candidate before combining")
        if reading["bad"]["satisfies"]:
            print(f"WARNING: {label}'s injected-bad candidate PASSED. "
                  f"Calibration is unsound; recording verdict=under_specified.", file=sys.stderr)

    both_gpu = a["device"] == "cuda" and b["device"] == "cuda"
    different_silicon = a["silicon"]["arch"] != b["silicon"]["arch"]
    inferred_status = "measured" if (both_gpu and different_silicon) else "illustrative"
    status = args.status or inferred_status
    if status == "measured" and not (both_gpu and different_silicon):
        sys.exit(f"--status measured requires both readings from --device cuda on different "
                  f"silicon (got device={a['device']!r}/{b['device']!r}, "
                  f"arch={a['silicon']['arch']!r}/{b['silicon']['arch']!r}). "
                  f"Omit --status to let it infer illustrative, or fix the readings.")

    sound = a["sound"] and b["sound"]
    verdict = "within_tolerance" if sound else "under_specified"

    entry_id = args.id or _next_entry_id()
    default_dir = "v0.1" if status == "measured" else "examples"
    out = args.out or str(REPO_ROOT / "corpus" / default_dir / f"{entry_id}.json")

    entry = {
        "entry_id": entry_id,
        "status": status,
        "operator": {
            "name": operator,
            "shape": spec["shape"](a["n"]),
            "dtype_in": a["dtype_in"],
            "dtype_accum": a["dtype_accum"],
        },
        "silicon_pair": {"a": a["silicon"], "b": b["silicon"]},
        "contract_class": spec["contract_class"],
        "primitive": spec["primitive"],
        "tolerance": {
            "structure": spec["tolerance_structure"],
            "eta": a["eta"],
            "derivation": f"derive_eta_from_unit_roundoff({a['dtype_accum']!r}), see harness/tolerance.py",
        },
        "measurement": {
            "divergence": max(a["bad"]["divergence"], b["bad"]["divergence"]),
            "method": "max(injected-bad divergence on silicon a, on silicon b), each vs float64 reference",
            "n_samples": a["n"],
        },
        "calibration": {
            "reference_conforming": {"result": "pass" if (a["conforming"]["satisfies"] and b["conforming"]["satisfies"]) else "fail"},
            "injected_bad": {
                "result": "fail" if (not a["bad"]["satisfies"] and not b["bad"]["satisfies"]) else "pass",
                "injection_method": a["bad"]["injection_method"],
            },
        },
        "verdict": verdict,
        "reproduction": {
            "command": " ".join(sys.argv),
            "seed": a["seed"],
            "harness_version": HARNESS_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    try:
        import jsonschema
        with open(REPO_ROOT / "schema" / "entry.schema.json") as f:
            schema = json.load(f)
        jsonschema.validate(entry, schema)
        schema_note = "schema: valid"
    except ImportError:
        schema_note = "schema: skipped (pip install jsonschema to validate)"
    except Exception as e:
        sys.exit(f"schema: INVALID -- {e}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"{schema_note}\nwrote {out}  [{entry_id}, status={status}, verdict={verdict}]")


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
        eta = tol.derive_eta_from_unit_roundoff(spec["default_dtype_accum"], correction)

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

    m = sub.add_parser("measure", help="run both candidates for one operator on this machine, write one reading")
    m.add_argument("operator", choices=list(OPERATOR_TABLE))
    m.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="'cpu' = numpy simulation, illustrative only. "
                         "'cuda' = real GPU via torch (CUDA or ROCm), required for status=measured.")
    m.add_argument("--dtype-accum", default=None, help="defaults to the operator's usual accumulator dtype")
    m.add_argument("--n", type=int, default=256, help="problem size (matrix dim / vector length)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=None, help="defaults to '<operator>_<arch>.json'")
    m.set_defaults(func=cmd_measure)

    c = sub.add_parser("combine", help="combine two silicons' readings into a corpus entry")
    c.add_argument("reading_a")
    c.add_argument("reading_b")
    c.add_argument("--id", default=None, help="defaults to the next KC-#### not already in corpus/")
    c.add_argument("--status", default=None, choices=["measured", "illustrative"],
                    help="defaults to an inferred value: measured iff both readings are --device cuda on different silicon")
    c.add_argument("--out", default=None, help="defaults to corpus/v0.1/<id>.json or corpus/examples/<id>.json")
    c.set_defaults(func=cmd_combine)

    st = sub.add_parser("selftest", help="run all operators' three-state calibration locally, no GPU needed")
    st.add_argument("--n", type=int, default=128)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
