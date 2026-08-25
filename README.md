# Kernel Contracts

An open corpus and harness for cross-silicon ML kernel divergence, annotated
against a written contract instead of left as an unexplained number.

## The problem

A matmul on AMD can produce a different result than the same matmul on
NVIDIA. Sometimes that's legitimate floating-point divergence. Sometimes
it's a silent error. Wen et al. (2025) measured the gap directly: model-level
output agreement of 99.8% on AMD MI300X, 94.9% on Huawei Ascend 910B, and
85.9% on Apple M4 Pro against an NVIDIA H200 reference, with roughly 1,700
out-of-bounds accesses silently passing on AMD where NVIDIA raises. Nobody
has published which of those divergences are which.

Single-silicon kernel correctness is being solved fast by other people, and
you should use their work, not this one, for that problem: see
[KernelBench-Verified](https://arxiv.org/html/2607.16241) for LLM-generated
kernel correctness, [SOL-ExecBench](https://github.com/nvidia/sol-execbench)
for correctness against a hardware speed-of-light model. Neither compares
across vendors. That's the gap this project is scoped to.

## What's here

- `schema/entry.schema.json` -- the corpus entry format. Every entry records
  an operator, a silicon pair, which of twelve contract classes it tests,
  a tolerance derived from first principles (never measured off the
  candidate under test), a three-state calibration result, and a verdict.
- `harness/` -- the tool that produces entries. Runnable today on CPU with
  no GPU (numpy only) for schema validation and calibration-soundness
  checks; producing a real `status: measured` entry needs the candidate run
  on actual silicon.
- `corpus/examples/` -- one worked example (`KC-0001.json`),
  `status: illustrative`, both readings from the same machine. Not a
  cross-silicon measurement -- a demonstration that the pipeline runs
  end to end.
- `corpus/v0.1/` -- where real, cross-silicon `status: measured` entries go.
  Empty until someone runs the harness on two different GPUs.

The underlying math -- the contract-triple, the twelve-class taxonomy, the
three physical failure primitives, backward-error tolerance -- is developed
in the companion paper, `kernel_contracts_v2.md` (not yet in this repo; see
Status below).

## Status

Two of five planned operators are implemented: `matmul` (C-PRC-01,
accumulator preservation) and `softmax` (C-PRC-02, cancellation/overflow).
Not yet implemented: `attention`, `reduction_order` (C-ORD-01),
`gather`/`index_select` (C-EXC-02). See `harness/operators.py`'s bottom
comment for what each needs.

The corpus itself is empty of `measured` entries as of this writing --
everything in `corpus/examples/` is `illustrative`. The two-week build plan
is: rent one NVIDIA and one AMD instance, run the harness on both for the
two implemented operators, produce the first real paired entries.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./kc selftest                              # no GPU needed
./kc measure matmul --device cuda --out a.json     # run on box 1
./kc measure matmul --device cuda --out b.json     # run on box 2
./kc combine a.json b.json                 # writes corpus/v0.1/KC-000N.json
```

That's the whole interface: three commands. `combine` figures out the
contract class, tolerance structure, entry id, and whether this counts as
`measured` or `illustrative` on its own -- see `harness/run.py --help` if
you need to override something.

## Why "corpus," and why open

The tool is easy to reproduce. Anyone can run it. What's hard to reproduce
is a structured, contract-annotated record of what actually diverges, kept
current. That's the part worth owning: not by keeping it closed, but by
being the one who keeps adding to it. See `CONTRIBUTING.md`.

## License

Harness code: Apache 2.0 (`LICENSE`). Corpus data: CC-BY-4.0
(`corpus/LICENSE`).
