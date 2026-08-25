# Contributing a corpus entry

1. Run `./kc selftest` first. If it reports UNSOUND for any operator
   you're about to use, fix that before producing entries with it -- an
   unsound calibration means the contract wouldn't catch its own injected
   failure, so any entry it produces is not evidence of anything.

2. Run `./kc measure <operator> --device cuda --out X.json` on each of two
   different physical silicons. Two JSON files total (one per box, each
   already containing both the conforming and bad candidate). Same machine
   twice does not count as a silicon pair -- `combine` will refuse to call
   it `measured` and will fall back to `illustrative` on its own.

3. Run `./kc combine A.json B.json`. It infers the entry id, the status,
   and all the contract metadata, and validates the result against
   `schema/entry.schema.json` automatically if `jsonschema` is installed --
   install it before opening a PR if you skipped it in requirements; an
   unvalidated entry will be rejected in review anyway.

4. Open a PR adding the entry `combine` wrote under `corpus/v0.1/`. Include
   the exact commands used (the `reproduction` field records the `combine`
   invocation automatically, but state the two `measure` commands too).

## Adding a new operator

`harness/operators.py` needs: a float64 reference implementation, a
reference-conforming candidate, and an injected-bad candidate whose failure
mode matches one of the twelve contract classes. `harness/run.py`'s
`OPERATOR_TABLE` needs an entry wiring these together with a tolerance
function and a way to derive `eta` from first principles -- not from
measuring the candidate, see `tolerance.py`'s docstring on why that's
disallowed.

Run `selftest` on your new operator before anything else. An operator
whose calibration is unsound on your own machine will not become sound by
being run on a rented GPU.

## What gets rejected

- Entries with `status: measured` where the two silicon identifiers in
  `silicon_pair` are identical, or where the reading files don't correspond
  to two actually-different `arch` values.
- Entries where `tolerance.derivation` says or implies the tolerance was
  measured from the candidate under test. See the paper's section 3.11 for
  why this specific mistake matters enough to reject on sight.
- Entries that fail schema validation.
- Undocumented reproduction commands.
