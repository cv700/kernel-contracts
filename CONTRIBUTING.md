# Contributing a corpus entry

1. Run `python3 -m harness.run selftest` first. If it reports UNSOUND for
   any operator you're about to use, fix that before producing entries with
   it -- an unsound calibration means the contract wouldn't catch its own
   injected failure, so any entry it produces is not evidence of anything.

2. Run `measure` twice on each of two different physical silicons
   (conforming + bad candidate on each). Four JSON readings total. Same
   machine twice does not count as a silicon pair -- label it
   `status: illustrative` if that's what you have, not `measured`.

3. Run `pair` to combine the four readings into one entry. It validates
   against `schema/entry.schema.json` automatically if `jsonschema` is
   installed; if you skipped that in requirements, install it before
   opening a PR -- an unvalidated entry will be rejected in review anyway.

4. Open a PR adding the entry to `corpus/v0.1/`. Include the exact command
   used to generate each reading (the `reproduction` field records this
   automatically, but state it in the PR description too).

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
