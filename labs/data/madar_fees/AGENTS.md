# madar_fees: notes for coding agents

A small library that implements Madar's fee (POL-02), SLA (POL-03) and refund (POL-04, POL-06) rules.
The policies in `docs/` are the source of truth; the tests encode their worked examples.

## Commands

- Run the tests: `python -m pytest -q` (from this folder)
- One file: `python -m pytest -q tests/test_fees.py`

## Rules

- Fix the library, never the tests. If a test looks wrong, say so in PROGRESS.md and leave it.
- Keep functions pure (no I/O, no clock). Amounts are whole SAR.
- One fix at a time: change one rule, run the tests, record the result in PROGRESS.md.
- Read the policy text before changing a rule; cite the policy ID in your note.
