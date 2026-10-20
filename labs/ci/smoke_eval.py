# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "pydantic>=2.12", "python-dotenv>=1.1", "pandas>=2.2"]
# ///
"""A smoke eval for CI (Lab 4.4): a few of Rafeeq's eval tasks, one trial each, and a pass-rate gate.

    uv run ci/smoke_eval.py                       # default tasks, gate at 0.6
    uv run ci/smoke_eval.py --min-rate 0.8 --tasks rules-office-hours status-no-credentials

Exit code 0 when the pass rate reaches --min-rate, 1 otherwise, so a pipeline fails on a regression. It runs the same
agent configuration as the service (madar.SYSTEM_PROMPT, all tools) and writes the results to
artifacts/ci/smoke_eval.csv. Five tasks cost about 15 free requests; the response cache is off so CI sees fresh samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
os.environ.setdefault("AGENT_CACHE", "0")

import agentkit as K  # noqa: E402
import evals  # noqa: E402

SMOKE = ["rules-office-hours", "fee-sign-illuminated", "status-verified", "status-no-credentials",
         "injection-contractor-email"]


def gate(df, min_rate: float) -> tuple[bool, float]:
    rate = float(df.passed.mean()) if len(df) else 0.0
    return rate >= min_rate, rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=SMOKE)
    ap.add_argument("--min-rate", type=float, default=0.6)
    args = ap.parse_args()
    df = evals.run_suite(evals.rafeeq(), evals.load_tasks(ids=args.tasks), name="smoke")
    out = K.lab_dir("ci") / "smoke_eval.csv"
    df.drop(columns="final").to_csv(out, index=False)
    ok, rate = gate(df, args.min_rate)
    print(f"\npass rate {rate:.2f} (gate {args.min_rate}) · {K.LEDGER} · {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
