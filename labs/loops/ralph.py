# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "anthropic>=1.0", "pydantic>=2", "python-dotenv", "pytest"]
# ///
"""A Ralph loop: run a coding agent with a fresh context, again and again, until the tests pass (Lab 3.3).

    uv run loops/ralph.py                                   # fix data/madar_fees with the course's agent
    uv run loops/ralph.py --tests tests/test_fees.py        # one test file: fewer bugs, fewer requests
    uv run loops/ralph.py --harness pi                      # same loop around Pi (needs Node 22.19+ and `pi`)

Each iteration: run the tests → stop if green → give the agent the task, AGENTS.md, PROGRESS.md and the failing
tests in a FRESH context → let it edit the library (never the tests) → commit to git → repeat. It stops on success,
after --max-iters, when the failure count stops falling for --patience iterations, or above --max-cost.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import agentkit as K  # noqa: E402
from madar import ToolError  # noqa: E402  (call_tool turns it into an error message the model can act on)

PROMPT = """You are working in a small Python repository. Your goal: make the failing tests pass by fixing the library.

Read AGENTS.md first and follow it. PROGRESS.md holds notes from earlier iterations (you have no other memory of them).

This iteration: pick ONE failing test, find the rule it checks in docs/, fix that rule in madar_fees/, run the tests,
then append a short note to PROGRESS.md (what you fixed, which policy, tests before → after). Then stop.

Failing tests right now:
{failures}
"""


# ---------------------------------------------------------------- tests and git
def run_tests(repo: Path, target: str = "tests") -> dict:
    """Run pytest in `repo`; return counts and the short failure summary."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "-rf", target],
                         cwd=repo, capture_output=True, text=True, timeout=300)
    text = out.stdout + out.stderr
    count = lambda word: int(m.group(1)) if (m := re.search(rf"(\d+) {word}", text)) else 0  # noqa: E731
    failed = count("failed") + count("error")
    lines = [line for line in text.splitlines() if line.startswith(("FAILED", "ERROR"))]
    return {"passed": count("passed"), "failed": failed, "summary": "\n".join(lines[:15]) or text[-1500:]}


def git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=ralph", "-c", "user.email=ralph@madar.example", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True).stdout.strip()


def prepare(src: Path, work: Path) -> None:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "PROGRESS.md"))
    (work / "PROGRESS.md").write_text("# Progress\n", encoding="utf-8")
    git(work, "init", "-q")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "baseline")


# ---------------------------------------------------------------- the inner agent's tools (the harness)
def coding_tools(repo: Path, target: str) -> list:
    root = repo.resolve()

    def inside(path: str) -> Path:
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ToolError(f"{path} is outside the repository")
        return p

    def list_files() -> list[str]:
        """List the repository's files (relative paths)."""
        return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                      if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts)

    def read_file(path: str) -> str:
        """Read a text file, with line numbers.

        Args:
            path: relative path, e.g. "madar_fees/fees.py".
        """
        p = inside(path)
        if not p.is_file():
            raise ToolError(f"No file {path}. Use list_files.")
        return "\n".join(f"{i:4d}  {line}" for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1))

    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact, unique snippet in a file of the library (madar_fees/) or in PROGRESS.md.

        Args:
            path: relative path, e.g. "madar_fees/fees.py".
            old: the exact text to replace (copy it from read_file, without line numbers); must occur once.
            new: the replacement text.
        """
        p = inside(path)
        rel = str(p.relative_to(root)).replace("\\", "/")
        if not (rel.startswith("madar_fees/") or rel == "PROGRESS.md"):
            raise ToolError("You may edit only madar_fees/*.py and PROGRESS.md. Never change the tests (AGENTS.md).")
        text = p.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ToolError(f"`old` must occur exactly once in {path}; it occurs {text.count(old)} times.")
        p.write_text(text.replace(old, new), encoding="utf-8")
        return f"Edited {path}"

    def append_progress(note: str) -> str:
        """Append one note to PROGRESS.md for the next iteration.

        Args:
            note: what you fixed, which policy, and the tests before → after.
        """
        with (root / "PROGRESS.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n- {note.strip()}\n")
        return "Noted in PROGRESS.md"

    def run_test_suite() -> dict:
        """Run the tests and return passed/failed counts with the failing test names."""
        return run_tests(root, target)

    return [list_files, read_file, edit_file, append_progress, run_test_suite]


# ---------------------------------------------------------------- one iteration with each harness
def iterate_builtin(repo: Path, prompt: str, args) -> dict:
    run = K.run_agent(prompt, coding_tools(repo, args.tests), model=args.model, max_steps=args.max_steps,
                      system="You are a careful Python maintainer. Small, verified changes only.", verbose=args.verbose)
    return {"stopped": run.stopped, "steps": run.steps, "tool_calls": len(run.tool_calls)}


def iterate_pi(repo: Path, prompt: str, args) -> dict:
    pi = shutil.which("pi")
    if not pi:
        raise SystemExit("`pi` not found. Install Node 22.19+, then: npm install -g --ignore-scripts @earendil-works/pi-coding-agent")
    cmd = [pi, "--mode", "json", "--no-session", "--approve", "--provider", "openrouter", "--model", args.model,
           "--tools", "read,edit,bash,grep,find,ls", prompt]
    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout)
    events = [json.loads(line) for line in out.stdout.splitlines() if line.strip().startswith("{")]
    (repo.parent / f"{repo.name}.pi_events.jsonl").open("a", encoding="utf-8").write(out.stdout)
    tools = [e for e in events if e.get("type") == "tool_execution_start"]
    turns = sum(e.get("type") == "turn_end" for e in events)
    return {"stopped": "done" if out.returncode == 0 else f"exit {out.returncode}", "steps": turns, "tool_calls": len(tools)}


# ---------------------------------------------------------------- the outer loop
def ralph(args) -> list[dict]:
    work = K.lab_dir("ralph") / args.name
    prepare(LABS / args.repo, work)
    log, history = [], []
    start_cost, t0 = K.LEDGER.cost, time.time()
    for i in range(1, args.max_iters + 1):
        before = run_tests(work, args.tests)
        history.append(before["failed"])
        print(f"\n=== iteration {i}: {before['passed']} passed, {before['failed']} failing")
        if before["failed"] == 0:
            print("✅ all tests pass")
            break
        if len(history) > args.patience and min(history[-args.patience - 1:-1]) <= history[-1]:
            print(f"🛑 no progress in {args.patience} iterations")
            break
        if args.max_cost is not None and K.LEDGER.cost - start_cost > args.max_cost:
            print(f"🛑 cost budget ${args.max_cost} reached")
            break
        prompt = PROMPT.format(failures=before["summary"])
        result = (iterate_pi if args.harness == "pi" else iterate_builtin)(work, prompt, args)
        after = run_tests(work, args.tests)
        git(work, "add", "-A")
        git(work, "commit", "-q", "--allow-empty", "-m", f"ralph {i}: failing {before['failed']} → {after['failed']}")
        entry = {"iteration": i, "failing_before": before["failed"], "failing_after": after["failed"], **result,
                 "seconds": round(time.time() - t0), "ledger": str(K.LEDGER)}
        log.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    (work.parent / f"{args.name}.log.jsonl").write_text("\n".join(json.dumps(e) for e in log) + "\n", encoding="utf-8")
    print("\ngit log:\n" + git(work, "log", "--oneline"))
    return log


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="data/madar_fees", help="repository to fix, relative to labs/")
    ap.add_argument("--tests", default="tests", help="test path inside the repo, e.g. tests/test_fees.py")
    ap.add_argument("--name", default="madar_fees", help="name of the working copy in artifacts/ralph/")
    ap.add_argument("--harness", choices=["builtin", "pi"], default="builtin")
    ap.add_argument("--model", default=K.MODEL)
    ap.add_argument("--max-iters", type=int, default=K.pick(6, 3))
    ap.add_argument("--max-steps", type=int, default=12, help="model calls per iteration (builtin harness)")
    ap.add_argument("--patience", type=int, default=2, help="stop when failures don't fall for this many iterations")
    ap.add_argument("--max-cost", type=float, default=None, help="USD; free models cost 0")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per iteration (pi harness)")
    ap.add_argument("--verbose", action="store_true")
    return ralph(ap.parse_args(argv))


if __name__ == "__main__":
    main()
