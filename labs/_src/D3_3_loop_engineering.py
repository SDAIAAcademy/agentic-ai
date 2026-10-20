# %% [markdown]
# # Lab 3.3 · A loop that fixes code
#
# Day 3 · Session 3 (Loop engineering) · 25 min. Some tasks are too big for one context window. The **Ralph loop**
# (Geoffrey Huntley's name for `while :; do cat PROMPT.md | agent; done`) runs an agent again and again, each time
# with a **fresh context**, and keeps state outside the model: in files, in git, and in the tests. In this lab you
# drive a coding agent on `madar_fees`, a tiny library with seeded bugs and failing tests:
# - the **harness** gives the agent file tools that can't touch the tests (the back-pressure must stay honest);
# - the **loop** runs iterations with a progress file, commits each one to git, and stops on success, on a budget,
#   or when it stops making progress.
#
# **Tasks**
# - [ ] 1 · The repository and its back-pressure (4')
# - [ ] 2 · One iteration with a fresh context (7')
# - [ ] 3 · The outer loop with stop conditions (8')
# - [ ] 4 · Read what the loop left behind (6')
# - [ ] ✱ Stretch · The same loop around Pi

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/agentic-ai.git"
    if not Path("/content/agentic-ai").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/agentic-ai"], check=True)
    os.chdir("/content/agentic-ai/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("agentkit.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import agentkit as K
print("provider:", K.PROVIDER, "| model:", K.MODEL, "| cache:", K.USE_CACHE)

# %%
import json

import pandas as pd

sys.path.insert(0, str(K.LABS / "loops"))
import ralph  # the finished loop runner (a PEP 723 script); this lab rebuilds its core

WORK = K.lab_dir("D3_3", reset=True) / "madar_fees"
ralph.prepare(K.DATA / "madar_fees", WORK)  # a copy with its own git repository and an empty PROGRESS.md
# The whole suite has 12 failing tests from 6 bugs. Free models allow 50 requests a day, so by default this lab works
# on the fee tests only; set TARGET = "tests" if you have the 1,000-a-day quota.
TARGET = "tests/test_fees.py"

# %% [markdown]
# ## 1 · The repository and its back-pressure (4')
#
# `AGENTS.md` tells the agent how to work; the **tests** tell it whether it worked. That only holds if the agent
# can't "fix" a test by editing it, so the harness enforces it: `edit_file` refuses anything outside `madar_fees/`
# and `PROGRESS.md`, and every path must stay inside the repository.

# %%
print((WORK / "AGENTS.md").read_text(encoding="utf-8"))
baseline = ralph.run_tests(WORK, TARGET)
print(f"{baseline['passed']} passed, {baseline['failed']} failing\n{baseline['summary']}")
tools = {t.__name__: t for t in ralph.coding_tools(WORK, TARGET)}
print("\n", K.call_tool(tools, "edit_file", {"path": "tests/test_fees.py", "old": "750", "new": "400"}))
print(K.call_tool(tools, "read_file", {"path": "../../../.env"}))
K.check(baseline["failed"] > 0, "the fee tests fail before the loop")

# %% [markdown]
# ## 2 · One iteration with a fresh context (7')
#
# Each iteration starts from **nothing** in the model's context. Everything it needs comes from the repository: the
# task prompt, `AGENTS.md`, `PROGRESS.md` (notes from earlier iterations) and the current test failures. That's the
# point: no context rot, no stale plans, and you can swap the model or the harness between iterations.
#
# ✏️ **Exercise** · Write `one_iteration(work)`: format `ralph.PROMPT` with the current failures, run
# `K.run_agent` with `ralph.coding_tools(work, TARGET)` (`max_steps=12`, the system prompt below), and return a dict
# with `failing_before`, `failing_after`, `steps` and `tool_calls`.

# %%
SYSTEM = "You are a careful Python maintainer. Small, verified changes only."


def one_iteration(work: Path) -> dict:
    # BEGIN SOLUTION
    before = ralph.run_tests(work, TARGET)
    prompt = ralph.PROMPT.format(failures=before["summary"])
    run = K.run_agent(prompt, ralph.coding_tools(work, TARGET), system=SYSTEM, max_steps=12, verbose=True)
    after = ralph.run_tests(work, TARGET)
    return {"failing_before": before["failed"], "failing_after": after["failed"], "steps": run.steps,
            "tool_calls": len(run.tool_calls)}
    # END SOLUTION


first = one_iteration(WORK)
print(first)
print((WORK / "PROGRESS.md").read_text(encoding="utf-8"))
K.check(first["failing_after"] <= first["failing_before"], "one iteration didn't make things worse")

# %% [markdown]
# ## 3 · The outer loop with stop conditions (8')
#
# A loop that never stops is a bill. Four exits:
# 1. **success**: no failing tests;
# 2. **iteration budget**: `max_iters`;
# 3. **no progress**: the failure count hasn't fallen for `patience` iterations (the agent is stuck or thrashing);
# 4. **cost budget**: `K.LEDGER.cost` grew by more than `max_cost` (0 on free models, but keep the habit).
#
# After every iteration, commit to git: a checkpoint you can diff, bisect or roll back.
#
# ✏️ **Exercise** · Write `stalled(history, patience)` (True when the last value is not lower than the minimum of
# the `patience` values before it) and `loop(work, max_iters, patience, max_cost)`, which returns the list of
# iteration dicts plus a `stop` reason.

# %%
def stalled(history: list[int], patience: int = 2) -> bool:
    # BEGIN SOLUTION
    return len(history) > patience and history[-1] >= min(history[-patience - 1:-1])
    # END SOLUTION


K.check(stalled([5, 4, 4, 4], 2) and not stalled([5, 4, 3], 2) and not stalled([5, 5], 2), "no-progress detection")


def loop(work: Path, max_iters: int = 4, patience: int = 2, max_cost: float = 0.10) -> tuple[list[dict], str]:
    log, history, start = [], [], K.LEDGER.cost
    # BEGIN SOLUTION
    for i in range(1, max_iters + 1):
        failing = ralph.run_tests(work, TARGET)["failed"]
        history.append(failing)
        if failing == 0:
            return log, "success"
        if stalled(history, patience):
            return log, "no progress"
        if K.LEDGER.cost - start > max_cost:
            return log, "cost budget"
        entry = {"iteration": i, **one_iteration(work)}
        ralph.git(work, "add", "-A")
        ralph.git(work, "commit", "-q", "--allow-empty", "-m",
                  f"iteration {i}: failing {entry['failing_before']} → {entry['failing_after']}")
        log.append(entry)
    return log, "iteration budget"
    # END SOLUTION


log, stop = loop(WORK, max_iters=K.pick(4, 2))
print("stopped:", stop)
pd.DataFrame(log)

# %% [markdown]
# ## 4 · Read what the loop left behind (6')
#
# The loop's real output is not the chat transcript: it is **commits**, **notes** and a **test result**.

# %%
print(ralph.git(WORK, "log", "--oneline"))
print("\n" + (WORK / "PROGRESS.md").read_text(encoding="utf-8"))
print(ralph.git(WORK, "diff", "--stat", ralph.git(WORK, "rev-list", "--max-parents=0", "HEAD"), "HEAD"))
final = ralph.run_tests(WORK, TARGET)
print(f"\nnow: {final['passed']} passed, {final['failed']} failing")

# %% [markdown]
# ✏️ **Exercise** · Verify the loop's work like a reviewer: `git diff` the baseline commit against `HEAD` and check
# that no file under `tests/` changed, and that every change is in `madar_fees/` or `PROGRESS.md`.

# %%
# BEGIN SOLUTION
base = ralph.git(WORK, "rev-list", "--max-parents=0", "HEAD")
changed = [f for f in ralph.git(WORK, "diff", "--name-only", base, "HEAD").splitlines() if f]
print(changed)
untouched_tests = not any(f.startswith("tests/") for f in changed)
only_allowed = all(f.startswith("madar_fees/") or f == "PROGRESS.md" for f in changed)
# END SOLUTION
K.check(untouched_tests and only_allowed, "the loop changed only the library and its notes")

# %% [markdown]
# 🔍 Read the diff of the commit that fixed a test. Is it the fix the policy asks for, or a fix that only satisfies
# the test? (A test suite is back-pressure, not a specification: Day 3 · S4 is about measuring the rest.)

# %% [markdown]
# ## ✱ Stretch · The same loop around Pi
#
# The outer loop doesn't care which harness runs inside it. In a terminal, from `labs/`:
#
# ```bash
# uv run loops/ralph.py --tests tests/test_fees.py --harness pi --name with-pi
# uv run loops/ralph.py --tests tests/test_fees.py --name builtin
# ```
#
# Compare the two logs in `artifacts/ralph/*.log.jsonl`: iterations, tool calls and failures per iteration. Pi's
# events for every iteration are saved next to the log.

# %% [markdown]
# ## Checkpoint
#
# 1. Why does a fresh context per iteration help, and what must the repository hold for it to work?
# 2. What would happen to this loop if `edit_file` could edit the tests?
# 3. Which stop condition fired, and was it the right one?
# 4. Where does loop engineering end and graph engineering (Day 4) begin?
