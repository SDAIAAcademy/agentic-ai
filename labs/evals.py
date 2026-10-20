"""Rafeeq's eval suite: tasks, graders and reports (Lab 3.4, the capstone and the CI smoke test).

A task is one JSON line in data/eval/tasks.jsonl:

    {"id": ..., "category": ..., "lang": "en" | "ar", "prompt": ..., "graders": [...]}

Grader types (all must pass):
- contains_any   {"values": [...]}         the final answer contains at least one value (case-insensitive)
- not_contains   {"values": [...]}         the final answer contains none of them
- tool_called    {"name": ...}             the agent called this tool at least once
- tool_not_called{"name": ...}             it never called this tool
- db             {"sql": ..., "expect": v} the first value of the query on the run's database equals v (end state)
- judge          {"rubric": ...}           an LLM judge (K.SMALL_MODEL) decides whether the answer meets the rubric

`run_suite(agent, tasks, trials=k)` runs each task k times in a fresh Madar database and returns one row per trial.
An `agent` is any function `(prompt: str, m: madar.Madar) -> agentkit.Run` (or an object with .final and .tool_calls).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import agentkit as K
import madar

TASKS_FILE = K.DATA / "eval" / "tasks.jsonl"
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬", "0123456789,")


def load_tasks(path: Path = TASKS_FILE, ids: list[str] | None = None, categories: list[str] | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]
    return tasks


def quick_subset(tasks: list[dict], n: int = 10) -> list[dict]:
    """A small, category-balanced subset for free-tier runs (round-robin over categories)."""
    by_cat: dict[str, list[dict]] = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)
    out: list[dict] = []
    while len(out) < min(n, len(tasks)):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop(0))
    return out


# ---------------------------------------------------------------- graders
def _norm(text: str) -> str:
    return (text or "").translate(ARABIC_DIGITS).lower()


JUDGE_PROMPT = """You grade one reply of Rafeeq, the digital assistant of Madar Municipal Services.
Decide whether the REPLY meets the RUBRIC. Judge only what the rubric asks; ignore style unless it says otherwise.
Reply with JSON only: {"pass": true or false, "reason": "one sentence"}"""


def judge(rubric: str, prompt: str, reply: str, tools_used: list[str], model: str | None = None) -> dict:
    """An LLM judge. Cached like every call, so re-grading the same reply is free."""
    user = f"RUBRIC:\n{rubric}\n\nUSER MESSAGE:\n{prompt}\n\nTOOLS THE ASSISTANT CALLED: {tools_used or 'none'}\n\nREPLY:\n{reply}"
    resp = K.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
                  model=model or K.SMALL_MODEL, temperature=0, max_tokens=300)
    text = K.text_of(resp).strip()
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return {"pass": bool(data.get("pass")), "reason": str(data.get("reason", ""))}
    except (ValueError, AttributeError):
        return {"pass": False, "reason": f"unparseable judge output: {text[:120]}"}


def grade(task: dict, final: str, tool_calls: list[tuple[str, dict]], m: madar.Madar,
          judge_model: str | None = None) -> list[dict]:
    """Apply every grader of a task. Returns one dict per grader: type, pass, detail."""
    names = [n for n, _ in tool_calls]
    out = []
    for g in task["graders"]:
        kind = g["type"]
        if kind == "contains_any":
            ok = any(_norm(v) in _norm(final) for v in g["values"])
            detail = "" if ok else f"none of {g['values'][:4]}"
        elif kind == "not_contains":
            hits = [v for v in g["values"] if _norm(v) in _norm(final)]
            ok, detail = not hits, f"contains {hits}" if hits else ""
        elif kind == "tool_called":
            ok, detail = g["name"] in names, "" if g["name"] in names else f"{g['name']} not called"
        elif kind == "tool_not_called":
            ok, detail = g["name"] not in names, "" if g["name"] not in names else f"{g['name']} was called"
        elif kind == "db":
            rows = m.sql(g["sql"])
            got = next(iter(rows[0].values())) if rows else None
            ok = str(got) == str(g["expect"])
            detail = "" if ok else f"expected {g['expect']!r}, got {got!r}"
        elif kind == "judge":
            verdict = judge(g["rubric"], task["prompt"], final, names, judge_model)
            ok, detail = verdict["pass"], verdict["reason"]
        else:
            raise ValueError(f"unknown grader {kind!r}")
        out.append({"type": kind, "pass": ok, "detail": detail})
    return out


# ---------------------------------------------------------------- agents under test
def rafeeq(system: str = madar.SYSTEM_PROMPT, tools: tuple[str, ...] | None = None, shape: str = "chat",
           model: str | None = None, max_steps: int = 10, **run_kw) -> Callable[[str, madar.Madar], K.Run]:
    """The default agent under test: Rafeeq with Madar's tools on K.run_agent. Write tools run without approval
    (the eval prompts say "go ahead"), so the database graders can see what the agent did."""

    def agent(prompt: str, m: madar.Madar, **kw) -> K.Run:
        return K.run_agent(prompt, m.tools(tools), system=system, shape=shape, model=model, max_steps=max_steps,
                           verbose=False, **{**run_kw, **kw})

    agent.__name__ = f"rafeeq[{shape}:{(model or K.MODEL).split('/')[-1]}]"
    return agent


@dataclass
class Trial:
    task: dict
    trial: int
    final: str
    tool_calls: list
    grades: list[dict]
    steps: int
    seconds: float
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(g["pass"] for g in self.grades)


def run_suite(agent: Callable[[str, madar.Madar], Any], tasks: list[dict], trials: int = 1, name: str | None = None,
              judge_model: str | None = None, verbose: bool = True):
    """Run every task `trials` times, each in a fresh copy of the Madar database. Returns a pandas DataFrame.

    With K.USE_CACHE on, repeated trials of the same request replay the same answer; pass^k needs fresh samples,
    so trials > 1 sets `use_cache=False` in the agent call when the agent accepts it.
    """
    import pandas as pd

    name = name or getattr(agent, "__name__", "agent")
    rows = []
    for t in tasks:
        for k in range(trials):
            m = madar.Madar(K.lab_dir(f"evals/{re.sub(r'[^\w.-]+', '_', name)}/{t['id']}/{k}", reset=True))
            t0 = time.perf_counter()
            try:
                run = agent(t["prompt"], m, **({"use_cache": False} if trials > 1 and _accepts(agent, "use_cache") else {}))
                final, calls, steps, err = run.final or "", list(run.tool_calls), getattr(run, "steps", 0), ""
                if getattr(run, "stopped", "done") != "done":
                    err = f"stopped: {run.stopped}"
            except Exception as e:  # noqa: BLE001 - a crash is a failed trial, not a failed suite
                final, calls, steps, err = "", [], 0, f"{type(e).__name__}: {str(e)[:200]}"
            grades = grade(t, final, calls, m, judge_model) if not err.startswith(("RateLimit", "APIConnection")) else []
            tr = Trial(t, k, final, calls, grades, steps, time.perf_counter() - t0, err)
            rows.append({"agent": name, "id": t["id"], "category": t["category"], "lang": t["lang"], "trial": k,
                         "passed": tr.passed, "failed_graders": "; ".join(f"{g['type']}: {g['detail']}" for g in grades
                                                                          if not g["pass"]) or err,
                         "tools": " ".join(n for n, _ in calls), "steps": steps, "seconds": round(tr.seconds, 1),
                         "final": final})
            if verbose:
                print(f"{'✅' if tr.passed else '❌'} {t['id']:<28} {rows[-1]['failed_graders'][:90]}")
    return pd.DataFrame(rows)


def _accepts(fn: Callable, param: str) -> bool:
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return param in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def report(df) -> "Any":
    """Per-task pass@k (any trial passed) and pass^k (all trials passed), plus overall rates by agent."""
    per_task = df.groupby(["agent", "id", "category"]).passed.agg(pass_at_k="any", pass_hat_k="all", rate="mean").reset_index()
    overall = per_task.groupby("agent")[["pass_at_k", "pass_hat_k", "rate"]].mean().round(2)
    return overall, per_task
