# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "pydantic>=2.12", "python-dotenv>=1.1", "pandas>=2.2", "mcp>=2.3"]
# ///
"""Capstone starter: one agent that already meets the minimum requirements' plumbing. Make it yours.

    uv run capstone/starter.py ask "What is the fee for an illuminated 3 m² sign?"
    uv run capstone/starter.py chat --session me              # a conversation, kept in artifacts/capstone/
    uv run capstone/starter.py eval capstone/tasks.example.jsonl --config A B

What is wired in (each one maps to a capstone requirement; search for "CAPSTONE:" to find where to change it):
1. tools: local Python functions plus the tools of an MCP server (default: servers/madar_mcp.py over stdio);
2. context: a spec file as the system prompt (AGENTS.md style), a session history with a compaction rule, and a cap
   on tool-result size;
3. harness: a permission policy (read → allow, write → ask a human), step and cost budgets, an audit log;
4. evals: the Day 3 eval runner over your tasks, on two configurations, with a per-task CSV for error analysis.
Security (requirement 5) is yours to analyse and defend: start from Lab 4.3's egress guard.

Run from labs/ (it imports agentkit, madar and evals from there).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

from mcp import StdioServerParameters  # noqa: E402

import agentkit as K  # noqa: E402
import evals  # noqa: E402
import madar  # noqa: E402

OUT = K.lab_dir("capstone")

# ---------------------------------------------------------------- CAPSTONE: your agent's spec (system prompt)
SPEC = madar.SYSTEM_PROMPT  # replace with Path("capstone/AGENTS.md").read_text() once you write your own spec

# Two configurations to compare in your eval (requirement 4): change ONE thing between them.
CONFIGS = {
    "A": {"system": SPEC, "model": None},
    "B": {"system": SPEC + "\nBefore any answer about fees or dates, call the tool that computes them.", "model": None},
}

MAX_STEPS, MAX_COST, MAX_RESULT_CHARS, MAX_HISTORY_MESSAGES = 8, 0.05, 3000, 24


# ---------------------------------------------------------------- CAPSTONE: tools
def local_tools(m: madar.Madar) -> list:
    """Tools that live in this process (here: Madar's write tools; the read tools come over MCP). Replace them with
    your domain's functions: typed, with Google-style docstrings, names that don't clash with the MCP tools."""
    return [m.book_inspection, m.reschedule_inspection, m.create_refund, m.escalate]


def mcp_server() -> StdioServerParameters:
    """The MCP server whose tools the agent also gets (requirement 1). Point it at your own server."""
    return StdioServerParameters(command="uv", args=["run", "--quiet", str(LABS / "servers" / "madar_mcp.py")])


WRITE_TOOLS = {"book_inspection", "reschedule_inspection", "create_refund", "escalate", "send_email"}


# ---------------------------------------------------------------- CAPSTONE: harness controls
def permission(name: str, args: dict) -> str:
    return "ask" if name in WRITE_TOOLS else "allow"


def approve_cli(name: str, args: dict) -> bool:
    return input(f"\n🛡️  Allow {name}({json.dumps(args, ensure_ascii=False)})? [y/N] ").strip().lower() == "y"


def cap_result(name: str, args: dict, result: str) -> str | None:
    """after_tool hook: keep tool results inside the context budget, and log every call."""
    with (OUT / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": name, "args": args, "chars": len(result)}, ensure_ascii=False) + "\n")
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n[… {len(result) - MAX_RESULT_CHARS} characters cut; ask for less]"
    return None


def compact(history: list) -> list:
    """Context rule: keep the last messages, never starting on a tool result (its call would be missing)."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    tail = history[-MAX_HISTORY_MESSAGES:]
    while tail and tail[0].get("role") != "user":
        tail = tail[1:]
    return tail


# ---------------------------------------------------------------- the agent
def make_agent(config: dict, mcp: K.MCPBridge, approve=approve_cli):
    def agent(prompt: str, m: madar.Madar, history: list | None = None, **kw) -> K.Run:
        tools = local_tools(m) + mcp.agent_tools()
        return K.run_agent(prompt, tools, system=config["system"], model=config["model"], history=history,
                           max_steps=MAX_STEPS, max_cost=MAX_COST, permission=permission, approve=approve,
                           after_tool=cap_result, verbose=kw.pop("verbose", False), **kw)
    return agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    c = sub.add_parser("chat")
    c.add_argument("--session", default="default")
    e = sub.add_parser("eval")
    e.add_argument("tasks", type=Path)
    e.add_argument("--config", nargs="+", default=["A", "B"])
    e.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    with K.MCPBridge(mcp_server()) as mcp:
        if args.cmd == "ask":
            run = make_agent(CONFIGS["A"], mcp)(args.question, madar.Madar(OUT / "db"), verbose=True)
            print("\n" + (run.final or f"(stopped: {run.stopped})"))
        elif args.cmd == "chat":
            path = OUT / f"session-{args.session}.json"
            history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            m, agent = madar.Madar(OUT / "db"), make_agent(CONFIGS["A"], mcp)
            print("Type your message (empty line to quit).")
            while (msg := input("\n🧑 ").strip()):
                run = agent(msg, m, history=compact(history))
                print("🤖", run.final or f"(stopped: {run.stopped})")
                history = run.state[1:]
                path.write_text(json.dumps(history, ensure_ascii=False, default=str), encoding="utf-8")
        else:
            tasks = evals.load_tasks(args.tasks)
            consent = lambda name, a: True  # noqa: E731 - unattended: task prompts give consent; db graders check writes
            frames = [evals.run_suite(make_agent(CONFIGS[c], mcp, approve=consent), tasks, trials=args.trials, name=c)
                      for c in args.config]
            import pandas as pd

            df = pd.concat(frames)
            df.drop(columns="final").to_csv(OUT / "eval.csv", index=False)
            overall, per_task = evals.report(df)
            print("\n", overall, "\n\nper task written to", OUT / "eval.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
