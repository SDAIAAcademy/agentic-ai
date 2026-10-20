# %% [markdown]
# # Lab 3.1 · Harness teardown
#
# Day 3 · Session 1 (Anatomy of a harness) · 25 min. *Agent = model + harness.* The harness is everything around the
# model: the loop, the system prompt, the tools, the context files, sessions, compaction, permissions, hooks and the
# sandbox. In this lab you:
# - run **Pi**, a minimal open-source coding-agent harness, on a free OpenRouter model and read its JSON event stream
#   (or use the sample stream if you don't have Node.js);
# - map Pi's events onto your own loop's trace;
# - compare four harnesses (Pi, OpenClaw, Hermes Agent, Mercury Agent) on the same dimensions;
# - add the harness controls you just saw (permissions, hooks, an audit log) to your own loop.
#
# **Tasks**
# - [ ] 1 · Pi from the outside: its event stream (6')
# - [ ] 2 · Pi from the inside: prompt and loop (5')
# - [ ] 3 · Four harnesses, one matrix (4')
# - [ ] 4 · Permissions, hooks and an audit log for your loop (10')
# - [ ] ✱ Stretch · Pi in read-only mode on a real bug

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
import re
import shutil

import pandas as pd

import madar

WORK = K.lab_dir("D3_1", reset=True)

# %% [markdown]
# ## 1 · Pi from the outside: its event stream (6')
#
# [Pi](https://github.com/earendil-works/pi) (MIT, TypeScript) gives the model four tools by default (`read`, `bash`,
# `edit`, `write`), a short system prompt, and a session stored as a JSONL tree. OpenClaw is built on it. Install it
# once (Node.js 22.19+), outside this notebook:
#
# ```bash
# npm install -g --ignore-scripts @earendil-works/pi-coding-agent
# pi update --models          # cache OpenRouter's model list (print and JSON modes use the cached list)
# ```
#
# The cell below runs Pi on a copy of `madar_fees` (the Lab 3.3 repository) in **JSON mode**, with read-only tools,
# your `OPENROUTER_API_KEY` and the course's free model. Without `pi` or a key it falls back to a **synthetic** sample
# stream in Pi's documented format (`data/traces/README.md`).

# %%
repo = WORK / "madar_fees"
shutil.copytree(K.DATA / "madar_fees", repo, ignore=shutil.ignore_patterns("__pycache__"))
TASK = ("Why does tests/test_fees.py::test_small_illuminated_sign_surcharge_after_minimum fail? "
        "Read the code and the policy, explain the cause and the fix. Do not edit files.")
pi = shutil.which("pi")
events_file = WORK / "pi_events.jsonl"
if pi and os.getenv("OPENROUTER_API_KEY"):
    cmd = [pi, "--mode", "json", "--no-session", "--no-approve", "--provider", "openrouter", "--model", K.MODEL,
           "--tools", "read,grep,find,ls", TASK]
    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=600)
    events_file.write_text(out.stdout, encoding="utf-8")
    print("live run, exit code", out.returncode, out.stderr[-300:])
else:
    shutil.copy(K.DATA / "traces" / "pi_sample_events.jsonl", events_file)
    print("pi or OPENROUTER_API_KEY not found: using the synthetic sample stream")
events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
print(len(events), "events:", pd.Series([e["type"] for e in events]).value_counts().to_dict())

# %% [markdown]
# ✏️ **Exercise** · From `events`, build:
# - `tools_df`: one row per `tool_execution_end` with `tool`, `args` (from the matching `tool_execution_start`),
#   `ms` (`durationMs`) and `error` (`isError`);
# - `turns_df`: one row per assistant `message_end` with `stop` (`stopReason`), `input` and `output` tokens (`usage`);
# - `final`: the text of the last assistant message.

# %%
# BEGIN SOLUTION
starts = {e["toolCallId"]: e["args"] for e in events if e["type"] == "tool_execution_start"}
tools_df = pd.DataFrame([{"tool": e["toolName"], "args": json.dumps(starts.get(e["toolCallId"], {}))[:70],
                          "ms": e.get("durationMs"), "error": e["isError"]}
                         for e in events if e["type"] == "tool_execution_end"])
assistant_msgs = [e["message"] for e in events if e["type"] == "message_end" and e["message"]["role"] == "assistant"]
turns_df = pd.DataFrame([{"stop": msg["stopReason"], "input": msg["usage"]["input"], "output": msg["usage"]["output"]}
                         for msg in assistant_msgs])
final = "".join(b.get("text", "") for b in assistant_msgs[-1]["content"] if b["type"] == "text")
# END SOLUTION
print(tools_df, "\n")
print(turns_df, "\n")
print(final[:600])
K.check(len(tools_df) > 0 and "stop" in set(turns_df.stop), "parsed tool executions, turns and the final answer")

# %% [markdown]
# 🔍 Compare `turns_df.input` from turn to turn: the same growth you measured in Lab 2.1. Which `stopReason` values
# appear? (Pi's are `stop`, `length`, `toolUse`, `error`, `aborted`.)

# %% [markdown]
# ## 2 · Pi from the inside: prompt and loop (5')
#
# Pi's **system prompt** is a small set of named sections (from `dist/core/system-prompt.js`, Pi 1.3):
#
# ```text
# You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files,
# executing commands, editing code, and writing new files.
# <tools>  - read: …  - bash: …  - edit: …  - write: …  </tools>
# <rules>  - Use bash for file operations like ls, rg, find  - Be concise in your responses  - Show file paths clearly … </rules>
# <docs>   where Pi's own documentation lives, read only when asked about pi </docs>
# <project_context>  the AGENTS.md / CLAUDE.md files found in the project </project_context>
# <skills> names and descriptions; bodies are read on demand </skills>
# <cwd>    the working directory </cwd>
# ```
#
# Its **loop** (`packages/agent/src/agent-loop.ts`) is two nested `while` loops: the inner one runs while the last
# assistant message has tool calls (or steering messages are queued); the outer one picks up follow-up messages. One
# detail you met on Day 1: when a reply stops with `length`, Pi does not run the truncated tool calls.
#
# ✏️ **Exercise** · Write `to_trace(events)`: convert Pi's events into `K.Event`s, the same trace format as
# `K.run_agent`: the user message → `user`; each `tool_execution_start` → `call` (`name(args)`); each
# `tool_execution_end` → `result` or `error`; the final assistant text → `final`. Print it with `K.print_event`.

# %%
def to_trace(events: list[dict]) -> list[K.Event]:
    trace = []
    # BEGIN SOLUTION
    for e in events:
        if e["type"] == "message_end" and e["message"]["role"] == "user":
            content = e["message"]["content"]
            trace.append(K.Event("user", content if isinstance(content, str) else json.dumps(content)))
        elif e["type"] == "tool_execution_start":
            trace.append(K.Event("call", f"{e['toolName']}({json.dumps(e['args'], ensure_ascii=False)})", {"name": e["toolName"]}))
        elif e["type"] == "tool_execution_end":
            text = " ".join(c.get("text", "") for c in e["result"]["content"])
            trace.append(K.Event("error" if e["isError"] else "result", text, {"name": e["toolName"]}))
        elif e["type"] == "message_end" and e["message"]["role"] == "assistant" and e["message"]["stopReason"] == "stop":
            trace.append(K.Event("final", "".join(b.get("text", "") for b in e["message"]["content"] if b["type"] == "text")))
    # END SOLUTION
    return trace


pi_trace = to_trace(events)
for ev in pi_trace:
    K.print_event(ev, width=130)
K.check(pi_trace[0].kind == "user" and pi_trace[-1].kind == "final", "Pi's run as a course trace")

# %% [markdown]
# ## 3 · Four harnesses, one matrix (4')
#
# The open harnesses people actually run in 2026, on the dimensions from the session. Fill the empty cells from the
# slides and the projects' docs (one short phrase each), then compare with the solution. Where the solution says
# "check its README", the course didn't verify the detail: look it up, and note the date you checked.

# %%
DIMENSIONS = ["loop and interfaces", "default tools", "context files", "memory", "skills / extensions",
              "permissions and approvals", "sandbox", "MCP", "2026 security lesson"]
matrix = pd.DataFrame(index=DIMENSIONS, columns=["Pi", "OpenClaw", "Hermes Agent", "Mercury Agent", "your loop (K.run_agent)"])
# BEGIN SOLUTION
matrix["Pi"] = ["while-loop agent; TUI, print, JSON, RPC, SDK", "read, bash, edit, write (+grep, find, ls)",
                "AGENTS.md / CLAUDE.md", "none built in (sessions as JSONL tree, compaction)",
                "skills on demand; TypeScript extensions in-process", "none per call; project trust gates config",
                "none built in: run it in a container/VM", "built in (1.x)", "extensions run with your permissions"]
matrix["OpenClaw"] = ["Pi's loop behind an always-on gateway; chat channels; heartbeat wake-ups", "shell, files, browser, messaging",
                      "SOUL.md, AGENTS.md", "MEMORY.md and daily notes", "ClawHub skill registry",
                      "configurable allowlists", "optional Docker sandbox", "yes",
                      "CVE-2026-25253; 341 malicious ClawHub skills; exposed gateways"]
matrix["Hermes Agent"] = ["Python agent loop; CLI and chat gateways; scheduled tasks", "terminal, files, web, delegation",
                          "AGENTS.md, SOUL.md", "capped MEMORY.md (~2,200 chars) and USER.md (~1,375)",
                          "writes its own skills from experience", "approval for dangerous commands",
                          "local, Docker, SSH and cloud backends", "yes", "CVE-2026-9353: self-written skills are code too"]
matrix["Mercury Agent"] = ["agent loop with token budgets per task", "check its README", "check its README",
                           "check its README", "check its README", "permissions.yaml: scopes per tool",
                           "scopes, not isolation", "OpenAI-compatible model slot; MCP: check its README",
                           "a small project (~3.4k stars): check maintenance before adopting"]
matrix["your loop (K.run_agent)"] = ["25-line loop over 3 API shapes", "whatever functions you pass", "system prompt + AGENTS.md (Lab 2.3)",
                                     "memory tool contract (Lab 2.3)", "load_skill (Lab 2.3)", "permission + approve hooks (below)",
                                     "none: tools are Python functions in your process", "MCPBridge (Lab 2.4)",
                                     "Day 4: you attack it"]
# END SOLUTION
matrix

# %% [markdown]
# 🔍 The matrix is a snapshot (October 2026); these projects ship weekly. What doesn't change is the **list of
# dimensions**: use it to evaluate the next harness you meet. Which row would you check first before installing one
# on a machine that holds Madar's database?

# %% [markdown]
# ## 4 · Permissions, hooks and an audit log for your loop (10')
#
# `K.run_agent` has the same control points as Pi's and Claude Code's harnesses:
# - `permission(name, args)` → `"allow"`, `"ask"` or `"deny"`; `"ask"` calls `approve(name, args)`;
# - `before_tool(name, args)` → a string to **block** the call (returned to the model as the result), or `None`;
# - `after_tool(name, args, result)` → a replacement result (e.g. redacted), or `None`.
#
# ✏️ **Exercise** · Write Madar's harness policy:
# 1. `policy`: read tools `allow`; `book_inspection`, `reschedule_inspection`, `create_refund`, `escalate` → `ask`;
#    `send_email` → `ask`; anything else → `deny`.
# 2. `guard` (`before_tool`): block `send_email` unless `to` is the registered email of an applicant
#    (`m.sql("SELECT email FROM applicants")`), with a message the model can act on.
# 3. `audit` (`after_tool`): append `{"tool", "args", "ok"}` to `AUDIT` with any 10-digit number in `args` replaced by
#    `"[REDACTED]"`; return `None` (don't change the result).

# %%
m = madar.Madar(WORK / "db")
TOOLS = m.tools()
EMAILS = {r["email"] for r in m.sql("SELECT email FROM applicants")}
AUDIT: list[dict] = []

# BEGIN SOLUTION
def policy(name: str, args: dict) -> str:
    if name in m.READ_TOOLS:
        return "allow"
    if name in m.WRITE_TOOLS:
        return "ask"
    return "deny"


def guard(name: str, args: dict) -> str | None:
    if name == "send_email" and args.get("to", "").lower() not in EMAILS:
        return json.dumps({"error": f"Blocked by policy: {args.get('to')!r} is not an applicant's registered email. "
                                    "Madar only emails applicants at their registered address (POL-10)."})
    return None


def audit(name: str, args: dict, result: str) -> None:
    redacted = json.loads(re.sub(r"\b\+?\d{10,12}\b", "[REDACTED]", json.dumps(args, ensure_ascii=False)))
    AUDIT.append({"tool": name, "args": redacted, "ok": not result.startswith('{"error"')})
    return None
# END SOLUTION


def approver(allowed: set[str]):
    def approve(name: str, args: dict) -> bool:
        ok = name in allowed
        print(f"  🛡️ approval for {name}: {'granted' if ok else 'refused'}")
        return ok
    return approve


KHALID = ("I was charged twice for my sign permit SGN-40388. Please refund the second payment. "
          "ID ends 2277, mobile 0505502277. Yes, go ahead.")
r = K.run_agent(KHALID, TOOLS, system=madar.SYSTEM_PROMPT, permission=policy, approve=approver({"create_refund"}),
                before_tool=guard, after_tool=audit, max_steps=8)
K.check(len(m.sql("SELECT * FROM refunds WHERE permit_id = 'SGN-40388'")) == 1, "the approved refund was created")

# %% [markdown]
# Now the contractor's email from the inbox, with its hidden instruction to email personal data to an outside
# address. Approvals are refused for everything here, and the guard blocks the address even if someone approves it.

# %%
email = next(x for x in madar.load_inbox() if "buildfast" in x["sender"])
r = K.run_agent(f"Draft a reply to this email:\n\n{email['body']}", TOOLS, system=madar.SYSTEM_PROMPT,
                permission=policy, approve=approver({"send_email"}), before_tool=guard, after_tool=audit, max_steps=8)
sent = m.sql("SELECT to_addr FROM outbox")
print("outbox:", sent)
K.check(not any("buildfast" in s["to_addr"] for s in sent), "nothing was sent to the outside address")
K.check(all("0505502277" not in json.dumps(a) for a in AUDIT), "the audit log has no mobile numbers")
pd.DataFrame(AUDIT)

# %% [markdown]
# 🔍 The second run approves `send_email` on purpose: a tired human clicks "yes". The **guard** still stops it,
# because it encodes a rule (POL-10) in code. Approvals catch what you didn't foresee; guards enforce what you did.

# %% [markdown]
# ## ✱ Stretch · Pi in read-only mode on a real bug
#
# If you have `pi`, run it interactively in `artifacts/D3_1/madar_fees` with `pi --tools read,grep,find,ls` and ask
# it to explain one failing test in `tests/test_calendar.py`. Then open the session file under
# `~/.pi/agent/sessions/` and find the system prompt, the tool calls and the token usage of each turn. What would you
# change in its `AGENTS.md` to make it faster next time?

# %% [markdown]
# ## Checkpoint
#
# 1. Pi has no per-call approval and no sandbox. Under what conditions is that a reasonable design?
# 2. Which of your three hooks would have prevented OpenClaw's malicious-skill incidents, and which would not?
# 3. Why is the audit log written in `after_tool` rather than inside each tool?
# 4. Name one control that must live in the tool itself, not in the harness.
