# %% [markdown]
# # Lab 4.2 · Orchestrator and workers
#
# Day 4 · Session 2 (Multi-agent systems) · 25 min. One agent with every tool is the default, and often the right
# answer. Multi-agent designs pay off when work splits into **independent** parts that each need their own context
# (breadth, parallel reads), and cost you when parts are **coupled** (shared writes, decisions that depend on each
# other). In this lab you build both and measure:
# - three specialist **workers** (policy, records, scheduling) with the OpenAI Agents SDK;
# - an **orchestrator** that calls them as tools (agents-as-tools), and a **triage** agent that hands off to them;
# - the same question through the single agent, the orchestrator and the handoff: answer, requests, tokens, time;
# - the policy worker published over **A2A** (v1.0) as its own service, discovered by its agent card and called from
#   the orchestrator.
#
# **Tasks**
# - [ ] 1 · Workers and a meter (5')
# - [ ] 2 · Agents as tools vs. one agent (8')
# - [ ] 3 · Handoffs (5')
# - [ ] 4 · A worker over A2A (7')
# - [ ] ✱ Stretch · Parallel fan-out

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
import asyncio
import json
import time
import uuid

import httpx
import pandas as pd
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

import madar

WORK = K.lab_dir("D4_2", reset=True)
m = madar.Madar(WORK)
set_tracing_disabled(True)

# %% [markdown]
# ## 1 · Workers and a meter (5')
#
# To compare designs fairly we count **every** model request, including the ones a worker makes inside a tool call.
# The SDK's own usage counter covers one run; a worker called as a tool is a separate run. So we meter at the HTTP
# layer: an `httpx` hook on the client that reads `usage` from every response.

# %%
class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.requests, self.input_tokens, self.output_tokens = 0, 0, 0

    async def hook(self, response: httpx.Response):
        if response.request.url.path.endswith("/chat/completions"):
            await response.aread()
            usage = (response.json() or {}).get("usage") or {}
            self.requests += 1
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)


METER = Meter()
client = AsyncOpenAI(base_url=K.PROVIDERS[K.PROVIDER]["base_url"], api_key=K.api_key(), default_headers=K.headers(),
                     http_client=httpx.AsyncClient(timeout=120, event_hooks={"response": [METER.hook]}))
model = OpenAIChatCompletionsModel(model=K.MODEL, openai_client=client)
tool = {name: function_tool(getattr(m, name)) for name in m.READ_TOOLS + m.WRITE_TOOLS}

# %% [markdown]
# ✏️ **Exercise** · Create three workers, each with a narrow job, its own instructions and only its tools:
# - `policy` (search_policies): rules, fees in general, deadlines, cites POL IDs; knows nothing about applications;
# - `records` (get_permit, compute_fee): facts about one verified application;
# - `scheduling` (list_slots, book_inspection, reschedule_inspection): inspection slots and bookings.
#
# Give each a `handoff_description` (used in Task 3).

# %%
COMMON = "You work for Madar Municipal Services. Today is Sunday 1 November 2026. Reply in the language of the request, briefly."
# BEGIN SOLUTION
policy = Agent(name="policy", model=model, tools=[tool["search_policies"]],
               handoff_description="Questions about Madar's rules: fees, deadlines, documents, appeals, refunds.",
               instructions=f"{COMMON}\nYou answer from Madar's policies only, using search_policies, and cite the policy "
                            "ID, e.g. (POL-06). You know nothing about individual applications.")
records = Agent(name="records", model=model, tools=[tool["get_permit"], tool["compute_fee"]],
                handoff_description="Facts about one application (status, dates, fees) for a verified applicant.",
                instructions=f"{COMMON}\nYou look up one application with get_permit, only with the last 4 ID digits and "
                             "the registered mobile. Report status, dates and fees from the tool result only.")
scheduling = Agent(name="scheduling", model=model,
                   tools=[tool["list_slots"], tool["book_inspection"], tool["reschedule_inspection"]],
                   handoff_description="Booking or moving inspections.",
                   instructions=f"{COMMON}\nYou find inspection slots and book or move inspections for a verified applicant.")
# END SOLUTION
WORKERS = [policy, records, scheduling]
K.check(all(w.handoff_description for w in WORKERS) and len(records.tools) == 2, "three narrow workers")

# %% [markdown]
# ## 2 · Agents as tools vs. one agent (8')
#
# **Agents as tools**: the orchestrator keeps the conversation and calls workers like functions. Each worker runs in a
# **fresh context** with only the sub-question, and returns only its answer. That is context isolation: the
# orchestrator never sees a worker's tool results.
#
# ✏️ **Exercise** · Build `orchestrator` with the three workers as tools (`worker.as_tool(tool_name=..., tool_description=...)`),
# and instructions to split the request, call the workers it needs, and combine their answers. Then run `compare`.

# %%
# BEGIN SOLUTION
orchestrator = Agent(
    name="orchestrator", model=model,
    instructions=f"{COMMON}\nYou are Rafeeq's coordinator. Split the request into parts, ask the right specialist for each "
                 "part (pass the applicant's credentials to records and scheduling when given), then combine their "
                 "answers into one reply. Never invent facts the specialists did not give you.",
    tools=[w.as_tool(tool_name=f"ask_{w.name}", tool_description=w.handoff_description) for w in WORKERS])
# END SOLUTION
single = Agent(name="rafeeq", model=model, instructions=madar.SYSTEM_PROMPT, tools=list(tool.values()))

QUESTION = ("REN-10482: what is the status, when is the decision due, and if I withdraw it now how much of my fee do I "
            "get back? ID ends 4821, mobile 0501234821.")
EXPECT = {"due": ["2 November", "November 2", "2026-11-02", "2 Nov", "Nov 2"], "refund": ["167", "50%", "half"]}


async def measure(agent: Agent, question: str = QUESTION, label: str | None = None) -> dict:
    METER.reset()
    t0 = time.perf_counter()
    try:
        r = await Runner.run(agent, question, max_turns=10)
        final, last = str(r.final_output), r.last_agent.name
    except Exception as e:  # noqa: BLE001 - a failed run is a data point
        final, last = f"{type(e).__name__}: {e}", "-"
    ok = {k: any(v.lower() in final.lower() for v in vals) for k, vals in EXPECT.items()}
    return {"design": label or agent.name, "requests": METER.requests, "input_tokens": METER.input_tokens,
            "output_tokens": METER.output_tokens, "seconds": round(time.perf_counter() - t0, 1),
            "answered_by": last, **ok, "final": final}


results = [await measure(single, label="single agent"), await measure(orchestrator, label="agents as tools")]
compare = pd.DataFrame(results)
compare.drop(columns="final")

# %%
for r in results:
    print(f"── {r['design']}\n{r['final'][:600]}\n")
K.check(results[1]["requests"] > results[0]["requests"], "the orchestrator made more model requests than one agent")

# %% [markdown]
# 🔍 Read both answers. Did the orchestrator pass the credentials to `records`? Did a worker's answer lose a detail
# the final reply needed? Coordination costs requests and tokens, and every hand-over is a place to lose context.

# %% [markdown]
# ## 3 · Handoffs (5')
#
# A **handoff** transfers the conversation: the triage agent picks a specialist, and that specialist talks to the
# person directly with the conversation so far. Good when one specialist owns the whole request; bad when the request
# needs two of them (who answers the second half?).
#
# ✏️ **Exercise** · Build `triage` with `handoffs=WORKERS` and instructions to route each request to exactly one
# specialist. Measure it on a single-specialist request (`BOOKING`) and on `QUESTION`, and record who answered.

# %%
BOOKING = ("My food truck permit FTR-30311 was approved. Which health-inspection slots are free on Tuesday 3 or "
           "Wednesday 4 November in the morning? ID ends 6654, mobile 0532236654.")
# BEGIN SOLUTION
triage = Agent(name="triage", model=model, handoffs=WORKERS,
               instructions=f"{COMMON}\nYou route each request to exactly one specialist with a handoff. Don't answer yourself.")
handoff_runs = [await measure(triage, BOOKING, "handoff · booking"), await measure(triage, QUESTION, "handoff · question")]
# END SOLUTION
pd.DataFrame(handoff_runs).drop(columns="final")

# %%
K.check(any(r["answered_by"] != "triage" for r in handoff_runs), "triage handed off to a specialist")

# %% [markdown]
# ## 4 · A worker over A2A (7')
#
# Inside one program, a worker is an object. Across teams or vendors, it is a **service**. A2A (Agent2Agent, v1.0,
# Linux Foundation) standardises that: an **agent card** at `/.well-known/agent-card.json` says what the agent does
# and how to reach it; clients send **messages** and get back **tasks** with a state and **artifacts**.
#
# `servers/madar_a2a.py` publishes the policy worker over A2A (a PEP 723 script with its own loop: it doesn't even use
# the Agents SDK). Start it and read its card.

# %%
A2A_PORT = 8792
a2a_proc = K.serve(["uv", "run", "-q", "servers/madar_a2a.py", "--port", str(A2A_PORT)],
                   ready_url=f"http://127.0.0.1:{A2A_PORT}/.well-known/agent-card.json")
card = httpx.get(f"http://127.0.0.1:{A2A_PORT}/.well-known/agent-card.json").json()
print(json.dumps(card, indent=1, ensure_ascii=False)[:900])

# %% [markdown]
# The A2A SDK's client resolves the card and speaks the protocol. A synchronous answer comes back as a **task** in
# state `TASK_STATE_COMPLETED` with the answer in an artifact.
#
# ✏️ **Exercise** · Write `ask_remote_policy(question)`: create a client with `create_client(url)`, send a
# `SendMessageRequest` with a user `Message` (a `message_id`, `role=Role.ROLE_USER`, one text `Part`), and return the
# text of the responses (`get_stream_response_text`). Then wrap it as a function tool and give it to a new orchestrator
# in place of the local policy worker.

# %%
from a2a.client import create_client
from a2a.helpers import get_stream_response_text
from a2a.types import Message, Part, Role, SendMessageRequest

A2A_URL = f"http://127.0.0.1:{A2A_PORT}"


async def ask_remote_policy(question: str) -> str:
    """Ask Madar's policy specialist (a remote A2A agent) a question about permit rules; it cites policy IDs.

    Args:
        question: One self-contained question about Madar's rules (no personal data).
    """
    # BEGIN SOLUTION
    a2a = await create_client(A2A_URL)
    request = SendMessageRequest(message=Message(message_id=uuid.uuid4().hex, role=Role.ROLE_USER,
                                                 parts=[Part(text=question)]))
    texts = [get_stream_response_text(r) async for r in a2a.send_message(request)]
    return "\n".join(t for t in texts if t)
    # END SOLUTION


answer = await ask_remote_policy("If I withdraw a renovation application during review, how much of the fee is refunded?")
print(answer)
K.check(bool(answer.strip()), "the remote policy worker answered over A2A")

# %%
# BEGIN SOLUTION
orchestrator_a2a = orchestrator.clone(
    name="orchestrator+a2a",
    tools=[function_tool(ask_remote_policy), records.as_tool(tool_name="ask_records", tool_description=records.handoff_description),
           scheduling.as_tool(tool_name="ask_scheduling", tool_description=scheduling.handoff_description)])
# END SOLUTION
remote = await measure(orchestrator_a2a, label="agents as tools + A2A")
pd.DataFrame(results + handoff_runs + [remote]).drop(columns="final")

# %% [markdown]
# 🔍 The meter counts the orchestrator's requests but **not** the remote worker's: they happen in another process, on
# someone else's budget. What would you need from a remote agent to keep cost, latency and failures observable?

# %%
K.stop(a2a_proc)

# %% [markdown]
# ## ✱ Stretch · Parallel fan-out
#
# Agents-as-tools runs workers one at a time unless the model asks for several tools at once. When the parts are known
# up front, fan out in code: run `records` and `policy` concurrently with `asyncio.gather`, then let one final call
# combine them. Compare the time with the orchestrator.

# %%
# BEGIN SOLUTION
METER.reset()
t0 = time.perf_counter()
rec, pol = await asyncio.gather(
    Runner.run(records, "REN-10482: status and decision due date? ID ends 4821, mobile 0501234821.", max_turns=5),
    Runner.run(policy, "What share of the fee is refunded when a renovation application is withdrawn during review?", max_turns=5))
combine = Agent(name="combine", model=model, instructions=f"{COMMON}\nCombine the specialists' notes into one reply.")
final = await Runner.run(combine, f"Question: {QUESTION}\nRecords: {rec.final_output}\nPolicy: {pol.final_output}", max_turns=1)
print(final.final_output, f"\n{METER.requests} requests · {time.perf_counter() - t0:.1f} s")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. In your table, which design was cheapest and which answered best? Was the extra cost worth it for this question?
# 2. When does a handoff beat agents-as-tools, and when does it fail?
# 3. Two workers both want to write (book an inspection, create a refund). What could go wrong, and who should own writes?
# 4. What does an A2A agent card tell you, and what does it **not** tell you that you'd want before trusting the agent?
