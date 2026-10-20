# %% [markdown]
# # Lab 3.2 · Rafeeq on the OpenAI Agents SDK
#
# Day 3 · Session 2 (Agent SDKs and hosted agents) · 25 min. You've built a harness by hand. Now rebuild Rafeeq on
# a harness someone else wrote: the **OpenAI Agents SDK** (open source, `openai-agents`), running on a free
# OpenRouter model through Chat Completions. In this lab you:
# - define Rafeeq as an `Agent` with function tools made from Madar's methods;
# - add an **input guardrail** that stops prompt injections before the model runs;
# - require **human approval** for write tools, and resume the run after a decision;
# - keep conversation history in a **SQLite session**;
# - compare the SDK's run with your own loop's trace.
#
# **Tasks**
# - [ ] 1 · Agent, model and tools (6')
# - [ ] 2 · An input guardrail (5')
# - [ ] 3 · Approvals for write tools (8')
# - [ ] 4 · Sessions, and the SDK vs. your loop (6')
# - [ ] ✱ Stretch · Streaming events

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

import pandas as pd
from agents import (Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OpenAIChatCompletionsModel,
                    RunContextWrapper, Runner, SQLiteSession, function_tool, input_guardrail, set_tracing_disabled)
from openai import AsyncOpenAI

import madar

WORK = K.lab_dir("D3_2", reset=True)
m = madar.Madar(WORK)
set_tracing_disabled(True)  # tracing exports to OpenAI's dashboard by default; we use free OpenRouter models instead

# %% [markdown]
# ## 1 · Agent, model and tools (6')
#
# The SDK talks to OpenAI's Responses API by default. For any other provider, wrap an `AsyncOpenAI` client in
# `OpenAIChatCompletionsModel`. `function_tool` turns a typed, documented function into a tool, exactly like
# `K.tool_schema` (it reads the same Google-style docstrings).
#
# ✏️ **Exercise** · Create `model` for OpenRouter, the read tools with `function_tool`, and `rafeeq`, an `Agent`
# named "Rafeeq" with `madar.SYSTEM_PROMPT` as instructions. Then run Sara's question with `await Runner.run(...)`
# (`max_turns=6`).

# %%
SARA = "What is the status of REN-10482 and when is the decision due? ID ends 4821, mobile 0501234821."
# BEGIN SOLUTION
client = AsyncOpenAI(base_url=K.PROVIDERS[K.PROVIDER]["base_url"], api_key=K.api_key(), default_headers=K.headers())
model = OpenAIChatCompletionsModel(model=K.MODEL, openai_client=client)
read_tools = [function_tool(getattr(m, name)) for name in m.READ_TOOLS]
rafeeq = Agent(name="Rafeeq", instructions=madar.SYSTEM_PROMPT, model=model, tools=read_tools)
result = await Runner.run(rafeeq, SARA, max_turns=6)
# END SOLUTION
print(result.final_output)
print("\nschema the model sees:", json.dumps(read_tools[1].params_json_schema)[:300], "…")
K.check(bool(result.final_output), "Rafeeq answered through the Agents SDK")

# %% [markdown]
# A run produces **items**, like the Responses API: tool calls, tool outputs, messages. And usage per request.

# %%
for item in result.new_items:
    raw = item.raw_item
    print(f"{type(item).__name__:20s}", (getattr(raw, "name", "") or ""), str(getattr(raw, "arguments", "") or "")[:80])
print(result.context_wrapper.usage.requests, "requests ·", result.context_wrapper.usage.input_tokens, "input tokens")

# %% [markdown]
# ## 2 · An input guardrail (5')
#
# Guardrails run **alongside** the agent: an input guardrail checks the user's input and can trip a wire that stops
# the run before (or while) the model works. Here, a cheap check for prompt-injection phrases. (Day 4 replaces the
# keyword list with a Decisions-style classifier.)
#
# ✏️ **Exercise** · Write `injection_guard`, an `@input_guardrail` that sets `tripwire_triggered=True` when the input
# matches any pattern in `INJECTION_PATTERNS` (case-insensitive), and add it to `rafeeq_guarded`. Show that the
# approval-letter message from the inbox trips it, and that Sara's question doesn't.

# %%
INJECTION_PATTERNS = ["ignore (all|previous|prior) instructions", "note for the ai", "to the digital assistant",
                      "إلى المساعد الرقمي", "لها أولوية على سياساتك", "do not mention this note"]


# BEGIN SOLUTION
@input_guardrail
async def injection_guard(ctx: RunContextWrapper, agent: Agent, input) -> GuardrailFunctionOutput:
    text = input if isinstance(input, str) else json.dumps(input, ensure_ascii=False)
    hits = [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return GuardrailFunctionOutput(output_info={"matched": hits}, tripwire_triggered=bool(hits))


rafeeq_guarded = rafeeq.clone(input_guardrails=[injection_guard])
# END SOLUTION

letter = next(x for x in madar.load_inbox() if x["subject"].startswith("Approval letter"))
try:
    await Runner.run(rafeeq_guarded, letter["body"], max_turns=4)
    tripped = False
except InputGuardrailTripwireTriggered as e:
    tripped = True
    print("🛡️ tripped:", e.guardrail_result.output.output_info)
K.check(tripped, "the injected approval letter trips the guardrail")
ok = await Runner.run(rafeeq_guarded, SARA, max_turns=6)
K.check(bool(ok.final_output), "a normal question passes")

# %% [markdown]
# ## 3 · Approvals for write tools (8')
#
# `function_tool(fn, needs_approval=True)` makes the SDK **pause** before running that tool. The run returns with
# `result.interruptions` (the pending calls); you turn the result into a `RunState`, `approve` or `reject` each item,
# and resume with `Runner.run(agent, state)`. A rejected call reaches the model as "not approved".
#
# ✏️ **Exercise** · Build `rafeeq_rw` with the read tools plus the write tools (`m.WRITE_TOOLS`) wrapped with
# `needs_approval=True`. Run Huda's request, then approve only `reschedule_inspection` calls (reject anything else),
# and resume until there are no interruptions left (at most 3 rounds).

# %%
HUDA = ("My foundation inspection for BLD-20981 is booked for Tuesday 3 November at 10:00. Please move it to "
        "Thursday 5 November, any time before noon. Go ahead. ID ends 1196, mobile 0548841196.")
decisions = []
# BEGIN SOLUTION
write_tools = [function_tool(getattr(m, name), needs_approval=True) for name in m.WRITE_TOOLS]
rafeeq_rw = rafeeq.clone(tools=read_tools + write_tools)
result = await Runner.run(rafeeq_rw, HUDA, max_turns=8)
for _ in range(3):
    if not result.interruptions:
        break
    state = result.to_state()
    for item in result.interruptions:
        name = item.raw_item.name
        ok = name == "reschedule_inspection"
        decisions.append((name, item.raw_item.arguments, ok))
        state.approve(item) if ok else state.reject(item)
    result = await Runner.run(rafeeq_rw, state, max_turns=8)
# END SOLUTION
for d in decisions:
    print("🛡️", "approved" if d[2] else "rejected", d[0], d[1])
print(result.final_output)
booked = m.sql("SELECT slot, status FROM inspections WHERE permit_id = 'BLD-20981'")
print(booked)
K.check(any(d[0] == "reschedule_inspection" for d in decisions), "the reschedule waited for approval")
K.check(any(r["slot"] == "2026-11-05 11:00" and r["status"] == "booked" for r in booked), "moved to 5 Nov 11:00")

# %% [markdown]
# 🔍 Where would the pending `state` live in a real service, between "Rafeeq asks" and "an officer clicks approve"
# an hour later? (`RunState` serialises to JSON: `state.to_json()`.) That is the durable-execution problem Day 4's
# graph solves with checkpoints.

# %% [markdown]
# ## 4 · Sessions, and the SDK vs. your loop (6')
#
# A `SQLiteSession` stores the conversation so the next `Runner.run` continues it: no history to pass by hand.

# %%
session = SQLiteSession("sara-2026-11-01", str(WORK / "sessions.sqlite"))
r1 = await Runner.run(rafeeq, SARA, session=session, max_turns=6)
r2 = await Runner.run(rafeeq, "And if I add 35 m² of structural works, what would the fee be?", session=session, max_turns=6)
print(r2.final_output)
items = await session.get_items()
print(len(items), "items stored in the session")
K.check(len(items) > 4, "the session kept both turns")

# %% [markdown]
# ✏️ **Exercise** · Run the same question with `K.run_agent` and fill `compare` with, for each harness: model
# requests, tool calls and input tokens. Then list in `differences` three things the SDK gave you for free and one
# thing you can no longer see or change easily.

# %%
# BEGIN SOLUTION
K.LEDGER.reset()
mine = K.run_agent(SARA, m.tools(m.READ_TOOLS), system=madar.SYSTEM_PROMPT, max_steps=6, verbose=False)
sdk = await Runner.run(rafeeq, SARA, max_turns=6)
compare = pd.DataFrame([
    {"harness": "K.run_agent", "requests": mine.steps, "tool_calls": len(mine.tool_calls), "input_tokens": K.LEDGER.input_tokens},
    {"harness": "Agents SDK", "requests": sdk.context_wrapper.usage.requests,
     "tool_calls": sum(type(i).__name__ == "ToolCallItem" for i in sdk.new_items),
     "input_tokens": sdk.context_wrapper.usage.input_tokens},
])
differences = ["approvals that pause and resume with serialisable state", "guardrails that run alongside the agent",
               "sessions and handoffs built in",
               "less visible: the exact messages sent each turn and how tool errors are worded"]
# END SOLUTION
compare

# %% [markdown]
# ## ✱ Stretch · Streaming events
#
# `Runner.run_streamed(agent, input)` returns a result whose `stream_events()` yields raw model events (the semantic
# events you saw on Day 1) and higher-level `run_item_stream_event`s (tool called, tool output, message). Print one
# line per run-item event for Sara's question.

# %%
# BEGIN SOLUTION
streamed = Runner.run_streamed(rafeeq, SARA, max_turns=6)
async for ev in streamed.stream_events():
    if ev.type == "run_item_stream_event":
        print(ev.name, "·", type(ev.item).__name__)
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Which of today's SDK features did your Day 1 loop already have, and which would take you a day to build well?
# 2. The SDK pauses for approval. Where does a **hosted** runtime (OpenAI's Agents API, Claude Managed Agents) keep
#    that pending state, and what does that cost you in control?
# 3. Your guardrail is a keyword list. Name one injection from the inbox it would miss.
# 4. When would you pick the Claude Agent SDK instead, and what would you give up?
