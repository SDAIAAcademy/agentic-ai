# %% [markdown]
# # Lab 1.4 · The agent loop from scratch
#
# Day 1 · Session 4 (The agent loop) · 25 min. An agent is a model calling tools **in a loop** until it decides it
# is done. Everything else in this course (context engineering, harnesses, loops, graphs) wraps this loop. In this
# lab you:
# - write the loop yourself in about 25 lines of Chat Completions code;
# - give it Rafeeq's read tools and watch it chain several calls;
# - add the two guards every loop needs: a **step budget** and **tool errors the model can recover from**;
# - run the course's reference loop, `K.run_agent`, on all three API shapes and compare them.
#
# **Tasks**
# - [ ] 1 · Write `my_agent()` (8')
# - [ ] 2 · Multi-step questions and the trace (6')
# - [ ] 3 · One loop, three API shapes (5')
# - [ ] 4 · Break it (6')
# - [ ] ✱ Stretch · Run parallel tool calls concurrently

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

import madar

m = madar.Madar(K.lab_dir("D1_4", reset=True))
READ = m.tools(m.READ_TOOLS)  # search_policies, get_permit, compute_fee, list_slots
print([t.__name__ for t in READ])

# %% [markdown]
# ## 1 · Write `my_agent()` (8')
#
# ```text
# messages = [system, user]
# repeat up to max_steps:
#     reply = model(messages, tools)
#     append reply to messages
#     if reply has no tool calls: return reply.content        ← the model decided it is done
#     for each tool call: run it, append {"role": "tool", "tool_call_id": ..., "content": result}
# give up: return None                                         ← the loop decided it is done
# ```
#
# ✏️ **Exercise** · Implement `my_agent` with `K.chat(messages, tools=tools)` (it accepts Python functions) and
# `K.call_tool(by_name, name, arguments)` (it returns the result as JSON, or an error as JSON). Print one line per
# tool call. Append the assistant message with `msg.model_dump(exclude_none=True)`.

# %%
def my_agent(task: str, tools: list, system: str = madar.SYSTEM_PROMPT, max_steps: int = 8) -> str | None:
    by_name = {t.__name__: t for t in tools}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    # BEGIN SOLUTION
    for step in range(max_steps):
        msg = K.chat(messages, tools=tools).choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            print(f"  step {step + 1}: {call.function.name}({call.function.arguments})")
            result = K.call_tool(by_name, call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return None
    # END SOLUTION


answer = my_agent("How much is the yearly fee for an illuminated sign of 3 m²?", READ)
print(answer)
K.check(answer is not None and "750" in answer, "my_agent computed the fee with a tool (750 SAR)")

# %% [markdown]
# ## 2 · Multi-step questions and the trace (6')
#
# Fahad's permit is late. Answering him well needs **two** kinds of information: his permit (verified) and the
# escalation policy. Nobody tells the model the order; it plans as it goes.
#
# `K.run_agent` is the same loop as yours plus a **trace**: a list of events (`user`, `call`, `result`, `final`, ...)
# that you can print, count and test.

# %%
FAHAD = ("السلام عليكم، قدمت طلب رخصة بناء رقم BLD-20417 ولم يصلني رد. هل تأخر الطلب؟ وماذا أفعل؟ "
         "آخر أربعة أرقام من هويتي 7730 وجوالي 0555017730")
run = K.run_agent(FAHAD, READ, system=madar.SYSTEM_PROMPT, max_steps=8)
print(f"\n{run.stopped} after {run.steps} model calls · tools used: {[n for n, _ in run.tool_calls]}")

# %% [markdown]
# ✏️ **Exercise** · From `run.trace`, build `summary`: a dict with the number of model calls (`run.steps`), the number
# of tool calls, the number of tool errors (events of kind `error`), and whether `get_permit` was called **before**
# the final answer mentions an overdue period.

# %%
# BEGIN SOLUTION
kinds = [e.kind for e in run.trace]
summary = {"model_calls": run.steps, "tool_calls": kinds.count("call"), "tool_errors": kinds.count("error"),
           "verified_first": any(n == "get_permit" for n, _ in run.tool_calls)}
# END SOLUTION
print(summary)
K.check(summary["verified_first"], "Rafeeq looked up the permit (with verification) before answering")
K.check("9" in run.final or "٩" in run.final, "the answer says the permit is 9 working days overdue")

# %% [markdown]
# ## 3 · One loop, three API shapes (5')
#
# `K.run_agent(..., shape=...)` runs the same loop over Chat Completions, Responses or Messages. Only an **adapter**
# changes: how the request is built, where tool calls are found, and how results go back:
#
# | | `chat` | `responses` | `anthropic` |
# |---|---|---|---|
# | tool calls | `message.tool_calls` | `function_call` items | `tool_use` blocks |
# | results | one `role="tool"` message each | one `function_call_output` item each | one user message of `tool_result` blocks |
# | done when | no tool calls | no `function_call` items | no `tool_use` blocks |
#
# ✏️ **Exercise** · Run Sara's question on all three shapes and fill a table with: shape, `stopped`, model calls,
# tool calls, input tokens (from `K.LEDGER`, reset before each run), and the first 80 characters of the answer.

# %%
SARA = "What is the status of REN-10482 and when is the decision due? National ID ends 4821, mobile 0501234821."
rows = []
for shape in ["chat", "responses", "anthropic"]:
    # BEGIN SOLUTION
    K.LEDGER.reset()
    r = K.run_agent(SARA, READ, system=madar.SYSTEM_PROMPT, shape=shape, max_steps=6, verbose=False)
    rows.append({"shape": shape, "stopped": r.stopped, "model_calls": r.steps, "tool_calls": len(r.tool_calls),
                 "input_tokens": K.LEDGER.input_tokens, "answer": " ".join(r.final.split())[:80]})
    # END SOLUTION
shapes = pd.DataFrame(rows)
K.check(len(shapes) == 3 and (shapes.stopped == "done").all(), "all three shapes finished")
shapes

# %% [markdown]
# 🔍 Same model, same tools, same question: do the token counts differ? Each shape formats tools and history
# differently before the model's chat template turns them into tokens. Look at the adapters in `agentkit.py`
# (`ChatShape`, `ResponsesShape`, `MessagesShape`): about 25 lines each.

# %% [markdown]
# ## 4 · Break it (6')
#
# Real tools fail and real models loop. Try three failures and watch what the loop does.
#
# **(a) A tool that raises.** `get_permit` with a wrong mobile raises `ToolError`; `K.call_tool` turns it into
# `{"error": "..."}` so the model can ask for the right data instead of crashing the loop.

# %%
r = K.run_agent("Status of REN-10482? ID ends 4821, mobile 0509999999.", READ, system=madar.SYSTEM_PROMPT, max_steps=5)
K.check(any(e.kind == "error" for e in r.trace), "the tool error reached the model as a result")

# %% [markdown]
# **(b) A tool that never satisfies the model.** `check_queue` always answers "still processing, call again". Without
# a step budget, this loop would run until your quota is gone.
#
# ✏️ **Exercise** · Run `K.run_agent` with `[check_queue]` and `max_steps=4` on "Check the queue until the job is
# finished, then tell me the result." Confirm it stops with `max_steps`. Then write, in `fix`, one sentence you would
# add to the tool's **result** so the model stops on its own.

# %%
def check_queue(job_id: str = "J-1") -> dict:
    """Check whether a background job has finished.

    Args:
        job_id: the job to check.
    """
    return {"job_id": job_id, "status": "processing", "message": "Still processing. Call check_queue again."}


# BEGIN SOLUTION
r = K.run_agent("Check the queue until the job is finished, then tell me the result.", [check_queue], max_steps=4)
fix = "Jobs take about 10 minutes: tell the person to check back later instead of calling again."
# END SOLUTION
print("stopped:", r.stopped, "| fix:", fix)
K.check(r.stopped in ("max_steps", "done"), "the step budget bounds the loop")

# %% [markdown]
# **(c) A question with no tool.** Ask Rafeeq something outside its tools and policies ("Can you renew my driving
# licence?"). A good agent answers without calling any tool and says what it can't do (POL-10).

# %%
r = K.run_agent("Can you renew my driving licence?", READ, system=madar.SYSTEM_PROMPT, max_steps=4)
K.check(len(r.tool_calls) <= 1 and r.stopped == "done", "out-of-scope question answered without a tool spree")

# %% [markdown]
# ## ✱ Stretch · Run parallel tool calls concurrently
#
# When the model returns several tool calls in one turn, they are independent by definition. Write
# `my_parallel_agent`, a copy of `my_agent` that runs one turn's tool calls with a
# `concurrent.futures.ThreadPoolExecutor` and appends the results **in the original order**. Test it on a question
# that needs two fees.

# %%
from concurrent.futures import ThreadPoolExecutor

# BEGIN SOLUTION
def my_parallel_agent(task: str, tools: list, system: str = madar.SYSTEM_PROMPT, max_steps: int = 8) -> str | None:
    by_name = {t.__name__: t for t in tools}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(max_steps):
        msg = K.chat(messages, tools=tools).choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        print(f"  {len(msg.tool_calls)} call(s) this turn")
        with ThreadPoolExecutor() as pool:
            outs = list(pool.map(lambda c: K.call_tool(by_name, c.function.name, c.function.arguments), msg.tool_calls))
        messages.extend({"role": "tool", "tool_call_id": c.id, "content": o} for c, o in zip(msg.tool_calls, outs))
    return None


print(my_parallel_agent("Fees for a 3 m² illuminated sign and for a 420 m² building permit?", [m.compute_fee]))
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Who decides that the task is done: the model, the loop, or both? Point to the lines in `my_agent`.
# 2. Why does a tool error go back to the model as a **result** instead of raising?
# 3. Which line would you change to make `my_agent` speak the Messages API instead of Chat Completions?
# 4. Your agent costs too much on long tasks. Name two numbers in the trace you would watch first.
