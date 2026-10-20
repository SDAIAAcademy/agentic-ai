# %% [markdown]
# # Lab 2.1 · Budget the window
#
# Day 2 · Session 1 (Context is a budget) · 25 min. Every step of an agent resends the whole conversation: system
# prompt, tool definitions, every message and every tool result. What you put in that window decides cost, latency
# and, past a point, accuracy ("context rot"). In this lab you:
# - measure **what fills** Rafeeq's context after a short conversation;
# - **trim** oversized tool results before the model sees them;
# - **compact** an old history into a summary, without leaking personal data into it;
# - keep a **stable prefix** so prompt caching can work.
#
# **Tasks**
# - [ ] 1 · Anatomy of a context window (6')
# - [ ] 2 · Trim tool results (6')
# - [ ] 3 · Compact the history (7')
# - [ ] 4 · Stable prefixes and caching (6')
# - [ ] ✱ Stretch · Context editing: clear old tool results

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

import matplotlib.pyplot as plt
import pandas as pd

import madar

m = madar.Madar(K.lab_dir("D2_1", reset=True))
READ = m.tools(m.READ_TOOLS)
print("tokenizer for estimates:", K.TOKENIZER_ID, "| 'رسوم اللوحة المضيئة' =", K.count_tokens("رسوم اللوحة المضيئة"), "tokens")

# %% [markdown]
# ## 1 · Anatomy of a context window (6')
#
# Sara has a three-turn conversation with Rafeeq. We pass the history of each run into the next one, exactly as a
# chat app would.

# %%
TURNS = [
    "What is the status of REN-10482? National ID ends 4821, mobile 0501234821.",
    "If I make the renovation structural, 35 m², how much more would it cost?",
    "What documents do I need for a structural renovation, and what if one is missing?",
]
history = []
for q in TURNS:
    run = K.run_agent(q, READ, system=madar.SYSTEM_PROMPT, history=history, max_steps=6, verbose=False)
    history = run.state[1:]  # everything except the system message
    print(f"→ {q[:60]}…\n  {' '.join(run.final.split())[:160]}\n")
print(len(history), "messages in the history;", K.LEDGER)

# %% [markdown]
# ✏️ **Exercise** · Write `anatomy(system, tools, history)`: the estimated tokens (`K.count_tokens`) of each part of
# the **next** request: `system`, `tools` (their Chat Completions schemas, `K.tool_schema(t)`), `user` messages,
# `assistant` messages (text and tool calls), and `tool` results. Return a dict.

# %%
def anatomy(system: str, tools: list, history: list[dict]) -> dict:
    # BEGIN SOLUTION
    parts = {"system": K.count_tokens(system), "tools": K.count_tokens([K.tool_schema(t) for t in tools]),
             "user": 0, "assistant": 0, "tool": 0}
    for msg in history:
        parts[msg["role"]] += K.count_tokens({k: v for k, v in msg.items() if k != "role"})
    return parts
    # END SOLUTION


parts = anatomy(madar.SYSTEM_PROMPT, READ, history)
total = sum(parts.values())
print({k: f"{v} ({v / total:.0%})" for k, v in parts.items()}, "| total ≈", total)
pd.Series(parts).plot.barh(title="Where Rafeeq's next request spends its tokens", figsize=(7, 2.6))
plt.xlabel("estimated tokens")
plt.show()
K.check(set(parts) == {"system", "tools", "user", "assistant", "tool"} and total > 500, "anatomy of the window")

# %% [markdown]
# 🔍 Compare with what the provider counted for the last call (`usage.prompt_tokens`): our tokenizer is not the
# model's, and the chat template adds a little. Which part grows with every turn? Which part is paid on **every**
# call even when nothing changes?

# %% [markdown]
# ## 2 · Trim tool results (6')
#
# Tool results are the biggest and least controlled part: a search returns five long passages when one would do.
# `run_agent` takes an `after_tool(name, args, result)` hook that may return a replacement result.
#
# ✏️ **Exercise** · Write `trim(name, args, result, max_chars=900)`:
# - if the result is a JSON **list** longer than `max_chars`, keep as many items as fit, then add one item
#   `{"note": "N more results not shown; refine the query"}`;
# - if it is any other text longer than `max_chars`, cut it and append `" …[truncated, N chars omitted]"`;
# - otherwise return it unchanged.

# %%
def trim(name: str, args: dict, result: str, max_chars: int = 900) -> str:
    if len(result) <= max_chars:
        return result
    # BEGIN SOLUTION
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        kept, size = [], 0
        for item in data:
            s = len(json.dumps(item, ensure_ascii=False))
            if kept and size + s > max_chars:
                break
            kept.append(item)
            size += s
        if len(kept) < len(data):
            kept.append({"note": f"{len(data) - len(kept)} more results not shown; refine the query"})
        return json.dumps(kept, ensure_ascii=False)
    return result[:max_chars] + f" …[truncated, {len(result) - max_chars} chars omitted]"
    # END SOLUTION


big = m.search_policies("refund after rejection and withdrawal", k=5)
out = trim("search_policies", {}, json.dumps(big, ensure_ascii=False))
print(len(json.dumps(big, ensure_ascii=False)), "→", len(out), "chars |", out[-90:])
K.check(len(out) < len(json.dumps(big, ensure_ascii=False)) and "more results" in out, "long lists are trimmed with a note")

# %% [markdown]
# Now the same question with and without the hook. The system prompt asks for broad searches (`k=5`) to make the
# effect visible. Compare tokens **and** the answers: did trimming lose what the model needed?

# %%
Q = "Omar's sign permit was rejected. What refund does he get, and could he appeal instead?"
SYS = madar.SYSTEM_PROMPT + "\nWhen you search policies, always ask for k=5 results."
rows = []
for label, hook in [("raw", None), ("trimmed", trim)]:
    K.LEDGER.reset()
    r = K.run_agent(Q, READ, system=SYS, after_tool=hook, max_steps=5, verbose=False)
    rows.append({"variant": label, "input_tokens": K.LEDGER.input_tokens, "steps": r.steps,
                 "mentions_100": "100" in r.final, "answer": " ".join(r.final.split())[:110]})
pd.DataFrame(rows)

# %% [markdown]
# ## 3 · Compact the history (7')
#
# When the history passes a budget, replace the **old** turns with a short summary and keep the last few messages
# verbatim. Two things must survive compaction: facts the agent needs later (permit ID, what was decided) and
# **rules**: a summary must not copy personal data that the conversation no longer needs (PDPL data minimisation).
#
# ✏️ **Exercise** · Write `compact(history, keep_last=2)`:
# 1. split `history` into `old` and `recent` (the last `keep_last` messages; don't start `recent` with a `tool`
#    message: move the cut earlier until it starts with a `user` message);
# 2. ask `K.SMALL_MODEL` (with `K.chat`) to summarise `old` using `SUMMARY_PROMPT`;
# 3. return `[{"role": "user", "content": "Summary of the conversation so far: ..."}, *recent]`.

# %%
SUMMARY_PROMPT = """Summarise this conversation between a citizen and Rafeeq, Madar's assistant, for Rafeeq to continue it.
Keep: permit IDs, permit status and dates, amounts quoted, what the citizen asked for, and open questions.
Write "identity verified for <permit ID>" instead of any national ID digits or mobile numbers. At most 120 words."""


def compact(history: list[dict], keep_last: int = 2) -> list[dict]:
    # BEGIN SOLUTION
    cut = max(0, len(history) - keep_last)
    while cut > 0 and history[cut]["role"] != "user":
        cut -= 1
    old, recent = history[:cut], history[cut:]
    if not old:
        return history
    transcript = json.dumps(old, ensure_ascii=False)
    summary = K.text_of(K.chat([{"role": "system", "content": SUMMARY_PROMPT},
                                {"role": "user", "content": transcript}], model=K.SMALL_MODEL))
    return [{"role": "user", "content": f"Summary of the conversation so far: {summary}"}, *recent]
    # END SOLUTION


short = compact(history, keep_last=2)
print(short[0]["content"][:600])
before, after = K.count_tokens(history), K.count_tokens(short)
print(f"\nhistory ≈ {before} tokens → {after} tokens")
K.check(after < before / 2, "compaction at least halves the history")
K.check("4821" not in short[0]["content"] and "0501234821" not in short[0]["content"], "no ID digits or mobile in the summary")

# %% [markdown]
# Does Rafeeq still know what it needs? Continue the conversation from the compacted history.

# %%
r = K.run_agent("Remind me: which permit are we talking about, and when is its decision due?", READ,
                system=madar.SYSTEM_PROMPT, history=short, max_steps=4)
K.check("REN-10482" in r.final, "the permit ID survived compaction")

# %% [markdown]
# 🔍 Did the model call `get_permit` again? With the credentials gone from the summary it **can't**, unless it asks
# Sara again. Is that a bug or a feature? (Both APIs now offer server-side compaction: `/responses/compact` on OpenAI
# and compaction in the Anthropic Messages API. The trade-off is the same: what is kept is what the summary says.)

# %% [markdown]
# ## 4 · Stable prefixes and caching (6')
#
# Providers cache the **prefix** of a request: if the first N tokens are identical to a recent request, they are
# read from cache (faster, and billed at a fraction on paid models). OpenAI and most OpenRouter providers do this
# automatically for long prefixes; Anthropic needs `cache_control` breakpoints. Anything that changes near the top
# (a timestamp, a user name, reordered tools) breaks the cache for everything after it.
#
# ✏️ **Exercise** · Write `shared_prefix_tokens(req_a, req_b)`: serialize both requests with
# `json.dumps(..., ensure_ascii=False)`, find the length of their common character prefix, and return
# `K.count_tokens` of that prefix. Then compare the two system-prompt designs below.

# %%
def shared_prefix_tokens(req_a: dict, req_b: dict) -> int:
    # BEGIN SOLUTION
    a, b = json.dumps(req_a, ensure_ascii=False), json.dumps(req_b, ensure_ascii=False)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return K.count_tokens(a[:n])
    # END SOLUTION


POLICIES = "\n\n".join(f"## {p['id']} {p['title']}\n{p['text']}" for p in madar.load_policies())
tools = [K.tool_schema(t) for t in READ]


def request(system: str, question: str) -> dict:
    return {"model": K.MODEL, "tools": tools,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}


def bad_system(now: str) -> str:   # volatile first
    return f"Current time: {now}.\n{madar.SYSTEM_PROMPT}\n\n# Policies\n{POLICIES}"


def good_system(now: str) -> str:  # stable first, volatile last
    return f"{madar.SYSTEM_PROMPT}\n\n# Policies\n{POLICIES}\n\nCurrent time: {now}."


for name, fn in [("bad", bad_system), ("good", good_system)]:
    a = request(fn("2026-11-01 09:00"), "How much is an illuminated 3 m² sign?")
    b = request(fn("2026-11-01 09:05"), "Which food-truck zones are open on Fridays?")
    print(f"{name:5s} shared prefix ≈ {shared_prefix_tokens(a, b):5d} of {K.count_tokens(a)} tokens")

# %% [markdown]
# Now ask the provider: two calls with the long, stable prefix, a few seconds apart. Read `cached_tokens` from the
# second response's usage. Free providers on OpenRouter may or may not cache (and report `0`): the design rule
# holds anyway, because on paid models a cached prefix costs 10-50% of normal input.

# %%
for q in ["How much is an illuminated 3 m² sign?", "Which food-truck zones are open on Fridays?"]:
    resp = K.chat([{"role": "system", "content": good_system("2026-11-01 09:00")}, {"role": "user", "content": q}],
                  use_cache=False, max_tokens=150)
    details = resp.usage.prompt_tokens_details
    print(f"prompt {resp.usage.prompt_tokens} tokens | cached {getattr(details, 'cached_tokens', None) if details else None}"
          f" | {K.text_of(resp)[:70]!r}")

# %% [markdown]
# For reference, how each API marks or keys a cacheable prefix:
#
# ```python
# # Anthropic Messages: explicit breakpoints (up to 4); everything up to the marked block is cached
# system=[{"type": "text", "text": SYSTEM_AND_POLICIES, "cache_control": {"type": "ephemeral"}}]
#
# # OpenAI Responses / Chat Completions: automatic for long prefixes; a key improves the hit rate across users
# client.responses.create(..., prompt_cache_key="rafeeq-v3")
# ```

# %% [markdown]
# ## ✱ Stretch · Context editing: clear old tool results
#
# Anthropic's context editing (and many harnesses) clear **old tool results** but keep the call that produced them,
# so the model knows it already looked something up. Write `clear_old_results(history, keep=1)` that replaces the
# `content` of every `tool` message except the last `keep` with `"[cleared to save context; call the tool again if
# needed]"`. Measure the tokens saved on `history`.

# %%
# BEGIN SOLUTION
def clear_old_results(history: list[dict], keep: int = 1) -> list[dict]:
    tool_idx = [i for i, msg in enumerate(history) if msg["role"] == "tool"]
    old = set(tool_idx[:-keep] if keep else tool_idx)
    return [{**msg, "content": "[cleared to save context; call the tool again if needed]"} if i in old else msg
            for i, msg in enumerate(history)]


cleared = clear_old_results(history)
print(K.count_tokens(history), "→", K.count_tokens(cleared), "tokens")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Which part of Rafeeq's window would you cut first, and which would you never cut?
# 2. A summary dropped the national ID digits, so Rafeeq asks Sara to verify again. Is that acceptable for Madar?
# 3. Why does putting the current time at the **top** of a system prompt cost money?
# 4. Trimming, compaction, clearing: which one loses information the model can't get back by calling a tool?
