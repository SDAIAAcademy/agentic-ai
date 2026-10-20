# %% [markdown]
# # Lab 1.1 · Two providers, one client
#
# Day 1 · Session 1 (The agentic landscape) · 25 min. Every agent in this course talks to models through an
# **OpenAI-compatible HTTP API**. One client, a different `base_url`, and you can switch between OpenRouter, Hugging Face
# Inference Providers, vLLM, Ollama or a hosted vendor. In this lab you:
# - call a free model on OpenRouter with the plain `openai` SDK and read everything the response tells you;
# - stream a response and measure time to first token;
# - list the free models that support tool calling, and shortlist them;
# - make a call survive a busy free model with **fallbacks**;
# - see what Arabic costs in tokens compared with English.
#
# **Tasks**
# - [ ] 1 · First call: read the whole response (5')
# - [ ] 2 · Stream it; time to first token (4')
# - [ ] 3 · The free model menu (6')
# - [ ] 4 · Fallbacks and routing (5')
# - [ ] 5 · Tokens: Arabic vs. English (5')
# - [ ] ✱ Stretch · Check your shortlist on a Madar question with a known answer

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
import time

import httpx
import pandas as pd
from openai import OpenAI

pd.set_option("display.max_colwidth", 70)
MODEL = K.MODEL  # a free model, e.g. google/gemma-4-31b-it:free

# %% [markdown]
# ## 1 · First call: read the whole response (5')
#
# In this lab we use the SDK **directly** (no `agentkit` helpers) so you see every moving part. OpenRouter only needs a
# different `base_url` and your key; the optional headers name your app on openrouter.ai.
#
# ✏️ **Exercise** · Create `client` for OpenRouter and send one Chat Completions request to `MODEL` with a system
# message ("You answer in one short sentence.") and a user message asking what a building permit is, in Arabic.
# Store the response in `resp`.

# %%
# BEGIN SOLUTION
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                default_headers=K.headers("openrouter"))
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "ما هي رخصة البناء؟"},
    ],
)
# END SOLUTION
print(resp.choices[0].message.content)

# %% [markdown]
# A response is more than its text. Find, in the dump below: the model that **actually** served you (free routers may
# pick a variant), `finish_reason`, the token counts, and the `cost` (0 for free models).

# %%
K.show(resp)

# %%
u = resp.usage
print(f"model: {resp.model} | finish: {resp.choices[0].finish_reason} | "
      f"tokens in/out: {u.prompt_tokens}/{u.completion_tokens} | cost: {getattr(u, 'cost', None)}")
K.check(resp.choices[0].finish_reason in ("stop", "length"), "the model finished")

# %% [markdown]
# ## 2 · Stream it; time to first token (4')
#
# Agents feel fast when the first tokens arrive quickly. With `stream=True` the SDK yields **chunks**; each carries a
# `delta` with the next piece of text.
#
# ✏️ **Exercise** · Stream the answer to "List five kinds of municipal permits in one line each." Print the text as it
# arrives and measure `ttft` (seconds until the first non-empty text delta) and `total` (seconds until the end).

# %%
t0 = time.perf_counter()
ttft = None
# BEGIN SOLUTION
stream = client.chat.completions.create(
    model=MODEL, stream=True,
    messages=[{"role": "user", "content": "List five kinds of municipal permits in one line each."}],
)
for chunk in stream:
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        if ttft is None:
            ttft = time.perf_counter() - t0
        print(delta, end="", flush=True)
# END SOLUTION
total = time.perf_counter() - t0
print(f"\n\ntime to first token: {ttft:.2f}s | total: {total:.2f}s")
K.check(ttft is not None and ttft <= total, "measured time to first token")

# %% [markdown]
# ## 3 · The free model menu (6')
#
# OpenRouter's model list is public: no key needed. Each model says what it supports (`supported_parameters`) and what
# it costs. Free models end in `:free`.
#
# ✏️ **Exercise** · Build `free` as a DataFrame of free models that support **tool calling** (`"tools"` in
# `supported_parameters`), with columns `id`, `context_length`, `structured` (supports `structured_outputs`), sorted
# by `context_length` descending.

# %%
models = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
print(len(models), "models listed")
# BEGIN SOLUTION
free = pd.DataFrame([
    {"id": m["id"], "context_length": m.get("context_length"),
     "structured": "structured_outputs" in (m.get("supported_parameters") or [])}
    for m in models
    if m["id"].endswith(":free") and "tools" in (m.get("supported_parameters") or [])
]).sort_values("context_length", ascending=False, ignore_index=True)
# END SOLUTION
K.check(len(free) > 0 and {"id", "context_length", "structured"} <= set(free.columns), f"{len(free)} free tool-calling models")
free

# %% [markdown]
# **Shortlisting.** A model list tells you what a model *accepts*, not how *good* it is. Before choosing, look at
# independent benchmarks, then test on **your** task (the ✱ Stretch, and the eval suite on Day 3):
#
# | Source | What it measures |
# |---|---|
# | [Artificial Analysis](https://artificialanalysis.ai/models) | intelligence index, speed (tokens/s, time to first token) and price, across providers |
# | [LMArena](https://lmarena.ai/leaderboard) | human preference votes (text, vision, WebDev) |
# | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | function calling: simple, parallel, multi-turn, relevance |
# | [τ²-bench](https://taubench.com) | tool-using agents in customer-service domains, `pass^k` reliability |
# | [Terminal-Bench](https://www.tbench.ai/leaderboard) | agents doing real terminal tasks (the harness matters as much as the model) |
#
# ✏️ **Exercise** · Open two of these sources and write down, for three models from `free` (or their paid parent
# models), one quality number and one speed number. Put them in the `shortlist` dict below.

# %%
shortlist = {
    # "google/gemma-4-31b-it:free": {"source": "...", "quality": "...", "speed": "..."},
}

# %% [markdown]
# ## 4 · Fallbacks and routing (5')
#
# Free models are shared and often **rate-limited** (HTTP 429). OpenRouter can try a list of models in order with the
# `models` parameter, and pick providers with the `provider` object. These are OpenRouter extensions, so they go in
# `extra_body`.
#
# ✏️ **Exercise** · Ask "Reply with one word: ready" with `models` = `[MODEL]` followed by `K.FALLBACKS`, and the
# provider preference `{"sort": "latency"}`. Print which model served the request.

# %%
# BEGIN SOLUTION
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Reply with one word: ready"}],
    extra_body={"models": [MODEL, *K.FALLBACKS], "provider": {"sort": "latency"}},
)
# END SOLUTION
print("asked for:", MODEL, "| served by:", r.model)
K.check(bool(r.choices[0].message.content), "a model in the fallback list answered")

# %% [markdown]
# **Same code, another provider.** Hugging Face Inference Providers speaks the same API at
# `https://router.huggingface.co/v1`, with model IDs from the Hub (`Qwen/Qwen3.6-35B-A3B`, optionally `:cheapest` or
# `:fastest`). It needs a token with *Make calls to Inference Providers* and credits (HF PRO includes a small amount),
# so this cell only runs if `HF_TOKEN` is set.

# %%
if os.getenv("HF_TOKEN"):
    hf = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
    try:
        r = hf.chat.completions.create(model=K.DEFAULT_MODELS["hf"] + ":cheapest",
                                       messages=[{"role": "user", "content": "Reply with one word: ready"}], max_tokens=50)
        print("HF served by:", r.model, "|", r.choices[0].message.content)
    except Exception as e:  # noqa: BLE001 - no credits is a normal outcome
        print("HF call failed (often: no credits):", str(e)[:150])
else:
    print("No HF_TOKEN: skipping (that's fine).")

# %% [markdown]
# ## 5 · Tokens: Arabic vs. English (5')
#
# You pay for, and wait for, **tokens**, not characters. Tokenizers trained mostly on English split Arabic into more
# pieces, which costs context and money.
#
# ✏️ **Exercise** · For each pair below, send the text with the instruction "Repeat exactly:" and read
# `usage.prompt_tokens`. Compute `ratio` = Arabic prompt tokens / English prompt tokens for each pair.

# %%
pairs = [
    ("Applicants have 14 calendar days to upload a missing document.", "لدى مقدم الطلب 14 يومًا لرفع المستند الناقص."),
    ("Only a licensing officer can approve or reject an application.", "الموظف المرخِّص وحده يوافق على الطلب أو يرفضه."),
    ("Illuminated signs pay a 25% surcharge on the sign fee.", "تضاف 25% إلى رسوم اللوحة المضيئة."),
]
rows = []
for en, ar in pairs:
    # BEGIN SOLUTION
    tok = {}
    for lang, text in (("en", en), ("ar", ar)):
        rr = client.chat.completions.create(model=MODEL, max_tokens=60,
                                            messages=[{"role": "user", "content": f"Repeat exactly: {text}"}])
        tok[lang] = rr.usage.prompt_tokens
    rows.append({"en_chars": len(en), "ar_chars": len(ar), "en_tokens": tok["en"], "ar_tokens": tok["ar"],
                 "ratio": tok["ar"] / tok["en"]})
    # END SOLUTION
tokens = pd.DataFrame(rows)
K.check(len(tokens) == 3 and "ratio" in tokens, "token ratio per pair")
tokens

# %% [markdown]
# 🔍 The prompt tokens include a fixed overhead (the chat template), so short texts understate the difference. The
# ratio is a property of the **tokenizer**: run the same pairs with another model to compare.

# %% [markdown]
# ## ✱ Stretch · Check your shortlist on a Madar question with a known answer
#
# A benchmark score is someone else's task. Ask three free models from `free` the question below (the right answer,
# from POL-02, is **750 SAR**: 3 × 200 = 600, plus 25%). Count how many get it right over 3 runs each, with
# `K.USE_CACHE` off for those calls (`temperature` left at its default).

# %%
QUESTION = ("Madar fee rules: signs cost 200 SAR per m² per year, minimum 400 SAR; illuminated signs pay 25% on top "
            "after the minimum. What is the yearly fee for an illuminated 3 m² sign? Answer with the number only.")
# BEGIN SOLUTION
results = {}
for model_id in free.id.head(3):
    ok = 0
    for _ in range(3):
        try:
            rr = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": QUESTION}], max_tokens=400)
            ok += "750" in (rr.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001 - busy free models
            print(model_id, "failed:", str(e)[:80])
    results[model_id] = f"{ok}/3"
results
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Which three lines would you change to move this lab from OpenRouter to a vLLM server on your own GPU?
# 2. A free model answers in 0.4 s on Monday and fails with 429 on Tuesday. What do you put in place before an agent
#    depends on it?
# 3. Your Arabic prompts cost about how many times the English ones on this model? Where would that matter in an agent?
# 4. Which benchmark in the table is closest to what Rafeeq, Madar's assistant, has to do? Why is it still not enough?
