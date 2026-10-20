# %% [markdown]
# # Lab 2.2 · Search the policies
#
# Day 2 · Session 2 (Retrieval for agents) · 25 min. Rafeeq answers from ten bilingual policies. How it **finds** the
# right passage decides whether it answers correctly and how many tokens it spends. In this lab you:
# - build a small retrieval eval: 16 questions in English and Arabic with the policy that answers each;
# - compare **BM25** (keywords), **embeddings** (a multilingual static model from the Hugging Face Hub) and a
#   **hybrid** of the two;
# - give the agent **agentic search** tools (`list`, `grep`, `read`) instead of a search engine, and compare;
# - check that answers **cite** the policy they used.
#
# **Tasks**
# - [ ] 1 · A retrieval eval and the baseline (4')
# - [ ] 2 · BM25 with Arabic normalisation (6')
# - [ ] 3 · Embeddings and hybrid search (6')
# - [ ] 4 · Agentic search vs. a search tool (9')
# - [ ] ✱ Stretch · When the corpus fits, skip retrieval

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

import bm25s
import numpy as np
import pandas as pd
from model2vec import StaticModel

import madar

m = madar.Madar(K.lab_dir("D2_2", reset=True))
CHUNKS = madar.chunk_policies()
TEXTS = [f"{c['title']}\n{c['text']}" for c in CHUNKS]
print(len(CHUNKS), "chunks from", len(madar.load_policies()), "policies; e.g.", CHUNKS[3]["chunk_id"], "→", TEXTS[3][:80].replace("\n", " "))

# %% [markdown]
# ## 1 · A retrieval eval and the baseline (4')
#
# A retriever is good if the passage that answers the question is in its top *k*. The **gold** answer here is the
# policy ID. `recall@k` = the share of questions whose gold policy appears among the top-*k* chunks. With only 24
# chunks, recall@3 is near 100% for every method, so we compare **recall@1**: is the first result the right one?

# %%
QUESTIONS = [  # (question, gold policy IDs)
    ("How much does an illuminated shop sign cost?", {"POL-02"}),
    ("How many working days does a renovation decision take?", {"POL-03"}),
    ("Can I move my inspection to another day?", {"POL-05"}),
    ("What happens if I miss my inspection?", {"POL-05", "POL-02"}),
    ("Which documents are needed for a building permit?", {"POL-01"}),
    ("What is the deadline to appeal a rejection?", {"POL-06"}),
    ("I put up a sign without a permit. What is the fine?", {"POL-07"}),
    ("Where can food trucks park near the university?", {"POL-09"}),
    ("Can you tell me whether my neighbour has a permit?", {"POL-08"}),
    ("What happens if I don't upload a missing document in time?", {"POL-04"}),
    ("كم رسوم رخصة البناء؟", {"POL-02"}),
    ("ما مدة الاعتراض على قرار الرفض؟", {"POL-06"}),
    ("هل يستطيع المساعد الرقمي الموافقة على طلبي؟", {"POL-10"}),
    ("ما هي أيام العمل الرسمية؟", {"POL-03"}),
    ("كم غرامة وضع لوحة بدون تصريح؟", {"POL-07"}),
    ("كم يومًا لدي لرفع مستند ناقص؟", {"POL-04"}),
]


def recall_at_k(search, k: int = 1) -> pd.DataFrame:
    """Run search(question, k) → list of chunk indices over QUESTIONS; one row per question."""
    rows = []
    for q, gold in QUESTIONS:
        got = [CHUNKS[i]["policy_id"] for i in search(q, k)]
        rows.append({"question": q[:45], "lang": "ar" if re.search("[؀-ۿ]", q) else "en",
                     "hit": bool(gold & set(got)), "top": " ".join(got)})
    return pd.DataFrame(rows)


def baseline(q: str, k: int) -> list[int]:  # madar.keyword_search: shared 5-letter word stems
    ids = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}
    return [ids[c["chunk_id"]] for c in madar.keyword_search(q, k)]


base = recall_at_k(baseline)
print("baseline recall@1:", base.hit.mean().round(2), "| by language:", base.groupby("lang").hit.mean().round(2).to_dict())
base[~base.hit]

# %% [markdown]
# ## 2 · BM25 with Arabic normalisation (6')
#
# BM25 scores documents by matching terms, weighting rare terms higher. Arabic needs a little normalisation first:
# the same word appears with different alef forms (أ إ آ ا), with or without diacritics (تَشْكِيل), with taa marbuta
# or haa (ة / ه) and with alef maqsura or yaa (ى / ي).
#
# ✏️ **Exercise** · Write `normalize(text)`: lowercase; remove Arabic diacritics (U+064B–U+0652) and tatweel (ـ);
# map أ إ آ → ا, ة → ه, ى → ي. Then build `bm25_search(q, k)` with `bm25s`: index `normalize(t)` for every text,
# tokenize with `bm25s.tokenize(..., return_ids=False, stopwords="en", show_progress=False)`, and return the top-k
# chunk indices.

# %%
# BEGIN SOLUTION
DIACRITICS = re.compile("[ً-ْـ]")


def normalize(text: str) -> str:
    text = DIACRITICS.sub("", text.lower())
    return text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"}))


bm25 = bm25s.BM25()
bm25.index(bm25s.tokenize([normalize(t) for t in TEXTS], return_ids=False, stopwords="en", show_progress=False),
           show_progress=False)


def bm25_search(q: str, k: int) -> list[int]:
    tokens = bm25s.tokenize([normalize(q)], return_ids=False, stopwords="en", show_progress=False)
    idx, _ = bm25.retrieve(tokens, k=k, show_progress=False)
    return [int(i) for i in idx[0]]
# END SOLUTION


K.check(normalize("الرُّسوم أُضيفت للمدينة") == "الرسوم اضيفت للمدينه", "Arabic normalisation")
bm = recall_at_k(bm25_search)
print("BM25 recall@1:", bm.hit.mean().round(2), "| by language:", bm.groupby("lang").hit.mean().round(2).to_dict())

# %% [markdown]
# ## 3 · Embeddings and hybrid search (6')
#
# **Embeddings** map text to vectors so that meaning, not spelling, decides similarity, including across languages:
# an Arabic question can match an English passage. `potion-multilingual-128M` is a *static* embedding model
# (model2vec): no transformer at query time, so it runs on any CPU in milliseconds.
#
# **Hybrid search** combines rankings. Reciprocal Rank Fusion (RRF) adds `1 / (60 + rank)` from each ranker; it
# needs no score calibration.
#
# ✏️ **Exercise** · Write `embed_search(q, k)` (cosine similarity against `E`, the normalised chunk vectors), and
# `hybrid_search(q, k)` that fuses the top-10 of `bm25_search` and `embed_search` with RRF.

# %%
emb = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
E = emb.encode(TEXTS)
E = E / np.linalg.norm(E, axis=1, keepdims=True)

# BEGIN SOLUTION
def embed_search(q: str, k: int) -> list[int]:
    v = emb.encode([q])[0]
    scores = E @ (v / np.linalg.norm(v))
    return [int(i) for i in np.argsort(-scores)[:k]]


def hybrid_search(q: str, k: int, depth: int = 10) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in (bm25_search(q, depth), embed_search(q, depth)):
        for rank, i in enumerate(ranking):
            fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
    return sorted(fused, key=fused.get, reverse=True)[:k]
# END SOLUTION


results = {"baseline": base, "bm25": bm, "embeddings": recall_at_k(embed_search), "hybrid": recall_at_k(hybrid_search)}
summary = pd.DataFrame({name: {"all": r.hit.mean(), "en": r[r.lang == "en"].hit.mean(), "ar": r[r.lang == "ar"].hit.mean()}
                        for name, r in results.items()}).T.round(2)
K.check(summary.loc["hybrid", "all"] >= summary.loc["baseline", "all"], "hybrid search beats the keyword baseline")
summary

# %% [markdown]
# 🔍 Which method wins on Arabic questions, and why? (The policies are mostly English with an Arabic summary at the
# end of each.) Look at the misses of the best method: is the gold label wrong, or the retriever?

# %%
results["hybrid"][~results["hybrid"].hit]

# %% [markdown]
# ## 4 · Agentic search vs. a search tool (9')
#
# Coding agents mostly don't use vector search: they **list**, **grep** and **read** files, like a person would, and
# decide what to read next. That works well when documents are well-structured and the agent can afford a few more
# steps. Here are both designs as tools.
#
# ✏️ **Exercise** · Write the three agentic tools (typed, with docstrings, so `K.tool_schema` can describe them):
# - `list_policies()` → `[{"policy_id", "title", "title_ar"}]`;
# - `grep_policies(pattern)` → up to 20 matches `{"policy_id", "line_no", "line"}` (case-insensitive regex, applied to
#   the raw lines of each policy);
# - `read_policy(policy_id)` → the full text, or an error dict if it doesn't exist.

# %%
POLICY_DOCS = {p["id"]: p for p in madar.load_policies()}


def search_policies(query: str, k: int = 3) -> list[dict]:
    """Search Madar's policies (English or Arabic) and return the best passages with their policy ID.

    Args:
        query: what you are looking for, e.g. "refund after rejection" or "رسوم اللوحة".
        k: number of passages (1-5).
    """
    return [{"policy_id": CHUNKS[i]["policy_id"], "text": CHUNKS[i]["text"]} for i in hybrid_search(query, max(1, min(k, 5)))]


# BEGIN SOLUTION
def list_policies() -> list[dict]:
    """List all Madar policies with their IDs and titles (English and Arabic)."""
    return [{"policy_id": p["id"], "title": p["title"], "title_ar": p["title_ar"]} for p in POLICY_DOCS.values()]


def grep_policies(pattern: str) -> list[dict]:
    """Find lines matching a regular expression (case-insensitive) in all policies.

    Args:
        pattern: a regex or plain word, e.g. "appeal" or "غرامة|fine".
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return [{"error": f"Invalid regex: {e}"}]
    hits = []
    for pid, p in POLICY_DOCS.items():
        for n, line in enumerate(p["text"].splitlines(), 1):
            if rx.search(line):
                hits.append({"policy_id": pid, "line_no": n, "line": line.strip()[:200]})
    return hits[:20] or [{"note": "No match. Try another word, an Arabic term, or list_policies."}]


def read_policy(policy_id: str) -> dict:
    """Read one policy in full.

    Args:
        policy_id: e.g. "POL-06".
    """
    p = POLICY_DOCS.get(policy_id.upper())
    return {"policy_id": p["id"], "text": p["text"]} if p else {"error": f"No policy {policy_id}. Use list_policies."}
# END SOLUTION


K.check(len(list_policies()) == 10 and any(h["policy_id"] == "POL-06" for h in grep_policies("appeal"))
        and "error" in read_policy("POL-99"), "agentic search tools work")

# %% [markdown]
# Run both designs on four questions with known answers. Record correctness, citations, tool calls and input tokens.

# %%
TESTS = [  # (question, expected text in the answer, policy that should be cited)
    ("What is the fine for a shop sign put up without a permit?", "1,000", "POL-07"),
    ("How many days do I have to upload a missing document, and what if I don't?", "14", "POL-04"),
    ("Which food-truck zone is closed during exam weeks?", "FZ-2", "POL-09"),
    ("كم يومًا لدي للاعتراض على رفض طلبي؟", "30", "POL-06"),
][: K.pick(4, 2)]
SYS = madar.SYSTEM_PROMPT + "\nAnswer only from the policies you found with your tools, and cite the policy ID."
rows = []
for q, expected, pid in TESTS:
    for design, tools in [("search tool", [search_policies]), ("agentic", [list_policies, grep_policies, read_policy])]:
        K.LEDGER.reset()
        r = K.run_agent(q, tools, system=SYS, max_steps=8, verbose=False)
        answer = r.final.replace("٬", ",")
        rows.append({"question": q[:40], "design": design, "correct": expected.replace(",", "") in answer.replace(",", ""),
                     "cites": pid in answer, "tool_calls": len(r.tool_calls), "input_tokens": K.LEDGER.input_tokens})
compare = pd.DataFrame(rows)
print(compare.groupby("design")[["correct", "cites", "tool_calls", "input_tokens"]].mean().round(2))
compare

# %% [markdown]
# 🔍 When does agentic search win? Think of exact identifiers (`FZ-2`, `POL-07`), tables, and documents whose
# structure the agent can navigate. When does an index win? Thousands of documents, fuzzy phrasing, cross-lingual
# questions, and tight latency budgets. Many production agents offer both.

# %% [markdown]
# ## ✱ Stretch · When the corpus fits, skip retrieval
#
# All ten policies are about 4,400 tokens. Put them **all** in the system prompt (no tools) and ask the four
# questions again. Compare correctness and input tokens with the two designs above. When is "just put it in
# context" the right answer, and what does Session 1's caching lesson add to it?

# %%
# BEGIN SOLUTION
ALL = "\n\n".join(f"## {p['id']} · {p['title']}\n{p['text']}" for p in POLICY_DOCS.values())
for q, expected, pid in TESTS:
    K.LEDGER.reset()
    resp = K.chat([{"role": "system", "content": SYS + "\n\n# Policies\n" + ALL}, {"role": "user", "content": q}])
    ans = K.text_of(resp).replace("٬", ",")
    print(f"{q[:40]:42s} correct={expected.replace(',', '') in ans.replace(',', '')} cites={pid in ans} tokens={K.LEDGER.input_tokens}")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Which retriever would you ship for Madar's 10 policies? For 10,000 municipal documents?
# 2. An Arabic question misses with BM25 but hits with embeddings. What does the embedding model do that BM25 can't?
# 3. Give one question where `grep` beats any embedding search, and one where it fails.
# 4. Why ask the agent to cite policy IDs, and how would you check the citations automatically?
