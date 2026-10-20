# %% [markdown]
# # Lab 4.1 · A permit-review graph
#
# Day 4 · Session 1 (Graph engineering) · 25 min. Rafeeq answers citizens; behind it, Madar's licensing officers review
# applications. That review is a **process** with known steps, rules, one human decision and a letter at the end: a
# graph, not an open-ended agent. In this lab you build it with **LangGraph**:
# - typed **state** that every node reads and updates;
# - **nodes** that are plain code where rules are clear and a model call where judgement is needed;
# - a **router** (conditional edge) and a **human gate** (`interrupt`) for the officer's decision;
# - a **checkpointer** (SQLite) so a run survives a crash and resumes where it stopped.
#
# ```
# intake → document_check → fee_check → assess_risk ─┬─ (missing documents) → request_info ──┬→ draft_letter → record
#                                                    └─ (complete) → officer_review ⏸ ───────┘
# ```
#
# **Tasks**
# - [ ] 1 · State and the rule-based nodes (6')
# - [ ] 2 · Model nodes and the router (6')
# - [ ] 3 · The human gate: interrupt and resume (7')
# - [ ] 4 · Crash, restart, resume (6')
# - [ ] ✱ Stretch · Run the two checks in parallel

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
import operator
import sqlite3
from collections import Counter
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import madar

WORK = K.lab_dir("D4_1", reset=True)
m = madar.Madar(WORK, actor="review-graph")
# One case for every branch. SGN-40764 was charged by the old fee calculator, which forgot the 25% surcharge.
m.sql("UPDATE payments SET amount_sar = 1800 WHERE permit_id = 'SGN-40764'")
UPLOADS = {
    "REN-10275": ["صك_الملكية.pdf", "وصف_الأعمال.docx", "صور_الوضع_الحالي.zip"],
    "REN-10482": ["title-deed-scan.pdf", "kitchen works description.pdf", "photos_kitchen.jpg"],
    "SGN-40764": ["CR_1010456789.pdf", "shop_lease_2026.pdf", "sign_design_8.5m2.pdf", "electrical_safety_cert.pdf"],
    "BLD-20558": ["title_deed.pdf", "drawings_stamped_AlBina_Engineering.pdf", "site_plan.dwg", "contractor_contract.pdf"],
}
CALLS = Counter()  # how many times each node ran (Task 4 uses it)

# %% [markdown]
# ## 1 · State and the rule-based nodes (6')
#
# The **state** is a typed dict. Each node returns only the keys it changes; LangGraph merges them. A key with a
# **reducer** (`Annotated[list, operator.add]`) accumulates instead of being replaced: here, a log of what happened.
#
# Rules from POL-01 and POL-02 are **code**, not prompts: required documents and fees are exact.

# %%
class Review(TypedDict, total=False):
    permit_id: str
    uploads: list[str]
    permit: dict
    required: list[str]
    missing: list[str]
    fee_expected: int
    fee_paid: int
    risk: str
    risk_reason: str
    decision: dict
    letter: str
    log: Annotated[list[str], operator.add]


def required_documents(p: dict) -> list[str]:
    """POL-01, with its conditional documents."""
    d = p["details"]
    docs = {"BLD": ["title deed", "stamped engineering drawings", "site plan", "contractor contract"],
            "REN": ["title deed", "description of works", "photos of the current state"],
            "SGN": ["commercial registration", "shop lease or title deed", "sign design with dimensions"],
            "FTR": ["vehicle registration", "health certificates for every worker"],
            "EVT": ["site plan", "safety plan"]}[p["type"]]
    if p["type"] == "REN" and d.get("structural"):
        docs.append("structural engineer's report")
    if p["type"] == "SGN" and d.get("illuminated"):
        docs.append("electrical safety certificate")
    if p["type"] == "EVT" and d.get("attendees", 0) > 500:
        docs.append("Civil Defense approval")
    return docs


def intake(s: Review) -> Review:
    CALLS["intake"] += 1
    row = m.sql("SELECT permit_id, type, status, district, details, submitted FROM permits WHERE permit_id = ?",
                (s["permit_id"],))[0]
    permit = {**row, "details": json.loads(row["details"])}
    return {"permit": permit, "required": required_documents(permit), "log": [f"intake {s['permit_id']}"]}


# %% [markdown]
# ✏️ **Exercise** · Write `fee_check`: compute the expected fee with `madar.fee_for` from the permit's details (the
# keyword names match: `area_m2`, `structural`, `sign_m2`, `illuminated`, `event_days`, `social_security`), read the
# amount paid as the sum of `payments` for the permit, and return `fee_expected`, `fee_paid` and a log line.

# %%
FEE_ARGS = ("area_m2", "structural", "sign_m2", "illuminated", "event_days", "social_security")


def fee_check(s: Review) -> Review:
    CALLS["fee_check"] += 1
    # BEGIN SOLUTION
    p = s["permit"]
    expected = madar.fee_for(p["type"], **{k: v for k, v in p["details"].items() if k in FEE_ARGS})
    paid = m.sql("SELECT coalesce(sum(amount_sar), 0) AS paid FROM payments WHERE permit_id = ?", (p["permit_id"],))[0]["paid"]
    return {"fee_expected": expected, "fee_paid": paid, "log": [f"fee expected {expected}, paid {paid}"]}
    # END SOLUTION


p0 = intake({"permit_id": "SGN-40764"})
f0 = fee_check({"permit": p0["permit"]})
print(p0["required"], f0)
K.check(f0["fee_expected"] == 2250 and f0["fee_paid"] == 1800, "SGN-40764 is underpaid by 450 SAR")

# %% [markdown]
# ## 2 · Model nodes and the router (6')
#
# Two steps need judgement, so they call a model through `K.structured` (a Pydantic schema + `response_format`, with
# one repair: Day 1.2's pattern):
# - **document_check**: do the uploaded file names cover the required documents? (Names are messy and bilingual.)
# - **assess_risk**: a Decisions-style typed choice, `low` or `high` with a reason. Hard rules run **first**; the model
#   only judges what the rules don't settle.

# %%
class DocCheck(BaseModel):
    missing: list[str] = Field(description="required documents with no matching upload, copied exactly from the list")
    notes: str = Field(description="one short sentence")


class Risk(BaseModel):
    risk: Literal["low", "high"]
    reason: str = Field(description="one short sentence")


def document_check(s: Review) -> Review:
    CALLS["document_check"] += 1
    out = K.structured(f"Required documents: {s['required']}\nUploaded files: {s['uploads']}\n"
                       "Which required documents have no matching upload? File names may be Arabic or English.",
                       DocCheck, system="You check permit applications for Madar Municipal Services.")
    missing = [d for d in out.missing if d in s["required"]]  # never trust a model to stay inside the list
    return {"missing": missing, "log": [f"documents: {len(missing)} missing"]}


def assess_risk(s: Review) -> Review:
    CALLS["assess_risk"] += 1
    d = s["permit"]["details"]
    rules = [(d.get("structural"), "structural works"), (d.get("floors", 0) >= 3, "three or more floors"),
             (d.get("sign_m2", 0) > 12, "sign over 12 m² (Urban Design Committee)"),
             (s["fee_paid"] != s["fee_expected"], "fee mismatch")]
    hits = [why for hit, why in rules if hit]
    if hits:
        return {"risk": "high", "risk_reason": "; ".join(hits), "log": [f"risk high by rule: {hits}"]}
    out = K.structured(f"Permit {s['permit']['type']} in {s['permit']['district']}: {json.dumps(d, ensure_ascii=False)}\n"
                       "Is this application low or high risk for public safety and neighbours?", Risk,
                       system="You triage permit applications for Madar Municipal Services. Be conservative.")
    return {"risk": out.risk, "risk_reason": out.reason, "log": [f"risk {out.risk} by model"]}


# %% [markdown]
# ✏️ **Exercise** · Write the router `route_after_checks(s)`: return `"request_info"` when documents are missing,
# otherwise `"officer_review"`. Every application with all its documents goes to an officer, whatever the risk:
# only a licensing officer approves or rejects (POL-01). Risk decides **how** it is presented, not **whether**.

# %%
def route_after_checks(s: Review) -> Literal["request_info", "officer_review"]:
    # BEGIN SOLUTION
    return "request_info" if s.get("missing") else "officer_review"
    # END SOLUTION


K.check(route_after_checks({"missing": ["site plan"]}) == "request_info"
        and route_after_checks({"missing": [], "risk": "low"}) == "officer_review", "router")

# %% [markdown]
# ## 3 · The human gate: interrupt and resume (7')
#
# `interrupt(payload)` **pauses** the graph inside a node. The state is saved by the checkpointer, the caller gets the
# payload under `__interrupt__`, and the run can wait an hour or a week. `graph.invoke(Command(resume=value), config)`
# continues the **same node**, where `interrupt(...)` now returns `value`. The `thread_id` in the config names the run.
#
# ✏️ **Exercise** · Write `officer_review`: interrupt with a summary for the officer (permit, fee expected and paid,
# risk and reason), then validate the answer (`outcome` must be `approve`, `reject` or `needs_info`; `officer` is
# required) and return it as `decision`.

# %%
def officer_review(s: Review) -> Review:
    CALLS["officer_review"] += 1
    # BEGIN SOLUTION
    answer = interrupt({"permit_id": s["permit_id"], "type": s["permit"]["type"], "details": s["permit"]["details"],
                        "fee_expected": s["fee_expected"], "fee_paid": s["fee_paid"],
                        "risk": s["risk"], "risk_reason": s["risk_reason"]})
    if answer.get("outcome") not in ("approve", "reject", "needs_info") or not answer.get("officer"):
        raise ValueError(f"Invalid officer decision: {answer}")
    return {"decision": answer, "log": [f"officer {answer['officer']}: {answer['outcome']}"]}
    # END SOLUTION


def request_info(s: Review) -> Review:
    CALLS["request_info"] += 1
    return {"decision": {"outcome": "needs_info", "officer": "review-graph", "note": f"Missing: {', '.join(s['missing'])}"},
            "log": ["request info (POL-01: incomplete applications move to needs_info)"]}


FAIL_ONCE: set[str] = set()  # Task 4 simulates a crash here


def draft_letter(s: Review) -> Review:
    CALLS["draft_letter"] += 1
    if s["permit_id"] in FAIL_ONCE:
        FAIL_ONCE.discard(s["permit_id"])
        raise RuntimeError("model provider timed out (simulated)")
    d = s["decision"]
    resp = K.chat([{"role": "system", "content": "You write short, formal letters for Madar Municipal Services, in Arabic "
                                                 "then English. State the outcome and the next step. No invented facts."},
                   {"role": "user", "content": f"Permit {s['permit_id']} ({s['permit']['type']}). Outcome: {d['outcome']}. "
                                               f"Officer note: {d.get('note', '')}. Fee expected {s['fee_expected']} SAR, "
                                               f"paid {s['fee_paid']} SAR."}], max_tokens=500)
    return {"letter": K.text_of(resp), "log": ["letter drafted"]}


def record(s: Review) -> Review:
    CALLS["record"] += 1
    status = {"approve": "approved", "reject": "rejected", "needs_info": "needs_info"}[s["decision"]["outcome"]]
    m.sql("UPDATE permits SET status = ?, reason = ?, needs_info_since = CASE WHEN ? = 'needs_info' THEN ? END "
          "WHERE permit_id = ?", (status, s["decision"].get("note", ""), status, madar.TODAY.isoformat(), s["permit_id"]))
    m._audit("review_decision", {"permit_id": s["permit_id"], **s["decision"]})
    return {"log": [f"recorded {status}"]}


def build(checkpointer):
    g = StateGraph(Review)
    for fn in (intake, document_check, fee_check, assess_risk, request_info, officer_review, draft_letter, record):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "document_check")
    g.add_edge("document_check", "fee_check")
    g.add_edge("fee_check", "assess_risk")
    g.add_conditional_edges("assess_risk", route_after_checks, ["request_info", "officer_review"])
    g.add_edge("request_info", "draft_letter")
    g.add_edge("officer_review", "draft_letter")
    g.add_edge("draft_letter", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer)


DB = WORK / "checkpoints.sqlite"
graph = build(SqliteSaver(sqlite3.connect(DB, check_same_thread=False)))
print(graph.get_graph().draw_mermaid())  # paste into mermaid.live to see it

# %% [markdown]
# Run all four applications. The complete ones stop at the officer's gate; the incomplete one goes straight to a
# request for information.

# %%
paused = {}
for pid, files in UPLOADS.items():
    cfg = {"configurable": {"thread_id": f"review-{pid}"}}
    out = graph.invoke({"permit_id": pid, "uploads": files, "log": []}, cfg)
    if "__interrupt__" in out:
        paused[pid] = out["__interrupt__"][0].value
        print(f"⏸  {pid}: waiting for an officer · risk {paused[pid]['risk']} ({paused[pid]['risk_reason']})")
    else:
        print(f"✉️  {pid}: {out['decision']['outcome']} · {out['log']}")
K.check("REN-10482" not in paused, "REN-10482 (no structural report) asks for information instead of a decision")
K.check({"SGN-40764", "BLD-20558"} <= set(paused), "high-risk applications wait for an officer")

# %% [markdown]
# The officers decide (in a real service: a review screen that lists paused threads). Resume each thread with its
# decision.

# %%
DECISIONS = {
    "REN-10275": {"outcome": "approve", "officer": "officer.salem", "note": "Non-structural works; complete file."},
    "SGN-40764": {"outcome": "needs_info", "officer": "officer.reem", "note": "Pay the remaining 450 SAR (illuminated surcharge)."},
    "BLD-20558": {"outcome": "approve", "officer": "eng.majed", "note": "Drawings checked; foundation inspection required."},
}
for pid, decision in DECISIONS.items():
    if pid in paused:
        out = graph.invoke(Command(resume=decision), {"configurable": {"thread_id": f"review-{pid}"}})
        print(f"{pid}: {out['log'][-1]}\n{out['letter'][:300]}\n")
print(m.sql("SELECT permit_id, status, reason FROM permits WHERE permit_id IN ('REN-10275','REN-10482','SGN-40764','BLD-20558')"))
K.check(m.sql("SELECT status FROM permits WHERE permit_id = 'BLD-20558'")[0]["status"] == "approved", "BLD-20558 approved by an officer")

# %% [markdown]
# ## 4 · Crash, restart, resume (6')
#
# Every step is checkpointed. If the process dies (deploy, timeout, laptop lid), a **new** process with the same
# checkpoint database can continue the thread from the last completed step, without redoing the model calls before it.
#
# Below, the letter step fails once for `SGN-40010`. ✏️ **Exercise** · After the crash, build a **new** graph object
# with a **new** SQLite connection (a fresh process), check `get_state(cfg).next` (where it stopped), and resume with
# `invoke(None, cfg)`. Then show from `CALLS` that `document_check` did not run again.

# %%
PID = "SGN-40010"
cfg = {"configurable": {"thread_id": f"review-{PID}"}}
FAIL_ONCE.add(PID)
graph.invoke({"permit_id": PID, "uploads": ["CR.pdf", "title_deed.pdf", "sign_design_7.8m2.pdf"], "log": []}, cfg)
try:
    graph.invoke(Command(resume={"outcome": "approve", "officer": "officer.reem", "note": "Complete."}), cfg)
except RuntimeError as e:
    print("💥", e)
before = CALLS.copy()

# BEGIN SOLUTION
graph2 = build(SqliteSaver(sqlite3.connect(DB, check_same_thread=False)))
print("stopped before:", graph2.get_state(cfg).next)
out = graph2.invoke(None, cfg)
# END SOLUTION
print(out["log"])
K.check(out.get("letter") and CALLS["document_check"] == before["document_check"],
        "resumed at the letter without re-running the earlier model calls")

# %% [markdown]
# The checkpoints are also an audit trail: every step of the thread, with the state after it.

# %%
for snap in graph2.get_state_history(cfg):
    print(snap.metadata.get("step"), snap.next, (snap.values.get("log") or [""])[-1])

# %% [markdown]
# ## ✱ Stretch · Run the two checks in parallel
#
# `document_check` and `fee_check` don't depend on each other. Build a variant where both start from `intake` and
# `assess_risk` waits for both (two edges into it). Run it on `REN-10275` and compare `get_state_history`: the two
# checks happen in the **same** step. Why must their output keys not overlap (or use a reducer)?

# %%
# BEGIN SOLUTION
def build_parallel(checkpointer):
    g = StateGraph(Review)
    for fn in (intake, document_check, fee_check, assess_risk, request_info, officer_review, draft_letter, record):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "document_check")
    g.add_edge("intake", "fee_check")
    g.add_edge(["document_check", "fee_check"], "assess_risk")
    g.add_conditional_edges("assess_risk", route_after_checks, ["request_info", "officer_review"])
    g.add_edge("request_info", "draft_letter")
    g.add_edge("officer_review", "draft_letter")
    g.add_edge("draft_letter", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer)


par = build_parallel(SqliteSaver(sqlite3.connect(WORK / "parallel.sqlite", check_same_thread=False)))
pcfg = {"configurable": {"thread_id": "parallel-REN-10275"}}
par.invoke({"permit_id": "REN-10275", "uploads": UPLOADS["REN-10275"], "log": []}, pcfg)
for snap in par.get_state_history(pcfg):
    print(snap.metadata.get("step"), snap.next)
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Which nodes are code and which call a model? Why is `required_documents` not a prompt?
# 2. The router never sends a complete application straight to `record`. Which policy requires that?
# 3. What exactly is stored when the graph pauses at `officer_review`, and what would you add to show an officer a
#    queue of paused reviews?
# 4. Day 3's Ralph loop also survived restarts. What does a graph give you that a loop with a progress file doesn't?
