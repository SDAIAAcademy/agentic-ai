# %% [markdown]
# # Lab 4.4 · Ship it
#
# Day 4 · Session 4 (Agents in production) · 25 min. A notebook agent becomes a service when other programs can call
# it safely, its cost is bounded, a failing provider doesn't take it down, someone can see what it did, and a
# regression stops a release. In this lab you put Rafeeq behind **FastAPI** and add, one at a time:
# - an **API key**, **sessions** and a **per-session token budget**;
# - **least privilege** for write tools (queued for an officer, never run by the service);
# - **fallbacks** across models;
# - **OpenTelemetry traces** with the GenAI conventions;
# - a **smoke eval** that gates CI.
#
# You build it in-process (FastAPI's `TestClient`, no server needed); `servers/rafeeq_api.py` is the finished
# service, which you start for real at the end.
#
# **Tasks**
# - [ ] 1 · The endpoint: key, session, budget (8')
# - [ ] 2 · Fallbacks (4')
# - [ ] 3 · Traces (6')
# - [ ] 4 · A smoke eval that gates CI (5')
# - [ ] 5 · Run the real service, plan your capstone (2')

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
import functools
import json

import httpx
import openai
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import evals
import madar

WORK = K.lab_dir("D4_4", reset=True)
m = madar.Madar(WORK, actor="rafeeq-api")
KEYS = {"key-portal", "key-callcenter"}

# %% [markdown]
# ## 1 · The endpoint: key, session, budget (8')
#
# The request and response are Pydantic models, so FastAPI validates input (no 50,000-character messages, no odd
# session IDs) and documents the API at `/docs`.
#
# ✏️ **Exercise** · Complete `make_app(budget)`:
# - `api_key`: a dependency that raises `HTTPException(401)` unless the `X-API-Key` header is in `KEYS`;
# - `chat`: load the session's history and spent tokens from `SESSIONS` (key: `f"{key}:{session_id}"`, so one client
#   can't read another's session); raise `HTTPException(429)` when `spent >= budget`; run Rafeeq with `history=`;
#   save `run.state[1:]` (everything but the system prompt) and the new token total. Use the `permission` given: write
#   tools are **denied** and recorded in `pending` for an officer.

# %%
class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[\w.-]+$")
    message: str = Field(min_length=1, max_length=4000)


class ChatOut(BaseModel):
    session_id: str
    reply: str
    tools: list[str]
    pending_actions: list[dict]
    tokens: int


SESSIONS: dict[str, dict] = {}  # the reference service uses SQLite; a dict is enough to learn the shape
TOOLS = m.tools()


def run_rafeeq(message: str, history: list, permission, **kw) -> K.Run:
    return K.run_agent(message, TOOLS, system=madar.SYSTEM_PROMPT, history=history, max_steps=6,
                       permission=permission, verbose=False, **kw)


def make_app(budget: int = 40_000, runner=run_rafeeq) -> FastAPI:
    app = FastAPI(title="Rafeeq API (lab)")

    # BEGIN SOLUTION
    def api_key(x_api_key: str = Header(default="")) -> str:
        if x_api_key not in KEYS:
            raise HTTPException(401, "Missing or invalid X-API-Key")
        return x_api_key
    # END SOLUTION

    @app.post("/v1/chat", response_model=ChatOut)
    def chat(req: ChatIn, key: str = Depends(api_key)) -> ChatOut:
        pending: list[dict] = []

        def permission(name: str, args: dict) -> str:
            if m.is_write(name):
                pending.append({"tool": name, "args": args})
                return "deny"
            return "allow"

        # BEGIN SOLUTION
        sid = f"{key}:{req.session_id}"
        session = SESSIONS.get(sid, {"history": [], "tokens": 0})
        if session["tokens"] >= budget:
            raise HTTPException(429, f"Session budget of {budget} tokens used up")
        before = K.LEDGER.input_tokens + K.LEDGER.output_tokens
        run = runner(req.message, session["history"], permission)
        used = K.LEDGER.input_tokens + K.LEDGER.output_tokens - before
        SESSIONS[sid] = {"history": run.state[1:], "tokens": session["tokens"] + used}
        # END SOLUTION
        reply = run.final or "I couldn't finish this request. An officer will follow up."
        if pending:
            reply += "\n\n(An officer will review and confirm: " + ", ".join(p["tool"] for p in pending) + ".)"
        return ChatOut(session_id=req.session_id, reply=reply, tools=[n for n, _ in run.tool_calls],
                       pending_actions=pending, tokens=used)

    return app


# %% [markdown]
# Contract tests: the API behaves as promised whatever the model says. These run in CI before any model is called.

# %%
api = TestClient(make_app())
H = {"X-API-Key": "key-portal"}
K.check(api.post("/v1/chat", json={"session_id": "s1", "message": "hi"}).status_code == 401, "no key → 401")
K.check(api.post("/v1/chat", headers=H, json={"session_id": "bad id!", "message": "hi"}).status_code == 422,
        "invalid session id → 422")

r1 = api.post("/v1/chat", headers=H, json={"session_id": "sara", "message":
              "What is the status of REN-10482? ID ends 4821, mobile 0501234821."}).json()
r2 = api.post("/v1/chat", headers=H, json={"session_id": "sara", "message": "And when is the decision due?"}).json()
print(r1["reply"][:300], "\n---\n", r2["reply"][:300])
K.check(len(SESSIONS["key-portal:sara"]["history"]) > 2, "the session carries the conversation")
K.check("key-callcenter:sara" not in SESSIONS, "sessions are scoped to the API key")

# %%
tight = TestClient(make_app(budget=1))
tight.post("/v1/chat", headers=H, json={"session_id": "b", "message": "What are your office hours?"})
K.check(tight.post("/v1/chat", headers=H, json={"session_id": "b", "message": "Thanks"}).status_code == 429,
        "budget exhausted → 429")

r3 = api.post("/v1/chat", headers=H, json={"session_id": "huda", "message":
              "Move my BLD-20981 foundation inspection to Thursday 5 November before noon. Go ahead. "
              "ID ends 1196, mobile 0548841196."}).json()
print(r3["reply"][:400], r3["pending_actions"])
K.check(not m.sql("SELECT 1 FROM inspections WHERE permit_id = 'BLD-20981' AND slot LIKE '2026-11-05%'"),
        "the service never executed a write tool itself")

# %% [markdown]
# ## 2 · Fallbacks (4')
#
# Free models get rate-limited and disappear. Two layers:
# - **gateway routing**: `K.chat` already sends OpenRouter a `models` list (`K.FALLBACKS`); OpenRouter tries them in
#   order on errors and rate limits;
# - **application fallback**: if the call still fails, retry the run on another model (or provider) and record it.
#
# ✏️ **Exercise** · Write `with_fallbacks(models)`, returning a `runner` for `make_app` that tries `run_rafeeq(...,
# model=model, fallbacks=False)` for each model in turn, catching `openai.APIError`, and records which model answered
# in `ANSWERED_BY`. Test it with a model that doesn't exist first.

# %%
ANSWERED_BY: list[str] = []


def with_fallbacks(models: list[str]):
    # BEGIN SOLUTION
    def runner(message: str, history: list, permission) -> K.Run:
        last = None
        for model in models:
            try:
                run = run_rafeeq(message, history, permission, model=model, fallbacks=False)
                ANSWERED_BY.append(model)
                return run
            except openai.APIError as e:
                last = e
        raise last
    return runner
    # END SOLUTION


fb = TestClient(make_app(runner=with_fallbacks(["madar/no-such-model:free", K.MODEL])))
out = fb.post("/v1/chat", headers=H, json={"session_id": "fb", "message": "What are your office hours?"})
print(out.status_code, ANSWERED_BY)
K.check(out.status_code == 200 and ANSWERED_BY[-1] == K.MODEL, "answered by the second model")

# %% [markdown]
# ## 3 · Traces (6')
#
# A trace is a tree of **spans**: one `invoke_agent` span per request, with a `chat` span per model call and an
# `execute_tool` span per tool call. The **OpenTelemetry GenAI semantic conventions** name the attributes
# (`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, ...), so any
# backend (Jaeger, Grafana Tempo, Phoenix, Langfuse, Datadog) can read them. Here spans go to memory so we can look
# at them; the reference service writes JSON lines, and production would use an OTLP exporter.
#
# `TracedShape` (from the reference service) opens a `chat` span around every model call. ✏️ **Exercise** · Write
# `traced(fn)`: a wrapper that runs the tool inside an `execute_tool <name>` span with `gen_ai.operation.name` and
# `gen_ai.tool.name` set. Keep the function's name, signature and docstring (`functools.wraps`): the tool schema is
# built from them.

# %%
sys.path.insert(0, str(K.LABS / "servers"))
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rafeeq_api import TracedShape

SPANS = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SPANS))
tracer = provider.get_tracer("rafeeq-lab")


# BEGIN SOLUTION
def traced(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with tracer.start_as_current_span(f"execute_tool {fn.__name__}") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", fn.__name__)
            return fn(*a, **kw)
    return wrapper
# END SOLUTION


def traced_runner(message: str, history: list, permission) -> K.Run:
    with tracer.start_as_current_span("invoke_agent Rafeeq") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "Rafeeq")
        return K.run_agent(message, [traced(t) for t in TOOLS], system=madar.SYSTEM_PROMPT, history=history,
                           max_steps=6, permission=permission, verbose=False, shape=TracedShape(tracer))


SPANS.clear()
TestClient(make_app(runner=traced_runner)).post("/v1/chat", headers=H, json={"session_id": "t", "message":
    "What is the fee for an illuminated 3 m² sign, and is REN-10482 decided yet? ID ends 4821, mobile 0501234821."})
spans = sorted(SPANS.get_finished_spans(), key=lambda s: s.start_time)
for s in spans:
    depth = 0 if s.parent is None else 1
    attrs = {k.removeprefix("gen_ai."): v for k, v in (s.attributes or {}).items() if "usage" in k or "tool.name" in k}
    print("  " * depth + f"{s.name:<40} {(s.end_time - s.start_time) / 1e6:7.0f} ms  {attrs}")
K.check(any(s.name.startswith("execute_tool") for s in spans) and any(s.name.startswith("chat") for s in spans),
        "model and tool spans inside the agent span")

# %% [markdown]
# ✏️ **Exercise** · From `spans`, compute `summary`: model time vs. tool time (ms) and total input/output tokens. Where
# does a request spend its time?

# %%
# BEGIN SOLUTION
ms = lambda s: (s.end_time - s.start_time) / 1e6  # noqa: E731
summary = {
    "model_ms": round(sum(ms(s) for s in spans if s.name.startswith("chat"))),
    "tool_ms": round(sum(ms(s) for s in spans if s.name.startswith("execute_tool"))),
    "input_tokens": sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in spans),
    "output_tokens": sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in spans),
}
# END SOLUTION
summary

# %% [markdown]
# ## 4 · A smoke eval that gates CI (5')
#
# Contract tests check the API; a **smoke eval** checks the agent. A handful of fast, important tasks from Day 3's
# suite run on every pull request; the pipeline fails below a pass rate. `ci/smoke_eval.py` (PEP 723) does exactly
# this, and `.github/workflows/smoke-eval.yml` runs it with the OpenRouter key from the repository's secrets.
#
# ✏️ **Exercise** · Write `gate(df, min_rate)` → `(ok, rate)` as the CI script does, and run the smoke tasks.

# %%
print((K.LABS / "ci" / "smoke_eval.py").read_text(encoding="utf-8").split('"""')[1])
SMOKE = ["rules-office-hours", "fee-sign-illuminated", "status-verified", "status-no-credentials",
         "injection-contractor-email"]


def gate(df: pd.DataFrame, min_rate: float) -> tuple[bool, float]:
    # BEGIN SOLUTION
    rate = float(df.passed.mean()) if len(df) else 0.0
    return rate >= min_rate, rate
    # END SOLUTION


smoke = evals.run_suite(evals.rafeeq(), evals.load_tasks(ids=SMOKE), name="smoke")
ok, rate = gate(smoke, 0.6)
print(f"pass rate {rate:.2f} → {'ship' if ok else 'block the release'}")
K.check(gate(pd.DataFrame({"passed": [True, True, False]}), 0.6) == (True, 2 / 3), "gate")

# %% [markdown]
# ## 5 · Run the real service, plan your capstone (2')
#
# The reference service adds SQLite sessions, a second-model retry, JSONL traces and `/healthz`. Start it with `uv`
# and call it over HTTP, exactly as a portal or a call-centre app would.

# %%
PORT = 8793
service = K.serve(["uv", "run", "-q", "servers/rafeeq_api.py", "--port", str(PORT)],
                  ready_url=f"http://127.0.0.1:{PORT}/healthz")
resp = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat", headers={"X-API-Key": "dev-key-1"}, timeout=180,
                  json={"session_id": "demo", "message": "ما هي ساعات عمل المكتب؟"})
print(resp.status_code, json.dumps(resp.json(), ensure_ascii=False, indent=1)[:600])
K.stop(service)
print((K.ARTIFACTS / "traces" / "rafeeq_api.jsonl").read_text(encoding="utf-8").splitlines()[-1][:300])

# %% [markdown]
# **Capstone.** Open `capstone/README.md`: pick a track, copy `capstone/TEMPLATE_README.md`, and write down tonight
# your agent's job, its tools, its trifecta, and 15 eval tasks with expected outcomes.
#
# ## Checkpoint
#
# 1. Which of today's controls are enforced by the **service** and which still depend on the **model**?
# 2. A citizen's session hits the budget mid-conversation. What should the portal show, and who should be alerted?
# 3. Which span attribute would you alert on first in production, and at what threshold?
# 4. Your smoke eval passes at 0.8 today. Tomorrow a free model is swapped and it drops to 0.6. What happens in your
#    pipeline, and what do you do next?
