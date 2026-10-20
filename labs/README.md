# Agentic AI Systems Engineering: labs

Hands-on notebooks for the SDAIA Academy **Agentic AI Systems Engineering (هندسة أنظمة الذكاء الاصطناعي التوكيلي)** course (AI Engineering track). Every lab builds a piece of **Rafeeq (رفيق)**, the digital assistant of **Madar Municipal Services**, a fictional municipality that issues building, renovation, signage, food-truck and event permits. The labs start from a single API call and end with an evaluated, secured agent service.

All labs use **free models over an API**. Nothing runs a model on your machine, and a laptop CPU or a Colab CPU runtime is enough.

## Quick start

**1 · Get a free OpenRouter key.** Create one at <https://openrouter.ai/keys>. Free models (IDs ending in `:free`) allow 50 requests a day and 20 a minute. A one-time purchase of 10 credits raises the limit to 1,000 free-model requests a day; the credits are not spent by free models. Identical requests are replayed from `labs/cache/`, so re-running a cell costs nothing.

**2a · Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/agentic-ai/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/<notebook>.ipynb`.
2. In the 🔑 **Secrets** panel add `OPENROUTER_API_KEY` and turn on notebook access.
3. Run the first cell. It clones this repository, installs `requirements-colab.txt` and generates the data.

**2b · Local**
```bash
git clone https://github.com/SDAIAAcademy/agentic-ai
cd agentic-ai/labs
cp .env.example .env           # then paste your key into .env
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version)
uv run python prepare_data.py  # generates data/madar.sqlite and data/inbox.jsonl (a second)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1. It checks your key against all three API shapes, downloads the Day 2 embedding model and checks `uv` and Node.js.

**Standalone programs** (the MCP server, the loop runner, the capstone starter) are single-file [PEP 723](https://peps.python.org/pep-0723/) scripts that declare their own dependencies. Run them with `uv run <file>.py`; no `pip install` needed.

**Lab 3.1** runs [Pi](https://github.com/earendil-works/pi), an open-source coding-agent harness, which needs Node.js 22.19+. Without Node, use the recorded trace in the lab.

## Models and providers

| Setting (`.env`) | Default | Notes |
|---|---|---|
| `AGENT_PROVIDER` | `openrouter` | `openrouter`, `hf` (Hugging Face Inference Providers, needs credits) or `custom` |
| `AGENT_MODEL` | `google/gemma-4-31b-it:free` | any free tool-calling model from <https://openrouter.ai/models?max_price=0&supported_parameters=tools> |
| `AGENT_SMALL_MODEL` | `google/gemma-4-26b-a4b-it:free` | cheaper steps: triage, judges, summaries |
| `AGENT_FALLBACKS` | `google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3-super-120b-a12b:free` | tried in order when a free model is rate-limited |
| `AGENT_CACHE` | `1` | `0` sends every request to the model |
| `AGENT_QUICK` | `0` | `1` shrinks eval runs and samples to save free requests |

Free models change often. If the default disappears or is busy, pick another free model with tool support and check it on [Artificial Analysis](https://artificialanalysis.ai/models), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) or [τ²-bench](https://taubench.com) before relying on it.

**Any other engine** that serves the OpenAI and/or Anthropic API shapes works with `AGENT_PROVIDER=custom`: vLLM, SGLang, llama.cpp, Ollama, LM Studio or LiteLLM. Set `LLM_BASE_URL` (OpenAI shape, ending in `/v1`), `LLM_ANTHROPIC_BASE_URL` (Messages shape), `LLM_API_KEY` and `LLM_MODEL`. A small Qwen or Gemma model is enough for local experiments; the labs are tuned for the free API models.

## Notebooks

| Day | Session | Notebook | API shapes |
|---|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/00_setup_check.ipynb) | all three |
| 1 | Two providers, one client | [`D1_1_two_providers.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D1_1_two_providers.ipynb) | Chat Completions |
| 1 | Triage with the Responses API | [`D1_2_responses_api.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D1_2_responses_api.ipynb) | Responses |
| 1 | Rafeeq's first tools, three ways | [`D1_3_messages_and_tools.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D1_3_messages_and_tools.ipynb) | Messages |
| 1 | The agent loop from scratch | [`D1_4_agent_loop.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D1_4_agent_loop.ipynb) | all three |
| 2 | Budget the window | [`D2_1_context_budget.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D2_1_context_budget.ipynb) | Chat Completions |
| 2 | Search the policies | [`D2_2_retrieval.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D2_2_retrieval.ipynb) | Chat Completions |
| 2 | Remember and specialise | [`D2_3_memory_skills.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D2_3_memory_skills.ipynb) | Chat Completions |
| 2 | Madar as an MCP server | [`D2_4_mcp.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D2_4_mcp.ipynb) | MCP (stdio, Streamable HTTP) |
| 3 | Harness teardown | [`D3_1_harness_teardown.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D3_1_harness_teardown.ipynb) | Pi (optional, Node.js 22.19+) |
| 3 | Rafeeq on the OpenAI Agents SDK | [`D3_2_agents_sdk.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D3_2_agents_sdk.ipynb) | Chat Completions via the SDK |
| 3 | A loop that fixes code | [`D3_3_loop_engineering.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D3_3_loop_engineering.ipynb) | Chat Completions + git + pytest |
| 3 | Score Rafeeq | [`D3_4_evals.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D3_4_evals.ipynb) | Chat Completions |
| 4 | A permit-review graph | [`D4_1_review_graph.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D4_1_review_graph.ipynb) | LangGraph + SQLite checkpoints |
| 4 | Orchestrator and workers | [`D4_2_multi_agent.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D4_2_multi_agent.ipynb) | Agents SDK + A2A |
| 4 | Attack and defend | [`D4_3_security.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D4_3_security.ipynb) | Chat Completions + harness hooks |
| 4 | Ship it | [`D4_4_ship_it.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/agentic-ai/blob/main/labs/D4_4_ship_it.ipynb) | FastAPI + OpenTelemetry |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in. Complete reference versions are in `solutions/`. Cells marked **✱ Stretch** are optional. Self-checks print ✅ or ❌. With free models, an occasional ❌ on an answer-quality check can be the model rather than your code: re-run, or compare with the solution.

Labs don't depend on each other. Each one opens a fresh copy of the Madar database in `artifacts/<lab>/`, so write tools never change the shared data.

## The case study: Madar Municipal Services

`prepare_data.py` generates everything in `data/`. It is synthetic and deterministic, dated around **1 November 2026** (the labs' "today"), and contains no real personal data.

| Data | File | Contents |
|---|---|---|
| Permit database (SQLite) | `madar.sqlite` | 249 applicants, 424 permits, payments, inspections, 540 inspection slots, refunds, escalations, an outbox and an audit log |
| Shared inbox | `inbox.jsonl` | 30 messages in Arabic and English with gold triage labels; two contain prompt injections |
| Policies | `policies/POL-01 … POL-10` | bilingual Markdown: permit types, fees, SLAs and working days, lifecycle, inspections, appeals and refunds, violations, privacy, food-truck zones, assistant scope |
| Eval suite | `eval/tasks.jsonl` | 30 tasks with graders (string checks, tool-call checks, database checks, LLM judge) |
| Skills and context file | `skills/*/SKILL.md`, `AGENTS.md` | four Agent Skills (refund, inspection booking, escalation, appeal) and Rafeeq's AGENTS.md, for Lab 2.3 |
| Loop-engineering repo | `madar_fees/` | a small Python package with seeded bugs and failing tests, for the Ralph loop in Lab 3.3 |

`madar.py` exposes the case study to agents: `Madar(workdir)` is a database copy with typed, documented **tools** (read: `search_policies`, `get_permit`, `compute_fee`, `list_slots`; write: `book_inspection`, `reschedule_inspection`, `create_refund`, `escalate`, `send_email`). Tools enforce Madar's rules themselves (identity verification, the 2-working-day booking rule, refund approval limits) and raise `ToolError` messages the model can act on.

Nine applicants have known stories for the labs and the eval suite:

| Applicant | Permit | Story | Verify with |
|---|---|---|---|
| Sara | REN-10482 | under review, decision due 2 Nov | 4821 · 0501234821 |
| Fahad | BLD-20417 | 640 m² building permit, 9 working days overdue | 7730 · 0555017730 |
| Noura | FTR-30311 | approved food truck, needs a health inspection in Al-Rawdah | 6654 · 0532236654 |
| Omar | SGN-40220 | rejected 14 m² sign; appeal deadline 7 Nov; 2,700 SAR refund pending approval | 0093 · 0540410093 |
| Layla | EVT-50077 | needs Civil Defense approval by 3 Nov | 8012 · 0568988012 |
| Khalid | SGN-40388 | paid twice | 2277 · 0505502277 |
| Maha | REN-10519 | social-security beneficiary, no fee | — |
| Huda | BLD-20981 | wants to move inspection INS-9001 to 5 Nov | 1196 · 0548841196 |
| Yousef | (agent for Fahad) | authorised agent for BLD-20417 | 5530 · +966571475530 |

## Standalone programs

| Script | What it is | Run |
|---|---|---|
| `servers/madar_mcp.py` | Madar's tools, policies (resources) and triage prompt as an MCP server | `uv run servers/madar_mcp.py` (stdio) or `--http 8765` |
| `loops/ralph.py` | A Ralph loop: a coding agent fixes `madar_fees` until its tests pass, one commit per iteration | `uv run loops/ralph.py --tests tests/test_fees.py` (`--harness pi` for Pi) |
| `servers/madar_a2a.py` | Madar's policy worker as an A2A agent (agent card + JSON-RPC) | `uv run servers/madar_a2a.py --port 8792` |
| `servers/rafeeq_api.py` | Rafeeq as an HTTP service: API keys, sessions, budgets, fallbacks, write tools queued for an officer, OpenTelemetry traces | `RAFEEQ_API_KEYS=dev-key-1 uv run servers/rafeeq_api.py` |
| `ci/smoke_eval.py` | A smoke eval with a pass-rate gate for CI (`.github/workflows/smoke-eval.yml`) | `uv run ci/smoke_eval.py --min-rate 0.6` |
| `capstone/starter.py` | Capstone starter: local + MCP tools, spec, compaction, permissions, budgets, eval on two configurations (see `capstone/README.md`) | `uv run capstone/starter.py eval capstone/tasks.example.jsonl` |

## Helpers: `agentkit.py`

| Helper | What it does |
|---|---|
| `client()`, `anthropic_client()` | OpenAI and Anthropic SDK clients pointed at the configured provider |
| `chat(messages, tools=...)` | one cached, retried Chat Completions call with free-model fallbacks |
| `tool_schema(fn, shape)` | a tool definition from a typed, documented Python function, for `chat`, `responses`, `anthropic` or `mcp` |
| `call_tool(tools, name, args)` | runs a tool call; returns JSON, or a JSON error the model can recover from |
| `run_agent(task, tools, shape=...)` | the course's reference agent loop over all three API shapes, with budgets, permissions and hooks |
| `count_tokens(obj)` | token estimate of a string or a request part, with an open tokenizer from the Hub |
| `MCPBridge(server)` | an MCP client usable from synchronous code; `agent_tools()` turns MCP tools into loop tools |
| `structured(prompt, Schema)` | one call whose answer validates as a Pydantic model (`response_format` + one repair) |
| `serve(cmd, ready_url)` / `stop(proc)` | start a background server (e.g. `uv run servers/...`) and stop it with its child processes |
| `LEDGER` | calls, tokens, cached tokens and cost so far |

`evals.py` runs the eval suite: `load_tasks()`, `run_suite(agent, tasks, trials=k)`, `report(df)` (pass@k and pass^k), with code graders and an LLM judge.
