# Agentic AI Systems Engineering: course outline

**هندسة أنظمة الذكاء الاصطناعي التوكيلي** · **SDAIA Academy · AI Engineering Track**: 4 teaching days × 4 sessions + 1 capstone day.
Each day runs Session 1 → 10' break → Session 2 → 20' break → Session 3 → 10' break → Session 4 (4 × 50 min).
Every session is roughly 20–25 minutes of concepts and demos followed by a 25-minute lab.

> **From a model call to a governed agent system.** Build agents from first principles in Python, then climb the
> engineering ladder: **prompt → context → harness → loop → graph**. Each layer wraps the one before it.
> Models come from **OpenRouter** and **Hugging Face**; the agent harnesses people use today (**Pi, OpenClaw, Hermes,
> Mercury**) are taken apart and compared with the one you build.

## Who it is for

Python developers, ML and data engineers who have called an LLM API before (Applied NLP or equivalent).
No prior agent framework, MCP or LangGraph experience is needed.

## Why this outline (research summary, October 2026)

| Signal | What it means for the course |
|---|---|
| **"Agent = model + harness."** Anthropic (*Effective harnesses for long-running agents*, Nov 2025), OpenAI (*Harness engineering*, Feb 2026) and Thoughtworks (Böckeler, *Harness engineering*, Apr 2026) all locate agent quality in the harness: tools, context files, permissions, sensors (tests, linters), progress files. LangChain moved Terminal-Bench 2.0 from 52.8 to 66.5 by changing only the harness. | Build a harness by hand (Day 1 loop → Day 3 harness) before using any framework, so every framework afterwards is "a harness someone else wrote". |
| **The engineering ladder.** Practitioners describe the stack as prompt → context → harness → loop → graph engineering (Pachaar, Jul 2026; Steinberger, Husain, Perez on loops vs. graphs; Osmani, *Loop engineering*, Jun 2026). The terms are new and contested, but each names a real layer with its own artifacts. | The course is organised as nested layers, not as a hype sequence: Day 1 prompt + tools, Day 2 context, Day 3 harness + loop, Day 4 graph + production. Slides show the canonical sources behind each layer. |
| **Context engineering** replaced prompt engineering as the core skill (Karpathy and Lütke, Jun 2025; Anthropic, Sep 2025; Manus; Breunig's four failure modes; LangChain's write/select/compress/isolate). | A full day: context budgets, compaction, just-in-time retrieval, memory, Agent Skills (SKILL.md, open standard Dec 2025) and MCP. |
| **Open harnesses went mainstream in 2026**: OpenClaw (~390k GitHub stars; gateway, channels, heartbeat), Hermes Agent by Nous Research (~250k; self-written skills, capped memory files), Pi by Mario Zechner (~117k; 4 tools, minimal prompt, the base OpenClaw was built on), Mercury Agent (token budgets and permission scopes as settings). Their security record (CVE-2026-25253, 341+ malicious ClawHub skills, exposed gateways) is the best agent-security case study available. | Day 3 opens with a harness teardown: participants run Pi against OpenRouter, read its event stream, and compare four harnesses on loop, tools, context files, memory, permissions and sandboxing. Day 4 security uses their incidents. |
| **Protocols consolidated under the Agentic AI Foundation** (Linux Foundation): MCP (spec 2026-07-28, stateless; Python SDK v2 with `MCPServer`), AGENTS.md, goose, and A2A (v1.0, joined Aug 2026). | MCP server and client in Day 2 with the current SDK (most tutorials still show v1); AGENTS.md and skills in Day 2–3; A2A in the multi-agent session. |
| **Graphs and durable execution** for production workflows: LangGraph 1.x leads framework downloads (≈45M/month); checkpoints, interrupts and human approval are table stakes. Anthropic's workflow patterns (chain, route, parallel, orchestrator-workers, evaluator-optimizer) remain the reference vocabulary. | Graph engineering session with LangGraph: typed state, checkpointing, interrupt for human approval, resume after a crash. |
| **Evals separate good builders from the rest** (Ng, *Agentic AI*, 2025; Anthropic, *Demystifying evals*; Berkeley's evaluator-first projects). | Eval-driven development from Day 3: a task suite with code graders and an LLM judge, pass^k, error analysis, traces; the capstone is scored on the same suite. |
| **Agent security** is now its own discipline: OWASP Top 10 for Agentic Applications (Dec 2025), the lethal trifecta (Willison), prompt injection through tools and documents. | A security session where participants attack their own agent with an injected email, then defend it (least privilege, approvals, quarantine of untrusted content, egress control), and red-team it with an eval. |
| **Providers**: OpenRouter routes 450+ models through one OpenAI-compatible API with provider routing, fallbacks, structured outputs and per-request cost. Hugging Face Inference Providers exposes 130+ open models through `router.huggingface.co/v1`; the Hub hosts the embedding models and datasets. Free HF accounts no longer include inference credits. | One `client()` helper switches between OpenRouter and HF with an env var. Labs run on **free** models only: Gemma 4 on OpenRouter (`google/gemma-4-31b-it:free`, with `google/gemma-4-26b-a4b-it:free` and `openrouter/free` as fallbacks), Qwen and Gemma on HF. Model choice is taught from public benchmarks (Artificial Analysis, LMArena, BFCL, τ²-bench, Terminal-Bench) plus our own eval, not from a fixed recommendation. A response cache keeps re-runs inside the free daily quota. Small Qwen and Gemma models from the Hub for optional local experiments; HF embedding models for retrieval. |
| **KSA context**: PDPL, SDAIA AI Ethics Principles and Generative AI Guidelines, SDAIA's report on agentic AI (Aug 2025), NCA's draft AI cybersecurity guidelines (2026). | The case study is a government service: identity verification before disclosure, human-only decisions, data minimisation, and cross-border/model-provider choices are part of the design. |

Sources are listed with links in the [README](README.md#sources).

## One case study, all week: **Madar Municipal Services**

Madar is a fictional municipality that issues five permit types (building, renovation, signage, food truck, event).
Participants build **Rafeeq**, its permit assistant, and the systems around it.

| Asset | Content | Used for |
|---|---|---|
| `policies/` | 10 bilingual (EN + AR) policy documents: permit types, fees, SLAs, lifecycle, inspections, appeals and refunds, violations, privacy and identity, food-truck zones, assistant scope | context, retrieval, grounding, evals |
| `madar.sqlite` | applicants, permits, inspections, inspection slots, payments, refund requests, escalations (synthetic) | tools with side effects, approvals |
| `inbox.jsonl` | 30 citizen and staff emails in Arabic and English, two with hidden prompt injections | loops, triage, security |
| `madar_fees/` | a small Python package (fees and SLA dates) with a failing test suite | coding agent, loop engineering |
| `eval/tasks.jsonl` | 30 tasks with graders: answers, database end states, refusals, required approvals | evals, capstone scoring |

All data is synthetic and generated deterministically by `labs/prepare_data.py`. No real person's data is involved.

## Day 1 · From Model to Agent (prompt engineering)

| # | Session | Concepts | Lab (25') |
|---|---|---|---|
| 1 | **The agentic landscape** | What an agent is (and isn't); workflows vs. agents; autonomy levels; the ladder prompt → context → harness → loop → graph; the model market in Oct 2026 and how to read benchmarks (Artificial Analysis, LMArena, BFCL, τ²-bench); OpenRouter and HF Inference Providers; free tiers and quotas; tokens, cost, latency | **1.1 Two providers, one client**: call free Gemma and Qwen models through OpenRouter and HF; streaming; fallbacks with a `models` list; read tokens and cost from `usage`; shortlist models from benchmark data and check them on your own task |
| 2 | **Prompting for agents** | System prompts as specifications; instruction hierarchy; structured outputs (JSON Schema, strict mode); validation and repair; reasoning models and effort; Arabic and bilingual prompts | **1.2 From email to structured request**: classify and extract Madar emails into a Pydantic schema; strict structured outputs vs. prompt-only JSON; validate, retry, measure accuracy on 20 labelled emails |
| 3 | **Tool calling** | Function schemas; the tool-call message protocol; parallel calls; designing tools for agents (fewer, higher-level, namespaced, token-efficient, errors as instructions); read vs. write tools | **1.3 Rafeeq's first tools**: implement `search_policies`, `get_permit`, `compute_fee`, `list_slots`; run one tool-call round trip by hand; improve a bad tool description and measure the difference |
| 4 | **The agent loop from scratch** | The loop in 40 lines; stop conditions, step and cost budgets; errors as observations; traces; why most "agent frameworks" are this loop plus extras | **1.4 Build `run_agent()`**: the loop with budgets and a trace; answer multi-step questions ("what will I pay and when is it due?"); break it on purpose (tool error, infinite loop) and make it recover |

## Day 2 · Context Engineering

| # | Session | Concepts | Lab (25') |
|---|---|---|---|
| 1 | **Context is a budget** | Context windows vs. effective context; context rot; the four failures (poisoning, distraction, confusion, clash); write/select/compress/isolate; KV-cache and prompt caching; stable prefixes | **2.1 Budget the window**: measure what fills Rafeeq's context; cap and summarise tool results; compaction when the history crosses a threshold; cache hits with a stable prefix |
| 2 | **Retrieval for agents** | RAG as a tool vs. RAG as a pipeline; agentic search (grep/glob, read on demand) vs. embeddings; hybrid search; Arabic retrieval; citing sources | **2.2 Search the policies**: BM25 and multilingual embeddings (HF Hub model) over bilingual policies; agentic search tools (`list`, `grep`, `read`); compare answer quality and tokens; answers cite policy IDs |
| 3 | **Memory, skills and context files** | Session vs. long-term memory; memory files with caps (Hermes); AGENTS.md; Agent Skills (SKILL.md) and progressive disclosure; just-in-time loading | **2.3 Remember and specialise**: a capped memory file the agent edits; load AGENTS.md; a `refund-request` skill loaded only when needed; measure tokens saved by progressive disclosure |
| 4 | **Model Context Protocol** | MCP architecture (host, client, server); tools, resources, prompts; the 2026-07-28 stateless revision; stdio vs. Streamable HTTP; the MCP SDK v2 and FastMCP; HF MCP server; registry and supply-chain risk | **2.4 Madar as an MCP server**: a PEP 723 MCP server (`uv run`) exposing Madar tools and policies as resources; connect it to your loop; list and call tools; inspect messages |

## Day 3 · Harness & Loop Engineering

| # | Session | Concepts | Lab (25') |
|---|---|---|---|
| 1 | **Anatomy of a harness** | Agent = model + harness; guides and sensors; system prompt, tools, context files, sessions, compaction, permissions, hooks, sandbox, sub-agents; a teardown of **Pi, OpenClaw, Hermes and Mercury** (+ Claude Code, Codex, OpenCode) | **3.1 Harness teardown**: run Pi on OpenRouter in print/JSON mode on a Madar task; read its event stream, system prompt and tools; fill a comparison matrix for four harnesses; map each part to your own loop |
| 2 | **Build the harness** | Permission policies (allow/ask/deny); human approval for write tools; pre/post tool hooks; sandboxed code execution (dependency vs. security isolation); sub-agents with isolated context | **3.2 Harden Rafeeq**: a permission policy and approval prompt; hooks that log and block; a sandboxed `run_python` tool via `uv run`; a research sub-agent that returns a summary |
| 3 | **Loop engineering** | Long-running agents; the Ralph loop; fresh context per iteration; progress files and feature lists; tests as back-pressure; maker vs. checker; budgets and no-progress detection; git checkpoints | **3.3 A loop that fixes code**: an outer loop drives a coding agent on `madar_fees` until the test suite passes; progress file; iteration and cost limits; inspect the git history it leaves |
| 4 | **Evaluating agents** | Eval-driven development; task suites; code graders, state checks, LLM judges; pass@k vs. pass^k; trajectories; error analysis; tracing with OpenTelemetry GenAI conventions | **3.4 Score Rafeeq**: run 30 tasks; grade answers, database end states and refusals; compare two models and two prompts; read the failing traces and fix one root cause |

## Day 4 · Graph Engineering & Production

| # | Session | Concepts | Lab (25') |
|---|---|---|---|
| 1 | **Graph engineering** | Workflows vs. agents; the five workflow patterns as graphs; typed state; nodes, edges, routers, gates; loops inside nodes; checkpointing, interrupts and resume (durable execution); LangGraph | **4.1 A permit-review graph**: intake → document check → fee check → risk router → human approval (interrupt) → decision letter; checkpoint to SQLite; kill and resume a run |
| 2 | **Multi-agent systems** | When multi-agent helps (breadth, parallel reads) and hurts (coupled writes); orchestrator-workers; handoffs; context isolation; cost; A2A v1.0 agent cards and tasks | **4.2 Orchestrator and workers**: fan out three specialist workers (policy, records, scheduling) and merge; compare quality, tokens and latency with the single agent; publish one worker over A2A and call it |
| 3 | **Securing agents** | OWASP Top 10 for Agentic Applications; prompt injection (direct, indirect); the lethal trifecta; tool and skill supply chain; least privilege, egress control, quarantining untrusted content, approvals; PDPL and SDAIA principles; harness incidents of 2026 | **4.3 Attack and defend**: an injected email makes Rafeeq leak data; add defences (trust labels, tool allowlists per context, approval on outbound actions, egress allowlist); re-run a red-team suite to prove it |
| 4 | **Agents in production + capstone kickoff** | Serving agents (API, streaming, sessions); observability and cost dashboards; model routing, fallbacks and budgets; caching; versioning prompts and tools; rollout and monitoring | **4.4 Ship it**: serve Rafeeq with FastAPI behind an API key; per-session budgets and provider fallbacks; traces to a local collector; a smoke eval in CI; capstone planning |

## Day 5 · Capstone presentations

Each participant (or pair) presents a working agent system: 8 minutes talk + live demo, 4 minutes technical Q&A,
peer feedback (1 strength · 1 question · 1 suggestion), instructor review and closing.

### Capstone options

1. **Rafeeq in production**: extend the Madar assistant into a complete service agent (MCP tools, memory, approvals, security, evals, API).
2. **Your harness**: a domain harness for a real workflow in your entity (coding, analysis, operations), with context files, permissions, a loop and evals.
3. **Graph workflow**: a durable, human-in-the-loop multi-step process (e.g. procurement review, complaint handling) as a graph with agents inside nodes.

### Minimum requirements

1. An agent that completes multi-step tasks with at least **three tools**, one of them exposed or consumed over **MCP**.
2. Deliberate **context engineering**: a context file (AGENTS.md or system spec), one memory or retrieval mechanism, and a compaction or budget rule.
3. A **harness control**: a permission policy with human approval for at least one write action, and step/cost budgets.
4. An **eval suite** of at least 15 tasks with automatic graders, run on two configurations, plus an error analysis.
5. A **security note**: the lethal-trifecta analysis of your system, one attack you tried and the defence that stopped it.

### Rubric

| Criterion | Weight |
|---|---|
| Working system and live demo | 25% |
| Context and harness engineering decisions | 25% |
| Evaluation and error analysis | 20% |
| Security, governance and operability | 20% |
| Presentation and reproducibility | 10% |

## Lab stack

Python 3.12 via `uv` · `openai` client against OpenRouter and Hugging Face Inference Providers · `huggingface_hub` ·
Pydantic · model2vec / sentence-transformers embeddings from the Hub · `mcp` SDK v2 · LangGraph · `a2a-sdk` ·
FastAPI · OpenTelemetry · Pi (Node 22+) for the harness teardown. All LLM calls go to **free** API models (an OpenRouter
key; one-time 10-credit top-up recommended to raise the free quota from 50 to 1,000 requests a day); nothing runs a
local LLM unless a participant opts into the small Qwen/Gemma experiments.
Labs are Jupyter notebooks that run locally with `uv` or on Google Colab; standalone programs (MCP servers, the loop
runner, the capstone starter) are single-file **PEP 723** scripts run with `uv run`. Everything runs on a laptop CPU.
