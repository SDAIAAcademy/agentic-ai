# Capstone · Agentic AI Systems Engineering

Day 5 is yours: you present a working agent system that you designed, built, measured and secured. Pick a track on
Day 4, build it that evening, polish it in Day 5's first session, and present.

## Tracks

| Track | You build | Good if you … |
|---|---|---|
| **1 · Rafeeq in production** | the Madar assistant as a complete service: MCP tools, memory, approvals, security, evals, API | want to go deep on one system with all the data ready |
| **2 · Your harness** | a domain harness for a real workflow in your entity (coding, analysis, operations): context files, permissions, a loop, evals | have a workflow at work that an agent could take over |
| **3 · Graph workflow** | a durable, human-in-the-loop process (procurement review, complaint handling, …) as a LangGraph graph with agents inside nodes | care most about reliability and control |

Use synthetic or public data only. No personal data from your entity, no internal documents you aren't allowed to share.

## Minimum requirements

1. An agent that completes multi-step tasks with at least **three tools**, one of them exposed or consumed over **MCP**.
2. Deliberate **context engineering**: a context file (AGENTS.md or system spec), one memory or retrieval mechanism,
   and a compaction or budget rule.
3. A **harness control**: a permission policy with human approval for at least one write action, and step/cost budgets.
4. An **eval suite** of at least **15 tasks** with automatic graders, run on **two configurations**, plus an error analysis.
5. A **security note**: the lethal-trifecta analysis of your system, one attack you tried and the defence that stopped it.

## Rubric

| Criterion | Weight |
|---|---|
| Working system and live demo | 25% |
| Context and harness engineering decisions | 25% |
| Evaluation and error analysis | 20% |
| Security, governance and operability | 20% |
| Presentation and reproducibility | 10% |

## Getting started

```bash
cd labs
uv run capstone/starter.py ask "What is the fee for an illuminated 3 m² sign?"
uv run capstone/starter.py eval capstone/tasks.example.jsonl --config A B
```

`starter.py` already wires requirements 1–4 for Madar: local tools plus the MCP server's tools, a spec as the
system prompt, a session history with compaction, a cap on tool results, allow/ask permissions, step and cost
budgets, an audit log, and the Day 3 eval runner on two configurations. Search it for `CAPSTONE:` and replace those
parts with your own. Tracks 2 and 3 can start from it, or from the lab that is closest:

| You need | Start from |
|---|---|
| an MCP server | `servers/madar_mcp.py` (Lab 2.4) |
| memory or skills | Lab 2.3 (`Memory`, `load_skill`) |
| a loop that works until checks pass | `loops/ralph.py` (Lab 3.3) |
| an SDK with approvals and guardrails | Lab 3.2 (OpenAI Agents SDK) |
| a graph with a human gate | Lab 4.1 (LangGraph) |
| security defences and a red-team suite | Lab 4.3 |
| an HTTP service with traces | `servers/rafeeq_api.py` (Lab 4.4) |

**Eval tasks** use the format of `data/eval/tasks.jsonl` (see `evals.py` for the grader types). For a domain without a
Madar database, use `contains_any`, `not_contains`, `tool_called`, `tool_not_called` and `judge` graders, or give
your agent an object with a `.sql()` method for `db` graders.

**Free quota.** Two configurations × 15 tasks × ~3 requests ≈ 90 requests: more than the 50 free requests a day.
Buy the 10 OpenRouter credits (1,000 free requests a day), or develop on 5 tasks and run the full suite once. The
response cache makes re-runs of unchanged requests free.

## Timeline

| When | What |
|---|---|
| Day 4, end of Session 4 | pick a track; write the one-paragraph pitch and three example tasks |
| Day 4, evening | copy `TEMPLATE_README.md` to your project as `README.md`; fill in the problem, tools and trifecta; write 15 tasks; build; run the eval on configuration A; analyse failures; change one thing (B); re-run; try one attack |
| Day 5, Session 1 | final prep: re-run both evals, rehearse, record a demo backup, clean-clone test |
| Day 5, Sessions 2–4 | 8 minutes talk + live demo, 4 minutes Q&A, peer feedback |

## Presentation (8 minutes)

1. The problem and the user (1 min).
2. Live demo of one multi-step task, including an approval (2 min).
3. Your design across the ladder: prompt, context, harness, loop, graph. Which rungs did you need and why? (2 min)
4. Eval results for A vs. B, and the biggest failure category (2 min).
5. Your trifecta, the attack and the defence (1 min).

**Have a fallback**: record a short screen capture of the demo in case the free model is rate-limited on stage.

## Submission

A public or shared repository (or a zip) with: your `README.md` from the template, the code, `tasks.jsonl`, the eval
CSVs for both configurations, and your slides. Everything must run with `uv run` (PEP 723) or `uv sync`, and an
OpenRouter key.
