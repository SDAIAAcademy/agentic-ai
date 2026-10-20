# <Project name>

> One sentence: who uses this agent, and what does it do for them?

**Track:** 1 · Rafeeq in production / 2 · Your harness / 3 · Graph workflow
**Team:** <names>
**Model(s):** <e.g. google/gemma-4-31b-it:free via OpenRouter>, chosen because <benchmark evidence + your own eval>

## Problem

<The workflow today, who does it, what goes wrong, and what "done" looks like. 5–8 lines.>

## How to run

```bash
uv run <your_script>.py ask "<an example request>"
uv run <your_script>.py eval tasks.jsonl --config A B
```

Environment: `OPENROUTER_API_KEY` in `.env`. <Anything else.>

## Design across the ladder

| Rung | Your decision | Why |
|---|---|---|
| Prompt | <spec file, tool descriptions, output format> | |
| Context | <retrieval / memory / skills, compaction or budget rule> | |
| Harness | <permissions, approvals, hooks, budgets, sandbox> | |
| Loop | <outer loop? stop conditions? or "not needed because …"> | |
| Graph | <explicit workflow? checkpoints? or "not needed because …"> | |

<An architecture diagram: boxes for the model, harness, tools, MCP server(s), data, and the human.>

## Tools

| Tool | Read / write | Over MCP? | Approval? | Trifecta leg (private / untrusted / egress) |
|---|---|---|---|---|
| | | | | |

## Evaluation

**Suite:** <N> tasks in `tasks.jsonl` (<categories>). Graders: <which types and why>.
**Configurations:** A = <baseline>; B = <the one change>.

| Config | pass rate | pass^k (k = ) | avg requests | avg tokens |
|---|---|---|---|---|
| A | | | | |
| B | | | | |

### Error analysis

| Root cause | Count (A) | Count (B) | Example task | Fix tried |
|---|---|---|---|---|
| | | | | |

<What flipped between A and B, in both directions. What you'd change next.>

## Security note

- **Trifecta:** <which legs your system has, and where they meet>
- **Attack tried:** <the input, and what happened without defences>
- **Defence:** <what you added; is it enforced in code or does it depend on the model?>
- **Residual risk:** <what is still possible, and who should know>
- **Governance:** <PDPL / SDAIA AI principles considerations: data minimisation, human oversight, where data goes>

## Limitations and next steps

<Honest list: what doesn't work yet, what you'd need for production.>
